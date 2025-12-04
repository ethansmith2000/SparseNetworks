#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import os
import tempfile
import inspect as _inspect

# Make Triton and TorchInductor use stable, user-writable cache/tmp dirs on the cluster
os.environ.setdefault("TRITON_CACHE_DIR", "/home/ethan/.triton/cache")
os.environ.setdefault("TMPDIR", "/home/ethan/tmp")
tempfile.tempdir = os.environ["TMPDIR"]


def _safe_getsourcelines(obj):
    """
    Work around Triton+Python 3.11 \"source code not available\" issues by
    returning a dummy source snippet instead of raising, so torch.compile
    can proceed. This is only meant for tooling and does not affect numerics.
    """
    try:
        return _inspect._orig_getsourcelines(obj)  # type: ignore[attr-defined]
    except OSError as e:
        if "source code not available" in str(e):
            return ["# source code not available\n"], 0
        raise


if not hasattr(_inspect, "_orig_getsourcelines"):
    _inspect._orig_getsourcelines = _inspect.getsourcelines  # type: ignore[attr-defined]
    _inspect.getsourcelines = _safe_getsourcelines


import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.v8_api_enabled = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # 
torch.backends.cuda.allow_tensor_float_32 = True


# torch._inductor.config.triton.cudagraphs = True   # or False if capture causes overhead
# torch._inductor.config.use_mixed_mm = True        # enables faster matmul codegen
# torch._inductor.config.triton.cudagraphs = True
# torch._inductor.config.triton.cudagraph_trees = True  # More aggressive cudagraph usage
# torch._inductor.config.triton.autotune_pointwise = True
# # torch._inductor.config.triton.dense_indexing = True
# torch._inductor.config.triton.max_tiles = 8  # Increase tiling options
# torch._inductor.config.aggressive_fusion = True
# torch._inductor.config.pattern_matcher = True
# torch._inductor.config.permute_fusion = True
# torch._inductor.config.max_autotune = True
# torch._inductor.config.max_autotune_gemm = True

torch.set_num_threads(12)
torch.set_num_interop_threads(2)

# torch._inductor.config.autotune_in_subproc = True            # instead of exporting TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC
# torch._inductor.config.autotune_multi_device = True          # mirrors TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE
# # torch._inductor.config.max_autotune_gemm_search_space = "EXHAUSTIVE"



import argparse
import copy
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version
from types import SimpleNamespace

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import time

from models.sparse import SparseLinear
from param_tracker import ParameterTracker

from transformers.models.gpt2 import modeling_gpt2 as gpt2_modeling
from transformers import modeling_utils as hf_modeling_utils

import torch._dynamo as torch_dynamo

logger = get_logger(__name__)

# Work around torch.compile limitation: tell torch._dynamo which *callables* are safe to ignore,
# including HuggingFace module-level `logger.warning_once` usages in GPT2 and shared modeling_utils.
try:
    if hasattr(torch_dynamo.config, "ignore_logger_methods"):
        current = getattr(torch_dynamo.config, "ignore_logger_methods", set())
        # Start from any existing callable entries, drop non-callables that may have been added earlier.
        safe_methods = {fn for fn in current if callable(fn)}

        # Generic logging.Logger methods (ignore all instances)
        safe_methods.update(
            {
                logging.Logger.debug,
                logging.Logger.info,
                logging.Logger.warning,
                logging.Logger.error,
                logging.Logger.exception,
                logging.Logger.critical,
                logging.Logger.log,
                logging.Logger.warning_once,
            }
        )

        # Specific HF GPT2 logger.warning_once (module-level logger in modeling_gpt2)
        hf_gpt2_logger = getattr(gpt2_modeling, "logger", None)
        if hf_gpt2_logger is not None:
            gpt2_warning_once = getattr(hf_gpt2_logger, "warning_once", None)
            if callable(gpt2_warning_once):
                safe_methods.add(gpt2_warning_once)

        # Shared transformers modeling_utils logger.warning_once (used by many models)
        hf_shared_logger = getattr(hf_modeling_utils, "logger", None)
        if hf_shared_logger is not None:
            shared_warning_once = getattr(hf_shared_logger, "warning_once", None)
            if callable(shared_warning_once):
                safe_methods.add(shared_warning_once)

        torch_dynamo.config.ignore_logger_methods = safe_methods
except Exception:
    # If anything goes wrong, fall back silently – this only impacts compile-mode logging behavior
    pass



def patch_gpt(model, config):
    for n,m in model.named_modules():
        if hasattr(m, "mlp"):
            m.mlp.c_fc = SparseLinear(full_in_dim=config.hidden_size, full_out_dim=config.hidden_size*4, **config.sparse_kwargs_up, bias=True)
            m.mlp.c_proj = SparseLinear(full_in_dim=config.hidden_size*4, full_out_dim=config.hidden_size, **config.sparse_kwargs_down, bias=True)
        if hasattr(m, "attn"):
            m.attn.c_attn = SparseLinear(full_in_dim=config.hidden_size, full_out_dim=config.hidden_size*3, **config.sparse_kwargs_qkv, bias=True)
            m.attn.c_proj = SparseLinear(full_in_dim=config.hidden_size, full_out_dim=config.hidden_size, **config.sparse_kwargs_out, bias=True)


def _build_activation_probe_batch(batch, limit):
    """Return a shallow copy of `batch` with tensors truncated along batch dim."""
    if limit is None:
        return batch
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.size(0) > limit:
            sliced[key] = value[:limit]
        else:
            sliced[key] = value
    return sliced


logger = get_logger(__name__)

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--override_json",
        type=str,
        default=None,
        help="Optional path to a JSON file whose keys override the default args dict.",
    )
    cli_args = parser.parse_args()

    args = {
        "num_validation_batches": 25,
        "validate_every": 1000,
        # "dataset_name": "openwebtext",
        # "dataset_config_name": None,
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-103-v1",
        "train_file": None,
        "validation_file": None,
        "validation_split_percentage": 5,
        "model_name_or_path": "openai-community/gpt2-medium",
        # "model_name_or_path": "openai-community/gpt2",
        "config_name": None,
        "tokenizer_name": None,
        "use_slow_tokenizer": False,
        "per_device_train_batch_size": 32,
        "learning_rate": 1.0e-4,
        "weight_decay": 0.01,
        "num_train_epochs": 4,
        "max_train_steps": None,
        "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "linear",
        "num_warmup_steps": 250,
        "seed": 123,
        "model_type": None,
        "block_size": 1024,
        "preprocessing_num_workers": 10,
        "overwrite_cache": False,
        "no_keep_linebreaks": False,
        "trust_remote_code": False,
        "checkpointing_steps": None,
        "resume_from_checkpoint": None,
        "with_tracking": True,
        "report_to": "wandb",
        "low_cpu_mem_usage": False,
        "max_grad_norm": 1.0,
        "hf_path": None,
        "base_output_dir": "model-output",


        "beta1": 0.9,
        "beta2": 0.98,

        "compile": False,
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": True,

        "gradient_checkpointing": True,

        "num_workers": 12,

        "log_params_every_n": 100,
        "activation_sample_limit": 2,
        "activation_capture": "both",
        "activation_param_blacklist": ["bias", "gate"],
        "activation_clear_cuda_cache": True,
        "activation_force_python_gc": False,
        "activation_probe_batch_size": 2,

        # model parameters
        "hidden_size": 2048,
        "depth": 12,
        "n_head": 16,

        # sparse and lora parameters
        "lr_mult_sparse": True,
        "lr_mult_lora": True,
        "lr_mult_gate": True,
        "lora_rank": 128 // 2,
        "sparse_heads": 8,
        "sparse_weight_init_mode": "per_block_xavier",
        # "sparse_weight_init_mode": "global_xavier",
        # "permute_init_mode": "classic",
        "permute_init_mode": "variance_matched",

        "down_permute_in_mode": None,

        "sparse": True,
        "sparse_kwargs_up":dict(
            permute_in_mode="lora", 
            # permute_in_mode=None,
            permute_out_mode="lora", 
            # permute_out_mode=None,
            ),
        "sparse_kwargs_down":dict(
            # permute_in_mode="lora", 
            permute_in_mode=None,
            permute_out_mode="lora", 
            ),
        "sparse_kwargs_qkv":dict(
            permute_in_mode="lora", 
            permute_out_mode="lora", 
            # permute_out_mode=None,
            ),
        "sparse_kwargs_out":dict(
            permute_in_mode="lora", 
            # permute_in_mode=None,
            permute_out_mode="lora", 
            # permute_out_mode=None,
            ),
    }

    # Optional overrides from a JSON file for sweep scripts or manual runs
    if cli_args.override_json is not None:
        with open(cli_args.override_json, "r") as f:
            json_overrides = json.load(f)
        # Shallow override: top-level keys in the JSON replace entries in args
        args.update(json_overrides)

    for key in ("sparse_kwargs_up", "sparse_kwargs_down", "sparse_kwargs_qkv", "sparse_kwargs_out"):
        args[key].setdefault("lr_mult_sparse", args["lr_mult_sparse"])
        args[key].setdefault("lr_mult_lora", args["lr_mult_lora"])
        args[key].setdefault("lr_mult_gate", args["lr_mult_gate"])
        args[key].setdefault("sparse_heads", args["sparse_heads"])
        args[key].setdefault("init_mode", args["sparse_weight_init_mode"])
        args[key].setdefault("rank_in", args["lora_rank"])
        args[key].setdefault("rank_out", args["lora_rank"])
        args[key].setdefault("permute_in_init_mode", args["permute_init_mode"])
        args[key].setdefault("permute_out_init_mode", args["permute_init_mode"])

    # Allow overriding the `permute_in_mode` only for the "down" sparse block
    for key in ("sparse_kwargs_down",):
        args[key].setdefault("permute_in_mode", args["down_permute_in_mode"])


    config = AutoConfig.from_pretrained(
        args['model_name_or_path'],
        trust_remote_code=args['trust_remote_code'],
    )
    config.attn_pdrop=0.0
    config.resid_pdrop=0.0
    config.embd_pdrop=0.0
    config.n_embd = args['hidden_size']
    config.n_layer = args['depth']
    config.n_head = args['n_head']

    base_str = f"base-{args['hidden_size']}"
    wandb_run_name = base_str
    if args["sparse"]:
        base_str = f"sparse-{args['hidden_size']}"
        sparse_init = args["sparse_weight_init_mode"]
        permute_init = "var_match" if args["permute_init_mode"] == "variance_matched" else args["permute_init_mode"]
        sparse_heads = args["sparse_heads"]
        lora_rank = args["lora_rank"]
        bool_to_tag = lambda flag: "on" if flag else "off"
        sparse_init = "pbx" if sparse_init == "per_block_xavier" else "gx"
        wandb_suffix = "-".join(
            [
                f"wInt-{sparse_init}",
                f"pInt-{permute_init}",
                f"hds-{sparse_heads}",
                f"rnk-{lora_rank}",
                f"dw-{args['down_permute_in_mode']}",
                # f"lrSprs-{bool_to_tag(args['lr_mult_sparse'])}",
                # f"lrLra-{bool_to_tag(args['lr_mult_lora'])}",
                # f"lrGte-{bool_to_tag(args['lr_mult_gate'])}",
            ]
        )
        wandb_run_name = f"{base_str}-{wandb_suffix}"
    args["output_dir"] = f"{args['base_output_dir']}/{base_str}"
    args["wandb_run_name"] = wandb_run_name

    args = SimpleNamespace(**args)

    print("Running with the following arguments:")
    print(json.dumps(vars(args), indent=2))

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.output_dir is None:
        args.output_dir = time.strftime("run_%Y%m%d_%H%M%S")

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, 
                                                            mixed_precision="bf16",
                                                            **accelerator_log_kwargs)

    param_tracker = ParameterTracker(
        activation_sample_limit=args.activation_sample_limit,
        activation_capture=args.activation_capture,
        activation_param_blacklist=args.activation_param_blacklist,
        activation_clear_cuda_cache=args.activation_clear_cuda_cache,
        activation_force_python_gc=args.activation_force_python_gc,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.hf_path)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.hf_path
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.hf_path
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=args.hf_path, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.hf_path
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.hf_path
                **dataset_args,
            )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )
    # gpt_conf = GPTConfig(
    #     vocab_size=len(tokenizer),
    #     n_layer=args.depth,
    #     n_head=args.n_head,
    #     n_embd=args.hidden_size,
    #     block_size=args.block_size,
    #     gradient_checkpointing=args.gradient_checkpointing,
    # )
    # model = GPT(gpt_conf)

    model = AutoModelForCausalLM.from_config(
        config,
        # args.model_name_or_path,
        # from_tf=bool(".ckpt" in args.model_name_or_path),
        # config=config,
        # low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
    )

    model.gradient_checkpointing_enable()

    if args.sparse:
        patch_gpt(model, args)

    print(model)


    print("num parameters", sum(p.numel() for p in model.parameters()))
    activation_probe_model = copy.deepcopy(model)
    for param in activation_probe_model.parameters():
        param.requires_grad_(False)
    model = model.to(accelerator.device)
    activation_probe_model = activation_probe_model.to(accelerator.device)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.transformer.wte.weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("resizing token embeddings", len(tokenizer), embedding_size)
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers, pin_memory=True
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_mult = getattr(param, "lr_mult", 1.0)
        optimizer_grouped_parameters.append(
            {
                "params": [param],
                "weight_decay": 0.0 if any(nd in name for nd in no_decay) else args.weight_decay,
                "lr": args.learning_rate * lr_mult,
            }
        )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1, args.beta2), fused=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # compile
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, fullgraph=args.compile_fullgraph)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        init_kwargs = {
            "wandb": {
                "name": args.wandb_run_name,
            }
        }
        accelerator.init_trackers("clm_no_trainer", experiment_config, init_kwargs=init_kwargs)
        # accelerator.init_trackers("sparse_gpt", experiment_config, init_kwargs=init_kwargs)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)


    # allocated and reserved memory
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    progress_bar.set_postfix(vram=f"{reserved_memory / (1024 ** 3):.2f} GB")

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    data_start = torch.cuda.Event(enable_timing=True)
    data_end = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        
        dataloader_iter = iter(active_dataloader)
        for step in range(len(active_dataloader)):
            model.train()
            
            # Time data loading
            data_start.record()
            batch = next(dataloader_iter)
            data_end.record()
            
            with accelerator.accumulate(model):
                should_profile_stats = (completed_steps % args.log_params_every_n == 0)
                
                if should_profile_stats:
                    state_dict = accelerator.unwrap_model(model).state_dict()
                    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                    activation_probe_model.load_state_dict(
                        state_dict
                    )
                    activation_batch = _build_activation_probe_batch(
                        batch, args.activation_probe_batch_size
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    activation_probe_model.to(accelerator.device, non_blocking=True)
                    activation_probe_model.eval()
                    activation_context = param_tracker.activation_capture_context(activation_probe_model)
                    with torch.inference_mode(), accelerator.autocast():
                        with activation_context:
                            _ = activation_probe_model(**activation_batch)
                    # activation_probe_model.to("cpu", non_blocking=True)
                    del activation_batch
                    # garbage collect
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                
                # Time forward pass
                forward_start.record()
                # logits, loss = model(idx=batch["input_ids"], targets=batch["labels"])
                outputs = model(**batch)
                loss = outputs.loss
                forward_end.record()
                
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                
                # Time backward pass
                backward_start.record()
                accelerator.backward(loss)
                backward_end.record()
                
                # Synchronize to get accurate timings
                torch.cuda.synchronize()
                
                # Calculate elapsed times in milliseconds
                data_time = data_start.elapsed_time(data_end)
                forward_time = forward_start.elapsed_time(forward_end)
                backward_time = backward_start.elapsed_time(backward_end)
                
                # clip the gradients
                mini_logs ={
                        "step_loss": loss.detach().float(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "timer/data_load_ms": data_time,
                        "timer/forward_ms": forward_time,
                        "timer/backward_ms": backward_time,
                    }


                if should_profile_stats:
                    param_tracker.update(model, completed_steps)
                    param_tracker.log_parameter_stats_to_wandb(accelerator, completed_steps)

                if args.max_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    mini_logs["grad_norm"] = grad_norm
                
                # Time optimizer step
                optimizer_start.record()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_end.record()
                torch.cuda.synchronize()
                
                optimizer_time = optimizer_start.elapsed_time(optimizer_end)
                mini_logs["timer/optimizer_ms"] = optimizer_time
                
                accelerator.log(
                        mini_logs,
                        step=completed_steps,
                    )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

            
            if completed_steps % args.validate_every == 0:
                model.eval()
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        # logits, loss = model(idx=batch["input_ids"], targets=batch["labels"])
                        outputs = model(**batch)
                        loss = outputs.loss

                    losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_train_batch_size)))
                    if args.num_validation_batches is not None:
                        if step >= args.num_validation_batches:
                            break

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

                if args.with_tracking:
                    accelerator.log(
                        {
                            "perplexity": perplexity,
                            "eval_loss": eval_loss,
                            "train_loss": total_loss.item() / len(train_dataloader),
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        print("Saving model to", args.output_dir)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
