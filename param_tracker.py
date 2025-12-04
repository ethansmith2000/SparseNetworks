"""
Parameter and Gradient Tracking System

Tracks statistics (mean, std, norm) for all parameters and their gradients
over the entire training run, creating individual plots for each parameter.
"""

import gc
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
import math

import torch

class _ActivationRunningStats:
    """Lightweight running stats container for activation tensors."""
    
    def __init__(self, threshold, sample_limit=None):
        self.threshold = threshold
        self.sample_limit = sample_limit
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.near_zero = 0.0
    
    def update(self, tensor):
        if tensor is None:
            return
        tensor = tensor.detach()
        if tensor.is_sparse:
            tensor = tensor.to_dense()
        if self.sample_limit is not None and tensor.ndim > 0:
            tensor = tensor[: self.sample_limit]
        # Move to CPU before computing stats so large activation buffers
        # do not accumulate on GPU and trigger fragmented cache growth.
        # tensor = tensor.to(device="cpu", dtype=torch.float32)
        tensor = tensor.float()
        numel = tensor.numel()
        if numel == 0:
            return
        self.count += numel
        tensor_flat = tensor.view(-1)
        self.sum += tensor_flat.sum().item()
        squared = tensor_flat.square()
        self.sum_sq += squared.sum().item()
        self.near_zero += (tensor_flat.abs() < self.threshold).sum().item()
    
    def summary(self):
        if self.count == 0:
            return None
        mean = self.sum / self.count
        second_moment = self.sum_sq / self.count
        # variance = max(second_moment - mean ** 2, 0.0)
        # std = math.sqrt(variance)
        return {
            # 'mean': mean,
            # 'std': std,
            'norm': math.sqrt(self.sum_sq / self.count),
            # 'zero_frac': self.near_zero / self.count,
        }


class ParameterTracker:
    """
    Tracks parameter and gradient statistics over time.
    
    For each parameter, maintains history of:
    - param_mean, param_std, param_norm
    - grad_mean, grad_std, grad_norm
    - grad_to_param_ratio: |grad| / |param| - shows relative update magnitude
    - grad_snr: |grad_mean| / grad_std - gradient signal-to-noise ratio
    - near_zero_frac: fraction of values close to zero
    - activation_mean/std/norm/near_zero_frac when cached activations are available
    """
    
    def __init__(
        self,
        near_zero_threshold=1e-3,
        activation_sample_limit=None,
        activation_capture="output",
        activation_param_blacklist=None,
        activation_clear_cuda_cache=False,
        activation_force_python_gc=False,
    ):
        self.activation_capture_locations = self._normalize_capture_locations(activation_capture)
        self.activation_stat_bases = (
            # 'mean',
            # 'std',
            'norm',
            # 'zero_frac',
        )
        self.activation_stat_keys = [
            f"{base}_{loc}"
            for loc in self.activation_capture_locations
            for base in self.activation_stat_bases
        ]
        self.activation_sample_limit = activation_sample_limit
        self.activation_param_blacklist = self._normalize_blacklist(activation_param_blacklist)
        self.activation_clear_cuda_cache = activation_clear_cuda_cache
        self.activation_force_python_gc = activation_force_python_gc
        
        def _history_factory():
            entry = {
                'steps': [],
                # 'param_mean': [],
                # 'param_std': [],
                'param_norm': [],
                # 'grad_mean': [],
                # 'grad_std': [],
                'grad_norm': [],
                'grad_to_param_ratio': [],
                'grad_snr': [],
                'grad_zero_frac': [],
                'grad_cosine': [],  # Cosine similarity with previous gradient
            }
            for key in self.activation_stat_keys:
                entry[key] = []
            return entry
        
        # Dictionary mapping param_name -> {stat_name: [values]}
        self.history = defaultdict(_history_factory)
        self.near_zero_threshold = near_zero_threshold
        self.prev_grads = {}  # For computing gradient direction stability
        self.activation_reference = {}

    def _cleanup_activation_pass(self):
        """Optionally release caches after the extra activation probe."""
        if self.activation_force_python_gc:
            gc.collect()
        if self.activation_clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    def _normalize_capture_locations(self, capture):
        """Normalize capture spec into an ordered tuple of valid locations."""
        if isinstance(capture, str):
            capture = capture.lower()
            if capture == "both":
                locations = ("input", "output")
            else:
                locations = (capture,)
        else:
            locations = tuple(loc.lower() for loc in capture)
        valid = {"input", "output"}
        for loc in locations:
            if loc not in valid:
                raise ValueError(f"Invalid activation capture location '{loc}'. Expected 'input', 'output', or 'both'.")
        deduped = []
        for loc in locations:
            if loc not in deduped:
                deduped.append(loc)
        return tuple(deduped)
    
    def _normalize_blacklist(self, patterns):
        """Force blacklist specification into a tuple of substrings."""
        if patterns is None:
            return ()
        if isinstance(patterns, str):
            return (patterns,)
        return tuple(str(p) for p in patterns if p is not None)

    def _should_track_param(self, module_name, param_name):
        full_name = param_name if not module_name else f"{module_name}.{param_name}"
        for pattern in self.activation_param_blacklist:
            if pattern and pattern in full_name:
                return False
        return True
    
    def _append_activation_stats(self, history, stats):
        """Append activation stats (or placeholders) to the history."""
        for loc in self.activation_capture_locations:
            summary = stats.get(loc) if stats is not None else None
            for base in self.activation_stat_bases:
                key = f"{base}_{loc}"
                value = summary.get(base) if summary is not None else None
                history[key].append(value)
    
    def _extract_tensor_from_payload(self, payload):
        """Return the first tensor found inside nested forward hook payloads."""
        if payload is None:
            return None
        if torch.is_tensor(payload):
            return payload
        if isinstance(payload, (tuple, list)):
            for item in payload:
                tensor = self._extract_tensor_from_payload(item)
                if tensor is not None:
                    return tensor
        if isinstance(payload, Mapping):
            for item in payload.values():
                tensor = self._extract_tensor_from_payload(item)
                if tensor is not None:
                    return tensor
        return None
    
    def _make_activation_hook(self, stats_by_location):
        capture_input = 'input' in stats_by_location
        capture_output = 'output' in stats_by_location

        def hook(_module, inputs, outputs):
            if capture_output:
                tensor = self._extract_tensor_from_payload(outputs)
                stats_by_location['output'].update(tensor)
            if capture_input:
                tensor = self._extract_tensor_from_payload(inputs)
                stats_by_location['input'].update(tensor)
        return hook
    
    def _build_activation_reference(self, module_data):
        activation_reference = {}
        for module_name, meta in module_data.items():
            prefix = f"{module_name}." if module_name else ""
            for param_name in meta['param_names']:
                entry_key = f"{prefix}{param_name}"
                for loc, stats in meta['stats'].items():
                    summary = stats.summary()
                    if summary is None:
                        continue
                    activation_reference.setdefault(entry_key, {})[loc] = summary.copy()
        return {name: stats for name, stats in activation_reference.items() if stats}
    
    def activation_capture_context(self, model):
        """
        Create a context manager that registers forward hooks on every module
        containing parameters for the duration of a single forward pass.
        """
        if model is None:
            return nullcontext()
        
        module_data = {}
        handles = []
        for module_name, module in model.named_modules():
            param_names = [name for name, _ in module.named_parameters(recurse=False)]
            if not param_names:
                continue
            tracked_param_names = [
                name for name in param_names
                if self._should_track_param(module_name, name)
            ]
            if not tracked_param_names:
                continue
            stats_by_location = {
                loc: _ActivationRunningStats(
                    threshold=self.near_zero_threshold,
                    sample_limit=self.activation_sample_limit,
                )
                for loc in self.activation_capture_locations
            }
            module_data[module_name] = {
                'param_names': tracked_param_names,
                'stats': stats_by_location,
            }
            handles.append(module.register_forward_hook(self._make_activation_hook(stats_by_location)))
        
        tracker = self
        
        @contextmanager
        def _context():
            tracker.activation_reference = {}
            try:
                yield
            finally:
                for handle in handles:
                    handle.remove()
                tracker.activation_reference = tracker._build_activation_reference(module_data)
                module_data.clear()
                tracker._cleanup_activation_pass()
        
        return _context()
    
    def _candidate_param_names(self, param_name):
        """Generate possible matches accounting for common wrapper prefixes."""
        seen = set()
        def _yield(name):
            if name not in seen:
                seen.add(name)
                yield name
        yield from _yield(param_name)
        prefixes = ('module.', 'model.', 'wrapped_module.', '_orig_mod.')
        for prefix in prefixes:
            if param_name.startswith(prefix):
                yield from _yield(param_name[len(prefix):])
        if '.' in param_name:
            yield from _yield(param_name.split('.', 1)[1])
    
    def _get_activation_stats_for_param(self, param_name):
        for candidate in self._candidate_param_names(param_name):
            stats = self.activation_reference.get(candidate)
            if stats is not None:
                return stats
        return None
    
    def update(self, model, step):
        """
        Collect statistics for all parameters at current step.
        
        Args:
            model: PyTorch model (can be wrapped or unwrapped)
            step: Current training step
        """
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            history = self.history[name]
            history['steps'].append(step)
            
            # Parameter statistics
            # param_mean = param.data.mean().item()
            # param_std = param.data.std().item()
            param_numel = param.data.numel()
            param_norm = param.data.norm().item()
            if param_numel > 0:
                param_norm /= math.sqrt(param_numel)
            
            # history['param_mean'].append(param_mean)
            # history['param_std'].append(param_std)
            history['param_norm'].append(param_norm)
            
            # Fraction of near-zero values
            near_zero = (param.data.abs() < self.near_zero_threshold).float().mean().item()
            history['grad_zero_frac'].append(near_zero)
            
            # Activation statistics sourced from the warmup hook pass (if available)
            activation_stats = self._get_activation_stats_for_param(name)
            self._append_activation_stats(history, activation_stats)
            
            # Gradient statistics
            if param.grad is not None:
                grad_tensor = param.grad.detach().float()
                grad_numel = grad_tensor.numel()
                grad_mean = grad_tensor.mean().item()
                grad_std = grad_tensor.std().item()
                grad_norm = grad_tensor.norm().item()
                if grad_numel > 0:
                    grad_norm /= math.sqrt(grad_numel)
                
                # history['grad_mean'].append(grad_mean)
                # history['grad_std'].append(grad_std)
                history['grad_norm'].append(grad_norm)
                
                # Gradient-to-parameter ratio
                grad_to_param = grad_norm / (param_norm + 1e-8)
                history['grad_to_param_ratio'].append(grad_to_param)
                
                # Signal-to-noise ratio for gradients
                grad_snr = abs(grad_mean) / (grad_std + 1e-8)
                history['grad_snr'].append(grad_snr)
                
                # Gradient direction stability (cosine similarity with previous gradient)
                grad_flat = grad_tensor.view(-1)
                if name in self.prev_grads:
                    prev_grad_flat = self.prev_grads[name]
                    # Compute cosine similarity on CPU to avoid holding GPU buffers
                    cos_sim = torch.nn.functional.cosine_similarity(
                        grad_flat.unsqueeze(0),
                        prev_grad_flat.unsqueeze(0),
                        dim=1,
                        eps=1e-8,
                    ).item()
                    history['grad_cosine'].append(cos_sim)
                else:
                    # First step, no previous gradient
                    history['grad_cosine'].append(1.0)
                
                # Store current gradient for next step (CPU to keep VRAM usage flat)
                self.prev_grads[name] = grad_flat.clone()
                
            else:
                # history['grad_mean'].append(0.0)
                # history['grad_std'].append(0.0)
                history['grad_norm'].append(0.0)
                history['grad_to_param_ratio'].append(0.0)
                history['grad_snr'].append(0.0)
                history['grad_cosine'].append(0.0)
        
        # Avoid reusing stale activation stats on the next step
        self.activation_reference = {}
    
    
    def get_param_names(self):
        """Get list of all tracked parameter names."""
        return list(self.history.keys())
    
    def clear(self):
        """Clear all tracked history."""
        self.history.clear()


    def log_parameter_stats_to_wandb(self, accelerator, step):
        """
        Log parameter and gradient statistics as scalar metrics to wandb.
        This creates time series that can be compared across training runs.
        
        Args:
            tracker: ParameterTracker instance
            wandb_module: wandb module (or None if not available)
            accelerator: Accelerator instance for logging
            step: Current training step
        """
        
        log_dict = {}
        
        # Log all parameter statistics as scalars
        for param_name in self.get_param_names():
            history = self.history[param_name]
            if len(history['steps']) == 0:
                continue
            
            # Get the most recent values
            latest_idx = -1
            
            # Clean up parameter name for wandb (use / for hierarchy)
            clean_name = param_name.replace('.', '/')

            # get rid of _orig_mod/
            clean_name = clean_name.replace('_orig_mod/', '')
            
            # Log parameter statistics
            # log_dict[f'params/{clean_name}/mean'] = history['param_mean'][latest_idx]
            # log_dict[f'params/{clean_name}/std'] = history['param_std'][latest_idx]
            log_dict[f'params/{clean_name}/norm'] = history['param_norm'][latest_idx]
            log_dict[f'params/{clean_name}/zero_frac'] = history['grad_zero_frac'][latest_idx]
            
            # Log gradient statistics
            # log_dict[f'grads/{clean_name}/mean'] = history['grad_mean'][latest_idx]
            # log_dict[f'grads/{clean_name}/std'] = history['grad_std'][latest_idx]
            # log_dict[f'grads/{clean_name}/norm'] = history['grad_norm'][latest_idx]
            log_dict[f'grads/{clean_name}/norm'] = history['grad_norm'][latest_idx]
            log_dict[f'grads/{clean_name}/to_param_ratio'] = history['grad_to_param_ratio'][latest_idx]
            log_dict[f'grads/{clean_name}/snr'] = history['grad_snr'][latest_idx]
            log_dict[f'grads/{clean_name}/cosine'] = history['grad_cosine'][latest_idx]
            
            # Log activation statistics when available
            for metric_name in self.activation_stat_keys:
                value = history[metric_name][latest_idx]
                if value is not None:
                    log_dict[f'activations/{clean_name}/{metric_name}'] = value
        
        if log_dict:
            accelerator.log(log_dict, step=step)


