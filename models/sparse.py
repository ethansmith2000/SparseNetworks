import math
from typing import Optional, Sequence

import torch 
from torch import nn
import torch.nn.functional as F


class LoRAPermuter(nn.Module):
    """
    LoRAPermuter
    """
    def __init__(
        self,
        dim,
        rank,
        lora_alpha: Optional[float] = None,
        init_weights: Optional[dict] = None,
        invert_weights: bool = False,
        inverse_strategy: str = "woodbury",
        inverse_eps: float = 1e-6,
        lora_init_mode: str = "variance_matched",
    ):
        super().__init__()
        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)
        self.rank = rank
        self.lr_mult = math.sqrt(dim / rank)
        setattr(self.lora_down.weight, "lr_mult", self.lr_mult)
        setattr(self.lora_up.weight, "lr_mult", self.lr_mult)

        if lora_init_mode == "classic":
            nn.init.normal_(self.lora_down.weight, mean=0.0, std=1.0 / self.rank)
            nn.init.zeros_(self.lora_up.weight)
        elif lora_init_mode == "variance_matched":
            # Match the variance of a dense Linear(dim, dim) initialized with Xavier.
            # For W = lora_up @ lora_down we need rank * Var(lora_up) * Var(lora_down) = 1 / dim.
            target_var = 1.0 / dim  # same as Xavier for square weight matrices
            down_std = 1.0 / math.sqrt(dim)  # keep down-projection well-scaled
            nn.init.normal_(self.lora_down.weight, mean=0.0, std=down_std)
            up_std = math.sqrt(target_var / (self.rank * down_std ** 2))
            nn.init.normal_(self.lora_up.weight, mean=0.0, std=up_std)
        else:
            raise ValueError(f"Unknown lora_init_mode '{lora_init_mode}'")

        if self.lora_up.bias is not None:
            self.lora_up.bias.data.zero_()
        if self.lora_down.bias is not None:
            self.lora_down.bias.data.zero_()

        self.gate_x = nn.Parameter(torch.ones(1) * 0.85)
        self.gate_lora = nn.Parameter(torch.ones(1) * 0.15)

        setattr(self.gate_x, "lr_mult", 10.0)
        setattr(self.gate_lora, "lr_mult", 10.0)

        if init_weights is not None:
            self._load_custom_weights(
                init_weights,
                invert_weights=invert_weights,
                inverse_strategy=inverse_strategy,
                inverse_eps=inverse_eps,
            )
        elif invert_weights:
            raise ValueError("invert_weights=True requires init_weights to be provided.")

        self.forward_op = self.forward_var_match if lora_init_mode == "variance_matched" else self.forward_classic

    def forward_var_match(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_up(self.lora_down(x)) * self.gate_lora + x * self.gate_x

    def forward_classic(self, x: torch.Tensor):
        return self.lora_up(self.lora_down(x)) + x

    def forward(self, x: torch.Tensor):
        return self.forward_op(x)

    def _load_custom_weights(
        self,
        weights_cfg,
        *,
        invert_weights: bool,
        inverse_strategy: str,
        inverse_eps: float,
    ) -> None:
        """
        Load external weights into the LoRA stack.
        weights_cfg can be a dict with tensors/arrays or another LoRAPermuter.
        """
        if isinstance(weights_cfg, LoRAPermuter):
            weights = {
                "down": weights_cfg.lora_down.weight.data.detach().clone(),
                "up": weights_cfg.lora_up.weight.data.detach().clone(),
                "gate1": weights_cfg.gate1.data.detach().clone(),
                "gate2": weights_cfg.gate2.data.detach().clone(),
                "scaling": weights_cfg.scaling,
            }
        else:
            weights = weights_cfg

        required_keys = ("down", "up")
        for key in required_keys:
            if key not in weights:
                raise ValueError(f"Custom LoRA weights require '{key}' to be provided.")

        device = self.lora_down.weight.device
        dtype = self.lora_down.weight.dtype

        def _as_tensor(value, shape):
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            tensor = tensor.to(device=device, dtype=dtype)
            if tensor.shape != shape:
                raise ValueError(f"Expected shape {shape} for custom weight, got {tensor.shape}.")
            return tensor

        down = _as_tensor(weights["down"], self.lora_down.weight.shape)
        up = _as_tensor(weights["up"], self.lora_up.weight.shape)

        if invert_weights:
            up, down = self._compute_inverse_weights(
                up,
                down,
                strategy=inverse_strategy,
                eps=inverse_eps,
            )
            # Ensure the residual path stays identity when acting as an inverse.
            self.gate1.data.fill_(1.0)
            self.gate2.data.fill_(1.0)
            self.scaling = 1.0

        self.lora_down.weight.data.copy_(down)
        self.lora_up.weight.data.copy_(up)

        if "gate1" in weights:
            self.gate1.data.copy_(_as_tensor(weights["gate1"], self.gate1.data.shape))
        if "gate2" in weights:
            self.gate2.data.copy_(_as_tensor(weights["gate2"], self.gate2.data.shape))
        if "bias" in weights:
            self.lora_up.bias.data.copy_(_as_tensor(weights["bias"], self.lora_up.bias.data.shape))
        if "scaling" in weights:
            self.scaling = float(weights["scaling"])

    def _compute_inverse_weights(
        self,
        up: torch.Tensor,
        down: torch.Tensor,
        *,
        strategy: str,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produce a low-rank inverse using either a Woodbury-style update or a pseudoinverse.
        Returns (up_tensor, down_tensor) suitable for assigning back into LoRA layers.
        """
        rank = self.rank
        eye = torch.eye(rank, device=up.device, dtype=up.dtype)

        if strategy == "woodbury":
            vu = down @ up  # (rank x rank)
            mat = eye + vu
            mat_inv = torch.linalg.solve(mat, eye)
            up_inv = -up @ mat_inv
            down_inv = down
        elif strategy == "pseudoinverse":
            gram_v = down @ down.transpose(0, 1) + eps * eye
            gram_u = up.transpose(0, 1) @ up + eps * eye
            gram_v_inv = torch.linalg.solve(gram_v, eye)
            gram_u_inv = torch.linalg.solve(gram_u, eye)
            v_pinv = down.transpose(0, 1) @ gram_v_inv
            u_pinv = gram_u_inv @ up.transpose(0, 1)
            up_inv = v_pinv  # shape (dim, rank)
            down_inv = u_pinv  # shape (rank, dim)
        elif strategy == "svd":
            up_inv, down_inv = self._svd_inverse(up, down, eps=eps)
        else:
            raise ValueError(f"Unknown inverse strategy '{strategy}'.")

        return up_inv.contiguous(), down_inv.contiguous()

    def _svd_inverse(
        self,
        up: torch.Tensor,
        down: torch.Tensor,
        *,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build an inverse by computing a low-rank SVD without materializing the dense matrix.
        """
        q_up, r_up = torch.linalg.qr(up, mode="reduced")
        q_down_t, r_down_t = torch.linalg.qr(down.transpose(0, 1), mode="reduced")
        s_mat = r_up @ r_down_t.transpose(0, 1)
        u_s, sigma, vh_s = torch.linalg.svd(s_mat, full_matrices=False)
        sigma = sigma.clamp_min(eps)
        left_basis = q_up @ u_s
        right_basis = q_down_t @ vh_s.transpose(0, 1)

        sigma_inv_sqrt = sigma.rsqrt()
        up_inv = right_basis * sigma_inv_sqrt.unsqueeze(0)
        down_inv = sigma_inv_sqrt.unsqueeze(1) * left_basis.transpose(0, 1)
        return up_inv, down_inv

class Permute(nn.Module):

    def __init__(self, 
                full_dim,
                sparse_heads,
                mode="chunk_random", # random, roll, chunk_random
                roll=0.4,
                chunks=4, # must divide the chunk dim evenly
                ):
        super().__init__()
        dim = full_dim // sparse_heads
        roll = int(roll * full_dim)
        if mode == "random":
            permute = torch.randperm(full_dim)
        elif mode == "roll":
            permute = torch.roll(torch.arange(full_dim), roll)
        elif mode == "chunk_random":
            assert dim % chunks == 0, "chunks must divide the dim evenly"
            chunk_indices = torch.randperm(full_dim // (dim // chunks))
            permute = torch.cat([torch.arange((dim // chunks)) + i * (dim // chunks) for i in chunk_indices])
        else:
            raise NotImplementedError("mode not implemented")
        self.register_buffer("permute", permute)

    def forward(self, x):
        return x[:, self.permute]


class Unpermute(nn.Module):

    def __init__(self, indices):
        super().__init__()
        perm_matrix = F.one_hot(indices, num_classes=indices.shape[0]).float()
        unperm_matrix = perm_matrix.inverse()
        unperm = unperm_matrix.argmax(dim=-1).long()
        self.register_buffer("unperm", unperm)

    def forward(self, x):
        return x[:, self.unperm]

class BiasAdd(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x + self.bias

class SparseLinear(nn.Module):
    
    def __init__(
        self,
        full_in_dim=1024,
        full_out_dim=1024,
        sparse_heads=8,
        bias=True,
        permute_in_mode="chunk_random",  # random, roll, chunk_random, lora
        roll_in=0.4,
        chunks_in=4,
        rank_in=16,
        permute_out_mode="chunk_random",  # random, roll, chunk_random, lora, unpermute
        roll_out=0.4,
        chunks_out=4,
        rank_out=16,
        init_mode="global_xavier",  # per_block_xavier, global_xavier, masked_kaiming, structure_adaptive
        init_gain=None,
        structure_gains: Optional[Sequence[float]] = None,
        lora_alpha_in: Optional[float] = None,
        lora_alpha_out: Optional[float] = None,
        use_block_gain: bool = False,
        permute_in_init_mode: str = "variance_matched",
        permute_out_init_mode: str = "variance_matched",
    ):
        super(SparseLinear, self).__init__()
        self.full_in = full_in_dim
        self.full_out = full_out_dim
        self.in_dim = full_in_dim // sparse_heads
        self.out_dim = full_out_dim // sparse_heads
        self.h = sparse_heads
        self.weight = nn.Parameter(torch.empty(self.h, self.in_dim, self.out_dim))
        self._init_weight_tensor(init_mode, init_gain, structure_gains)
        self.weight_lr_mult = math.sqrt(self.full_in / self.in_dim)
        setattr(self.weight, "lr_mult", self.weight_lr_mult)
        self.bias_add = BiasAdd(self.full_out) if bias else nn.Identity()

        if permute_in_mode == "lora":
            alpha_in = lora_alpha_in if lora_alpha_in is not None else rank_in
            self.permute_in = LoRAPermuter(
                self.full_in,
                rank_in,
                lora_alpha=alpha_in,
                lora_init_mode=permute_in_init_mode,
            )
        elif permute_in_mode is None:
            self.permute_in = nn.Identity()
        else:
            self.permute_in = Permute(self.in_dim, sparse_heads, mode=permute_in_mode, roll=roll_in, chunks=chunks_in)

        if permute_out_mode == "lora":
            alpha_out = lora_alpha_out if lora_alpha_out is not None else rank_out
            self.permute_out = LoRAPermuter(
                self.full_out,
                rank_out,
                lora_alpha=alpha_out,
                lora_init_mode=permute_out_init_mode,
            )
        elif permute_out_mode == "unpermute":
            self.permute_out = Unpermute(self.permute_in.permute)
        elif permute_out_mode is None:
            self.permute_out = nn.Identity()
        else:
            self.permute_out = Permute(self.out_dim, sparse_heads, mode=permute_out_mode, roll=roll_out, chunks=chunks_out)

    def _init_weight_tensor(
        self,
        init_mode: str,
        init_gain: str,
        structure_gains: Optional[Sequence[float]],
    ) -> None:
        if init_gain is None:
            gain = 1.0
        else:
            gain = torch.nn.init.calculate_gain(init_gain)
        weight = self.weight.data
        if init_mode == "per_block_xavier":
            for head in range(self.h):
                torch.nn.init.xavier_normal_(weight[head], gain=gain)

        elif init_mode == "global_xavier":
            fan_in = self.full_in
            fan_out = self.full_out
            std = gain * math.sqrt(2.0 / (fan_in + fan_out))
            torch.nn.init.normal_(weight, mean=0.0, std=std)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

    def forward(self, x):
        if x.dim() == 2:
            b, full_in = x.shape
            x = x.unsqueeze(1)
            tokens = 1
        elif x.dim() == 3:
            b, tokens, full_in = x.shape
        else:
            raise ValueError("SparseLinear expects input of shape (b, dim) or (b, seq, dim).")

        x = self.permute_in(x)

        x = x.reshape(b * tokens, self.h, self.in_dim)

        x = torch.einsum('bhd,hdl->bhl', x.float(), self.weight.float()).to(x.dtype)

        x = x.reshape(b, tokens, self.h * self.out_dim)
        x = self.permute_out(x)
        x = self.bias_add(x)
        if tokens == 1:
            x = x.squeeze(1)
        return x

