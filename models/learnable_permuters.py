import math

import torch
from torch import nn


def _sinkhorn(logits: torch.Tensor, num_iters: int = 7) -> torch.Tensor:
    """Project logits onto the Birkhoff polytope with Sinkhorn iterations."""
    log_p = logits
    for _ in range(num_iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return log_p.exp()


def _sample_gumbel(shape, device, dtype):
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u + 1e-9) + 1e-9)


class ButterflyPermuter(nn.Module):
    """
    Structured O(d log d) parameterization that composes learnable 2x2 rotations.
    Matches a standard butterfly factorization; dim must be a power of two.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim & (dim - 1):
            raise ValueError("Butterfly permutation requires a power-of-two dimension.")
        self.dim = dim
        self.num_stages = int(math.log2(dim))
        self.angles = nn.Parameter(torch.zeros(self.num_stages, dim // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        y = x.reshape(-1, self.dim)
        for stage in range(self.num_stages):
            span = 2 ** (stage + 1)
            half = span // 2
            y = y.reshape(-1, self.dim // span, span)
            left, right = y[..., :half], y[..., half:]
            theta = self.angles[stage].reshape(1, self.dim // span, half)
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            left_new = cos_t * left + sin_t * right
            right_new = -sin_t * left + cos_t * right
            y = torch.cat([left_new, right_new], dim=-1)
            y = y.reshape(-1, self.dim)
        return y.reshape(orig_shape)


class BlockShufflePermuter(nn.Module):
    """
    Lightweight alternative: learn local mixings inside chunks and chunk shuffles.
    Uses Sinkhorn on a chunk-level assignment plus per-chunk doubly-stochastic maps.
    """

    def __init__(self, dim: int, chunks: int = 8, temperature: float = 0.15, sinkhorn_iters: int = 5):
        super().__init__()
        if dim % chunks != 0:
            raise ValueError("chunks must divide the dimension evenly.")
        self.dim = dim
        self.chunks = chunks
        self.chunk_size = dim // chunks
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.chunk_logits = nn.Parameter(torch.randn(chunks, chunks) * 0.01)
        self.intra_logits = nn.Parameter(torch.randn(chunks, self.chunk_size, self.chunk_size) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        y = x.reshape(-1, self.chunks, self.chunk_size)
        chunk_perm = _sinkhorn(self.chunk_logits / self.temperature, num_iters=self.sinkhorn_iters)
        y = torch.einsum('ij,bjd->bid', chunk_perm, y)
        intra_perm = _sinkhorn(self.intra_logits / self.temperature, num_iters=self.sinkhorn_iters)
        perm_t = intra_perm.transpose(-1, -2).unsqueeze(0)
        y = torch.matmul(y.unsqueeze(2), perm_t).squeeze(2)
        return y.reshape(orig_shape)



def demo(batch: int = 2, dim: int = 32):
    torch.manual_seed(0)
    x = torch.randn(batch, dim)
    modules = {
        "butterfly": ButterflyPermuter(dim),
        "block_shuffle": BlockShufflePermuter(dim, chunks=8),
    }
    for name, module in modules.items():
        y = module(x)
        print(f"{name:>16} | mean={y.mean():+.4f} std={y.std():+.4f}")


if __name__ == "__main__":
    demo()

