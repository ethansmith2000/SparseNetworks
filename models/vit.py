# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .sparse import SparseLinear

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.,
        # sparse_heads=8,
        # permute_in_mode="chunk_random",
        # roll_in=0.4,
        # chunks_in=4,
        # rank_in=16,
        # permute_out_mode="chunk_random",
        # roll_out=0.4,
        # chunks_out=4,
        # rank_out=16
        sparse_kwargs_up=None,
        sparse_kwargs_down=None,
    ):
        super().__init__()

        if sparse_kwargs_up is None:
            fc1 = nn.Linear(dim, hidden_dim)
        else:
            fc1 = SparseLinear(full_in_dim=dim, full_out_dim=hidden_dim, **sparse_kwargs_up)

        if sparse_kwargs_down is None:
            fc2 = nn.Linear(hidden_dim, dim)
        else:
            fc2 = SparseLinear(full_in_dim=hidden_dim, full_out_dim=dim, **sparse_kwargs_down)

        self.net = nn.Sequential(
            fc1,
            nn.GELU(),
            nn.Dropout(dropout),
            fc2,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0. , sparse_kwargs_qkv=None, sparse_kwargs_out=None):
        super().__init__()
        inner_dim = dim
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) if sparse_kwargs_qkv is None else SparseLinear(full_in_dim=dim, full_out_dim=inner_dim * 3, **sparse_kwargs_qkv)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim) if sparse_kwargs_out is None else SparseLinear(full_in_dim=inner_dim, full_out_dim=dim, **sparse_kwargs_out),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # Convert qkv from [b, n, h*d] to [b, n, h, d]
        q, k, v = [t.reshape(b, n, self.heads, -1).transpose(1, 2) for t in qkv]  # [b, h, n, d]
        # Use PyTorch's scaled dot-product attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0 if not self.training else self.to_out[1].p if isinstance(self.to_out, nn.Sequential) else 0.0
        )
        # out: [b, h, n, d] -> [b, n, h*d]
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0., sparse_kwargs_up=None, sparse_kwargs_down=None, sparse_kwargs_qkv=None, sparse_kwargs_out=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            mlp = FeedForward(dim, mlp_dim, dropout, sparse_kwargs_up=sparse_kwargs_up, sparse_kwargs_down=sparse_kwargs_down)
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, sparse_kwargs_qkv=sparse_kwargs_qkv, sparse_kwargs_out=sparse_kwargs_out)),
                PreNorm(dim, mlp)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, 
                    image_size, 
                    patch_size, 
                    num_classes, 
                    dim, 
                    depth, 
                    heads, 
                    mlp_mult, 
                    pool = 'cls', 
                    channels = 3, 
                    dropout = 0., 
                    emb_dropout = 0.,
                    sparse_kwargs_up=None,
                    sparse_kwargs_down=None,
                    sparse_kwargs_qkv=None,
                    sparse_kwargs_out=None,
                    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        mlp_dim = dim * mlp_mult
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, sparse_kwargs_up=sparse_kwargs_up, sparse_kwargs_down=sparse_kwargs_down, sparse_kwargs_qkv=sparse_kwargs_qkv, sparse_kwargs_out=sparse_kwargs_out)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def get_feat(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x

    def forward(self, img):
        x = self.get_feat(img)
        return self.mlp_head(x)
