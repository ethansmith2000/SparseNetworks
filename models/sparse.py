import torch 
from torch import nn
import torch.nn.functional as F

class PermuteIn(nn.Module):

    def __init__(self, 
                dim,
                heads,
                mode="random", # random, roll, chunk_random
                roll=0.4,
                chunks=4, # must divide the chunk dim evenly
                ):
        super().__init__()
        full_dim = dim * heads
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
    
    def __init__(self, in_dim=64, out_dim=64, heads=8, bias=True):
        super(SparseLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h = heads
        self.weight = nn.Parameter(torch.randn(heads, in_dim, out_dim))
        self.bias_add = BiasAdd(out_dim) if bias else nn.Identity()

    def forward(self, x):
        # b, h * in_dim
        b, h, in_dim = x.shape[0], self.h, self.in_dim
        x = x.reshape(b, h, in_dim).reshape(b, h, -1, in_dim)
        x = torch.vmap(lambda x: torch.bmm(x, self.weight))(x)
        x = x.squeeze(-2).reshape(b, h, * self.od)
        x = self.bias_add(x)
        return x



class SparseMLP(nn.Module):

    def __init__(self, dim=64, 
                        heads=8, 
                        act=nn.GELU, 
                        mlp_dim=256, 
                        unperm=False, 
                        residual=False, 
                        dropout=0., 
                        permute_mode="chunk_random", # ["random", "roll", "chunk_random", "linear"]
                        ):
        super(SparseMLP, self).__init__()
        self.d = dim
        self.h = heads
        self.residual = lambda x, y: x + y if residual else x

        self.up = nn.Parameter(torch.randn(heads, dim, mlp_dim))
        self.down = nn.Parameter(torch.randn(heads, mlp_dim, dim))
        self.act = act()

        self.unperm = nn.Identity()
        if permute_mode != "linear":
            self.perm = PermuteIn(dim, heads, mode=permute_mode)
            if unperm:
                self.unperm = Unpermute(self.perm.permute)
        else:
            self.perm = nn.Linear(dim * heads, dim * heads)
            if unperm:
                self.unperm = nn.Linear(dim * heads, dim * heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        b, h, d = x.shape[0], self.h, self.d

        x = self.perm(x) # reorder features to have different interactions
        x = x.reshape(b, h, d).reshape(b, h, -1, d) # b*h, 1, i bmm is annoying this way
        # x = torch.bmm(x, self.up.repeat(b, 1, 1))
        x = torch.vmap(lambda x: torch.bmm(x, self.up))(x)
        x = self.act(x)
        x = self.dropout(x)
        # x = torch.bmm(x, self.down.repeat(b, 1, 1)) # b*h, 1, d
        x = torch.vmap(lambda x: torch.bmm(x, self.down))(x)
        x = self.dropout(x)
        x = x.squeeze(-2).reshape(b, h * d)
        x = self.unperm(x)

        x = self.residual(x, residual)

        return x


class SparseFeedForward(nn.Module):

    def __init__(self, dim=64, heads=8, act=nn.GELU, mlp_dim=256, perm=True, unperm=True, residual=False, dropout=0.):
        super().__init__()
        self.d = dim
        self.h = heads

        self.residual = lambda x, y: x + y if residual else x

        self.up = nn.Parameter(torch.randn(heads, dim, mlp_dim))
        self.down = nn.Parameter(torch.randn(heads, mlp_dim, dim))
        self.act = act()

        self.unperm = nn.Identity()
        if perm:
            self.perm = PermuteIn(dim, heads)
            if unperm:
                self.unperm = Unpermute(self.perm.permute)
        else:
            self.perm = nn.Linear(dim * heads, dim * heads)
            if unperm:
                self.unperm = nn.Linear(dim * heads, dim * heads)

        self.dropout = nn.Dropout(dropout)

    # def forward(self, x):
    #     x = torch.utils.checkpoint.checkpoint(self.forward_, x)
    #     return x

    def chunk_bmm(self, x, y, chunks=1, total_rep=1):
        chunks = [torch.bmm(x_chunk, y.repeat(total_rep//chunks,1,1)) for x_chunk in x.chunk(chunks, dim=0)]
        return torch.cat(chunks, dim=0)

    def forward(self, x):
        residual = x
        b, l, D = x.shape
        h, d = self.h, self.d

        x = x.reshape(b * l, D) # batch and token combine
        x = self.perm(x) # reorder features to have different interactions
        x = x.reshape(b * l, h, d) 
        x = x.reshape(b * l, h, 1, d) 
        # x = x.reshape(b * l * h, -1, d) # b*h, 1, i bmm is annoying this way
        # x = torch.bmm(x, self.up.repeat(b * l, 1, 1)) # b*h, 1, d*mlp_mult
        x = torch.vmap(lambda x: torch.bmm(x, self.up))(x)
        x = self.act(x)
        x = self.dropout(x)
        # x = torch.bmm(x, self.down.repeat(b * l, 1, 1)) # b*h, 1, d
        x = torch.vmap(lambda x: torch.bmm(x, self.down))(x)
        x = self.dropout(x)
        x = x.squeeze(1).reshape(b * l, h * d)
        x = self.unperm(x)
        x = x.reshape(b, l, D)

        x = self.residual(x, residual)

        return x

    # def forward_(self, x):
    #     residual = x
    #     b, l, D = x.shape
    #     h, d = self.h, self.d

    #     x = x.reshape(b * l, D) # batch and token combine
    #     x = self.perm(x) # reorder features to have different interactions
    #     x = x.reshape(b * l, h, d) 
    #     x = x.reshape(b * l * h, -1, d) # b*h, 1, i bmm is annoying this way
    #     x = torch.bmm(x, self.up.repeat(b * l, 1, 1)) # b*h, 1, d*mlp_mult
    #     # x = self.chunk_bmm(x, self.up, chunks=8, total_rep=b * l)
    #     x = self.act(x)
    #     x = self.dropout(x)
    #     x = torch.bmm(x, self.down.repeat(b * l, 1, 1)) # b*h, 1, d
    #     # x = self.chunk_bmm(x, self.down, chunks=8, total_rep=b * l)
    #     x = self.dropout(x)
    #     x = x.squeeze(1).reshape(b * l, h * d)
    #     x = self.unperm(x)
    #     x = x.reshape(b, l, D)

    #     x = self.residual(x, residual)

    #     return x





# class Network(nn.Module):
#     def __init__(self, input_dim=3, 
#                     output_dim=3, 
#                     dim=64, 
#                     heads=8, 
#                     depth=2, 
#                     act=nn.GELU, 
#                     mlp_mult=4, 
#                     unperm=True, 
#                     residual=True
#                     ):
#         super().__init__()
#         total_dim = dim * heads
#         self.in_proj = nn.Linear(input_dim, total_dim)

#         self.layers = nn.ModuleList([ 
#             SparseMLP(dim=dim, heads=heads, act=act, mlp_mult=mlp_mult, unperm=unperm, residual=residual) for _ in range(depth)
#         ])
#         self.norms = nn.ModuleList([nn.LayerNorm(total_dim) for _ in range(depth)])

#         self.out_norm = nn.LayerNorm(total_dim)
#         self.out_proj = nn.Linear(total_dim, output_dim)

#     def forward(self, x):
#         x = self.in_proj(x)
#         for i, layer in enumerate(self.layers):
#             x = self.norms[i](x)
#             x = layer(x)
#         x = self.out_norm(x)
#         x = self.out_proj(x)
#         return x

