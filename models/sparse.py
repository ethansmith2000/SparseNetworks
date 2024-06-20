import torch 
from torch import nn
import torch.nn.functional as F

class PermuteIn(nn.Module):

    def __init__(self, 
                full_dim,
                heads,
                mode="chunk_random", # random, roll, chunk_random
                roll=0.4,
                chunks=4, # must divide the chunk dim evenly
                ):
        super().__init__()
        dim = full_dim // heads
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
    
    def __init__(self, full_in_dim=1024, full_out_dim=1024, heads=8, bias=True):
        super(SparseLinear, self).__init__()
        self.full_in = full_in_dim
        self.full_out = full_out_dim
        self.in_dim = full_in_dim // heads
        self.out_dim = full_out_dim // heads
        self.h = heads
        weights = [torch.randn(self.in_dim, self.out_dim) for _ in range(heads)]
        for i in range(len(weights)):
            torch.nn.init.xavier_uniform_(weights[i], gain=torch.nn.init.calculate_gain('relu'))
        self.weight = nn.Parameter(torch.stack(weights, dim=0))
        self.bias_add = BiasAdd(self.full_out) if bias else nn.Identity()

    def forward(self, x):
        b, h, in_dim = x.shape[0], self.h, self.in_dim
        x = x.reshape(b, h, in_dim)
        x = torch.einsum('bhd,hdl->bhl', x, self.weight)
        x = x.reshape(b, h * self.out_dim)
        x = self.bias_add(x)
        return x



class SparseMLP(nn.Module):

    def __init__(self, full_dim=1024, 
                        heads=8, 
                        act=nn.GELU, 
                        full_mlp_dim=4096, 
                        unperm=True, 
                        dropout=0., 
                        permute_mode="chunk_random", # ["random", "roll", "chunk_random", "linear"]
                        ):
        super(SparseMLP, self).__init__()
        self.up = SparseLinear(full_dim, full_mlp_dim, heads)
        self.down = SparseLinear(full_mlp_dim, full_dim, heads)
        self.act = act()

        self.unperm = nn.Identity()
        if permute_mode != "linear":
            self.perm = PermuteIn(full_dim, heads, mode=permute_mode)
            if unperm:
                self.unperm = Unpermute(self.perm.permute)
        else:
            self.perm = nn.Linear(full_dim, full_dim)
            if unperm:
                self.unperm = nn.Linear(full_dim, full_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.perm(x) # reorder features to have different interactions
        x = self.up(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.down(x)
        x = self.dropout(x)
        x = self.unperm(x)

        return x


class SparseFeedForward(SparseMLP):

    def forward(self, x):
        b, toks, d = x.shape
        x = x.reshape(b * toks, d)
        x = super().forward(x)
        x = x.reshape(b, toks, d)
        return x
