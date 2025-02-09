import torch
import torch.nn as nn
import torch.nn.functional as F

import mlx
import mlx.nn
import mlx.core as mx


class SimpleModule(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dtype=torch.float32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, h_dim, bias=False, dtype=dtype),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim, bias=False, dtype=dtype),
        )

    def forward(self, x):
        return self.layers(x)


class IndexCopyModule(nn.Module):
    def __init__(self, i, j, dtype=torch.float32):
        super().__init__()
        # along zeroth dim, copy a single ones row into the value at the zeroth dim (a row)
        self.register_buffer("copy_tensor", torch.ones((1, j), dtype=dtype))
        # copy along zeroth dim
        self.register_buffer("indices", torch.tensor([0]))

    def forward(self, x):
        # Copy half of static_buffer into output at specified indices
        return x.index_copy_(0, self.indices, self.copy_tensor)


class EmbeddingModule(nn.Module):
    def __init__(
        self, vocab_size=1000, embedding_dim=64, hidden_dim=32, dtype=torch.float32
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, dtype=dtype)
        self.fc = nn.Linear(embedding_dim, hidden_dim, dtype=dtype)
        self.out = nn.Linear(hidden_dim, vocab_size, dtype=dtype)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = F.relu(self.fc(x))  # (batch_size, seq_len, hidden_dim)
        x = self.out(x)  # (batch_size, seq_len, vocab_size)
        return x


class TestModule(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dtype=torch.float32):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, h_dim, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(h_dim, out_dim, bias=False, dtype=dtype)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = torch.triu(x)
        x = self.fc2(x)
        x = torch.sin(x)
        return x


class TestGraphBreakModule(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dtype=torch.float32):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, h_dim, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(h_dim, out_dim, bias=False, dtype=dtype)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.silu(x)

        # Add data dependent branching based on mean value
        mean_val = torch.mean(x)
        if mean_val == 0:
            x = torch.triu(x)
        else:
            x = torch.tril(x)
        x = self.fc2(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)

        # Self attention block
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feedforward block
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


def cool_mlx_fn(primals_1, primals_2, primals_3):
    t = mx.transpose(primals_1)
    mm = mx.matmul(primals_2, t)
    silu = mlx.nn.silu(mm)
    triu = mx.triu(silu)
    t_1 = mx.transpose(primals_3)
    mm_1 = mx.matmul(triu, t_1)
    sin = mx.sin(mm_1)
    return (mx.eval(sin),)
