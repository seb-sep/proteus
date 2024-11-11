import torch
import torch.nn as nn
from proteus.proteus import proteus
import torch.nn.functional as F


class TestModule(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, h_dim, bias=False)
        self.fc2 = nn.Linear(h_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = torch.triu(x)
        if x[0][0] == 0:
            return x
        x = self.fc2(x)
        x = torch.sin(x)
        return x


in_dim, h_dim, out_dim = 4092, 2048, 256
model = TestModule(in_dim, h_dim, out_dim)
test_input = torch.rand((16, in_dim))

for _ in range(1000):
    _ = model(test_input)

model = proteus(model)
_ = model(test_input)
