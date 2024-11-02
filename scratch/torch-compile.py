import time
import torch
from typing import List
import torch.nn as nn
import torch.fx as fx

class Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim),
            nn.Softmax()
        )


    def forward(self, x):
        return self.layers(x)

# print(torch._dynamo.list_backends(None))
def print_backend(graph: fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(type(graph))
    graph.graph.print_tabular()
    graph.print_readable()
    print(graph.graph)
    return graph



print('compiling...')
device='cpu'
model = Model(2048, 1024, 10).to(device)
input = torch.rand((2048,), device=device)
# model = torch.compile(model, backend=print_backend)
print('compiled')
print(fx.symbolic_trace(model, concrete_args={}))

# with torch.inference_mode():
#     start = time.time()
#     for _ in range(1):
#         x = model(input)
#     end = time.time()

# print(f'{end-start} s')
