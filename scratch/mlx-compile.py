import time
import mlx.core as mx
import mlx.nn as nn

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

    def __call__(self, x):
        return self.layers(x)

# print(torch._dynamo.list_backends(None))

print('compiling...')
mx.set_default_device(mx.gpu)
model = (Model(2048, 1024, 10))
print('compiled')

input = mx.random.uniform(shape=(2048,))
start = time.time()
for _ in range(10000):
    x = model(input)
end = time.time()

print(f'{end-start} s')
