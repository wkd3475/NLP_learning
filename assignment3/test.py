import torch

v = torch.Tensor([1, 0])
x = torch.Tensor([1, 1])

v_norm = (v*v).sum(0)**0.5
x_norm = (x*x).sum(0)**0.5

sim = torch.dot(v, x) / (v_norm * x_norm)

print(sim)
