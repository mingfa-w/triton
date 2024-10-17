import torch

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(x)
    return a + b
opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))