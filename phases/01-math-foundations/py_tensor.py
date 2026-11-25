import torch
x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

score = torch.dot(x, y)
score.backward()
print(x.grad)