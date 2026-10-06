import torch

class LinerarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Linerar = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.Linerar(x)
        return y_pred

x_data = torch.tensor([
    [1.],
    [2.],
    [3.]
])
y_data = torch.tensor([
    [2.],
    [4.],
    [6.]
])

model = LinerarModel()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.Linerar.parameters(), lr = 0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w=', model.Linerar.weight.item())
print('b=', model.Linerar.bias.item())

x_test = torch.tensor([[4.]])
y_test = model(x_test)
print('y_pred=', y_test)