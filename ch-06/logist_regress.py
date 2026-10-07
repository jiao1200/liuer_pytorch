import torch
import torch.nn.functional as F

#1 数据准备
#用逻辑回归实现二分类问题
x_data = torch.tensor([[1.], [2.], [3.]])
y_data = torch.tensor([[0.], [0.], [1.]])
#2 用类设计模型,
class BinaryLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linerar = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linerar(x))
        return y_pred
#3 损失函数和优化器
model = BinaryLinearModel()
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
#4 循环训练
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linerar.weight.item())
print('b = ', model.linerar.bias.item())
x_test = torch.tensor([[4.]])
y_test = model(x_test)
print('y_pred=', y_test.item())