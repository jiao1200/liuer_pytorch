#1 数据准备
import torch
import numpy as np

xy = np.loadtxt(r'datasets\diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])
# print(y_data)# 默认直接调用__str__

#2 用类构建模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.ReLU()
        self.sigmiod = torch.nn.Sigmoid()
        # self.sigmoid = torch.nn.functional.sigmoid() 不可以这样写，这不是对象而是一个函数

    def forward(self, x):
        x = self.activate(self.linear1(x))#旧的tensor仍然被grad引用着构建计算图
        x = self.activate(self.linear2(x))
        x = self.sigmiod(self.linear3(x))
        return x

model = Model()

#3 损失和优化器 
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

#4 训练循环
for epoch in range(100):
    #Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    #Backward
    optimizer.zero_grad()
    loss.backward()

    #Update
    optimizer.step()
