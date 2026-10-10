import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#1 Prepare dataset
#Dataset是一个抽象类，负责定义如何从已有数据集中取出样本信息
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):#支持下标,定义了他就可以被enumerate/for循环
        return self.x_data[index], self.y_data[index] # 因为数据集较小，所以我们直接读入到内存中了

    def __len__(self):#返回长度
        return self.len

dataset = DiabetesDataset(r'datasets\diabetes.csv.gz')
train_loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

#2 Design model using Class
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

#3 Construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

#4 Traning cycle
if __name__ == '__main__':
    for epoch in range(10):
        for i, data in enumerate(train_loader, start=0):#dataset每次取出一个数据元组，而dataload在此基础上每次取出一个minibatch，enumerate编号的也是minibatch
            # 1 Prepare data
            inputs, labels = data
            #2 Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            #3 Backward
            optimizer.zero_grad()
            loss.backward()
            #4 Update
            optimizer.step()

