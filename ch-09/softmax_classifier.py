import torch
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# import torch.nn.functional as F
import torch.optim as optim

#1 Prepare dataset
batch_size = 64
# Compose/ToTensor/Normalize->Class
transform = transforms.Compose([#compose 组合变换
    transforms.ToTensor(),# 传进来的必须是可以调用的对象obj(img)
    transforms.Normalize((0.1307,), (0.3081,))# 参数是元组，代表了多个通道的mean和std
])

train_dataset = datasets.MNIST(
    root='datasets/mnist/',
    train=True,
    transform=transform,
    download=True
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
    )
test_dataset = datasets.MNIST(
    root='datasets/mnist/',
    train=False,
    transform=transform,
    download=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
    )
# train_dataset = train_dataset.view(-1, 784)# -1代表自动填入，和reshape关系：连续直接view,不连续先copy在view
# test_dataset = test_dataset.view(-1, 784)
#2 Design model using Class
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
        self.ReLU = torch.nn.ReLU()# 没有参数可以复用

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.ReLU(self.l1(x))
        x = self.ReLU(self.l2(x))
        x = self.ReLU(self.l3(x))
        x = self.ReLU(self.l4(x))
        x = self.l5(x)
        return x

model = Net()
#3 Construct criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#4 Training cycle
def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader):
        inputs, targets = data
        optimizer.zero_grad()
        #Forward, Backward, Update
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _,predicted = torch.max(outputs.data, dim=1)# 只要索引，不要最大值
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on test set: {100 * correct / total}%')


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()