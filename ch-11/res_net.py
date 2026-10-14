import torch
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F

#1 Prepare dataset
batch_size = 64
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
#2 Design model using Class
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)
    
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5) 
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.shape[0]
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
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