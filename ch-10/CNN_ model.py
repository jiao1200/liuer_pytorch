import torch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.ReLu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.pooling(self.ReLu(self.conv1(x)))
        x = self.pooling(self.ReLu(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

in_channels = 1
width, height = 28, 28
batch_size = 10

input = torch.randn(
    batch_size,
    in_channels,
    width,
    height
)
input.to(device)
output = model(input)
print(output.shape)
