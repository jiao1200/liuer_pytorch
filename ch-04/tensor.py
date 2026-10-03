import torch

x_data = [1, 2, 3]
y_data = [2, 4, 6]

w = torch.tensor([1.])
w.requires_grad = True

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print('predict before training:', 4, forward(4).item()) # 这里forward返回的是一个tensor对象

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y) # forward就会自动建立计算图
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data -= 0.01 * w.grad.data
        w.grad.data.zero_()

    print('progress:', epoch, l.item())

print('predict after training:', 4, forward(4).item())
