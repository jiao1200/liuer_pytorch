# 实现一个线性模型的梯度下降算法
x_data = [1., 2., 3.]
y_data = [2., 4., 6.]

w = 1.0  # 初始化权重

def forward(x):
    return w * x

def loss(x, y): # 随机梯度下降是单个数据的loss就去对w进行更新
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (x * w - y) 

print('Predict (before training)', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad_val = gradient(x, y)
        w -= 0.01 * grad_val
        print('\tgrad', x, y, grad_val)
        loss_val = loss(x, y)
    print(f'Epoch:{epoch},w={w},loss={loss_val}')

print('Predict (after training)', 4, forward(4))
