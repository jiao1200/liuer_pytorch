# 实现一个线性模型的梯度下降算法
x_data = [1., 2., 3.]
y_data = [2., 4., 6.]

w = 1.0  # 初始化权重

def forward(x):
    return w * x

def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) * (y_pred - y)

    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * (w * x -  y) * x
    return grad / len(xs) 

print('Predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print(f'Epoch:{epoch},w={w},loss={cost_val}')

print('Predict (after training)', 4, forward(4))
