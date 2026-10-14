import torch

in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3 # 一般都用正方形
batch_size = 1

input = torch.randn(
    batch_size,
    in_channels,
    width,
    height
)
conv_layer = torch.nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel_size
)
maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)
output = conv_layer(input)
pooling_output = maxpooling_layer(output)

print(input.shape)# shape是实例化对象的属性，类型是torch.Size
print(output.shape)
print(conv_layer.weight.shape)
print(pooling_output.shape)