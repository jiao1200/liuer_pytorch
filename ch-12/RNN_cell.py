import torch
#1 Prepare data
seqlen = 3
batch_size = 1
input_size = 4
hidden_size = 2
dataset = torch.randn(seqlen, batch_size, input_size)

# class Net(torch.nn.Module):
#     def __init__(self):
#         super().__init__(self, input_size, hidden_size)
#         cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

#     def forward(self, input, hidden):
#         hidden = self.cell(input, hidden)
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

hidden = torch.zeros(batch_size, hidden_size)

for idx, input in enumerate(dataset):
    print('*' * 20, idx, '*'*20)
    print('Input size:', input.shape)
    hidden = cell(input, hidden)
    print('output size:', hidden.shape)
    print(hidden)
