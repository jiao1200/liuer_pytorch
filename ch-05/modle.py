import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        #linear是一个实例对象，其中定义了__call__
        '''
        model(x)
 → Module.__call__(x)
   → forward_pre_hooks
   → self.forward(x)
   → forward_hooks
 → 返回结果
        '''
        self.linear = torch.nn.Linear(1, 1) #  输入输出特征数

    def forward(self, x):
        # linear(x),相当于linear.__call__(x)
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()