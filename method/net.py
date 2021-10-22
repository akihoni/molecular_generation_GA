from torch import nn
from torch.nn import Module, Sequential

# nn model
class Net(Module):
    # constructor
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = Sequential()
        self.linear1.add_module('linear1', nn.Linear(2048, 1000))
        self.linear1.add_module('relu1', nn.ReLU())
        self.linear1.add_module('batch_norm1', nn.BatchNorm1d(1000))

        self.linear2 = Sequential()
        self.linear2.add_module('linear2', nn.Linear(1000, 1))
        # self.linear2.add_module('sigmoid', nn.Sigmoid())

    # forward propagation
    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        return out
