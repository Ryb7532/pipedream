import torch
from .stage0 import Stage0
from .stage1 import Stage1

class resnet50Partitioned(torch.nn.Module):
    def __init__(self):
        super(resnet50Partitioned, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self._initialize_weights()

    

    def forward(self, input0):
        (out1, out0) = self.stage0(input0)
        out2 = self.stage1(out1, out0)
        return out2
