import torch
from .stage0 import Stage0
from .stage1 import Stage1

class alexnetPartitioned(torch.nn.Module):
    def __init__(self):
        super(alexnetPartitioned, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()

    

    def forward(self, input0):
        out1 = self.stage0(input0)
        out2 = self.stage1(out1)
        return out2
