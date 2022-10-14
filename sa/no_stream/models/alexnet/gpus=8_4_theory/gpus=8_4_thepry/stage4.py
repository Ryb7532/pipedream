import torch


class Stage4(torch.nn.Module):
    def __init__(self):
        super(Stage4, self).__init__()
        self.layer1 = torch.nn.ReLU(inplace=True)

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        return out1
