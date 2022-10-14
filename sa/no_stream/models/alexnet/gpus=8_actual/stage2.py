import torch


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.layer1 = torch.nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        return out1
