import torch


class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer2 = torch.nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))

    

    def forward(self, input0):
        out0 = input0.clone()
        out2 = self.layer2(out0)
        return out2
