import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer1 = torch.nn.ReLU(inplace=True)
        self.layer2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer5 = torch.nn.Dropout(p=0.5)
        self.layer6 = torch.nn.Linear(in_features=9216, out_features=4096, bias=True)

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = out2.size(0)
        out4 = out2.view(out3, 9216)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        return out6
