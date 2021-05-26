import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer1 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer6 = torch.nn.Dropout(p=0.5)
        self.layer7 = torch.nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.layer8 = torch.nn.ReLU(inplace=True)
        self.layer9 = torch.nn.Dropout(p=0.5)
        self.layer10 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer11 = torch.nn.ReLU(inplace=True)
        self.layer12 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = out3.size(0)
        out5 = out3.view(out4, 9216)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        return out12
