import torch


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.layer3 = torch.nn.Dropout(p=0.5)
        self.layer4 = torch.nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.layer5 = torch.nn.ReLU(inplace=True)
        self.layer6 = torch.nn.Dropout(p=0.5)
        self.layer7 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer8 = torch.nn.ReLU(inplace=True)
        self.layer9 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)

    

    def forward(self, input1):
        out0 = input1.clone()
        out1 = out0.size(0)
        out2 = out0.view(out1, 9216)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        return out9
