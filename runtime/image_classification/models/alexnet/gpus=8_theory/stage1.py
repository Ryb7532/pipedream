import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer3 = torch.nn.Dropout(p=0.5)
        self.layer4 = torch.nn.Linear(in_features=9216, out_features=4096, bias=True)

    

    def forward(self, input1):
        out0 = input1.clone()
        out1 = out0.size(0)
        out2 = out0.view(out1, 9216)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out4
