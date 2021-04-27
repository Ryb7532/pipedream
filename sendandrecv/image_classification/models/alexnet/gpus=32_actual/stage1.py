import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.Dropout(p=0.5)

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return out3
