import torch


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer2 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer5 = torch.nn.ReLU(inplace=True)
        self.layer6 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer7 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer8 = torch.nn.ReLU(inplace=True)
        self.layer9 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer10 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer11 = torch.nn.ReLU(inplace=True)
        self.layer12 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer13 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer15 = torch.nn.ReLU(inplace=True)
        self.layer16 = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer17 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer18 = torch.nn.ReLU(inplace=True)
        self.layer19 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer20 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer21 = torch.nn.ReLU(inplace=True)
        self.layer22 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer23 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer24 = torch.nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer25 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer27 = torch.nn.ReLU(inplace=True)
        self.layer28 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer29 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer30 = torch.nn.ReLU(inplace=True)
        self.layer31 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer32 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer33 = torch.nn.ReLU(inplace=True)
        self.layer34 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer35 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer37 = torch.nn.ReLU(inplace=True)
        self.layer38 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer39 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer40 = torch.nn.ReLU(inplace=True)
        self.layer41 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer42 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer43 = torch.nn.ReLU(inplace=True)
        self.layer44 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer45 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer47 = torch.nn.ReLU(inplace=True)
        self.layer48 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer49 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer50 = torch.nn.ReLU(inplace=True)
        self.layer51 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer52 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer53 = torch.nn.ReLU(inplace=True)
        self.layer54 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer55 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer57 = torch.nn.ReLU(inplace=True)
        self.layer58 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer59 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer60 = torch.nn.ReLU(inplace=True)
        self.layer61 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer62 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer63 = torch.nn.ReLU(inplace=True)
        self.layer64 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer65 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer67 = torch.nn.ReLU(inplace=True)
        self.layer68 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer69 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer70 = torch.nn.ReLU(inplace=True)
        self.layer71 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer72 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer73 = torch.nn.ReLU(inplace=True)
        self.layer74 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer75 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer77 = torch.nn.ReLU(inplace=True)
        self.layer78 = torch.nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer79 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer80 = torch.nn.ReLU(inplace=True)
        self.layer81 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer82 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer83 = torch.nn.ReLU(inplace=True)
        self.layer84 = torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer85 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer86 = torch.nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer87 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer89 = torch.nn.ReLU(inplace=True)
        self.layer90 = torch.nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer91 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer92 = torch.nn.ReLU(inplace=True)
        self.layer93 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer94 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer95 = torch.nn.ReLU(inplace=True)
        self.layer96 = torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer97 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer99 = torch.nn.ReLU(inplace=True)
        self.layer100 = torch.nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer101 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer102 = torch.nn.ReLU(inplace=True)
        self.layer103 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer104 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer105 = torch.nn.ReLU(inplace=True)
        self.layer106 = torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer107 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer109 = torch.nn.ReLU(inplace=True)
        self.layer110 = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.layer113 = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, input1, input0):
        out0 = input0.clone()
        out1 = input1.clone()
        out2 = self.layer2(out0)
        out3 = self.layer3(out2)
        out3 = out3 + out1
        out5 = self.layer5(out3)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out13 = out13 + out5
        out15 = self.layer15(out13)
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out24 = self.layer24(out15)
        out25 = self.layer25(out24)
        out23 = out23 + out25
        out27 = self.layer27(out23)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        out31 = self.layer31(out30)
        out32 = self.layer32(out31)
        out33 = self.layer33(out32)
        out34 = self.layer34(out33)
        out35 = self.layer35(out34)
        out35 = out35 + out27
        out37 = self.layer37(out35)
        out38 = self.layer38(out37)
        out39 = self.layer39(out38)
        out40 = self.layer40(out39)
        out41 = self.layer41(out40)
        out42 = self.layer42(out41)
        out43 = self.layer43(out42)
        out44 = self.layer44(out43)
        out45 = self.layer45(out44)
        out45 = out45 + out37
        out47 = self.layer47(out45)
        out48 = self.layer48(out47)
        out49 = self.layer49(out48)
        out50 = self.layer50(out49)
        out51 = self.layer51(out50)
        out52 = self.layer52(out51)
        out53 = self.layer53(out52)
        out54 = self.layer54(out53)
        out55 = self.layer55(out54)
        out55 = out55 + out47
        out57 = self.layer57(out55)
        out58 = self.layer58(out57)
        out59 = self.layer59(out58)
        out60 = self.layer60(out59)
        out61 = self.layer61(out60)
        out62 = self.layer62(out61)
        out63 = self.layer63(out62)
        out64 = self.layer64(out63)
        out65 = self.layer65(out64)
        out65 = out65 + out57
        out67 = self.layer67(out65)
        out68 = self.layer68(out67)
        out69 = self.layer69(out68)
        out70 = self.layer70(out69)
        out71 = self.layer71(out70)
        out72 = self.layer72(out71)
        out73 = self.layer73(out72)
        out74 = self.layer74(out73)
        out75 = self.layer75(out74)
        out75 = out75 + out67
        out77 = self.layer77(out75)
        out78 = self.layer78(out77)
        out79 = self.layer79(out78)
        out80 = self.layer80(out79)
        out81 = self.layer81(out80)
        out82 = self.layer82(out81)
        out83 = self.layer83(out82)
        out84 = self.layer84(out83)
        out85 = self.layer85(out84)
        out86 = self.layer86(out77)
        out87 = self.layer87(out86)
        out85 = out85 + out87
        out89 = self.layer89(out85)
        out90 = self.layer90(out89)
        out91 = self.layer91(out90)
        out92 = self.layer92(out91)
        out93 = self.layer93(out92)
        out94 = self.layer94(out93)
        out95 = self.layer95(out94)
        out96 = self.layer96(out95)
        out97 = self.layer97(out96)
        out97 = out97 + out89
        out99 = self.layer99(out97)
        out100 = self.layer100(out99)
        out101 = self.layer101(out100)
        out102 = self.layer102(out101)
        out103 = self.layer103(out102)
        out104 = self.layer104(out103)
        out105 = self.layer105(out104)
        out106 = self.layer106(out105)
        out107 = self.layer107(out106)
        out107 = out107 + out99
        out109 = self.layer109(out107)
        out110 = self.layer110(out109)
        out111 = out110.size(0)
        out112 = out110.view(out111, -1)
        out113 = self.layer113(out112)
        return out113
