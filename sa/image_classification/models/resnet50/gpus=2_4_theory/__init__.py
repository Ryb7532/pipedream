from .resnet50 import resnet50Partitioned
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .stage4 import Stage4

def arch():
    return "resnet50"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out0", "out1"]),
        (Stage1(), ["out0", "out1"], ["out2", "out3"]),
        (Stage2(), ["out2", "out3"], ["out4", "out5"]),
        (Stage3(), ["out4", "out5"], ["out6", "out7"]),
        (Stage4(), ["out6", "out7"], ["out8"]),
        (criterion, ["out8"], ["loss"])
    ]

def full_model():
    return resnet50Partitioned()
