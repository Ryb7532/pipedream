from .vgg16 import vgg16Partitioned
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .stage4 import Stage4

def arch():
    return "vgg16"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out0"]),
        (Stage1(), ["out0"], ["out2"]),
        (Stage2(), ["out2"], ["out3"]),
        (Stage3(), ["out3"], ["out4"]),
        (Stage4(), ["out4"], ["out5"]),
        (criterion, ["out5"], ["loss"])
    ]

def full_model():
    return vgg16Partitioned()
