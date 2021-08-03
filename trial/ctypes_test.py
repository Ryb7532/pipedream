from ctypes import *
nccl = CDLL("/apps/t3/sles12sp2/free/nccl/2.4.2/gcc4.8.5/cuda9.2/lib/libnccl.so.2.4.2")
print(nccl.ncclGetVersion())
nccl.ncclGroupStart()
nccl.ncclGroupEnd()
