import sys
import torch
import torchvision
import time
import cudaprofile

"""
print(torch.__version__)
print(torchvision.__version__)
print(torch._C._cuda_getCompiledVersion(), 'cuda compiled version')
print(torch.cuda.nccl.version())

print(torch.cuda.is_available())
"""

tensor = torch.tensor(range(10), dtype=torch.float)
print(tensor)
print(tensor.data_ptr())
print(tensor.numel())
tensor2 = torch.tensor(range(10,20),dtype=torch.float)
cat_tensor = torch.cat([tensor,tensor2])
print(cat_tensor)
chunk_tensor = torch.chunk(tensor,2)
print(chunk_tensor)

#if torch.cuda.is_available():
#    torch.cuda.set_device(0)
#    time_0 = torch.cuda.Event(enable_timing=True)
#    time_1 = torch.cuda.Event(enable_timing=True)
#    time_0.record()
#    time.sleep(1)
#    time_1.record()
#    time = time_0.elapsed_time(time_1)
#    print(time)
#    print("time: %.6f" % time)

#tensor = torch.tensor([i for i in range(30)]).cuda()
#tensor_shape = torch.tensor(tensor.shape, dtype=torch.int).cuda()
#print(tensor_shape)
#received_tensor_shape = torch.zeros(len([2,1,0]), dtype=torch.int).cuda()
#print(received_tensor_shape)
#received_tensor_shape = list(map(lambda x: int(x), received_tensor_shape))
#print(received_tensor_shape)
#tensor = torch.zeros(received_tensor_shape, dtype=torch.float32).cuda()
#print(tensor)
