import argparse
import torch
import torch.distributed as dist
import copy
import os
import time

parser = argparse.ArgumentParser(description='test allreduce_comm')
parser.add_argument('--local_rank', default=0, type=int,
                    help="Local rank of worker")
parser.add_argument('--rank', default=0, type=int, help="")
parser.add_argument('--nrank', default=2, type=int, help="")
parser.add_argument('--master_addr', default='localhost', type=str, help="")


args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
print("start: rank %d" % args.rank)

master_port = '12345'
master_addr = args.master_addr
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port
dist.init_process_group('nccl', init_method="tcp://{}:{}".format(master_addr, master_port), rank=args.rank, world_size=args.nrank)

print("comm_init: rank %d" % args.rank)
#group = dist.GroupMember.WORLD
group = dist.new_group(ranks=list(range(args.nrank)))
"""
groups = []
for i in range(4):
    groups.append(dist.new_group(ranks=[i for i in range(args.nrank) if i % 4 == args.local_rank]))
group = groups[args.local_rank]
"""
for i in [1, 1, 4, 16, 64, 256]:
    time.sleep(2)
    if args.rank == 0:
        print()
        print('size: %d KB' % i)
    size = 1024 * i
    data = torch.ones(size, device='cuda')

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(100):
        dist.all_reduce(data, group=group, op=dist.ReduceOp.SUM)
        data /= args.nrank

    torch.cuda.synchronize()
    end = time.time()

    print("Rank %d: time %.6f ms" % (args.rank, (end-start)*10.0))


for i in [1, 4, 16, 64, 256, 1024]:
    time.sleep(2)
    if args.rank == 0:
        print()
        print('size: %d MB' % i)
    size = 1024 * 1024* i
    data = torch.ones(size, device='cuda')

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(100):
        dist.all_reduce(data, group=group, op=dist.ReduceOp.SUM)
        data /= args.nrank

    torch.cuda.synchronize()
    end = time.time()

    print("Rank %d: time %.6f ms" % (args.rank, (end-start)*10.0))
