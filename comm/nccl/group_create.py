# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import time
import torch
import torch.distributed as dist
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test lightweight communication library')
    parser.add_argument("--master_addr", required=True, type=str,
                        help="IP address of master")
    parser.add_argument("--rank", required=True, type=int,
                        help="Rank of current worker")
    parser.add_argument('-p', "--master_port", default=12345,
                        help="Port used to communicate tensors")
    parser.add_argument("--broadcast", action='store_true',
                        help="Broadcast within a server")

    args = parser.parse_args()

    num_ranks_in_server = 1
    if args.broadcast:
        num_ranks_in_server = 2
    local_rank = args.rank
    torch.cuda.set_device(local_rank)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    dist.init_process_group("nccl", rank=local_rank, world_size=2)

    groups = []
    for i in range(5):
        groups.append(dist.new_group(ranks=[0,1]))

    print("create process groups in rank %d" % local_rank, flush=True)
    if local_rank == 0:
        tensor = torch.tensor(range(1000000000), dtype=torch.float32).cuda(local_rank)
        dist.broadcast(tensor=tensor, group=groups[0], src=0)
    else:
        tensor = torch.zeros((1000000000,), dtype=torch.float32).cuda(local_rank)
        dist.broadcast(tensor=tensor, group=groups[0], src=0)
#    time.sleep(100)

    for i in reversed(range(5)):
        dist.destroy_process_group(groups[i])
