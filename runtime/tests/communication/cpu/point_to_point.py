# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import threading
import time
import torch
import torch.distributed as dist

NUM_TRIALS = 100

def receive_tensor_helper(tensor, src_rank, tag, num_iterations):
    start_time = time.time()
    for i in range(num_iterations):
        dist.recv(tensor=tensor, src=src_rank, tag=tag)
    end_time = time.time()
    size = tensor.size()[0]
    throughput = (size * 4. * num_iterations) / (
        (end_time - start_time) * 10**9)
    print("Time to receive %s MB: %.3f seconds" %
        ((size * 4.) / 10**6,
         (end_time - start_time) / num_iterations))
    print("Throughput: %.3f GB/s" % throughput)

def send_tensor_helper(tensor, dst_rank, tag, num_iterations):
    for i in range(num_iterations):
        dist.send(tensor=tensor, dst=dst_rank, tag=tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test lightweight communication library')
    parser.add_argument("--backend", type=str, default='gloo',
                        help="Backend")
    parser.add_argument("--master_addr", required=True, type=str,
                        help="IP address of master")
    parser.add_argument("--rank", required=True, type=int,
                        help="Rank of current worker")
    parser.add_argument('-p', "--master_port", default=12345,
                        help="Port used to communicate tensors")

    args = parser.parse_args()

    args.send = args.rank == 0
    local_rank = args.rank
    print("Local rank: %d" % local_rank)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    world_size = 2
    dist.init_process_group(args.backend, rank=args.rank, world_size=world_size)

    tensor_sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000,
                    100000000, 800000000]

    for tag, tensor_size in enumerate(tensor_sizes):
        if args.send:
            tensor = torch.tensor(range(tensor_size), dtype=torch.float32)
            send_tensor_helper(tensor, 1-args.rank, tag, NUM_TRIALS)
        else:
            tensor = torch.zeros((tensor_size,), dtype=torch.float32)
            receive_tensor_helper(tensor, 1-args.rank, tag, NUM_TRIALS)
