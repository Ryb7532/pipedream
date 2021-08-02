# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import threading
import torch
import torch.distributed as dist
import sys
import time
import queue

import threadsafe_counter
import threadsafe_queue


NCCL='nccl'
GLOO='gloo'


class CommunicationHandler(object):
    """ Handles communication between stages.

    For stages on different machines, use send/recv.
    For stages on same machine, use broadcast.
    """
    def __init__(self, master_addr, master_port, rank,
                 local_rank, num_ranks_in_server,
                 world_size, fp16, backend):
        """ Set up process groups.

        Note: To turn off broadcasting, set num_ranks_in_server = 1.
        """
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self.num_ranks_in_server = num_ranks_in_server
        self.world_size = world_size
        self.fp16 = fp16
        assert num_ranks_in_server > 0

        # Initialize the distributed environment.
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        assert dist.get_world_size() == self.world_size
        print("Finished initializing process group; backend: %s, rank: %d, "
              "world_size: %d" % (backend, rank, world_size))

        # Stores information about tensors sent directly GPU-to-GPU.
        self.connection_list = []

        # Stores process groups (for broadcast() connections).
        self.process_groups = {}

    def initialize(self, receive_ranks, send_ranks,
                   tensor_tags, target_tensor_names,
                   training_tensor_dtypes,
                   rank_in_stage,
                   num_ranks_in_stage,
                   ranks_in_previous_stage,
                   ranks_in_next_stage,
                   num_warmup,
                   num_previous_warmup):
        """
        Initialize state needed for CommunicationHandler.
        """
        self.receive_ranks = receive_ranks
        self.send_ranks = send_ranks
        self.tensor_tags = tensor_tags
        self.target_tensor_names = target_tensor_names
        self.training_tensor_dtypes = training_tensor_dtypes
        self.rank_in_stage = rank_in_stage
        self.num_ranks_in_stage = num_ranks_in_stage
        self.ranks_in_previous_stage = ranks_in_previous_stage
        self.num_ranks_in_previous_stage = len(ranks_in_previous_stage)
        self.ranks_in_next_stage = ranks_in_next_stage
        self.num_ranks_in_next_stage = len(ranks_in_next_stage)
        self.num_warmup = num_warmup
        self.num_previous_warmup = num_previous_warmup

        self.setup_queues()
        self.create_process_groups()

    def setup_queues(self):
        """
        Setup queues for communication between main compute thread
        and helper communication threads. One queue per tensor
        in forward / backward direction.
        """
        self.forward_receive_queues = {}
        self.backward_receive_queues = {}
        self.forward_send_queues = {}
        self.backward_send_queues = {}

        # Setup queues for each tensor to be received and sent.
        for input_name in self.receive_ranks:
            self.forward_receive_queues[input_name] = []
            self.backward_send_queues[input_name] = []
            for i in range(len(self.receive_ranks[input_name])):
                self.forward_receive_queues[input_name].append(
                    threadsafe_queue.Queue())
                self.backward_send_queues[input_name].append(
                    threadsafe_queue.Queue())
                target_receive_rank = self.receive_ranks[input_name][i]
                self.register_tensor(
                    connected_rank=target_receive_rank,
                    tag=self.tensor_tags[input_name])
        for output_name in self.send_ranks:
            self.backward_receive_queues[output_name] = []
            self.forward_send_queues[output_name] = []
            for i in range(len(self.send_ranks[output_name])):
                self.backward_receive_queues[output_name].append(
                    threadsafe_queue.Queue())
                self.forward_send_queues[output_name].append(
                    threadsafe_queue.Queue())
                target_send_rank = self.send_ranks[output_name][i]
                self.register_tensor(
                    connected_rank=target_send_rank,
                    tag=self.tensor_tags[output_name])

        for target_tensor_name in self.target_tensor_names:
            # Queues for target in forward pass.
            self.forward_receive_queues[target_tensor_name] = []
            self.forward_send_queues[target_tensor_name] = []

            if self.num_ranks_in_previous_stage > 0:
                self.receive_ranks[target_tensor_name] = self.ranks_in_previous_stage
                for i in range(len(self.receive_ranks[target_tensor_name])):
                    self.register_tensor(
                        connected_rank=self.receive_ranks[target_tensor_name][i],
                        tag=self.tensor_tags[target_tensor_name])
                    self.forward_receive_queues[target_tensor_name].append(
                        threadsafe_queue.Queue())

            if self.num_ranks_in_next_stage > 0:
                self.send_ranks[target_tensor_name] = self.ranks_in_next_stage
                for i in range(len(self.send_ranks[target_tensor_name])):
                    self.register_tensor(
                        connected_rank=self.send_ranks[target_tensor_name][i],
                        tag=self.tensor_tags[target_tensor_name])
                    self.forward_send_queues[target_tensor_name].append(
                        threadsafe_queue.Queue())

        print ("Send ranks: ", self.send_ranks)
        print ("Receive ranks: ", self.receive_ranks)

    def register_tensor(self, connected_rank, tag):
        """
        Builds connections list of tensors that are communicated GPU to GPU.

        For tensors that are sent GPU-to-GPU (intra-server for GLOO backend),
        make a list of destination/source ranks and the corresponding tag.
        This information is then used to crate process groups.
        """
        connection_info = [tag, connected_rank]
        self.connection_list.append(connection_info)

    def create_process_groups(self):
        """ Create process groups in the same order across all ranks.

        To create process groups in the same order, each worker collects
        the connection_list of all other workers. To do this, every worker
        gathers the largest size of all other worker's connection_lists (L).
        Then every worker creates a tensor of size Lx2, where each row
        represents a connection, and fills up this tensor depending on how
        large its own connection list is. The worker(s) w/ the largest
        connection list will fill up the entire tensor.

        After constructing this list, an all_gather is performed, after which
        each worker has an identical NxLx2 output, where N is the number of
        workers (world_size), and each index of output represents a worker's
        connection list. For i=self.rank, the output will be identical to the
        workers local connection list.

        Each worker then iterates in the same order over the connections list,
        checking if each connection has been created yet (every connection will
        appear twice in the output), and creating a new process group if one
        doesn't exist for that connection, for both the forward and backward
        direction. Since ranks within process groups must always be identical,
        the smaller rank always goes first, followed by the larger rank.
        """
        print("Setting up process groups for broadcasts...")

        # Figure out the size of the largest connection list that any worker
        # has (L).
        connection_list_size = torch.tensor(
            len(self.connection_list), dtype=torch.int)
        if self.backend == NCCL:
            connection_list_size = connection_list_size.cuda()
        gathered_connection_list_sizes = [
            torch.ones_like(connection_list_size)
            for _ in range(self.world_size)]
        dist.all_gather(gathered_connection_list_sizes,
                        connection_list_size)
        max_connection_list_size = max(
            gathered_connection_list_sizes)

        if max_connection_list_size == 0:
            return

        # Build tensor to send local connection list to all other workers.
        connection_list_tensor = torch.ones([max_connection_list_size, 2],
                                            dtype=torch.int) * -1
        if self.backend == NCCL:
            connection_list_tensor = connection_list_tensor.cuda()
        if len(self.connection_list) > 0:
            connection_list_tensor[0:len(self.connection_list)] = \
                torch.IntTensor(self.connection_list)

        # Gather connection lists of all workers.
        aggregated_connection_list = [
            torch.ones_like(connection_list_tensor)
            for _ in range(self.world_size)]
        dist.all_gather(aggregated_connection_list,
                        connection_list_tensor)

        # Construct identical process groups on each worker.
        for src_rank in range(len(aggregated_connection_list)):
            for connection in aggregated_connection_list[src_rank]:
                tag = int(connection[0])
                dst_rank = int(connection[1])

                if tag == -1:
                    assert dst_rank == -1
                    continue

                min_rank = min(src_rank, dst_rank)
                max_rank = max(src_rank, dst_rank)
                assert min_rank != max_rank

                if min_rank not in self.process_groups:
                    self.process_groups[min_rank] = {}

                if max_rank not in self.process_groups[min_rank]:
                    # self.process_groups[min_rank][max_rank] = \
                    #     dist.new_group(ranks=[min_rank, max_rank])
                    self.process_groups[min_rank][max_rank] = {}
                    self.process_groups[min_rank][max_rank]['forward'] = \
                        dist.new_group(ranks=[min_rank, max_rank])
                    self.process_groups[min_rank][max_rank]['backward'] = \
                        dist.new_group(ranks=[min_rank, max_rank])

    def set_tensor_shapes(self, tensor_shapes):
        self.tensor_shapes = tensor_shapes

    def set_counter(self, counter):
        self.counter = threadsafe_counter.Counter(counter)

    def wait(self):
        self.counter.wait()

    def schedule_for_helper_threads(self, num_iterations):
        """ Scales the number of iterations a helper thread is run for.

        Since we start a helper thread for each worker in previous/next stage,
        the number of iterations for each thread should be scaled by
        the number of workers in previous/next stage.

        TODO: don't current support uneven configurations.
        """
        forward_schedules = []
        backward_schedules = []

        if self.num_ranks_in_next_stage > 0:
            assert num_iterations % self.num_ranks_in_next_stage == 0
            for i in range(self.num_ranks_in_next_stage):
                forward_schedules.append([])
            send = self.rank_in_stage % self.num_ranks_in_next_stage
            for i in range(num_iterations):
                forward_schedules[send].append(
                    self.rank_in_stage + i * self.num_ranks_in_stage)
                send = (send + self.num_ranks_in_stage) % \
                    self.num_ranks_in_next_stage

        if self.num_ranks_in_previous_stage > 0:
            assert num_iterations % self.num_ranks_in_previous_stage == 0
            num_iterations = (num_iterations // self.num_ranks_in_previous_stage) \
                * self.num_ranks_in_stage
            for j in range(self.num_ranks_in_previous_stage):
                backward_schedules.append([])
                recv = j % self.num_ranks_in_stage
                for i in range(num_iterations):
                    if recv == self.rank_in_stage:
                        backward_schedules[j].append(
                            j + i * self.num_ranks_in_previous_stage)
                    recv = (recv + self.num_ranks_in_previous_stage) % \
                        self.num_ranks_in_stage

        return forward_schedules, backward_schedules

    def start_helper_threads(self, num_iterations, forward_only, epoch):
        """
        Start helper communication threads, one for each queue.
        """
        self.set_counter((self.num_ranks_in_previous_stage + self.num_ranks_in_next_stage)*2)
        if "ack" in self.receive_ranks:
            del self.receive_ranks["ack"]
        if "ack" in self.send_ranks:
            del self.send_ranks["ack"]

        (forward_schedules, backward_schedules) = \
            self.schedule_for_helper_threads(
                num_iterations=num_iterations)
        dtype = torch.float16 if self.fp16 else torch.float32

        for i in range(self.num_ranks_in_previous_stage):
            self.start_helper_thread_(
                self.send_helper_thread_args,
                send_helper_thread,
                [i, True,
                backward_schedules[i],
                epoch])
            self.start_helper_thread_(
                self.recv_helper_thread_args,
                recv_helper_thread,
                [i, False,
                backward_schedules[i],
                epoch])

        for i in range(self.num_ranks_in_next_stage):
            self.start_helper_thread_(
                self.send_helper_thread_args,
                send_helper_thread,
                [i, False,
                forward_schedules[i],
                epoch])
            self.start_helper_thread_(
                self.recv_helper_thread_args,
                recv_helper_thread,
                [i, True,
                forward_schedules[i],
                epoch])

    def start_helper_thread_(self, args_func, func, args_func_args):
        """
        Start passed-in func on a helper thread.
        """
        args = args_func(*args_func_args)
        helper_thread = threading.Thread(target=func,
                                         args=args)
        helper_thread.start()

    def increment_messaging_index(self, sending):
        return

    def recv_helper_thread_args(self, index, backward, schedule, epoch):
        tensor_names = []
        targets = []
        recv_queues = {}
        tensor_shapes = {}
        dtypes = {}
        tags = {}

        if backward:
            src_rank = self.ranks_in_next_stage[index]
            for tensor_name in self.send_ranks:
                if tensor_name == "ack" or tensor_name in self.target_tensor_names:
                    continue
                tensor_names.append(tensor_name)
                recv_queues[tensor_name] = self.backward_receive_queues[tensor_name][index]
                tensor_shapes[tensor_name] = self.tensor_shapes[tensor_name]
                dtypes[tensor_name] = self.training_tensor_dtypes[tensor_name]
                tags[tensor_name] = self.tensor_tags[tensor_name]
                assert src_rank == self.send_ranks[tensor_name][index]
        else:
            src_rank = self.ranks_in_previous_stage[index]
            for tensor_name in self.receive_ranks:
                if tensor_name == "ack":
                    continue
                if tensor_name not in self.target_tensor_names:
                    tensor_names.append(tensor_name)
                else:
                    targets.append(tensor_name)
                recv_queues[tensor_name] = self.forward_receive_queues[tensor_name][index]
                tensor_shapes[tensor_name] = self.tensor_shapes[tensor_name]
                dtypes[tensor_name] = self.training_tensor_dtypes[tensor_name]
                tags[tensor_name] = self.tensor_tags[tensor_name]
                assert src_rank == self.receive_ranks[tensor_name][index]

        sub_process_group = None
        min_rank = min(self.rank, src_rank)
        max_rank = max(self.rank, src_rank)
        if backward:
            sub_process_group = \
                self.process_groups[min_rank][max_rank]['backward']
        else:
            sub_process_group = \
                self.process_groups[min_rank][max_rank]['forward']
        assert sub_process_group

        return (recv_queues, self.counter, self.local_rank, tensor_names,
                targets, self.rank, src_rank, tags, tensor_shapes, dtypes,
                sub_process_group, schedule, epoch)

    def send_helper_thread_args(self, index, backward, schedule, epoch):
        tensor_names = []
        targets = []
        send_queues = {}
        tags = {}

        if backward:
            dst_rank = self.ranks_in_previous_stage[index]
            for tensor_name in self.receive_ranks:
                if tensor_name == "ack" or tensor_name in self.target_tensor_names:
                    continue
                tensor_names.append(tensor_name)
                send_queues[tensor_name] = self.backward_send_queues[tensor_name][index]
                tags[tensor_name] = self.tensor_tags[tensor_name]
                assert dst_rank == self.receive_ranks[tensor_name][index]
        else:
            dst_rank = self.ranks_in_next_stage[index]
            for tensor_name in self.send_ranks:
                if tensor_name == "ack":
                    continue
                if tensor_name not in self.target_tensor_names:
                    tensor_names.append(tensor_name)
                else:
                    targets.append(tensor_name)
                send_queues[tensor_name] = self.forward_send_queues[tensor_name][index]
                tags[tensor_name] = self.tensor_tags[tensor_name]
                assert dst_rank == self.send_ranks[tensor_name][index]

        sub_process_group = None
        min_rank = min(self.rank, dst_rank)
        max_rank = max(self.rank, dst_rank)
        if backward:
            sub_process_group = \
                self.process_groups[min_rank][max_rank]['backward']
        else:
            sub_process_group = \
                self.process_groups[min_rank][max_rank]['forward']
        assert sub_process_group

        return (send_queues, self.counter, self.local_rank, tensor_names,
                targets, self.rank, dst_rank, tags,
                sub_process_group, schedule, epoch)

    def send_and_recv_helper_thread_args(self, index, comm_across_previous, schedule, epoch):
        dst_rank = None
        tensor_names = []
        targets = []
        recv_queues = {}
        send_queues = {}
        tensor_shapes = {}
        dtypes = {}
        tags = {}
        sub_process_group = None

        if comm_across_previous:
            dst_rank = self.ranks_in_previous_stage[index]
            for tensor_name in self.receive_ranks:
                if tensor_name == "ack":
                    continue
                if tensor_name not in self.target_tensor_names:
                    tensor_names.append(tensor_name)
                    send_queues[tensor_name] = self.backward_send_queues[tensor_name][index]
                else:
                    targets.append(tensor_name)
                recv_queues[tensor_name] = self.forward_receive_queues[tensor_name][index]
                tensor_shapes[tensor_name] = self.tensor_shapes[tensor_name]
                dtypes[tensor_name] = self.training_tensor_dtypes[tensor_name]
                tags[tensor_name] = self.tensor_tags[tensor_name]
                assert dst_rank == self.receive_ranks[tensor_name][index]

        else:
            dst_rank = self.ranks_in_next_stage[index]
            for tensor_name in self.send_ranks:
                if tensor_name == "ack":
                    continue
                if tensor_name not in self.target_tensor_names:
                    tensor_names.append(tensor_name)
                    recv_queues[tensor_name] = self.backward_receive_queues[tensor_name][index]
                    tensor_shapes[tensor_name] = self.tensor_shapes[tensor_name]
                    dtypes[tensor_name] = self.training_tensor_dtypes[tensor_name]
                else:
                    targets.append(tensor_name)
                send_queues[tensor_name] = self.forward_send_queues[tensor_name][index]
                tags[tensor_name] = self.tensor_tags[tensor_name]
                assert dst_rank == self.send_ranks[tensor_name][index]

        min_rank = min(self.rank, dst_rank)
        max_rank = max(self.rank, dst_rank)
        sub_process_group = \
            self.process_groups[min_rank][max_rank]
        assert sub_process_group

        return (recv_queues, send_queues, self.counter, self.local_rank, tensor_names, targets,
                self.rank, dst_rank, tags, tensor_shapes, dtypes,
                sub_process_group, schedule, epoch)

    def recv(self, tensor_name, forward_minibatch_id,
             backward_minibatch_id, backward=False):
        if backward:
            index = backward_minibatch_id % self.num_ranks_in_next_stage
            tensor = self.backward_receive_queues[tensor_name][
                index].remove()
            return tensor
        else:
            index = forward_minibatch_id % self.num_ranks_in_previous_stage
            tensor = self.forward_receive_queues[tensor_name][
                index].remove()
            if tensor.dtype == torch.float32:
                tensor = tensor.requires_grad_()
            return tensor

    def send(self, tensor_name, tensor, forward_minibatch_id,
             backward_minibatch_id, backward=False):
        if backward:
            index = backward_minibatch_id % self.num_ranks_in_previous_stage
            self.backward_send_queues[tensor_name][index].add(tensor)
        else:
            index = forward_minibatch_id % self.num_ranks_in_next_stage
            self.forward_send_queues[tensor_name][index].add(tensor)

def recv_helper_thread(recv_queues, counter, local_rank, tensor_names,
                        targets, dst_rank, src_rank, tags, tensor_shapes, dtypes,
                        sub_process_group, schedule, epoch):
    if not schedule:
        counter.decrement()
        return

    torch.cuda.set_device(local_rank)
    # This method is to be executed from a helper daemon thread.
    recv_list = targets + tensor_names
    for i in schedule:
        q = queue.Queue()
        dist.barrier(sub_process_group)
        for tensor_name in recv_list:
            tensor, work = _recv_async(tensor_name, src_rank,
                    tensor_shape=tensor_shapes[tensor_name],
                    dtype=dtypes[tensor_name], tag=tags[tensor_name],
                    sub_process_group=sub_process_group)
            q.put((tensor, work))
        for tensor_name in recv_list:
            tensor, work = q.get()
            work.wait()
            torch.cuda.synchronize()
            print("%.6lf : Rank %d : _recv : epoch %d : iter %d : from %d : %s" % (time.time(), dst_rank, epoch, i, src_rank, tensor_name))
            recv_queues[tensor_name].add(tensor)
    counter.decrement()

def send_helper_thread(send_queues, counter, local_rank, tensor_names,
                        targets, src_rank, dst_rank, tags,
                        sub_process_group, schedule, epoch):
    if not schedule:
        counter.decrement()
        return

    torch.cuda.set_device(local_rank)
    # This method is to be executed from a helper daemon thread.
    send_list = []
    if src_rank < dst_rank:
        send_list += targets
    send_list += tensor_names
    for i in schedule:
        q = queue.Queue()
        dist.barrier(sub_process_group)
        for tensor_name in send_list:
            tensor = send_queues[tensor_name].remove()
            size = tensor.element_size() * tensor.nelement()
            torch.cuda.synchronize()
            print("%.6lf : Rank %d : _send : epoch %d : iter %d : to %d : %.2lf bytes : %s" % (time.time(), src_rank, epoch, i, dst_rank, size, tensor_name))
            work = _send_async(tensor, tensor_name, src_rank, dst_rank,
                    tag=tags[tensor_name],
                    sub_process_group=sub_process_group)
            q.put(work)
        for tensor_name in send_list:
            work = q.get()
            work.wait()
            torch.cuda.synchronize()
            print("%.6lf : Rank %d : comp_send : epoch %d : iter %d : to %d : %s" % (time.time(), src_rank, epoch, i, dst_rank, tensor_name))
    counter.decrement()

def send_and_recv_helper_thread_(recv_queues, send_queues, counter, local_rank,
                        tensor_names, targets, src_rank, dst_rank, tags,
                        tensor_shapes, dtypes, sub_process_group,
                        schedule, epoch):
    if not schedule:
        counter.decrement()
        return

    torch.cuda.set_device(local_rank)
    comm_across_previous = src_rank > dst_rank
    for is_send, i in schedule:
        q = queue.Queue()
        if is_send:
            send_list = []
            if not comm_across_previous:
                send_list += targets
            send_list += tensor_names
            dist.barrier(sub_process_group)
            for tensor_name in send_list:
                tensor = send_queues[tensor_name].remove()
                size = tensor.element_size() * tensor.nelement()
                torch.cuda.synchronize()
                print("%.6lf : Rank %d : _send : epoch %d : iter %d : to %d : %.2lf bytes : %s" % (time.time(), src_rank, epoch, i, dst_rank, size, tensor_name))
                work = _send_async(tensor, tensor_name, src_rank, dst_rank,
                        tag=tags[tensor_name],
                        sub_process_group=sub_process_group)
                q.put(work)
            for tensor_name in send_list:
                work = q.get()
                work.wait()
                torch.cuda.synchronize()
                print("%.6lf : Rank %d : comp_send : epoch %d : iter %d : to %d : %s" % (time.time(), src_rank, epoch, i, dst_rank, tensor_name))
        else:
            recv_list = []
            if comm_across_previous:
                recv_list += targets
            recv_list += tensor_names
            dist.barrier(sub_process_group)
            for tensor_name in recv_list:
                tensor, work = _recv_async(tensor_name, dst_rank,
                        tensor_shape=tensor_shapes[tensor_name],
                        dtype=dtypes[tensor_name], tag=tags[tensor_name],
                        sub_process_group=sub_process_group)
                q.put((tensor, work))
            for tensor_name in recv_list:
                tensor, work = q.get()
                work.wait()
                torch.cuda.synchronize()
                print("%.6lf : Rank %d : _recv : epoch %d : iter %d : from %d : %s" % (time.time(), src_rank, epoch, i, dst_rank, tensor_name))
                recv_queues[tensor_name].add(tensor)
    counter.decrement()

def _recv_async(tensor_name, src_rank, tensor_shape=None, dtype=torch.float32,
          tensor=None, tag=None, sub_process_group=None):
    """
    Receives tensor by calling PyTorch's recv() call.

    Tensor will be copied to GPU prior to return.
    """
    assert tag is not None
    if tensor is None:
        assert tensor_shape is not None
        assert dtype is not None
        assert dtype != torch.float16

    if sub_process_group is not None:
        # Receive tensor shape.
        received_tensor_shape = torch.zeros((len(tensor_shape),),
                                            dtype=torch.int)#.cuda()
        dist.broadcast(tensor=received_tensor_shape,
                       src=src_rank,
                       group=sub_process_group)
        received_tensor_shape = list(map(lambda x: int(x),
                                         received_tensor_shape))

        # Receive tensor.
        tensor = torch.zeros(received_tensor_shape, dtype=dtype).cuda()
        work = dist.broadcast(tensor=tensor,
                       src=src_rank,
                       group=sub_process_group,
                       async_op=True)

        return tensor, work
    else:
        # Receive tensor shape.
        received_tensor_shape = torch.zeros(len(tensor_shape),
                                            dtype=torch.int)
        dist.recv(tensor=received_tensor_shape,
                  src=src_rank,
                  tag=tag)
        received_tensor_shape = list(map(lambda x: int(x),
                                         received_tensor_shape))

        # Receive tensor.
        tensor = torch.zeros(received_tensor_shape, dtype=dtype)
        dist.recv(tensor=tensor,
                  src=src_rank,
                  tag=tag)
        tensor = tensor.cuda()

    assert tensor.is_cuda
    return tensor

def _send_async(tensor, tensor_name, src_rank, dst_rank, tag, sub_process_group=None):
    """
    Sends tensor by calling PyTorch's send() call.

    If tensor is being sent not via broadcast(), it will
    be first copied to the CPU.
    """
    if sub_process_group is not None:
        assert tensor.is_cuda

        # Send tensor shape.
        tensor_shape = torch.tensor(tensor.shape, dtype=torch.int)#.cuda()
        dist.broadcast(tensor=tensor_shape, src=src_rank,
                      group=sub_process_group)

        # Send tensor.
        contiguous_tensor = tensor.detach().clone().contiguous()
        work = dist.broadcast(tensor=contiguous_tensor,
                       src=src_rank,
                       group=sub_process_group,
                       async_op=True)

        return work
    else:
        assert tensor.is_cuda
        tensor = tensor.cpu()

        # Send tensor shape.
        tensor_shape = torch.tensor(tensor.shape, dtype=torch.int)
        dist.send(tensor=tensor_shape, dst=dst_rank, tag=tag)

        # Send tensor.
        dist.send(tensor=tensor, dst=dst_rank, tag=tag)
