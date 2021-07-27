import argparse
from collections import OrderedDict
import collections
import os
import itertools
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sgd


NCCL = 'nccl'
backend = NCCL

parser = argparse.ArgumentParser(description='PyTorch Image Classification Training')
# parser.add_argument('--data_dir', type=str,
#                     help='path to dataset')
# parser.add_argument('--distributed_backend', type=str,
#                     help='distributed backend to use (gloo|nccl)')
# parser.add_argument('--module', '-m', required=True,
#                     help='name of module that contains model and tensor_shapes definition')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--eval-batch-size', default=100, type=int,
#                     help='eval mini-batch size (default: 100)')
# parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                     metavar='LR', help='initial learning rate')
# parser.add_argument('--lr_policy', default='step', type=str,
#                     help='policy for controlling learning rate')
# parser.add_argument('--lr_warmup', action='store_true',
#                     help='Warmup learning rate first 5 epochs')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--print-freq', '-p', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='static loss scale, positive power of 2 to improve fp16 convergence')
parser.add_argument('--master_addr', default=None, type=str,
                    help="IP address of master (machine with rank 0)")
# parser.add_argument('--config_path', default=None, type=str,
#                     help="Path of configuration file")
parser.add_argument('--no_input_pipelining', action='store_true',
                    help="No pipelining of inputs")
parser.add_argument('--rank', default=None, type=int,
                    help="Rank of worker")
parser.add_argument('--local_rank', default=0, type=int,
                    help="Local rank of worker")
# parser.add_argument('--forward_only', action='store_true',
#                     help="Run forward pass only")
# parser.add_argument('--num_minibatches', default=None, type=int,
#                     help="Number of minibatches to run")
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
#                     help='path to directory to save checkpoints')
# parser.add_argument('--checkpoint_dir_not_nfs', action='store_true',
#                     help='checkpoint dir is not on a shared NFS server')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")
parser.add_argument('--num_ranks_in_server', default=1, type=int,
                    help="number of gpus per machine")
parser.add_argument('--recompute', action='store_true',
                    help='Recompute tensors in backward pass')
# parser.add_argument('--macrobatch', action='store_true',
#                     help='Macrobatch updates to save memory')



class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer2 = torch.nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.layer3 = torch.nn.ReLU(inplace=True)
        self.layer4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer5 = torch.nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.layer6 = torch.nn.ReLU(inplace=True)
        self.layer7 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer8 = torch.nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer9 = torch.nn.ReLU(inplace=True)
        self.layer10 = torch.nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer11 = torch.nn.ReLU(inplace=True)

    def forward(self, input0):
        out0 = input0.clone()
        out2 = self.layer2(out0)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        return out11

class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer1 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer6 = torch.nn.Dropout(p=0.5)
        self.layer7 = torch.nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.layer8 = torch.nn.ReLU(inplace=True)
        self.layer9 = torch.nn.Dropout(p=0.5)
        self.layer10 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer11 = torch.nn.ReLU(inplace=True)
        self.layer12 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = out3.size(0)
        out5 = out3.view(out4, 9216)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        return out12

class alexnetPartitioned(torch.nn.Module):
    def __init__(self):
        super(alexnetPartitioned, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()

    def forward(self, input0):
        out0 = self.stage0(input0)
        out1 = self.stage1(out0)
        return out1

class module__:
    def arch():
        return "alexnet"

    def model(criterion):
        return [
            (Stage0(), ["input0"], ["out0"]),
            (Stage1(), ["out0"], ["out1"]),
            (criterion, ["out1"], ["loss"])
        ]

    def full_model():
        return alexnetPartitioned()



class ModulesWithDependencies:
    def __init__(self, modules_with_dependencies):
        self._modules = []
        self._all_input_names = []
        self._all_output_names = []
        for (module, input_names, output_names) in modules_with_dependencies:
            self._modules.append(module)
            self._all_input_names.append(input_names)
            self._all_output_names.append(output_names)

    def modules(self):
        return self._modules

    def all_input_names(self):
        return self._all_input_names

    def all_output_names(self):
        return self._all_output_names

    def is_input_tensor(self, tensor_name):
        for module_input_names in self._all_input_names:
            if tensor_name in module_input_names:
                return True



class StageRuntime:
    def __init__(self, model, distributed_backend, fp16, loss_scale,
                 training_tensor_shapes,
                 training_tensor_dtypes, inputs_module_destinations,
                 target_tensor_names, configuration_maps, master_addr,
                 rank, local_rank, verbose_freq,
                 enable_recompute=False):
        self.tensors = []
        self.gradients = {}
        self.distributed_backend = distributed_backend
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.training_tensor_shapes = training_tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.target_tensor_names = target_tensor_names

        self.initialize(model, inputs_module_destinations, configuration_maps,
                        master_addr, rank, local_rank)

        self.verbose_freq = verbose_freq
        self.forward_only = False
        self.enable_recompute = enable_recompute

    def initialize(self, model, inputs_module_destinations,
                   configuration_maps, master_addr, rank,
                   local_rank):
        self.send_ranks = {}
        self.receive_ranks = {}
        self.rank = rank
        self.local_rank = local_rank
        self.stage = None
        self.tensor_tags = {}
        self.forward_id = {}
        self.backward_id = {}
        self.criterion_input_name = str(model[-1][1][0])
        self.group = None

        tensor_tag = 1
        for (_, input_tensors, output_tensors) in model:
            for input_tensor in input_tensors:
                if input_tensor not in self.tensor_tags:
                    self.tensor_tags[input_tensor] = tensor_tag
                    tensor_tag += 1
            for output_tensor in output_tensors:
                if output_tensor not in self.tensor_tags:
                    self.tensor_tags[output_tensor] = tensor_tag
                    tensor_tag += 1
        for target_tensor_name in sorted(self.target_tensor_names):
            self.tensor_tags[target_tensor_name] = tensor_tag
            tensor_tag += 1

        module_to_stage_map = configuration_maps['module_to_stage_map']
        stage_to_rank_map = configuration_maps['stage_to_rank_map']
        stage_to_depth_map = configuration_maps['stage_to_depth_map']

        assert len(module_to_stage_map) == len(model)
        assert self.rank is not None

        stage_to_module_map = collections.defaultdict(list)
        for module in range(len(module_to_stage_map)):
            stage_to_module_map[module_to_stage_map[module]].append(module)

        rank_to_stage_map = {}
        for stage in stage_to_rank_map:
            for rank in stage_to_rank_map[stage]:
                rank_to_stage_map[rank] = stage

        assert 0 <= self.rank < len(rank_to_stage_map)
        self.num_ranks = len(rank_to_stage_map)
        self.num_stages = len(stage_to_module_map)
        self.stage = rank_to_stage_map[self.rank]
        self.rank_in_stage = stage_to_rank_map[self.stage].index(self.rank)
        self.num_ranks_in_stage = len(stage_to_rank_map[self.stage])
        self.num_ranks_in_first_stage = len(stage_to_rank_map[0])
        self.num_ranks_in_previous_stage = 0
        self.ranks_in_previous_stage = []
        if self.stage > 0:
            self.num_ranks_in_previous_stage = len(
                stage_to_rank_map[self.stage - 1])
            self.ranks_in_previous_stage = stage_to_rank_map[self.stage - 1]
        self.num_ranks_in_next_stage = 0
        self.ranks_in_next_stage = []
        if self.stage < self.num_stages - 1:
            self.num_ranks_in_next_stage = len(
                stage_to_rank_map[self.stage + 1])
            self.ranks_in_next_stage = stage_to_rank_map[self.stage + 1]
        modules = stage_to_module_map[self.stage]
        self.modules_with_dependencies = ModulesWithDependencies(
            [model[module] for module in modules])
        self.is_criterion = self.stage == (self.num_stages - 1)
        self.num_warmup_minibatches = stage_to_depth_map[str(self.stage)]

        for i in range(len(model)):
            for j in range(i+1, len(model)):
                for tensor_name in model[i][2]:
                    if tensor_name in model[j][1]:
                        if module_to_stage_map[i] == \
                            module_to_stage_map[j]:
                            continue
                        if module_to_stage_map[j] == self.stage:
                            self.receive_ranks[tensor_name] = \
                                stage_to_rank_map[module_to_stage_map[i]]
                        if module_to_stage_map[i] == self.stage:
                            self.send_ranks[tensor_name] = \
                                stage_to_rank_map[module_to_stage_map[j]]

        for model_inputs in inputs_module_destinations.keys():
            destination_stage = module_to_stage_map[
                inputs_module_destinations[model_inputs]]
            if destination_stage > self.stage:
                self.send_ranks[model_inputs] = \
                    self.ranks_in_next_stage

            if 0 < self.stage <= destination_stage:
                self.receive_ranks[model_inputs] = \
                    self.ranks_in_previous_stage

            if destination_stage > 0:
                if model_inputs not in self.tensor_tags:
                    self.tensor_tags[model_inputs] = tensor_tag
                    tensor_tag += 1

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()

        master_port = 12345
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        dist.init_process_group(backend, rank=args.rank, world_size=self.num_ranks)

        if stage_to_rank_map is not None:
            groups = []
            for stage in range(self.num_stages):
                ranks = stage_to_rank_map[stage]
                if len(ranks) > 1:
                    groups.append(dist.new_group(ranks=ranks))
                else:
                    groups.append(None)
            self.group = groups[self.stage]

        connection_list = []
        for input_name in self.receive_ranks:
            for i in range(len(self.receive_ranks[input_name])):
                target_receive_rank = self.receive_ranks[input_name][i]
                connection_list.append([self.tensor_tags[input_name], target_receive_rank])
        for output_name in self.send_ranks:
            for i in range(len(self.send_ranks[output_name])):
                target_send_rank = self.send_ranks[output_name][i]
                connection_list.append([self.tensor_tags[output_name], target_send_rank])
        for target_tensor_name in self.target_tensor_names:
            if self.num_ranks_in_previous_stage > 0:
                self.receive_ranks[target_tensor_name] = self.ranks_in_previous_stage
                for i in range(len(self.receive_ranks[target_tensor_name])):
                    connection_list.append([self.tensor_tags[target_tensor_name], \
                        self.receive_ranks[target_tensor_name][i]])
            if self.num_ranks_in_next_stage > 0:
                self.send_ranks[target_tensor_name] = self.ranks_in_next_stage
                for i in range(len(self.send_ranks[target_tensor_name])):
                    connection_list.append([self.tensor_tags[target_tensor_name], \
                            self.send_ranks[target_tensor_name][i]])

        connection_list_size = torch.tensor(
            len(connection_list), dtype=torch.int)
        if backend == NCCL:
            connection_list_size = connection_list_size.cuda()
        gathered_connection_list_sizes = [
            torch.ones_like(connection_list_size)
            for _ in range(self.num_ranks)]
        dist.all_gather(gathered_connection_list_sizes,
                        connection_list_size)
        max_connection_list_size = max(
            gathered_connection_list_sizes)

        if max_connection_list_size == 0:
            return

        connection_list_tensor = torch.ones([max_connection_list_size, 2],
                                            dtype=torch.int) * -1
        if backend == NCCL:
            connection_list_tensor = connection_list_tensor.cuda()
        if len(connection_list) > 0:
            connection_list_tensor[0:len(connection_list)] = \
                torch.IntTensor(connection_list)

        aggregated_connection_list = [
            torch.ones_like(connection_list_tensor)
            for _ in range(self.num_ranks)]
        dist.all_gather(aggregated_connection_list,
                        connection_list_tensor)

        self.process_groups = {}
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
                    self.process_groups[min_rank][max_rank] = dist.new_group(
                        ranks=[min_rank, max_rank])

        broadcast_bucket_size = 250 * 1024 * 1024
        num_parameters = 0
        for i in range(len(modules)):
            if self.group is not None:
                if ((i < (len(modules)-1) and self.is_criterion)
                    or not self.is_criterion):
                    num_parameters += \
                        sum(x.size()[0] * x.size()[1]
                            if len(x.size()) > 1 else x.size()[0]
                            for x in modules[i].parameters() if x.size())
                    module_states = list(modules[i].state_dict().values())
                    if len(module_states) > 0:
                        dist._dist_broadcast_coalesced(self.group, module_states, broadcast_bucket_size, False)
        if self.num_ranks_in_stage > 1:
            module_size = 4. * num_parameters
            print("Replicating stage: ranks=%d, module_size=%.3f" % (
                self.num_ranks_in_stage, module_size))

        parameter_iterators = []
        for module in modules:
            parameter_iterators.append(module.parameters())
        parameters = itertools.chain(*parameter_iterators)
        self.master_parameters = list(parameters)
        self.model_parameters = None

        self.epoch = -1

    def modules(self):
        return self.modules_with_dependencies.modules()

    def train(self):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.training_tensor_shapes
        self.forward_only = False

        self.forward_id = {
            'send':self.rank_in_stage,
            'run':self.rank_in_stage,
            'recv':self.rank_in_stage}
        self.backward_id = {
            'send':self.rank_in_stage,
            'run':self.rank_in_stage,
            'recv':self.rank_in_stage}

        self.epoch += 1

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].train()

    def set_loader(self, loader):
        if loader is not None:
            self.loader_iter = iter(loader)
        else:
            self.loader_iter = None

    def receive_tensors_forward(self, tensors):
        forward_minibatch_id = self.forward_id['recv']
        torch.cuda.synchronize()
        t_1 = time.time()
        print("%.6lf : Rank %d : receive_forward : epoch %d :  iter %d" % (t_1, self.rank, self.epoch, forward_minibatch_id))
        if self.loader_iter is not None:
            input = next(self.loader_iter)
            (input, target) = input
            if self.fp16:
                input = input.half()
            tensors["input0"] = input.cuda(non_blocking=True)
            tensors["target"] = target.cuda(non_blocking=True)
        else:
            src_rank = self.ranks_in_previous_stage[forward_minibatch_id % \
                self.num_ranks_in_previous_stage]
            pg = self.process_groups[src_rank][self.rank]
            for input_name in self.receive_ranks:
                tensor_shape = self.tensor_shapes[input_name]
                dtype = self.training_tensor_dtypes[input_name]
                tensor = torch.zeros(tensor_shape, dtype=dtype).cuda()
                dist.broadcast(tensor=tensor,
                    src=src_rank,
                    group=pg)
                tensors[input_name] = tensor
                torch.cuda.synchronize()
                print("%.6lf : Rank %d : _recv : epoch %d : iter %d : from %d : %s" % (time.time(), self.rank, self.epoch, forward_minibatch_id, src_rank, input_name))
        self.forward_id['recv'] += self.num_ranks_in_stage
        torch.cuda.synchronize()
        t = time.time()
        print("%.6lf : Rank %d : receive_forward__ : epoch %d :  iter %d" % (t, self.rank, self.epoch, forward_minibatch_id))

    def send_tensors_forward(self, activations=None):
        if self.num_ranks_in_next_stage == 0:
            return
        forward_minibatch_id = self.forward_id['send']
        torch.cuda.synchronize()
        t_3 = time.time()
        print("%.6lf : Rank %d : send_forward : epoch %d : iter %d" % (t_3, self.rank, self.epoch, forward_minibatch_id))
        dst_rank = self.ranks_in_next_stage[forward_minibatch_id % \
            self.num_ranks_in_next_stage]
        pg = self.process_groups[self.rank][dst_rank]
        for output_name in self.send_ranks:
            if activations is None:
                contiguous_tensor = self.tensors[-1][output_name].detach().clone()
            else:
                contiguous_tensor = activations[output_name].detach().clone()
            size = contiguous_tensor.element_size() * contiguous_tensor.nelement()
            torch.cuda.synchronize()
            start = time.time()
            print("%.6lf : Rank %d : _send : epoch %d : iter %d : to %d : %.2lf bytes : %s" % (start, self.rank, self.epoch, forward_minibatch_id, dst_rank, size, output_name))
            dist.broadcast(tensor=contiguous_tensor.contiguous(),
                src=self.rank,
                group=pg)
            torch.cuda.synchronize()
            end = time.time()
            print("%.6lf : Rank %d : comp_send : epoch %d : iter %d : to %d : %s" % (end, self.rank, self.epoch, forward_minibatch_id, dst_rank, output_name))
        self.forward_id['send'] += self.num_ranks_in_stage
        torch.cuda.synchronize()
        t = time.time()
        print("%.6lf : Rank %d : send_forward__ : epoch %d : iter %d" % (t, self.rank, self.epoch, forward_minibatch_id))

    def receive_tensors_backward(self):
        if self.num_ranks_in_next_stage == 0:
            return
        backward_minibatch_id = self.backward_id['recv']
        torch.cuda.synchronize()
        t_1 = time.time()
        print("%.6lf : Rank %d : receive_backward : epoch %d : iter %d" % (t_1, self.rank, self.epoch, backward_minibatch_id))
        src_rank = self.ranks_in_next_stage[backward_minibatch_id % \
            self.num_ranks_in_next_stage]
        pg = self.process_groups[self.rank][src_rank]
        for output_name in self.send_ranks:
            if output_name in self.target_tensor_names:
                continue
            tensor_shape = self.tensor_shapes[output_name]
            dtype = self.training_tensor_dtypes[output_name]
            tensor = torch.zeros(tensor_shape, dtype=dtype).cuda()
            dist.broadcast(tensor=tensor,
                src=src_rank,
                group=pg)
            self.gradients[output_name] = tensor
            torch.cuda.synchronize()
            print("%.6lf : Rank %d : _recv : epoch %d : iter %d : from %d : %s" % (time.time(), self.rank, self.epoch, backward_minibatch_id, src_rank, output_name))
        self.backward_id['recv'] += self.num_ranks_in_stage
        torch.cuda.synchronize()
        t = time.time()
        print("%.6lf : Rank %d : receive_backward__ : epoch %d : iter %d" % (t, self.rank, self.epoch, backward_minibatch_id))

    def send_tensors_backward(self, gradients=None):
        if self.num_ranks_in_previous_stage == 0:
            return
        backward_minibatch_id = self.backward_id['send']
        torch.cuda.synchronize()
        t_3 = time.time()
        print("%.6lf : Rank %d : send_backward : epoch %d : iter %d" % (t_3, self.rank, self.epoch, backward_minibatch_id))
        dst_rank = self.ranks_in_previous_stage[backward_minibatch_id % \
            self.num_ranks_in_previous_stage]
        pg = self.process_groups[dst_rank][self.rank]
        for input_name in self.receive_ranks:
            if input_name in self.target_tensor_names:
                continue
            if gradients is None:
                tensor = self.gradients[input_name].detach().clone()
            else:
                tensor = gradients[input_name].detach().clone()
            size = tensor.element_size() * tensor.nelement()
            torch.cuda.synchronize()
            start = time.time()
            print("%.6lf : Rank %d : _send : epoch %d : iter %d : to %d : %.2lf bytes : %s" % (start, self.rank, self.epoch, backward_minibatch_id, dst_rank, size, input_name))
            dist.broadcast(tensor=tensor.contiguous(),
                src=self.rank,
                group=pg)
            torch.cuda.synchronize()
            end = time.time()
            print("%.6lf : Rank %d : comp_send : epoch %d : iter %d : to %d : %s" % (end, self.rank, self.epoch, backward_minibatch_id, dst_rank, input_name))
        self.backward_id['send'] += self.num_ranks_in_stage
        torch.cuda.synchronize()
        t = time.time()
        print("%.6lf : Rank %d : send_backward__ : epoch %d : iter %d" % (t, self.rank, self.epoch, backward_minibatch_id))

    def run_forward(self):
        self.tensors.append({})
        tensors = self.tensors[-1]
        self.receive_tensors_forward(tensors)
        self._run_forward(tensors)
        self.send_tensors_forward()

    def _run_forward(self, tensors):
        forward_minibatch_id = self.forward_id['run']
        torch.cuda.synchronize()
        t_2 = time.time()
        print("%.6lf : Rank %d : run_forward : epoch %d : iter %d" % (t_2, self.rank, self.epoch, forward_minibatch_id))
        modules = self.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()
        for i, (module, input_names, output_names) in \
                enumerate(zip(modules, all_input_names, all_output_names)):
            if i == (len(modules) - 1) and self.is_criterion:
                module_outputs = [module(tensors[input_name],
                                            tensors["target"])
                                    for input_name in input_names]
                module_outputs = [sum(module_outputs)]
            else:
                module_outputs = module(*[tensors[input_name]
                                          for input_name in input_names])
                if not isinstance(module_outputs, tuple):
                    module_outputs = (module_outputs,)
                module_outputs = list(module_outputs)

            for (output_name, module_output) in zip(output_names, module_outputs):
                tensors[output_name] = module_output

        self.output = tensors[input_names[0]]
        if self.is_criterion:
            self.loss = tensors[output_names[0]]
        else:
            self.loss = 1
        torch.cuda.synchronize()
        t_4 = time.time()
        print("%.6lf : Rank %d : end_forward : epoch %d : iter %d" % (t_4, self.rank, self.epoch, forward_minibatch_id))
        self.forward_id['run'] += self.num_ranks_in_stage

    def run_backward(self):
        self.receive_tensors_backward()
        self._run_backward()
        if self.group is not None:
            num_replicas = self.num_ranks_in_stage
            torch.cuda.synchronize()
            start = time.time()
            print("%.6lf : Rank %d : _sync_reduction " % (start, self.rank))
            for module in self.modules():
                for param in module.parameters():
                    dist.all_reduce(param.grad.data, group=self.group, op=dist.ReduceOp.SUM)
                    param.grad.data /= num_replicas
            torch.cuda.synchronize()
            end = time.time()
            print("%.6lf : Rank %d : end_sync_reduction " % (end, self.rank))
        self.send_tensors_backward()

    def _run_backward(self):
        backward_minibatch_id = self.backward_id['run']
        torch.cuda.synchronize()
        t_2 = time.time()
        print("%.6lf : Rank %d : run_backward : epoch %d : iter %d" % (t_2, self.rank, self.epoch, backward_minibatch_id))
        inputs = {}
        outputs = {}
        gradients = {}
        output_gradients = {}

        all_input_names_set = set()
        all_output_names_set = set()

        modules = self.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()

        for (input_names, output_names) in zip(all_input_names, all_output_names):
            for input_name in input_names:
                all_input_names_set.add(input_name)
            for output_name in output_names:
                all_output_names_set.add(output_name)

        tensors = self.tensors.pop(0)
        for (_, input_names, output_names) in \
            zip(reversed(modules), reversed(all_input_names), reversed(all_output_names)):
            for output_name in output_names:
                if output_name not in all_input_names_set:
                    if output_name not in self.gradients:
                        output_gradients[output_name] = None
                    else:
                        output_gradients[output_name] = self.gradients[output_name]
                    if tensors[output_name].requires_grad:
                        outputs[output_name] = tensors[output_name]
            for input_name in input_names:
                if input_name not in all_output_names_set:
                    inputs[input_name] = tensors[input_name]

        def hook_wrapper(input_name):
            def hook(input_gradient):
                gradients[input_name] = input_gradient
            return hook

        for input_name in inputs:
            if input_name != "input0" and input_name != "input1" and input_name != "input2" \
                    and inputs[input_name].requires_grad:
                inputs[input_name].register_hook(hook_wrapper(input_name))

        if "loss" in outputs:
            outputs["loss"] *= self.loss_scale

        torch.autograd.backward(tuple([outputs[output_name] for output_name in outputs]),
                                grad_tensors=tuple([output_gradients[output_name]
                                                    for output_name in outputs]))

        for input_name in inputs:
            if not inputs[input_name].requires_grad:
                self.gradients[input_name] = inputs[input_name]
                continue

            if input_name != "input0" and input_name != "input1" and input_name != "input2" and input_name != "input":
                self.gradients[input_name] = gradients[input_name]
        torch.cuda.synchronize()
        t_4 = time.time()
        print("%.6lf : Rank %d : end_backward : epoch %d : iter %d" % (t_4, self.rank, self.epoch, backward_minibatch_id))
        self.backward_id['run'] += self.num_ranks_in_stage

    def num_iterations(self, loader_size):
        if self.stage == 0 or self.stage is None:
            return loader_size

        num_iterations = loader_size * self.num_ranks_in_first_stage
        assert num_iterations % self.num_ranks_in_stage == 0
        num_iterations = num_iterations // self.num_ranks_in_stage

        return num_iterations



class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length



def is_first_stage():
    return args.stage is None or (args.stage == 0)



def main():
    global args
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)

    torch.cuda.synchronize()
    start = time.time()
    print("%.6lf : Rank %d : start " % (start, args.rank))
    criterion = nn.CrossEntropyLoss()
    model = module__.model(criterion)
    input_size = [args.batch_size, 3, 224, 224]
    training_tensor_shapes = {"input0": input_size, "target": [args.batch_size]}
    dtypes = {"input0": torch.int64, "target": torch.int64}
    inputs_module_destinations = {"input": 0}
    target_tensor_names = {"target"}

    for (stage, inputs, outputs) in model[:-1]:  # Skip last layer (loss).
        input_tensors = []
        for input in inputs:
            input_tensor = torch.zeros(tuple(training_tensor_shapes[input]),
                                       dtype=torch.float32)
            input_tensors.append(input_tensor)
        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype

    configuration_maps = {
        'module_to_stage_map': [0,1,1],
        'stage_to_rank_map': {0: [0,1,2], 1: [3]},
        'stage_to_depth_map': {str(1): 0, str(0): 1}
    }

    r = StageRuntime(
        model=model, distributed_backend=backend,
        fp16=args.fp16, loss_scale=args.loss_scale,
        training_tensor_shapes=training_tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr, rank=args.rank,
        local_rank=args.local_rank,
        verbose_freq=args.verbose_frequency,
        enable_recompute=args.recompute)

    args.stage = r.stage
    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks
    if not is_first_stage():
        args.synthetic_data = True

    if args.no_input_pipelining:
        num_versions = 1
    else:
        num_versions = r.num_warmup_minibatches + 1

    optimizer = sgd.SGDWithWeightStashing(r.modules(), r.master_parameters,
                                          r.model_parameters, 1,
                                          num_versions=num_versions,
                                          lr=0.1,
                                          momentum=0.9,
                                          weight_decay=1e-4,
                                          verbose_freq=0,
                                          macrobatch=False)

    cudnn.benchmark = True

    train_dataset = SyntheticDataset((3, 224, 224), 122880)
    distributed_sampler = False
    train_sampler = None
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank)
            distributed_sampler = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    torch.cuda.synchronize()
    end = time.time()
    print("Initialize_time (Rank %d ): %.6f seconds" % (args.rank, end-start))

    average_epoch_time = 0
    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.synchronize()
        print("%.6lf : Rank %d : _epoch : epoch %d" % (time.time(), args.rank, epoch))
        if epoch == args.start_epoch+1:
            torch.cuda.synchronize()
            average_epoch_time = time.time()
        #    os.system("nvidia-smi")
        #    os.system("numactl -s")
        #    os.system("ps aux")
        if distributed_sampler:
            train_sampler.set_epoch(epoch)

        train(train_loader, r, optimizer, epoch)

        torch.cuda.synchronize()
        print("%.6lf : Rank %d : end_epoch : epoch %d" % (time.time(), args.rank, epoch))

    torch.cuda.synchronize()
    end = time.time()
    total_time = end - start
    num_epoch = args.epochs - args.start_epoch
    print("Average_epoch_time (Rank %d ): %.6f seconds" % (args.rank, (end-average_epoch_time)/(num_epoch-1)))
    print("Total_time (Rank %d ): %d minutes %.6f seconds" % (args.rank, total_time // 60, total_time % 60))
    print("%.6lf : Rank %d : end " % (time.time(), args.rank))



def train(train_loader, r, optimizer, epoch):
    n = r.num_iterations(loader_size=len(train_loader))
    r.train()
    if not is_first_stage(): train_loader = None
    r.set_loader(train_loader)

    if args.no_input_pipelining:
        num_warmup_minibatches = 0
    else:
        num_warmup_minibatches = r.num_warmup_minibatches

    if args.verbose_frequency > 0:
        print("Letting in %d warm-up minibatches" % num_warmup_minibatches)
        print("Running training for %d minibatches" % n)

    torch.cuda.synchronize()
    end = time.time()
    epoch_start_time = end

    if args.rank == 3:
        tensors = {}
        gradients = {}

        for input_name in r.receive_ranks:
            tensors[input_name] = []
        # receive forward
        for _ in range(3):
            receive_tensors = {}
            r.receive_tensors_forward(receive_tensors)
            for input_name in r.receive_ranks:
                tensors[input_name] += [receive_tensors[input_name]]
        r.tensors.append({})
        for input_name in r.receive_ranks:
            r.tensors[-1][input_name] = torch.cat(tensors[input_name])
        ##

        for _ in range(num_warmup_minibatches):
            r._run_forward(r.tensors[-1])
            r.forward_id['run'] += 2

            # receive forward
            for input_name in r.receive_ranks:
                tensors[input_name] = []
            for _ in range(3):
                receive_tensors = {}
                r.receive_tensors_forward(receive_tensors)
                for input_name in r.receive_ranks:
                    tensors[input_name] += [receive_tensors[input_name]]
            r.tensors.append({})
            for input_name in r.receive_ranks:
                r.tensors[-1][input_name] = torch.cat(tensors[input_name])
            ##

        for _ in range(n//3-num_warmup_minibatches-1):
            r._run_forward(r.tensors[-1])
            r.forward_id['run'] += 2

            optimizer.zero_grad()
            optimizer.load_old_params()
            r._run_backward()
            r.backward_id['run'] += 2
            optimizer.load_new_params()
            optimizer.step()

            # receive forward
            for input_name in r.receive_ranks:
                tensors[input_name] = []
            for _ in range(3):
                receive_tensors = {}
                r.receive_tensors_forward(receive_tensors)
                for input_name in r.receive_ranks:
                    tensors[input_name] += [receive_tensors[input_name]]
            r.tensors.append({})
            for input_name in r.receive_ranks:
                r.tensors[-1][input_name] = torch.cat(tensors[input_name])
            ##

            # send backward
            for input_name in r.receive_ranks:
                if input_name in r.target_tensor_names:
                    continue
                gradients[input_name] = torch.chunk(r.gradients[input_name],3)
            for i in range(3):
                gradient = {}
                for input_name in r.receive_ranks:
                    if input_name in r.target_tensor_names:
                        continue
                    gradient[input_name] = gradients[input_name][i]
                r.send_tensors_backward(gradient)
            ##

        r._run_forward(r.tensors[-1])
        r.forward_id['run'] += 2

        for _ in range(num_warmup_minibatches+1):
            optimizer.zero_grad()
            optimizer.load_old_params()
            r._run_backward()
            r.backward_id['run'] += 2
            optimizer.load_new_params()
            optimizer.step()

            # send backward
            for input_name in r.receive_ranks:
                if input_name in r.target_tensor_names:
                    continue
                gradients[input_name] = torch.chunk(r.gradients[input_name],3)
            for i in range(3):
                gradient = {}
                for input_name in r.receive_ranks:
                    if input_name in r.target_tensor_names:
                        continue
                    gradient[input_name] = gradients[input_name][i]
                r.send_tensors_backward(gradient)
            ##
    else:
        for _ in range(num_warmup_minibatches):
            r.run_forward()

        for _ in range(n - num_warmup_minibatches):
            r.run_forward()

            optimizer.zero_grad()
            optimizer.load_old_params()
            r.run_backward()
            optimizer.load_new_params()
            optimizer.step()

        for _ in range(num_warmup_minibatches):
            optimizer.zero_grad()
            optimizer.load_old_params()
            r.run_backward()
            optimizer.load_new_params()
            optimizer.step()

    torch.cuda.synchronize()
    end = time.time()
    print("Epoch_%d (Rank %d ): %.6f seconds" % (epoch, args.rank, end - epoch_start_time))
    print("Epoch start time: %.6f, epoch end time: %.6f (Rank %d)" % (epoch_start_time, end, args.rank))



if __name__ == '__main__':
    main()
