# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class RuntimeStats:
    def __init__(self, forward, rank, epoch):
        self.stats = {
            'compute_time': 0.0,
            'send_tensors': 0.0,
            'send_tensors_size': 0,
            'receive_tensors': 0.0,
            'receive_tensors_size': 0,
            'total_time': 0.0
        }
        self.forward = forward
        self.rank = rank
        self.epoch = epoch

    def print_stats(self):
        compute_pass = 'Backward'
        if self.forward:
            compute_pass = 'Forward'
            print("Forward Stats:")
        else:
            print("Backward Stats:")
        for i in sorted(self.stats):
            units = 'ms'
            if i == 'receive_tensors_size' or i == 'send_tensors_size':
                units = 'bytes'
            print("\t %s (Epoch %d, Rank %d, %s) %.3f %s" % (i, self.epoch, self.rank, compute_pass, self.stats[i], units))

    def reset_stats(self):
        for i in self.stats.keys():
            self.stats[i] = 0.0
        self.epoch += 1
