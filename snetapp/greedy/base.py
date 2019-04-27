# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: base
# @Date: 2019-04-18-12-55
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


from snetapp import Worker


class GreedyBaseWorker(Worker):
    """
    Greedy training worker.

    options:
    Greedy Training, P&B Phases
    MNIST 28x28 0,1,2
    """

    def __init__(self, options=None):
        if options is None:
            self.options = self.get_default_options()
        else:
            self.options = self.infer(options)

        super(GreedyBaseWorker, self).__init__(self.options)

    def get_default_options(self):
        options = super(GreedyBaseWorker, self).get_default_options()

        options.update({
            'greedy': True,
            'pattern_firing_rate': 1.0,  # unit: spikes/dt
            'background_firing_rate': 12.,
            't_background_phase': 20,  # unit: dt
        })

        return self.infer(options)


if __name__ == "__main__":
    worker = GreedyBaseWorker()
    worker.train()

    # path = r'E:\Projects\SNet-apps\snetapp\greedy\results\base.py\Sat-Apr-27-12-58-53-2019'
    # worker = GreedyBaseWorker.load(path)

    worker.test(worker.vote_test)
