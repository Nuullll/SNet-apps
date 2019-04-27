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
        options = {
            'image_size': (28, 28),
            'include_categories': [0, 1, 2],
            # input_number (inferred)
            'output_number': 12,

            'dt': 1e-3,     # unit: s

            't_training_image': 200,    # unit: dt
            't_testing_image': 300,

            'tracker_size': 100,

            # synapses
            'w_min': 0.1,
            'w_max': 5.0,
            'w_init': 'random',
            'learning_rate_p': 0.8,
            'learning_rate_m': 0.8,
            'tau_p': 5,         # unit: dt
            'tau_m': 5,

            # LIF
            'v_th_rest': 0.4,
            'tau': 200.,        # unit: dt
            'refractory': 0,
            'res': 1.,
            'adapt_factor': 1.,

            'greedy': True,
            'pattern_firing_rate': 1.0,     # unit: spikes/dt
            'background_firing_rate': 12.,
            't_background_phase': 20,       # unit: dt
        }

        return self.infer(options)


if __name__ == "__main__":
    worker = GreedyBaseWorker()
    worker.train()

    # path = r'E:\Projects\SNet-apps\snetapp\greedy\results\base.py\Sat-Apr-27-12-58-53-2019'
    # worker = GreedyBaseWorker.load(path)

    worker.test(worker.vote_test)
