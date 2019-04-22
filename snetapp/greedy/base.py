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
            't_testing_image': 1000,

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

            'greedy': True,
            'pb_phases': True,
            'pattern_firing_rate': 1.0,     # unit: spikes/dt
            'background_firing_rate': 20.,
            't_background_phase': 20,       # unit: dt
        }

        return self.infer(options)

    def train(self):
        self.logger.info("Start training.")

        self.network.training_mode()

        for i, (image, label) in enumerate(self.dataset.training_set):

            log_prefix = f"[#{i}] label={label} "

            start_time = self.network.time

            self.logger.info(log_prefix + "Feeding image. " + f"@{start_time}")
            self.network.feed_image(image)

            self.network.learn_current_image()

            finish_time = self.network.time
            self.logger.info(log_prefix + "Learned. " + f"@{finish_time} (dt={finish_time-start_time})")

            if self.network.time % 220 == 0:
                self.network.W.plot_weight_map()


if __name__ == "__main__":
    worker = GreedyBaseWorker()

    worker.train()
