# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: base
# @Date: 2019-04-18-12-55
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


import os
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
            self.logger.info(log_prefix + f"Thresholds={self.network.OUTPUT.v_th.numpy()}")

            # if self.network.time % 10 == 0:
            #     self.network.W.plot_weight_map()

        self.post_train()

    def post_train(self):
        """
        Save model after training.
        """

        self.network.save_model(self.result_dir)

        self.summarize('train')

        self.send()

    def summarize(self, phase='train'):
        def to_html(dict):
            html = ""
            for key, value in dict.items():
                html += f"{key}: {value}<br>"

            return html

        self.summary['options'] = to_html(self.options)

        image_file = os.path.join(self.result_dir, 'weights.jpg')
        self.network.W.plot_weight_map(image_file)

        self.summary['image_file'] = image_file

        phase_summary = {
            'v_th': self.network.OUTPUT.v_th,
            'time': self.network.time,
            'sample_count': len(self.dataset.training_set if phase == 'train' else len(self.dataset.testing_set)),
        }

        self.summary[phase] = to_html(phase_summary)

    def test(self):
        # TODO: evaluation
        pass


if __name__ == "__main__":
    worker = GreedyBaseWorker()

    worker.train()
