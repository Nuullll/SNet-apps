# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: epochs
# @Date: 2019-05-18-13-49
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


from snetapp.greedy.base import GreedyBaseWorker


class TrainForEpochs(GreedyBaseWorker):

    def __init__(self, options=None):
        if options is None:
            self.options = self.get_default_options()
        else:
            self.options = options

        super(TrainForEpochs, self).__init__(self.options)

    def get_default_options(self):
        options = super(TrainForEpochs, self).get_default_options()

        options.update({
            'greedy': True,
            'pattern_firing_rate': 1.,  # unit: spikes/dt
            'background_firing_rate': 10.,
            't_background_phase': 10,  # unit: dt

            'include_categories': list(range(10)),
            't_training_image': 200,
            'v_th_rest': 0.4,
            'output_number': 50,
        })

        return options

    def post_epoch(self, i, prefix=''):
        rate_drop = (self.options.get('background_firing_rate') - 3.) / 60

        if i % 1000 == 0:
            self.network.W.plot_weight_map(out_file=self.get_path(f'{prefix}{i}-weights.jpg'))
            self.network.W.plot_update_map(out_file=self.get_path(f'{prefix}{i}-updates.jpg'))

            self.network.INPUT.background_firing_rate -= rate_drop
            self.logger.info(f"Rate dropped to: {self.network.INPUT.background_firing_rate}")


if __name__ == '__main__':
    worker = TrainForEpochs()
    worker.train()
    worker.test(worker.greedy_test)
