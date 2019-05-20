# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: epochs
# @Date: 2019-05-18-13-49
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


from snetapp.greedy.base import GreedyBaseWorker


class TrainWithVariation(GreedyBaseWorker):

    def __init__(self, options=None):
        if options is None:
            self.options = self.get_default_options()
        else:
            self.options = options

        super(TrainWithVariation, self).__init__(self.options)

    def get_default_options(self):
        options = super(TrainWithVariation, self).get_default_options()

        options.update({
            'greedy': True,
            'pattern_firing_rate': 1.,  # unit: spikes/dt
            'background_firing_rate': 5.,
            't_background_phase': 10,  # unit: dt
            'training_samples': 10000,
            'testing_samples': 1000,

            'include_categories': [0, 1, 2],
            't_training_image': 200,
            'v_th_rest': 0.4,
            'output_number': 12,
        })

        return options

    def post_epoch(self, i, prefix=''):

        if i % 1000 == 0:
            self.save(filename=f"{prefix}{i}-worker.pickle")

            self.network.W.plot_weight_map(out_file=self.get_path(f'{prefix}{i}-weights.png'))
            self.network.W.plot_update_map(out_file=self.get_path(f'{prefix}{i}-updates.png'))

            # self.network.INPUT.background_firing_rate -= rate_drop
            # self.logger.info(f"Rate dropped to: {self.network.INPUT.background_firing_rate}")


def train_baseline():
    worker = TrainWithVariation()
    worker.train(prefix='baseline-')
    worker.test(worker.greedy_test)


def train_learning_rate_d2d():
    options = TrainWithVariation().options

    for v in [0.1, 0.25, 0.5, 1.0]:

        options.update({'learning_rate_d2d_variation': v})
        worker = TrainWithVariation(options)

        prefix = f'lr-d2d-{v}-'
        worker.train(prefix=prefix)
        worker.test(worker.greedy_test)

        # additional test
        for res in ['8000', '9000']:
            tester = TrainWithVariation.load(worker.result_dir, filename=f"{prefix}{res}-worker.pickle")
            tester.test(tester.greedy_test, rerun=True, prefix=f"{prefix}{res}-")


def train_learning_rate_c2c():
    options = TrainWithVariation().options

    for v in [0.1, 0.25, 0.5, 1.0]:

        options.update({'learning_rate_c2c_variation': v})
        worker = TrainWithVariation(options)

        prefix = f'lr-c2c-{v}-'
        worker.train(prefix=prefix)
        worker.test(worker.greedy_test)

        # additional test
        for res in ['8000', '9000']:
            tester = TrainWithVariation.load(worker.result_dir, filename=f"{prefix}{res}-worker.pickle")
            tester.test(tester.greedy_test, rerun=True, prefix=f"{prefix}{res}-")


def train_learning_rate_combine():
    options = TrainWithVariation().options

    for v in [0.1, 0.25, 0.5, 1.0]:
        options.update({
            'learning_rate_c2c_variation': v,
            'learning_rate_d2d_variation': v,
        })
        worker = TrainWithVariation(options)

        prefix = f'lr-combine-{v}-'
        worker.train(prefix=prefix)
        worker.test(worker.greedy_test)

        # additional test
        for res in ['8000', '9000']:
            tester = TrainWithVariation.load(worker.result_dir, filename=f"{prefix}{res}-worker.pickle")
            tester.test(tester.greedy_test, rerun=True, prefix=f"{prefix}{res}-")


if __name__ == '__main__':
    train_learning_rate_combine()
