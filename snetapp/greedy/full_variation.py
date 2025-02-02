# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: epochs
# @Date: 2019-05-24-15-27
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


from snetapp.greedy.variation import TrainWithVariation


class FullJobWithVariation(TrainWithVariation):

    eval_list = ['58000', '59000']

    def __init__(self, options=None):

        self.options = self.get_default_options()

        if options:
            self.options.update(options)

        super(FullJobWithVariation, self).__init__(self.options)

    def get_default_options(self):
        options = super(FullJobWithVariation, self).get_default_options()

        options.update({
            'greedy': True,
            'pattern_firing_rate': 1.,  # unit: spikes/dt
            'background_firing_rate': 7.,
            't_background_phase': 10,  # unit: dt

            'include_categories': list(range(10)),
            't_training_image': 200,
            'v_th_rest': 0.4,
            'output_number': 50,
        })

        return options

    @classmethod
    def train_baseline(cls):

        prefix = 'baseline-'

        worker = cls()
        worker.train(prefix=prefix)
        worker.test(worker.greedy_test)

        # additional test
        for res in worker.eval_list:
            tester = cls.load(worker.result_dir, filename=f"{prefix}{res}-worker.pickle")
            tester.test(tester.greedy_test, rerun=True, prefix=f"{prefix}{res}-")

    @classmethod
    def train_lr_d2d_variation(cls):

        for v in [0.1, 0.3, 0.5]:

            prefix = f'lr-d2d-{v}-'

            worker = cls({
                'learning_rate_d2d_variation': v
            })

            worker.train(prefix)
            worker.test(worker.greedy_test)

            # additional test
            for res in worker.eval_list:
                tester = cls.load(worker.result_dir, filename=f"{prefix}{res}-worker.pickle")
                tester.test(tester.greedy_test, rerun=True, prefix=f"{prefix}{res}-")

    @classmethod
    def train_lr_c2c_variation(cls):

        for v in [0.1, 0.3, 0.5]:

            prefix = f'lr-c2c-{v}-'

            worker = cls({
                'learning_rate_c2c_variation': v
            })

            worker.train(prefix)
            worker.test(worker.greedy_test)

            # additional test
            for res in worker.eval_list:
                tester = cls.load(worker.result_dir, filename=f"{prefix}{res}-worker.pickle")
                tester.test(tester.greedy_test, rerun=True, prefix=f"{prefix}{res}-")

    @classmethod
    def train_lr_combine_variation(cls):

        for v in [0.1, 0.3, 0.5]:

            prefix = f'lr-combine-{v}-'

            worker = cls({
                'learning_rate_d2d_variation': v,
                'learning_rate_c2c_variation': v
            })

            worker.train(prefix)
            worker.test(worker.greedy_test)

            # additional test
            for res in worker.eval_list:
                tester = cls.load(worker.result_dir, filename=f"{prefix}{res}-worker.pickle")
                tester.test(tester.greedy_test, rerun=True, prefix=f"{prefix}{res}-")

    @classmethod
    def train_window_d2d_variation(cls):

        for v in [0.1, 0.3, 0.5]:

            prefix = f'window-d2d-{v}-'

            worker = cls({
                'window_d2d_variation': v
            })

            worker.train(prefix)
            worker.test(worker.greedy_test)

            # additional test
            for res in worker.eval_list:
                tester = cls.load(worker.result_dir, filename=f"{prefix}{res}-worker.pickle")
                tester.test(tester.greedy_test, rerun=True, prefix=f"{prefix}{res}-")

    @classmethod
    def train_window_c2c_variation(cls):

        for v in [0.1, 0.3, 0.5]:

            prefix = f'window-c2c-{v}-'

            worker = cls({
                'window_c2c_variation': v
            })

            worker.train(prefix)
            worker.test(worker.greedy_test)

            # additional test
            for res in worker.eval_list:
                tester = cls.load(worker.result_dir, filename=f"{prefix}{res}-worker.pickle")
                tester.test(tester.greedy_test, rerun=True, prefix=f"{prefix}{res}-")

    @classmethod
    def train_window_combine_variation(cls):

        for v in [0.1, 0.3, 0.5]:

            prefix = f'window-combine-{v}-'

            worker = cls({
                'window_c2c_variation': v,
                'window_d2d_variation': v
            })

            worker.train(prefix)
            worker.test(worker.greedy_test)

            # additional test
            for res in worker.eval_list:
                tester = cls.load(worker.result_dir, filename=f"{prefix}{res}-worker.pickle")
                tester.test(tester.greedy_test, rerun=True, prefix=f"{prefix}{res}-")


if __name__ == '__main__':
    FullJobWithVariation.train_baseline()
