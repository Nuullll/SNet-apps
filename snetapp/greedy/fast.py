# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: fast
# @Date: 2019-05-07-12-02
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


from snetapp.greedy.speed import SpeedTester


class FastTester(SpeedTester):
    """
    Fast evaluation on 0,1 patterns, with 1000 samples.
    """
    def __init__(self, options=None):
        if options is None:
            self.options = self.get_default_options()
        else:
            self.options = options

        super(FastTester, self).__init__(self.options)

    def get_default_options(self):
        options = super(FastTester, self).get_default_options()

        options.update({
            'output_number': 4,
            'include_categories': [0, 1],
            'training_samples': 1000,
            'testing_samples': 1000,

            # variables
            'learning_rate_p': 0.8,
            'learning_rate_m': 0.8,

            'background_firing_rate': 5.,

            'w_max': 5.,
            'w_min': 1.0,

            # 'update_variation': 0.1,
            'failure_rate': 0.9
        })

        return options

    def post_epoch(self, i):
        if i % 100 == 0:
            # save worker
            self.save(filename=f"{i}-worker.pickle")

            # save and show maps
            self.network.W.plot_weight_map(out_file=self.get_path(f"{i}-weight.jpg"))


if __name__ == '__main__':
    worker = FastTester()
    worker.train()

    FastTester.series_test(worker.result_dir)
