# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: base
# @Date: 2019-04-28-17-19
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


from snetapp import Worker


class OriginalBaseWorker(Worker):
    """
    Conventional SNN training worker.
    """
    def __init__(self, options=None):
        if options is None:
            self.options = self.get_default_options()
        else:
            self.options = options

        super(OriginalBaseWorker, self).__init__(self.options)

    def get_default_options(self):
        options = super(OriginalBaseWorker, self).get_default_options()

        options.update({
            'pattern_firing_rate': 1.,  # unit: spikes/dt
            'v_th_rest': 0.4,
            'learning_rate_p': 1.0,
            'learning_rate_m': 1.0,
            'decay': 0.00001,
            'scale_factor': 1e-1,
            'adapt_factor': 1.
        })

        return options

    def infer(self, options):
        options = super(OriginalBaseWorker, self).infer(options)

        scale_factor = options.get('scale_factor', 1.)

        options['learning_rate_p'] *= scale_factor
        options['learning_rate_m'] *= scale_factor
        options['decay'] *= scale_factor
        # options['pattern_firing_rate'] /= scale_factor

        return options


if __name__ == '__main__':
    worker = OriginalBaseWorker()

    worker.train()
