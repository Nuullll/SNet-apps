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
        })

        return options
