# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: single
# @Date: 2019-04-27-15-39
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


from snetapp.greedy.base import GreedyBaseWorker


class GreedySingleLearner(GreedyBaseWorker):
    """
    Worker which learns only one single image repeatedly.
    """
    def __init__(self, options=None):
        if options is None:
            self.options = self.get_default_options()
        else:
            self.options = self.infer(options)

        super(GreedySingleLearner, self).__init__(self.options)

    def get_default_options(self):
        options = super(GreedySingleLearner, self).get_default_options()

        options.update({
            'single': True
        })

        return self.infer(options)
