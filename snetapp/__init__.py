# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: __init__.py
# @Date: 2019-04-18-12-43
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


import os
from datetime import datetime
import json


class Worker(object):
    """
    Abstract worker to do the training and evaluation work.
    """

    def __init__(self, options=None):

        if options is None:
            self.options = self.get_default_options()
        else:
            self.options = self.infer(options)

    @property
    def results_dir(self):
        return os.path.join(os.path.dirname(__file__), 'results')

    def infer(self, options):
        """
        Infers some fields of options.
        :param options:     <dict>
        :return:            <dict>      inferred options
        """
        options['input_number'] = options['image_size'][0] * options['image_size'][1]

        return options

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

        }

        return self.infer(options)

    def export_options(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.options, file)

    def prepare(self):
        """
        Creates `results` directory for training.
        """
        path = os.path.join(self.results_dir, os.path.basename(__file__),
                            datetime.now().strftime("%c").replace(" ", "-"))
        os.makedirs(path)

        # record options
        option_file = os.path.join(path, "options.json")
        self.export_options(option_file)
