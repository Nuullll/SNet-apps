# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: __init__.py
# @Date: 2019-04-18-12-43
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


import os
from datetime import datetime
import json
from snet.dataset.mnist import MNISTLoader
from snet.core import Network
import logging
import sys
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, Content, Email
import base64


class Worker(object):
    """
    Abstract worker to do the training and evaluation work.
    """

    def __init__(self, options=None):

        if options is None:
            self.options = self.get_default_options()
        else:
            self.options = self.infer(options)

        self.summary = {}

        self._prepare()

        self._init_logger()

        self._load_dataset()

        self._load_network()

        # sendgrid service
        self.sg_key = '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'

    def _init_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.result_dir, "worker.log"))
            ]
        )

        self.logger = logging.getLogger()

    def _load_dataset(self):
        self.dataset = MNISTLoader(self.options)

    def _load_network(self):
        self.network = Network(self.options)

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

    def _export_options(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.options, file)

    def _prepare(self):
        """
        Creates `results` directory for training.
        """
        file_path = os.path.abspath(sys.modules[self.__module__].__file__)

        path = os.path.join(os.path.dirname(file_path), 'results', os.path.basename(file_path),
                            datetime.now().strftime("%c").replace(" ", "-").replace(":", "-"))
        os.makedirs(path)

        # record options
        option_file = os.path.join(path, "options.json")
        self._export_options(option_file)

        self.result_dir = path

    def train(self):
        raise NotImplementedError("Worker.train() is not implemented.")

    def test(self):
        raise NotImplementedError("Worker.test() is not implemented.")

    def send(self):
        """
        Sends summary email.
        """
        sg = SendGridAPIClient(self.sg_key)

        def to_html():
            html = ""

            for key, value in self.summary.items():
                html += f"<p><strong>{key}: </strong>{value}</p>"

            html += '<img src="cid:weights"/>'

            return html

        message = Mail(
            from_email=Email('report@snet.com'),
            to_emails=Email('vfirst218@gmail.com'),
            subject='[SNET] Summary',
            html_content=Content('text/html', to_html())
        )

        image_file = self.summary['image_file']
        with open(image_file, 'rb') as f:
            data = f.read()
            f.close()

        encoded = base64.b64encode(data).decode()
        attachment = Attachment()
        attachment.content = encoded
        attachment.type = 'image/jpg'
        attachment.filename = os.path.basename(image_file)
        attachment.disposition = 'inline'
        attachment.content_id = 'weights'

        message.add_attachment(attachment)

        sg.client.mail.send.post(request_body=message.get())
