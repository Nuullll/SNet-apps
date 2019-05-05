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
from sendgrid.helpers.mail import Mail, Attachment, Content
import base64
import pickle
import torch
from sklearn.svm import SVC


class Worker(object):
    """
    Abstract worker to do the training and evaluation work.
    """

    # sendgrid service
    sg_key = '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'

    @classmethod
    def load(cls, path, filename='worker.pickle'):
        with open(os.path.join(path, filename), 'rb') as f:
            worker = pickle.load(f)

        return worker

    @classmethod
    def relay(cls, relay_path, filename='worker.pickle'):
        worker = Worker.load(relay_path, filename=filename)

        worker.options.update({
            'relay_worker': os.path.join(relay_path, filename)
        })

        worker.summary = {}
        worker.prepare()
        worker.init_logger()

        return worker

    def get_path(self, filename):
        return os.path.join(self.result_dir, filename)

    def __init__(self, options=None):

        if options is None:
            self.options = self.get_default_options()
        else:
            self.options = options

        self.options = self.infer(self.options)

        self.summary = {}

        self.prepare()

        self.init_logger()

        self.load_dataset()

        self._load_network()

        self.reset_test_states()

    def reset_test_states(self):
        self.train_responses = []
        self.test_responses = []
        self.train_labels = []
        self.test_labels = []
        self.train_fire_time = []
        self.test_fire_time = []

        self.save()

    def init_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.result_dir, "worker.log"))
            ]
        )

    @property
    def logger(self):
        return logging.getLogger()

    def load_dataset(self):
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
            't_testing_image': 300,

            'tracker_size': 1000,

            # synapses
            'w_min': 0.1,
            'w_max': 5.0,
            'w_init': 'random',
            'learning_rate_p': 0.8,
            'learning_rate_m': 0.8,
            'tau_p': 3,         # unit: dt
            'tau_m': 3,

            # LIF
            'v_th_rest': 0.4,
            'tau': 200.,        # unit: dt
            'refractory': 0,
            'res': 1.,
            'adapt_factor': 1.1
        }

        return options

    def _export_options(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.options, file, indent=4)

    def prepare(self):
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
            from_email='report@snet.com',
            to_emails=['vfirst218@gmail.com'],
            subject='[SNET] Summary',
            html_content=Content('text/html', to_html())
        )

        image_file = self.summary['image_file']
        with open(image_file, 'rb') as f:
            data = f.read()
            f.close()

        encoded = base64.b64encode(data).decode()
        attachment = Attachment(file_content=encoded, file_type='image/jpg', file_name=os.path.basename(image_file),
                                disposition='inline', content_id='weights')

        message.add_attachment(attachment)

        try:
            sg.client.mail.send.post(request_body=message.get())
        except Exception as e:
            self.logger.warning("Failed to send email: " + e)

    def save(self, filename='worker.pickle'):
        with open(os.path.join(self.result_dir, filename), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def __getstate__(self):
        state = self.__dict__.copy()

        # exclude dataset to speed up pickling
        del state['dataset']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.load_dataset()
        self.init_logger()

    def train(self):
        self.logger.info("Start training.")

        self.network.training_mode()

        for i, (image, label) in enumerate(self.dataset.training_set):

            log_prefix = f"[#{i}] label={label} "

            start_time = self.network.time

            self.logger.debug(log_prefix + "Feeding image. " + f"@{start_time}")
            self.network.feed_image(image)

            self.network.learn_current_image()

            finish_time = self.network.time
            self.logger.info(log_prefix + "Learned. " + f"@{finish_time} (dt={finish_time-start_time})")
            self.logger.debug(log_prefix + f"Response={self.network.OUTPUT.spike_counts_history[-1]}")
            self.logger.debug(log_prefix + f"Threshold={self.network.OUTPUT.v_th}")

            self.post_epoch(i)

        self.post_train()

    def post_epoch(self, i):
        if i % 1000 == 0:
            self.network.W.plot_weight_map(out_file=self.get_path(f'{i}-weights.jpg'))
            self.network.W.plot_update_map(out_file=self.get_path(f'{i}-updates.jpg'))

    def post_train(self):
        """
        Save model after training.
        """

        # self.network.save_model(self.result_dir)
        self.save()

        self.summarize('train')

        self.send()

    def summarize(self, phase='train'):
        def to_html(d):
            html = ""
            for key, value in d.items():
                html += f"{key}: {value}<br>"

            return html

        self.summary['options'] = to_html(self.options)

        image_file = os.path.join(self.result_dir, 'weights.jpg')
        self.network.W.plot_weight_map(image_file)

        self.summary['image_file'] = image_file

        phase_summary = {
            'v_th': self.network.OUTPUT.v_th,
            'time': self.network.time,
            'sample_count': len(self.dataset.training_set) if phase == 'train' else len(self.dataset.testing_set),
        }

        self.summary[phase] = to_html(phase_summary)

    def test(self, test_func, rerun=False):
        # change network mode
        self.network.inference_mode()
        # show weights
        self.network.W.plot_weight_map()

        # get responses
        if not rerun:
            self.load_responses('train')
            self.load_responses('test')
            # check train_responses
            if len(self.train_responses) == 0:
                self.get_responses('train')

            # check test_responses
            if len(self.test_responses) == 0:
                self.get_responses('test')
        else:
            # reset states
            self.reset_test_states()

            self.get_responses('train')
            self.get_responses('test')

        self.predict_labels = []
        self.hit_list = []
        test_func()

        self.export_summary(f"{test_func.__name__}-summary.json")
        self.send()

    def get_responses(self, flag='train'):

        if flag == 'train':
            dataset = self.dataset.training_set
            responses = self.train_responses
            labels = self.train_labels
        else:
            dataset = self.dataset.testing_set
            responses = self.test_responses
            labels = self.test_labels

        for i, (image, label) in enumerate(dataset):

            log_prefix = f"Inference: [#{i}] label={label} "

            start_time = self.network.time

            self.logger.info(log_prefix + "Feeding image. " + f"@{start_time}")
            self.network.feed_image(image)

            self.network.learn_current_image()

            finish_time = self.network.time
            self.logger.info(log_prefix + "Learned. " + f"@{finish_time} (dt={finish_time-start_time})")

            response = self.network.OUTPUT.spike_counts_history[-1]

            self.logger.info(log_prefix + f"Response={response.numpy()}")

            responses.append(response)
            labels.append(label)

        self.export_responses(flag)

    def export_responses(self, flag):
        setattr(self, flag + '_responses', torch.stack(getattr(self, flag + '_responses')))
        setattr(self, flag + '_labels', torch.tensor(getattr(self, flag + '_labels')))

        torch.save(getattr(self, flag + '_responses'), os.path.join(self.result_dir, flag + '_responses.pt'))
        torch.save(getattr(self, flag + '_labels'), os.path.join(self.result_dir, flag + '_labels.pt'))

    def load_responses(self, flag):
        response_file = os.path.join(self.result_dir, flag + '_responses.pt')
        label_file = os.path.join(self.result_dir, flag + '_labels.pt')

        if os.path.exists(response_file):
            setattr(self, flag + '_responses', torch.load(response_file))
        if os.path.exists(label_file):
            setattr(self, flag + '_labels', torch.load(label_file))

    def vote_test(self):

        # assess importance of each output neuron
        train_responses = self.train_responses

        common_scores = train_responses.sum(0)

        self.importances = 1 / common_scores
        self.importances = self.importances / self.importances.sum()
        self.confidence_map = torch.zeros(len(self.options.get('include_categories')), self.network.OUTPUT.size)

        for label, response in zip(self.train_labels, self.train_responses):
            score = response * self.importances
            if score.sum() > 0:
                self.confidence_map[label] += score / score.sum()

        for label, response in zip(self.test_labels, self.test_responses):
            if response.sum() > 0:
                score = torch.matmul(self.confidence_map, (response * self.importances).unsqueeze(0).t())
                _, max_ind = score.max(dim=0)

                predicted = self.options.get('include_categories')[max_ind]
            else:
                score = None
                predicted = label

            self.predict_labels.append(predicted)

            hit = predicted == label
            self.hit_list.append(hit)

            self.logger.info(f"Vote={predicted} score={score} truth={label} hit={hit}")

        hit_count = torch.tensor(self.hit_list).sum().item()
        total = len(self.hit_list)

        accuracy = hit_count / total

        self.logger.info(f"Vote test accuracy: {hit_count}/{total} = {accuracy * 100}%")

        self.summarize('test')
        self.summary['vote test accuracy'] = f"{hit_count}/{total} = {accuracy * 100}%"

    def export_summary(self, filename='summary.json'):
        with open(self.get_path(filename), "w") as f:
            json.dump(self.summary, f, indent=4)

    def svm_test(self):

        svm = SVC()

        svm.fit(self.train_responses, self.train_labels)

        self.predict_labels = svm.predict(self.test_responses)
        accuracy = svm.score(self.test_responses, self.test_labels)

        self.logger.info(f"SVM test accuracy: {accuracy * 100}%")

        self.summarize('test')
        self.summary['SVM test accuracy'] = f"{accuracy * 100}%"

    def legacy_test(self):

        self.confidence_map = torch.zeros(len(self.options.get('include_categories')), self.network.OUTPUT.size)

        for label, response in zip(self.train_labels, self.train_responses):
            if response.sum() > 0:
                self.confidence_map[label] += response / response.sum()

        neuron_activity = self.confidence_map.sum(0)

        # normalize confidence_map
        self.confidence_map = self.confidence_map / neuron_activity * neuron_activity.min() / neuron_activity

        for label, response in zip(self.test_labels, self.test_responses):
            if response.sum() > 0:
                score = torch.matmul(self.confidence_map, response.squeeze(0) / response.sum())

                _, max_ind = score.max(dim=0)

                predicted = self.options.get('include_categories')[max_ind]
            else:
                score = None
                predicted = label

            self.predict_labels.append(predicted)

            hit = predicted == label
            self.hit_list.append(hit)

            self.logger.info(f"Legacy vote={predicted} score={score} truth={label} hit={hit}")

        hit_count = torch.tensor(self.hit_list).sum().item()
        total = len(self.hit_list)

        accuracy = hit_count / total

        self.logger.info(f"Legacy vote test accuracy: {hit_count}/{total} = {accuracy * 100}%")

        self.summarize('test')
        self.summary['legacy vote test accuracy'] = f"{hit_count}/{total} = {accuracy * 100}%"
