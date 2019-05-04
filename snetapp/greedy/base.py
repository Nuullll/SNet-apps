# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: base
# @Date: 2019-04-18-12-55
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


from snetapp import Worker
import torch
import os


class GreedyBaseWorker(Worker):
    """
    Greedy training worker.

    options:
    Greedy Training, P&B Phases
    MNIST 28x28 0,1,2
    """

    def __init__(self, options=None):
        if options is None:
            self.options = self.get_default_options()
        else:
            self.options = options

        super(GreedyBaseWorker, self).__init__(self.options)

    def get_default_options(self):
        options = super(GreedyBaseWorker, self).get_default_options()

        options.update({
            'greedy': True,
            'pattern_firing_rate': 1.,  # unit: spikes/dt
            'background_firing_rate': 5.,
            't_background_phase': 10,  # unit: dt
        })

        return options

    def test(self, test_func, rerun=False):
        if not test_func == self.greedy_test:
            super(GreedyBaseWorker, self).test(test_func=test_func, rerun=rerun)

            return

        # change network mode
        self.network.inference_mode()
        # show weights
        self.network.W.plot_weight_map()

        # get responses
        if not rerun:
            self.load_greedy_responses('train')
            self.load_greedy_responses('test')
            # check train_responses
            if len(self.train_responses) == 0:
                self.get_greedy_responses('train')

            # check test_responses
            if len(self.test_responses) == 0:
                self.get_greedy_responses('test')
        else:
            # reset states
            self.reset_test_states()

            self.get_greedy_responses('train')
            self.get_greedy_responses('test')

        self.predict_labels = []
        self.hit_list = []
        test_func()

        self.export_summary(f"{test_func.__name__}-summary.json")
        self.send()

    def get_greedy_responses(self, flag='train'):

        if flag == 'train':
            dataset = self.dataset.training_set
            responses = self.train_responses
            labels = self.train_labels
            fire_time = self.train_fire_time
        else:
            dataset = self.dataset.testing_set
            responses = self.test_responses
            labels = self.test_labels
            fire_time = self.test_fire_time

        for i, (image, label) in enumerate(dataset):

            log_prefix = f"Greedy inference: [#{i}] label={label} "

            start_time = self.network.time

            self.logger.debug(log_prefix + f"Feeding image. @{start_time}")
            self.network.feed_image(image)

            self.network.learn_current_image(force_greedy=True)

            finish_time = self.network.time
            self.logger.info(log_prefix + f"Learned. @{finish_time} (dt={finish_time-start_time})")

            response = self.network.OUTPUT.spike_counts_history[-1]
            time_used = self.network.OUTPUT.time_history[-1]

            self.logger.debug(log_prefix + f"Response={response.numpy()}")

            responses.append(response)
            labels.append(label)
            fire_time.append(time_used)

        # export greedy responses
        setattr(self, flag + '_responses', torch.stack(getattr(self, flag + '_responses')))
        setattr(self, flag + '_labels', torch.tensor(getattr(self, flag + '_labels')))
        setattr(self, flag + '_fire_time', torch.tensor(getattr(self, flag + '_fire_time')))

        torch.save(getattr(self, flag + '_responses'), self.get_path(f"{flag}-greedy-responses.pt"))
        torch.save(getattr(self, flag + '_labels'), self.get_path(f"{flag}-greedy-labels.pt"))
        torch.save(getattr(self, flag + '_fire_time'), self.get_path(f"{flag}-greedy-fire-time.pt"))

    def load_greedy_responses(self, flag):
        response_file = self.get_path(f"{flag}-greedy-responses.pt")
        label_file = self.get_path(f"{flag}-greedy-labels.pt")
        time_file = self.get_path(f"{flag}-greedy-fire-time.pt")

        if os.path.exists(response_file):
            setattr(self, flag + '_responses', torch.load(response_file))
        if os.path.exists(label_file):
            setattr(self, flag + '_labels', torch.load(label_file))
        if os.path.exists(time_file):
            setattr(self, flag + '_fire_time', torch.load(time_file))

    def greedy_test(self):

        self.confidence_map = torch.zeros(len(self.options.get('include_categories')), self.network.OUTPUT.size)

        for label, response, time in zip(self.train_labels, self.train_responses, self.train_fire_time.float()):
            score = response / (time + 1)

            self.confidence_map[label] += score

        for label, response, time in zip(self.test_labels, self.test_responses, self.test_fire_time.float()):
            if response.sum() > 0:
                _, ind = response.max(dim=0)
                _, ind = self.confidence_map[:, ind].max(dim=0)

                predicted = ind
            else:
                predicted = label

            self.predict_labels.append(predicted)

            hit = predicted == label
            self.hit_list.append(hit)

            self.logger.info(f"Greedy vote={predicted} response={response} truth={label} hit={hit}")

        hit_count = torch.tensor(self.hit_list).sum().item()
        total = len(self.hit_list)

        accuracy = hit_count / total

        self.logger.info(f"Greedy test accuracy: {hit_count}/{total} = {accuracy * 100}%")

        self.summarize('test')
        self.summary['greedy test accuracy'] = f"{hit_count}/{total} = {accuracy * 100}%"


def train_whole_mnist():
    options = GreedyBaseWorker().options

    options.update({
        'include_categories': list(range(10)),
        'output_number': 50,
    })

    worker = GreedyBaseWorker(options)
    worker.train()

    worker.test(worker.greedy_test)


if __name__ == "__main__":
    # worker = GreedyBaseWorker()
    # worker.train()

    # path = r'E:\Projects\SNet-apps\snetapp\greedy\results\base.py\Fri-May--3-15-54-15-2019'
    # worker = GreedyBaseWorker.load(path)
    #
    # worker.test(worker.greedy_test)

    train_whole_mnist()
