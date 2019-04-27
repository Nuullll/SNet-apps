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
            'include_categories': list(range(10)),
            'single': True,
            'output_number': 1
        })

        return self.infer(options)

    def train(self):
        self.logger.info("Start single pattern learning.")

        self.network.training_mode()

        image, label = self.dataset.training_set[0]

        epochs = 100

        for i in range(epochs):

            log_prefix = f"[Epoch#{i}] label={label} "

            start_time = self.network.time

            self.network.feed_image(image)

            self.network.learn_current_image()

            finish_time = self.network.time

            self.logger.info(log_prefix + "Learned. " + f"@{finish_time} (dt={finish_time-start_time})")

            self.network.W.plot_weight_map()

        self.post_train()


if __name__ == '__main__':
    worker = GreedySingleLearner()

    worker.train()
