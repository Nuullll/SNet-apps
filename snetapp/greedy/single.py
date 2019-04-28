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
            self.options = options

        super(GreedySingleLearner, self).__init__(self.options)

        self.image = None
        self.label = None
        self.epochs = self.options.get('epochs', 100)

    def get_default_options(self):
        options = super(GreedySingleLearner, self).get_default_options()

        options.update({
            'include_categories': list(range(10)),
            'single': True,
            'output_number': 1,
            'epochs': 100,

            'background_firing_rate': 10.,
            'learning_rate_p': 1.0,
            'learning_rate_m': 1.0,
            'scale_factor': 1.
        })

        return options

    def infer(self, options):
        options = super(GreedySingleLearner, self).infer(options)

        options['learning_rate_p'] *= options.get('scale_factor', 1.)
        options['learning_rate_m'] *= options.get('scale_factor', 1.)

        return options

    def train(self):
        self.logger.info("Start single pattern learning.")

        self.network.training_mode()

        image, label = self.dataset.training_set[0]

        self.image = image
        self.label = label

        for i in range(self.epochs):

            log_prefix = f"[Epoch#{i}] label={label} "

            start_time = self.network.time

            self.network.feed_image(image)

            self.network.learn_current_image()

            finish_time = self.network.time

            pscore, bscore = self.score()
            self.logger.info(log_prefix + f"P={pscore}, B={bscore} " + f"@{finish_time} (dt={finish_time-start_time})")

            self.network.W.plot_weight_map()

        self.post_train()

    def score(self):
        image_size = self.options.get('image_size')

        def binarize(image, range):
            return image >= ((range[0] + range[1]) / 2)

        binarized_image = binarize(self.image.view(*image_size), (0, 1))

        binarized_weights = binarize(self.network.W.weights.view(*image_size),
                                     (self.network.W.w_min, self.network.W.w_max))

        pattern_score = binarized_weights[binarized_image].float().sum() / binarized_image.float().sum()
        background_score = (~binarized_weights[~binarized_image]).float().sum() / (~binarized_image).float().sum()

        return pattern_score, background_score


if __name__ == '__main__':
    worker = GreedySingleLearner()

    worker.train()
