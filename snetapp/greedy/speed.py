# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: speed
# @Date: 2019-04-30-13-29
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


from snetapp.greedy.base import GreedyBaseWorker
import glob
import os


class SpeedTester(GreedyBaseWorker):
    """
    Evaluates the convergence speed.
    """

    def get_default_options(self):
        options = super(SpeedTester, self).get_default_options()

        options.update({
            'background_firing_rate': 5.
        })

        return options

    def post_epoch(self, i):
        if i % 1000 == 0:
            # save worker
            self.save(filename=f"{i}-worker.pickle")

            # save and show maps
            self.network.W.plot_weight_map(out_file=self.get_path(f"{i}-weight.jpg"))
            self.network.W.plot_update_map(out_file=self.get_path(f"{i}-update.jpg"))

    @classmethod
    def series_test(cls, path):
        test_func = SpeedTester.greedy_test

        checkpoints = glob.glob(os.path.join(path, "*worker.pickle"))
        for checkpoint in [os.path.basename(x) for x in checkpoints]:
            worker = SpeedTester.load(path, checkpoint)
            if worker is None:
                continue
            worker.test(test_func, rerun=True)

            worker.export_summary(checkpoint.replace(".pickle", "-summary.json"))


if __name__ == '__main__':
    tester = SpeedTester()
    tester.train()

    SpeedTester.series_test(tester.result_dir)

    # path = r'E:\Projects\SNet-apps\snetapp\greedy\results\speed.py\Tue-Apr-30-13-42-45-2019'
    # SpeedTester.series_test(path)
