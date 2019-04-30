# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: speed
# @Date: 2019-04-30-13-29
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


from snetapp.greedy.base import GreedyBaseWorker


class SpeedTester(GreedyBaseWorker):
    """
    Evaluates the convergence speed.
    """

    def post_epoch(self, i):
        if i % 1000 == 0:
            # save worker
            self.save(filename=f"{i}-worker.pickle")

            # save and show maps
            self.network.W.plot_weight_map(out_file=self.get_path(f"{i}-weight.jpg"))
            self.network.W.plot_update_map(out_file=self.get_path(f"{i}-update.jpg"))

    @classmethod
    def series_test(cls, path):
        for i in range(0, 20000, 1000):
            worker = SpeedTester.load(path, f"{i}-worker.pickle")
            worker.test(worker.svm_test, rerun=True)

            worker.export_summary(f"{i}-summary.json")

        # final evaluation
        worker = SpeedTester.load(path)
        worker.test(worker.svm_test, rerun=True)
        worker.export_summary(f"final-summary.json")


if __name__ == '__main__':
    # tester = SpeedTester()
    # tester.train()

    path = r'E:\Projects\SNet-apps\snetapp\greedy\results\speed.py\Tue-Apr-30-13-42-45-2019'
    SpeedTester.series_test(path)
