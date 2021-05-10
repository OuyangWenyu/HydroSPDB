import unittest
import pandas as pd

from explore.stat import statError
from hydroDL.trainer import load_result, stat_ensemble_result, stat_result


class ModelEvaluateTests(unittest.TestCase):
    def setUp(self):
        self.save_dir = "/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp2/"
        self.epoch = 300
        self.save_dirs = ["/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp2/",
                          "/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp3/",
                          "/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp4/",
                          "/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp5/",
                          "/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp6/",
                          "/mnt/data/owen411/code/hydro-spdb-dl/example/gages/exp7/"]

    def test_load_result(self):
        save_dir = self.save_dir
        epoch = self.epoch
        pred, obs = load_result(save_dir, epoch)
        inds = statError(obs, pred)
        inds_df = pd.DataFrame(inds)
        print(inds_df.median())

    def test_stat_result(self):
        save_dir = self.save_dir
        epoch = self.epoch
        inds_df = stat_result(save_dir, epoch)
        print(inds_df.median())

    def test_stat_ensemble_results(self):
        print("load ensemble results")
        save_dirs = self.save_dirs
        test_epoch = self.epoch
        inds_df = stat_ensemble_result(save_dirs, test_epoch)
        print(inds_df.median())


if __name__ == '__main__':
    unittest.main()
