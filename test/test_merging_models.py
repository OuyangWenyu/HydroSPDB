from hydroDL.meta_models.merging_model import MergingModel
from hydroDL.custom.crits import make_criterion_functions
import unittest
import torch


class TestMerging(unittest.TestCase):
    def setUp(self):
        self.merging_model = MergingModel("Concat", {"cat_dim": 2, "repeat": True})

    def test_merger_runs(self):
        m = self.merging_model(torch.rand(2, 6, 10), torch.rand(4))
        self.assertEqual(m.shape[0], 2)
        self.assertEqual(m.shape[1], 6)
        self.assertEqual(m.shape[2], 14)

    def test_merger_two(self):
        m = self.merging_model(torch.rand(2, 6, 20), torch.rand(4))
        self.assertEqual(m.shape[2], 24)

    def test_crit_functions_list(self):
        res = make_criterion_functions(["MSE", "RMSE", "MAPE"])
        self.assertIsInstance(res, list)

    def test_crit_functions_dict(self):
        res = make_criterion_functions({"MASELoss": {"baseline_method": "mean"}, "MSE": {}})
        self.assertIsInstance(res, list)


if __name__ == '__main__':
    unittest.main()
