import os
import unittest

import definitions
from data.cache.cache_factory import cache_dataset
from data.config import default_config_file, cmd, update_cfg
from data.data_camels import Camels
from hydroDL.trainer import train_and_evaluate


class CamelsTrainEvaluateTests(unittest.TestCase):
    def setUp(self):
        camels_dir = os.path.join(definitions.DATASET_DIR, "camels")
        dataset_name = "CAMELS"
        project_name = "camels/exp1"
        # project_name = "camels/cache-671sites-19851001_19951001_19951001_20051001-17attr-6forcing"
        # project_name = "camels/exp2"
        self.config_data = default_config_file(camels_dir, dataset_name)
        weight_path = None
        # weight_path = os.path.join(config_data["dataset_params"]["test_path"], "26_March_202109_27AM_model.pth")
        self.args = cmd(sub=project_name, download=0, model_name="KuaiLSTM", cache_write=1,
                        gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                                 "01055000", "01057000", "01170100"], batch_size=5, rho=20,  # batch_size=100, rho=365,
                        var_t=["dayl", "prcp", 'srad', 'tmax', 'tmin', 'vp'],
                        data_loader="StreamflowDataModel", scaler="DapengScaler", n_feature=23, hidden_size=256,
                        weight_path=weight_path, train_epoch=20)

    def test_train_evaluate(self):
        config_data = self.config_data
        args = self.args
        update_cfg(config_data, args)

        if config_data["dataset_params"]["cache_write"]:
            dataset_params = config_data["dataset_params"]
            dataset = Camels(dataset_params["data_path"], dataset_params["download"])
            cache_dataset(dataset_params, dataset)

        train_and_evaluate(config_data)


if __name__ == "__main__":
    unittest.main()
