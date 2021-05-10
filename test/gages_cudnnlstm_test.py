import os
import unittest
import definitions
from data.cache.cache_factory import cache_dataset
from data.data_gages import Gages
from data.config import default_config_file, cmd, update_cfg
from hydroDL.trainer import train_and_evaluate


class GagesTrainEvaluateTests(unittest.TestCase):
    def setUp(self):
        gages_dir = os.path.join(definitions.DATASET_DIR, "gages")
        dataset_name = "GAGES"
        self.config_data = default_config_file(gages_dir, dataset_name)
        project_name = "gages/exp1"
        attrBasin = ['DRAIN_SQKM', 'ELEV_MEAN_M_BASIN', 'SLOPE_PCT']
        attrLandcover = ['DEVNLCD06', 'FORESTNLCD06', 'PLANTNLCD06', 'WATERNLCD06', 'SNOWICENLCD06', 'BARRENNLCD06',
                         'SHRUBNLCD06', 'GRASSNLCD06', 'WOODYWETNLCD06', 'EMERGWETNLCD06']
        attrSoil = ['AWCAVE', 'PERMAVE', 'RFACT', 'ROCKDEPAVE']
        attrGeol = ['GEOL_REEDBUSH_DOM', 'GEOL_REEDBUSH_DOM_PCT']
        attrHydro = ['STREAMS_KM_SQ_KM']
        attrHydroModDams = ['NDAMS_2009', 'STOR_NOR_2009', 'RAW_DIS_NEAREST_MAJ_DAM']
        attrHydroModOther = ['CANALS_PCT', 'RAW_DIS_NEAREST_CANAL', 'FRESHW_WITHDRAWAL', 'POWER_SUM_MW']
        attrPopInfrastr = ['PDEN_2000_BLOCK', 'ROADS_KM_SQ_KM', 'IMPNLCD06']
        varC = attrBasin + attrLandcover + attrSoil + attrGeol + attrHydro + attrHydroModDams + attrHydroModOther + attrPopInfrastr
        self.args = cmd(sub=project_name, download=0, model_name="KuaiLSTM", opt="Adadelta", rs=1234, cache_write=1,
                        scaler="DapengScaler", data_loader="StreamflowDataModel", batch_size=5, rho=20,
                        var_c=varC, n_feature=len(varC) + 7, train_epoch=20,
                        loss_func="RMSESum", train_period=["1992-01-01", "1994-01-01"],
                        test_period=["1995-10-01", "1996-10-01"], hidden_size=256,
                        gage_id=["01013500", "01022500", "01030500", "01031500", "01047000", "01052500", "01054200",
                                 "01055000", "01057000", "01170100"], )

    def test_gages_train_evaluate(self):
        config_data = self.config_data
        args = self.args
        update_cfg(config_data, args)

        if config_data["dataset_params"]["cache_write"]:
            dataset_params = config_data["dataset_params"]
            dataset = Gages(dataset_params["data_path"], dataset_params["download"])
            cache_dataset(dataset_params, dataset)

        train_and_evaluate(config_data)
        print("All processes are finished!")


if __name__ == '__main__':
    unittest.main()
