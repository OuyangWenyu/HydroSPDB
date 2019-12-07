from unittest import TestCase
import pandas as pd

from hydroDL.utils.dataset_format import trans_daymet_forcing_file_to_camels


class TestTransDaymetForcingFileToCamels(TestCase):
    def test_trans_daymet_forcing_file_to_camels(self):
        daymet_dir = ''
        output_dir = ''
        result = pd.read_csv(output_dir)
        self.assertEqual(trans_daymet_forcing_file_to_camels(daymet_dir, output_dir), result)
