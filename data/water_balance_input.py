from utils.hydro_math import concat_two_3darray


class GagesWaterBalanceDataModel(object):
    """TODO:data model for water balance informed neural network"""

    def __init__(self, data_model, gridmet_data_model, is_use_cropet=False):
        self.data_model = data_model
        self.gridmet_data_model = gridmet_data_model
        self.is_use_cropet = is_use_cropet

    def load_data(self):
        model_dict = self.data_model.data_source.data_config.model_dict
        x_daymet, y, c = self.data_model.load_data(model_dict)
        x, cet = self.gridmet_data_model.load_data()
        if self.is_use_cropet:
            # add cet data to x
            xet = concat_two_3darray(x, cet)
            return xet, y, c
        else:
            return x, y, c
