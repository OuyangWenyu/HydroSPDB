<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-04-20 11:52:37
 * @LastEditTime: 2023-04-20 23:02:54
 * @LastEditors: Wenyu Ouyang
 * @Description: README.md for HydroSPDB
 * @FilePath: /HydroSPDB/README.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# HydroSPDB

This is the Code for Streamflow Prediction in (Dammed) Basins (SPDB) with Deep Learning models.

If you use this code, please cite the following paper:

```BibTeX
@article{OUYANG2021126455,
title = {Continental-scale streamflow modeling of basins with reservoirs: Towards a coherent deep-learning-based strategy},
journal = {Journal of Hydrology},
volume = {599},
pages = {126455},
year = {2021},
issn = {0022-1694},
doi = {https://doi.org/10.1016/j.jhydrol.2021.126455},
url = {https://www.sciencedirect.com/science/article/pii/S0022169421005023},
author = {Wenyu Ouyang and Kathryn Lawson and Dapeng Feng and Lei Ye and Chi Zhang and Chaopeng Shen},
keywords = {Hydrologic modeling, Reservoir, Deep learning, LSTM, Degree of regulation},
abstract = {A large fraction of major waterways have dams influencing streamflow, which must be accounted for in large-scale hydrologic modeling. However, daily streamflow prediction for basins with dams is challenging for various modeling approaches, especially at large scales. Here we examined which types of dammed basins could be well represented by long short-term memory (LSTM) models using readily-available information, and delineated the remaining challenges. We analyzed data from 3557 basins (83% dammed) over the contiguous United States and noted strong impacts of reservoir purposes, degree of regulation (dor), and diversion on streamflow modeling. While a model trained on a widely-used reference-basin dataset performed poorly for non-reference basins, the model trained on the whole dataset presented a median Nash-Sutcliffe efficiency coefficient (NSE) of 0.74. The zero-dor, small-dor (with storage of approximately a month of average streamflow or less), and large-dor basins were found to have distinct behaviors, so migrating models between categories yielded catastrophic results, which means we must not treat small-dor basins as reference ones. However, training with pooled data from different sets yielded optimal median NSEs of 0.72, 0.79, and 0.64 for these respective groups, noticeably stronger than existing models. These results support a coherent modeling strategy where smaller dams (storing about a month of average streamflow or less) are modeled implicitly as part of basin rainfall-runoff processes; then, large-dor reservoirs of certain types can be represented explicitly. However, dammed basins must be present in the training dataset. Future work should examine separate modeling of large reservoirs for fire protection and irrigation, hydroelectric power generation, and flood control.}
}
```

## How to run

**Notice: ONLY tested in an "Ubuntu" machine with NVIDIA GPUs**

### Clone the repository

Fork this repository and clone it to your local machine.

```bash
# xxx is your github username
git clone git@github.com:xxxx/HydroSPDB.git
cd HydroSPDB
```

### Install dependencies

```bash
# if you have mamaba installed, it's faster to use mamba to create a new environment than conda
# if you don't have mamba, please install it
# conda install -c conda-forge mamba
mamba env create -f environment.yml
# after the environment is created, activate it
conda activate SPDB
# check if packages are installed correctly and HydroMTL is runnable
pytest tests
```

### Prepare data

Firstly, download data manually from my [google drive](https://drive.google.com/drive/folders/1MA5HKTa2e6ZCWIoTQkLMBmREpz6QjVDH?usp=share_link).

Then, put the data in a folder and set this fold in definitions.py.
 
A recommeded way to config the data path is to create a file named `definitions_private.py` in the root directory of the project, and set the data path in it.

You can set the data path in `definitions_private.py` as follows:

```python
# xxx is your path
DATASET_DIR = xxx # This is your Data source directory
RESULT_DIR = xxx # This is your result directory
```

Run the following command to unzip all data.

```bash
cd scripts
python prepare_data.py
```

### Train and test

Firstly, choose basins to train the model. A file `gage_id.csv` should be created in the `RESULT_DIR` folder. The file should contain the basin id of basins to train the model. For example, the file `gage_id.csv` for 7 basins is as follows:

```csv
GAUGE_ID
01407500
01591000
01669520
02046000
02051500
02077200
02143500
```

One can choose more basins. Notice the GAUGE_ID should be in order.


After data is ready, run the following command to train the model.

```bash
# if not in the scripts folder, cd to it
# cd scripts
# train models
# One can use --cache_path to avoid reading forcing, attribute and streamflow data again. cache_path is the directory to save forcing, attribute and streamflow data.
python train_model.py --exp exp001 --train_period 2001-10-01 2011-10-01 --test_period 2011-10-01 2016-10-01 --ctx 0 --random 1234
```

One can use the trained model to test in any period.

```bash
# if not in the scripts folder, cd to it
# cd scripts
# NOTE: We set test exp as trainexp+"0", for example, train exp is exp001, then, test exp is exp0010
# test_period can be any period
python evaluate_task.py --exp exp0010 --test_period 2016-10-01 2019-10-01 --cache_path /your/path/to/cache_directory_for_attributes_forcings_targets/or/None --weight_path /your/path/to/trained_model_pth_file
```

### Plot

To show the results visually, run the following command.

```bash
# if not in the scripts folder, cd to it
# cd scripts

```