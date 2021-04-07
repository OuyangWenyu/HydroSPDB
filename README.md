# Streamflow Prediction in Dammed Basins (SPDB) with Deep Learning models

This is the code for [this paper](https://arxiv.org/abs/2101.04423).

## Code

Notice: ONLY tested in an "Ubuntu" machine with NVIDIA GPUs

Clone this repo to your local directory.

Please use the "master" branch, but if you want to use another branch, for example, the "dev" branch, you can run the commands:

```git
git fetch origin dev
git checkout -b dev origin/dev
```

## Setup

Please move to the root directory of this repo and then use the following code to generate the python environment:

```Shell
conda env create -f environment.yml
```
 
The main packages are as follows:

```conda
pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 pydrive geopandas netcdf4 scipy tensorboard future matplotlib statsmodels seaborn cartopy geoplot easydict pyarrow xlrd
```

After installing the environment, activate it before running the code:

```Shell
conda activate SPDB
```

## Workflow

References:

- [machine-learning-project-walkthrough](https://github.com/WillKoehrsen/machine-learning-project-walkthrough)

Pipeline：

1. config, download and process the data
2. build and train the model
3. test the model
4. visualization

Directories:

- The scripts which can run main tasks were in "app" dir
- "data" dir was used for data pre-process
- All data were put in the "example" directory
- Statistics code was in "explore" dir
- "hydroDL" had the core code
- "refine" has not been used yet. It would be used for optimization of hyper-param
- All test scripts were written with Unittest framework, and in "test" dir
- Visualization codes were in "visual" dir

## Usage

- Please make sure that you have all input data. The data list is shown below.
    - basin_mean_forcing.zip
    - basinchar_and_report_sept_2011.zip
    - boundaries_shapefiles_by_aggeco.zip
    - camels531.zip
    - camels_attributes_v2.0.zip
    - gages_streamflow.zip
    - gagesII_9322_point_shapefile.zip
    - nid.zip
    - 59692a64e4b0d1f9f05f : this is the GAGES-II time series dataset, yet not used in [this paper](https://arxiv.org/abs/2101.04423).
- Download the data manually. (We'll publish them later)

Then make a directory:

```Shell
# /mnt/sdc/wvo5024/HydroSPDB/ is my root directory. Please change it to yours.
cd /mnt/sdc/wvo5024/HydroSPDB/example
mkdir data
cd data
mkdir gages
```
and put all downloaded zip files in it.
- Run the data/config.py, and then you are ready to run this repo
- Now you can run the script, such as "app/streamflow/gages_conus_analysis.py", using Ubuntu "screen" tool: 

```Shell
screen -S xxx (give the process a name)
conda activate SPDB
cd xx/app/streamflow （move to the directory）
screen -r xxx (use the id of the process to enter the process. you can use "screen -ls" to see what the id is)
python gages_conus_analysis.py --sub basic/exp37 --cache_state 1
```

All "xxx_xxx_analysis.py" scripts are run for training and evaluating, while all "xxx_xxx_result_sectionx.py" files are used for showing the results.
Please make sure you have run the corresponding training scripts and saved some cache for input data and trained model before running test scripts.
If there is no cache for input, it will take much time to test models.

Use the following code to generate some quickly-accessed binary data:

```Shell
python gages_conus_analysis.py --gen_quick_data 1 --train_mode 0
```

- The LSTM model we used is the "hydroDL.model.rnn.CudnnLstmModel"
- The calling sequence of functions is: "hydroDL.master.master.master_train" -> "hydroDL.model.model_run.model_train" -> "hydroDL.model.rnn.CudnnLstmModel"
- Other models in "hydroDL" directory have been tried, yet have NOT been used in [this paper](https://arxiv.org/abs/2101.04423).
