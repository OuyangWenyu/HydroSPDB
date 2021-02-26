# Streamflow Prediction in Dammed Basins (SPDB) with Deep Learning models

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

- Please make sure that you have all input data. The data list is shown below. If you miss any one, please connect with me: hust2014owen@gmail.com
    - basin_mean_forcing.zip 
    - basinchar_and_report_sept_2011.zip 
    - boundaries_shapefiles_by_aggeco.zip 
    - camels531.zip 
    - camels_attributes_v2.0.zip 
    - gages_streamflow.zip 
    - gagesII_9322_point_shapefile.zip
    - nid.zip 
    - wbdhu4-a-us-september2019-shpfile.zip
- Download the data manually if you don't have an access to my google drive or you don't know how to use PyDrive. 
Then you have to make a directory:
```Shell
# /mnt/sdc/wvo5024/hydro-anthropogenic-lstm/ is my root directory. you should change it to yours
cd /mnt/sdc/wvo5024/hydro-anthropogenic-lstm/example
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

All "xxx_xxx_analysis.py" scripts are run for training and testing, while all "xxx_xxx_result_sectionx.py" files are used for showing the results. 
To run the testing file, please make sure you have run the corresponding training scripts 
and saved some cache for input data and trained model. If there is no cache for input, it will take much time to test models.
You can use the following code to generate some quickly-accessed binary data:

```Shell
python gages_conus_analysis.py --gen_quick_data 1 --quick_data 0 --train_mode 0
```
