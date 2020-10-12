# Impact of Reservoirs on streamflow prediction with Deep Learning model

## Code

Clone this repo to your local directory.

You should use the "master" branch, but if you want to use another branch, for example, the "dev" branch, you can run the commands:

```git
git fetch origin dev
git checkout -b dev origin/dev
```

## Setup

The following are processes of generating environment.yml:

```Shell
conda create --prefix ./envs python=3.7
# /mnt/sdc/wvo5024/hydro-anthropogenic-lstm/ is my root directory. Other dirs also could be used.
conda activate /mnt/sdc/wvo5024/hydro-anthropogenic-lstm/envs
```
 
and then use conda to install package, for example:

```Shell
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install -c conda-forge pydrive
conda install -c conda-forge geopandas
conda install -c conda-forge netcdf4
conda install -c conda-forge scipy
conda install -c conda-forge tensorboard
conda install -c conda-forge future
conda install -c conda-forge matplotlib
conda install -c conda-forge statsmodels
conda install -c conda-forge seaborn
conda install -c conda-forge cartopy
conda install -c conda-forge geoplot
conda install -c conda-forge easydict
conda install -c conda-forge pyarrow
conda install -c conda-forge xlrd
# if any package cannot be installed by conda (for example, xlrd), pip could be tried after all others were installed: pip install xlrd
```

Finally, generate environment.yml file by conda:

```Shell
conda env export > environment.yml
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
- "refine" was not used yet. It would be used for optimization of hyper-param
- All test scripts were written with Unittest framework, and in "test" dir
- Visualization in "visual" dir

## Usage

- Please make sure that you have all input data. If you don't have, please connect with me: hust2014owen@gmail.com
- You should download the data manually if you don't have an access to my google drive or you don't know how to use PyDrive. 
Then you have to make a directory:
```Shell
# /mnt/sdc/wvo5024/hydro-anthropogenic-lstm/ is my root directory. you should change it to yours
cd /mnt/sdc/wvo5024/hydro-anthropogenic-lstm/example
mkdir data
mkdir gages
```
and put all downloaded zip files in it.
- Run the data/config.py, and then you are ready to run this repo
- Now you can run the script, such as "app/streamflow/gages_conus_analysis.py", using Ubuntu "screen" tool: 

```Shell
screen -S xxx (give the process a name)
conda activate xxxxx (the python environment you created for this repo. use "conda env list" to see the name)
cd xx/app/streamflow （move to the directory）
screen -r xxx (use the id of the process to enter the process. you can use "screen -ls" to see what the id is)
python gages_conus_analysis.py --sub basic/exp37 --cache_state 1
```

All "xxx_xxx_analysis" scripts are run for training and testing, while all "xxx_xxx_forecast" files are used for showing the results. 
To run the testing file, please make sure you have run the corresponding training scripts 
and saved some cache for input data and trained model. If there is no cache for input, it will take much time to do test.
You can use the following code to generate some quickly-accessed binary data:

```Shell
python gages_conus_analysis.py --gen_quick_data 1 --quick_data 0 --train_mode 0
```
