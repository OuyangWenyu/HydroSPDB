# Impact of Reservoirs on streamflow prediction with Deep Learning model

## Code

star/fork this project, and then clone this repo to your local directory.

For branch, for example, the "resequ" branch:

```git
git fetch origin resequ
git checkout -b resequ origin/resequ
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
# if xlrd cannot be installed by conda, pip could be tried: pip install xlrd
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

At first, please make sure that you have all input data. If you don't have, please connect with me: hust2014owen@gmail.com
