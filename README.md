# HydroSPDB

This is the Code for Streamflow Prediction in (Dammed) Basins (SPDB) with Deep Learning models.

## Code

Notice: ONLY tested in an "Ubuntu" machine with NVIDIA GPUs

Clone this repo to your local directory.

Please use the "master" branch, but if you want to use another branch, for example, the "dev" branch, you can run the commands:

```Shell
git fetch origin dev
git checkout -b dev origin/dev
```

Pipeline：

- config, download and process the data
- build and train the model
- test the model
- visualization

Directories:

- The scripts which can run main tasks were in "app" dir
- "data" dir was used for data pre-process
- All data were put in the "example" directory
- Statistics code was in "explore" dir
- "hydroDL" had the core code
- "refine" has not been used yet. It would be used for optimization of hyper-param
- All test scripts were written with Unittest framework, and in "test" dir
- Visualization codes were in "visual" dir

## Setup

Set the python virtual environment.

Make sure Conda (miniconda is recommended) has been installed, if not, please
see [here](https://github.com/OuyangWenyu/hydrus/blob/master/1-basic-envir/2-python-envir.md).

Then move to the root directory of this repo and run the following command in the terminal:

```Shell
conda env create -f environment.yml
```

Wait some minutes, and then all dependencies will be installed.

## Configuration

Download the source data and put them in a directory.

Then, assign it to the "DATASET_DIR" in definitions.py.

Next, Define the Configuration File.

The template Configuration File (now only a python-dict version is supported) is in config.py.

A configuration file is composed of four major required sub-parts:

1. model_params
2. dataset_params
3. training_params
4. metrics

You can easily see all the parameters that you specify to your model.

Now only some choices are supported (I will list them later).

## Usage

I recommend starting with CudnnLstmModel/Vanilla LSTM with the CAMELS dataset.

```Shell
screen -S xxx (give the process a name)
conda activate hydrodl
cd xx/app/streamflow （move to the directory）
screen -r xxx (use the id of the process to enter the process. you can use "screen -ls" to see what the id is)
python camels671_analysis.py --sub test/exp1 --download 1 --model_name KuaiLSTM --opt Adadelta --rs 1234 --cache_write 1 --scaler DapengScaler --data_loader StreamflowDataModel --batch_size 5 --rho 20 --n_feature 24 --gage_id 01013500 01022500 01030500 01031500 01047000 01052500 01054200 01055000 01057000 01170100
```

## Acknowledgement

Thanks to the following repositories:

- [mhpi/hydroDL](https://github.com/mhpi/hydroDL)
- [neuralhydrology/neuralhydrology](https://github.com/neuralhydrology/neuralhydrology)
- [AIStream-Peelout/flow-forecast](https://github.com/AIStream-Peelout/flow-forecast)
