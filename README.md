# Analyze impact of human activities on streamflow prediction with Deep Learning model

## setup

As the "environment.yml" file has existed, you can run the code with it to setup the environment (Firstly you need to modify "$prefix" of this file to your own):

```Shell
conda env create -f environment.yml
```

The following is procedure of generating environment.yml:

```Shell
conda create --prefix ./envs python=3.7
conda activate /mnt/sdc/wvo5024/hydro-anthropogenic-lstm/envs
```
create or modify the .condarc file in your system:

```Shell
conda config --set env_prompt '({name})'
```

Then add the following content of .condarc file (you can find it in your home directory):

```.condarc
channel_priority: strict
channels:
  - conda-forge
  - defaults
```
 
and then use conda to install package, for example:

```Shell
conda config --set channel_priority strict
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install -c conda-forge kaggle
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
pip install xlrd
```

Finally, generate environment.yml file by conda:

```Shell
conda env export > environment.yml
```

## workflow

主要参考了项目[machine-learning-project-walkthrough](https://github.com/WillKoehrsen/machine-learning-project-walkthrough)，
重新构建整体流程，便于项目调试和各个模块的测试。

机器学习的pipeline主要有如下部分：

1. 数据获取和数据清理，及格式化为输入格式
2. 探索性数据分析
3. 构建模型，即特征工程与选择
4. 比较不同机器学习模型的性能
5. 执行超参数调优得到最优模型
6. 测试集上评价最优模型

最后就是解释模型结果，并总结写出成果。针对上述6个需要coding的部分，结合本项目实际情况，
分步骤再简要阐述一些。
 
## 数据前处理

首先要明确所需数据内容，然后去查看相应的数据下载方法，尽量使用code下载以实现自动化。

然后需要对数据进行格式转换，转换成输入所需的格式，在这过程中还会有一些数据清洗过程。

## 探索性数据分析

为了辨别关键变量等，首先要对数据进行一些统计分析，先大致看看哪些数据的影响比较大。

## 建模

这部分是核心部分，很多前人工作 can be referred，并不最难。

## 比较不同模型性能

不同模型运行，还有要做模型结果可视化等，以便进行模型比较。

## 超参数调优

可以使用贝叶斯优化等来实现超参数调优，得到最优模型。

## 测试模型

最后一步执行测试，复用前面的代码，略加修改即可。

因此，总结一下code需要注意的部分，首先是数据下载和转换，其次是建模包括训练和测试，还有就是超参数调优。

第一部分放到data文件夹下。下载的数据放到example文件夹下。

第二部分的模型code在hydroDL中，结果可视化部分在visual文件夹中

第三部分在refine文件夹下。

另外如果需要简单探索性分析数据，放在explore文件夹下。

最后执行代码的脚本在app文件夹下。docker是最后制作项目镜像所用。
