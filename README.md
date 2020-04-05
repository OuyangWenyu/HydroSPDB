# Analyze impact of human activities to streamflow prediction with Deep Learning model

## Paper idea

选择85-05年，20年数据，选择无数据时间长度不超过10%的流域站点进行计算，单独计算和一起计算分别尝试。

- 目前得到的结果是分开计算的不太好，需要进一步检查；
- 另外，在其他文献中都能看到 basin area 是影响计算结果的关键指标，所以可以做一个histogram看看是什么样的。

有一些问题可以问：

1. 有些流域可预测，有些流域结果很差，这能说明什么？加州的比较差可能是因为水库很多，中部差是因为地表水会很快下渗，然后在地下流动到一定地方之后再反补出来，会造成径流预测不准。这可以有什么hypothesis？对加州地区的水库做进一步分析？
2. DL能不能分离各类影响，比如能分析climate change，是怎么分析的？DL的预测记录下了climate change？逻辑上，DL记录下了时间序列的趋势，所以从中可以看出climate change
所以land use如何分析呢？可能是类似地，如果能预测出结果，就能统计分析判断land use的变化。

进一步的理解和认识需要补充一些文献，比如2013年开启水文十年的关于水-社会-人类活动的综述文献等。

- [Change in hydrology and society—The IAHS Scientific Decade 2013–2022](https://doi.org/10.1080/02626667.2013.809088)
- [A Transdisciplinary Review of Deep Learning Research and Its Relevance for Water Resources Scientists](https://doi.org/10.1029/2018WR022643)

If a method of classification need to be incorporated to the pre-processing, this paper can be referred: https://doi.org/10.1016/j.jhydrol.2013.03.024

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

PCA等方式是相对比较常用的方法。不过对于深度学习算法，这一步可能不是太重要，因此这里可以先不展开，暂时仅以计算数据统计值为主。

## 建模

这部分是核心部分，不过因为有很多前人工作，可能并不是最难的部分。

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
