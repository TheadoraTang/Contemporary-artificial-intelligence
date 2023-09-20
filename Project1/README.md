# 文本分类
第一次实验：
利用包含逻辑回归/SVM/MLP/决策树等方法进行文本分类。



## Setup

运行实验代码需要安装以下依赖：

- json 版本: 2.0.9
- pandas 版本: 2.1.0
- scikit-learn 版本: 1.3.0
- re 版本: 2.2.1
- numpy 版本: 1.23.5
- torch 版本: 2.0.1+cpu
- keras 版本: 2.12.0
- matplotlib 版本: 3.7.2
  
在终端中使用如下命令即可安装所需依赖：

```shell
pip install pandas
pip install scikit-learn
pip install torch
pip install numpy
pip install keras
pip install matplotlib
```

 

## Repository Structure 

本仓库的文件结构如下：

```
|-- MLP.py	# MLP代码
|-- SVM.py  # SVM代码
|-- TextCNN.py  # TextCNN代码
|-- 决策树.py   # 决策树代码
|-- 逻辑回归.py # 逻辑回归代码
|-- Kfolds.png  # 折数测试结果
|-- test ratio.png  # 划分比例测试结果
|-- DecisionTree_results.txt    # 决策树结果
|-- LR results.txt  # 逻辑回归结果
|-- MLP results.txt # MLP结果
|-- SVM results.txt # SVM结果
|-- TextCNN results.txt # TextCNN结果
|-- train_data.txt	# 训练集
|-- test.txt	# 测试集
|-- submit_sample.txt	# 
|-- README.md   #对仓库的解释
|-- requirements.txt    #本次实验的环境
```



## Usage

在终端中输入

```shell
python MLP.py
python SVM.py
python TextCNN.py
python 决策树.py
python 逻辑回归.py
```

运行.py文件即可。

实验的所有内容都需要放于同一目录下。