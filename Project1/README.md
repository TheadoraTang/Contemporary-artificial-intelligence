# 文本分类
第一次实验：
利用包含逻辑回归/SVM/MLP/决策树等方法进行文本分类。



## Setup

运行实验代码需要安装以下依赖：

- pandas == 2.1.0
- scikit-learn == 1.3.0
- numpy == 1.23.5
- torch == 2.0.1
- keras == 2.12.0
- matplotlib == 3.7.2
  
在终端中使用如下命令即可安装所需依赖：

```shell
pip install pandas
pip install scikit-learn
pip install torch
pip install numpy
pip install keras
pip install matplotlib
```
本次实验用到的库：

1. json：用于处理JSON数据的库。
2. pandas：数据处理库。
3. sklearn：机器学习库。
4. TfidfVectorizer：Scikit-Learn的一部分，用于TF-IDF向量化。
5. MLPClassifier：Scikit-Learn中的多层感知器分类器。
6. SVC：Scikit-Learn中的SVM分类器。
7. numpy：数值计算库。
8. torch：PyTorch深度学习框架。
9. nn 和 optim 模块来自PyTorch。
10. keras.preprocessing.text：Keras文本预处理模块。
11. keras_preprocessing.sequence：Keras序列预处理模块。
12. DecisionTreeClassifier：Scikit-Learn中的决策树分类器。
13. LogisticRegression：Scikit-Learn中的逻辑回归分类器。
14. matplotlib.pyplot：用于创建图表和可视化的库。
 

## Repository Structure 

本仓库的文件结构如下：

```
|-- MLP.py	# MLP代码
|-- SVM.py  # SVM代码
|-- TextCNN.py  # TextCNN代码
|-- 决策树.py   # 决策树代码
|-- 逻辑回归.py # 逻辑回归代码
|-- results.txt # 最优(MLP)预测结果
|-- train_data.txt	# 训练集
|-- test.txt	# 测试集
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

## 参考资料
[1] http://t.csdn.cn/g4HmC \
[2] https://developer.aliyun.com/article/1118850 \
[3] https://monkeylearn.com/text-classification/ \
[4]https://stackoverflow.com/questions/72326025/cannot-import-name-pad-sequences-from-keras-preprocessing-sequence \
[5]https://monkeylearn.com/blog/what-is-tf-idf/ \
[6]https://github.com/chongzicbo/nlp-ml-dl-notes/blob/master/code/textclassification/tfidf_nb. \
[7]http://t.csdn.cn/NYiMx \
[8]http://t.csdn.cn/zFIfR \
[9]https://cloud.tencent.com/developer/article/2103987


## 还想说两点：
1.除了逻辑回归所有代码的运行时间都很长(因为分割训练集和数据集用的是Kfolds，再加上参数选择，代码中有很多循环，时间复杂度比较高),不是无法运行。\
2.为了缩减实验报告篇幅，有一些代码图片被我缩小了，可能不太利于观察。
## 请助教手下留情