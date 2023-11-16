# CNN图像分类
第三次实验：
MNIST手写数据识别

## Setup

运行实验代码需要安装以下依赖：

- torchvision
- torch
- argparse
  
在终端中使用如下命令即可安装所需依赖：

```shell
pip install torchvision
pip install torch
pip install argparse
```
 

## Repository Structure 

本仓库的文件结构如下：

```
|-- main.py #本次实验的脚本
|-- README.md   #对仓库的解释
|-- requirements.txt    #本次实验的环境
|-- 10215501437-唐小卉-project3.pdf #实验报告
```



## Usage

在终端中输入

```shell
python main.py --model lenet --lr 0.01 --dropout 0.0
python main.py --model resnet --lr 0.1 --dropout 0.5
python main.py --model alexnet --lr 0.01 --dropout 0.0
python main.py --model vgg16 --lr 0.1 --dropout 0.5
```
其中，模型和超参数的选择可以根据自己的需求进行更改

实验的所有内容都需要放于同一目录下。

本次实验的运行时间比较长，最长的需要将近一天的时间才能得出结果（CPU），最短的也需要十几分钟左右。并不是无法执行

## Reference
[1] http://t.csdnimg.cn/B8OYy

[2] http://t.csdnimg.cn/Zq3K2

[3] https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist/notebook