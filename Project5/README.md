# 多模态融合模型
第五次实验：
设计一个多模态融合模型。 自行从训练集中划分验证集，调整超参数。 预测测试集（test_without_label.txt）上的情感标签。


## Setup
 
在终端中使用如下命令即可安装所需依赖：

```shell
pip install -r requirements.txt
```

## Repository Structure 

本仓库的文件结构如下：

```
|-- multi.py #本次实验模型
|-- deal_data.py #本次实验的脚本
|-- README.md   #对仓库的解释
|-- requirements.txt    #本次实验的环境
|-- 10215501437-唐小卉-project5.pdf #实验报告
|-- test_with_label.csv  #预测结果
|-- data    #数据集
|-- dataset #处理后的数据集
```



## Usage

在终端中输入（按顺序）

```shell
python deal_data.py
python multi.py
```
实验的所有内容都需要放于同一目录下。

本次实验的运行时间比较长，并不是无法执行,强烈建议使用云主机。

ROBERTA模型需要下载Huggingface中的资源，如果本地网络连接不佳建议使用colab的GPU运行代码进行训练（GPU可能出现out of memory问题）。

data文件夹必须保留，dataset文件夹可根据deal_data运行得出。

## Reference
[1] https://github.com/RecklessRonan/GloGNN/blob/master/readme.md

[2] http://t.csdnimg.cn/Ep4oB

[3] https://github.com/lixiao-han/Multask_Fusion_CenterNet

[4] https://github.com/zdou0830/METER

