# Seq2seq文本摘要
第四次实验：
医疗数据文本摘要

## Setup
 
在终端中使用如下命令即可安装所需依赖：

```shell
pip install -r requirements.txt
```

## Repository Structure 

本仓库的文件结构如下：

```
|-- GRU.py #本次实验的脚本1
|-- bart.py #本次实验的脚本2
|-- README.md   #对仓库的解释
|-- requirements.txt    #本次实验的环境
|-- 10215501437-唐小卉-project4.pdf #实验报告
|-- BART_large_predict.csv  #预测结果
```



## Usage

在终端中输入

```shell
python GRU.py
python bart.py
```
实验的所有内容都需要放于同一目录下。

本次实验的运行时间比较长，并不是无法执行,强烈建议使用云主机。

BART模型需要下载Huggingface中的资源，如果本地网络连接不佳建议使用colab的TPU运行代码进行训练（GPU容易出现out of memory问题）。

GRU这里的从keras导入如果报错可以调整成from transformers.keras..............，根据个人需求进行调整。主要与版本有关。


## Reference
[1] https://cloud.tencent.com/developer/article/1543802

[2] http://t.csdnimg.cn/Ep4oB

[3] https://arxiv.org/pdf/1910.13461.pdf

[4] http://t.csdnimg.cn/4XTat

