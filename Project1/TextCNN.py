# TextCNN模型
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
MAX_SEQUENCE_LENGTH = 100

# 函数用于文本处理
def preprocess_text(text):
    # 去除特殊字符和标点符号，只保留字母和空格
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# TextCNN模型定义
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.fc(x)
        return x

# 读取训练数据
with open('train_data.txt', 'r') as file:
    data = [json.loads(line) for line in file]

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 设置交叉验证折数
num_folds = 16
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# 创建列表用于存储每个折叠的性能指标
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# 文本处理和标签编码
df['processed_text'] = df['raw'].apply(preprocess_text)
texts = df['processed_text'].tolist()
labels = df['label'].tolist()

# 设置超参数
vocab_size = 5000  # 词汇表大小
embedding_dim = 100  # 词嵌入维度
num_filters = 128  # 卷积核数量
filter_sizes = [3, 4, 5]  # 卷积核尺寸
num_classes = 10  # 类别数量
num_epochs = 5  # 训练周期数
batch_size = 32 # 批处理大小

for train_index, val_index in kf.split(texts, labels):
    train_texts, val_texts = np.array(texts)[train_index], np.array(texts)[val_index]
    train_labels, val_labels = np.array(labels)[train_index], np.array(labels)[val_index]

    # 创建词汇表和编码器
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    word_index = tokenizer.word_index

    # 文本转换为整数序列
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    val_sequences = tokenizer.texts_to_sequences(val_texts)

    # 填充序列
    train_padded = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    val_padded = pad_sequences(val_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    # 创建PyTorch张量数据集
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_padded), torch.tensor(train_labels))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_padded), torch.tensor(val_labels))

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # 初始化TextCNN模型
    model = TextCNN(vocab_size, embedding_dim, num_filters, filter_sizes, num_classes)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_text, batch_labels in train_loader:
            batch_labels = batch_labels.to(torch.int64)
            optimizer.zero_grad()
            outputs = model(batch_text)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

    # 评估模型性能
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch_text, batch_labels in val_loader:
            outputs = model(batch_text)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.tolist())
            val_true.extend(batch_labels.tolist())

    accuracy = accuracy_score(val_true, val_preds)
    precision = precision_score(val_true, val_preds, average='macro')
    recall = recall_score(val_true, val_preds, average='macro')
    f1 = f1_score(val_true, val_preds, average='macro')

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# 计算平均性能指标
avg_accuracy = sum(accuracy_scores) / num_folds
avg_precision = sum(precision_scores) / num_folds
avg_recall = sum(recall_scores) / num_folds
avg_f1 = sum(f1_scores) / num_folds

print(f"TextCNN准确率: {avg_accuracy}")
print(f"TextCNN精确率: {avg_precision}")
print(f"TextCNN召回率: {avg_recall}")
print(f"TextCNN F1分数: {avg_f1}")

# 在测试集上进行预测
with open('test.txt', 'r', encoding='utf-8') as file:
    file.readline()
    lines = file.readlines()

# 创建一个列表用于存储文本数据,一个列表存储序号
text_data = []
number_data = []

# 提取每行文本中第一个逗号后面的部分，并进行预处理
for line in lines:
    parts = line.strip().split(',', 1)  # 使用逗号分割，最多分割一次
    if len(parts) > 1:
        number_data.append(parts[0])
        text_data.append(preprocess_text(parts[1]))  # 对测试集文本进行预处理

# 文本转换为整数序列
test_sequences = tokenizer.texts_to_sequences(text_data)

# 填充序列
test_padded = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# 创建PyTorch张量数据集
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_padded))

# 创建数据加载器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# 预测测试集
test_preds = []
model.eval()
with torch.no_grad():
    for batch_text in test_loader:
        outputs = model(batch_text[0])
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.tolist())

# 将结果写入文件
with open('TextCNN results.txt', 'w', encoding='utf-8') as out_file:
    out_file.write('id, pred\n')
    for number, pred in zip(number_data, test_preds):
        out_file.write(f"{number}, {pred}\n")


# import json
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from keras.preprocessing.text import Tokenizer
# from keras_preprocessing.sequence import pad_sequences
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import make_scorer, f1_score
# from sklearn.base import BaseEstimator
# from sklearn.model_selection import train_test_split
#
# import re
# MAX_SEQUENCE_LENGTH = 1000
#
# # 函数用于文本处理
# def preprocess_text(text):
#     # 去除特殊字符和标点符号，只保留字母和空格
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     return text
#
# # TextCNN模型定义
# class TextCNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes):
#         super(TextCNN, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.convs = nn.ModuleList([
#             nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
#         ])
#         self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
#
#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.unsqueeze(1)
#         x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]
#         x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
#         x = torch.cat(x, 1)
#         x = self.fc(x)
#         return x
#
# # 自定义适配器以包装TextCNN模型
# class TextCNNWrapper(BaseEstimator):
#     def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, num_epochs, batch_size):
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.num_filters = num_filters
#         self.filter_sizes = filter_sizes
#         self.num_classes = num_classes
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         self.tokenizer = None
#         self.model = None  # 添加模型属性
#
#     def fit(self, X, y):
#         # 创建词汇表和编码器
#         self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
#         self.tokenizer.fit_on_texts(X)
#         word_index = self.tokenizer.word_index
#
#         # 文本转换为整数序列
#         sequences = self.tokenizer.texts_to_sequences(X)
#
#         # 填充序列
#         padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
#
#         # 创建PyTorch张量数据集
#         labels = torch.tensor(y)
#         dataset = torch.utils.data.TensorDataset(torch.tensor(padded_sequences), labels)
#
#         # 创建数据加载器
#         train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
#
#         # 初始化TextCNN模型
#         model = TextCNN(self.vocab_size, self.embedding_dim, self.num_filters, self.filter_sizes, self.num_classes)
#         optimizer = optim.Adam(model.parameters())
#         criterion = nn.CrossEntropyLoss()
#
#         # 训练模型
#         for epoch in range(self.num_epochs):
#             model.train()
#             total_loss = 0.0
#             for batch_text, batch_labels in train_loader:
#                 batch_labels = batch_labels.to(torch.int64)
#                 optimizer.zero_grad()
#                 outputs = model(batch_text)
#                 loss = criterion(outputs, batch_labels)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#
#             avg_loss = total_loss / len(train_loader)
#
#         self.model = model  # 存储训练好的模型
#
#     def predict(self, X):
#         self.model.eval()
#         with torch.no_grad():
#             sequences = self.tokenizer.texts_to_sequences(X)
#             padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
#             test_dataset = torch.utils.data.TensorDataset(torch.tensor(padded_sequences))
#             test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)
#
#             test_preds = []
#             for batch_text in test_loader:
#                 outputs = self.model(batch_text[0])
#                 preds = torch.argmax(outputs, dim=1)
#                 test_preds.extend(preds.tolist())
#             return test_preds
#
#
#
# # 读取训练数据
# with open('train_data.txt', 'r') as file:
#     data = [json.loads(line) for line in file]
#
# # 将数据转换为DataFrame
# df = pd.DataFrame(data)
#
# # 文本处理和标签编码
# df['processed_text'] = df['raw'].apply(preprocess_text)
# texts = df['processed_text'].tolist()
# labels = df['label'].tolist()
#
# # 划分训练集和验证集
# X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
#
# # 设置超参数
# vocab_size = 5000  # 词汇表大小
# embedding_dim = 100  # 词嵌入维度
# num_filters = 128  # 卷积核数量
# filter_sizes = [3, 4, 5]  # 卷积核尺寸
# num_classes = 10  # 类别数量
# num_epochs = 5  # 训练周期数
# batch_size = 32
#
# # 设置随机搜索的超参数空间
# param_dist = {
#     "vocab_size": [vocab_size],
#     "embedding_dim": [50, 100, 200],
#     "num_filters": [64, 128, 256],
#     "filter_sizes": [[3, 4, 5], [4, 5, 6], [3, 4, 5, 6]],
#     "num_classes": [num_classes],
#     "num_epochs": [num_epochs],
#     "batch_size": [batch_size]
# }
#
# # 定义F1评分为模型性能的评估指标
# scorer = make_scorer(f1_score, average='macro')
#
# # 创建TextCNNWrapper对象
# text_cnn = TextCNNWrapper(vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, num_epochs, batch_size)
#
# # 创建随机搜索对象
# random_search = RandomizedSearchCV(
#     text_cnn,  # 使用TextCNNWrapper作为estimator
#     param_distributions=param_dist,
#     scoring=scorer,
#     cv=5,  # 交叉验证折数
#     n_iter=10,  # 随机搜索迭代次数，可以根据需要进行调整
#     verbose=1,  # 控制输出详细程度，可以根据需要进行调整
#     n_jobs=-1  # 使用所有可用的CPU核心进行计算，可以根据需要进行调整
# )
#
# # 执行随机搜索
# random_search.fit(X_train, y_train)
#
# # 打印最佳超参数组合和最佳F1分数
# print("最佳组合:")
# print(random_search.best_params_)
# print("最佳F1分数:", random_search.best_score_)
#
# # import json
# # import pandas as pd
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from keras.preprocessing.text import Tokenizer
# # from keras_preprocessing.sequence import pad_sequences
# # from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
# # from sklearn.metrics import make_scorer, f1_score
# # from sklearn.base import BaseEstimator
# #
# # import re
# # MAX_SEQUENCE_LENGTH = 1000
# #
# # # 函数用于文本处理
# # def preprocess_text(text):
# #     # 去除特殊字符和标点符号，只保留字母和空格
# #     text = re.sub(r'[^a-zA-Z\s]', '', text)
# #     return text
# #
# # # TextCNN模型定义
# # class TextCNN(nn.Module):
# #     def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes):
# #         super(TextCNN, self).__init__()
# #         self.embedding = nn.Embedding(vocab_size, embedding_dim)
# #         self.convs = nn.ModuleList([
# #             nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
# #         ])
# #         self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
# #
# #     def forward(self, x):
# #         x = self.embedding(x)
# #         x = x.unsqueeze(1)
# #         x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]
# #         x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
# #         x = torch.cat(x, 1)
# #         x = self.fc(x)
# #         return x
# #
# # # 自定义适配器以包装TextCNN模型
# # class TextCNNWrapper(BaseEstimator):
# #     def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, num_epochs, batch_size):
# #         self.vocab_size = vocab_size
# #         self.embedding_dim = embedding_dim
# #         self.num_filters = num_filters
# #         self.filter_sizes = filter_sizes
# #         self.num_classes = num_classes
# #         self.num_epochs = num_epochs
# #         self.batch_size = batch_size
# #         self.tokenizer = None  # 初始化 tokenizer 属性为 None
# #
# #     def fit(self, X, y):
# #         # 创建词汇表和编码器
# #         self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
# #         self.tokenizer.fit_on_texts(X)
# #         word_index = self.tokenizer.word_index
# #
# #         # 省略其他代码...
# #
# #     def predict(self, X):
# #         # 在 predict 方法中使用 self.tokenizer
# #         self.model.eval()
# #         with torch.no_grad():
# #             sequences = self.tokenizer.texts_to_sequences(X)
# #             padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
# #             test_dataset = torch.utils.data.TensorDataset(torch.tensor(padded_sequences))
# #             test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)
# #
# #             test_preds = []
# #             for batch_text in test_loader:
# #                 outputs = self.model(batch_text[0])
# #                 preds = torch.argmax(outputs, dim=1)
# #                 test_preds.extend(preds.tolist())
# #             return test_preds
# #
# #
# # # 读取训练数据
# # with open('train_data.txt', 'r') as file:
# #     data = [json.loads(line) for line in file]
# #
# # # 将数据转换为DataFrame
# # df = pd.DataFrame(data)
# #
# # # 文本处理和标签编码
# # df['processed_text'] = df['raw'].apply(preprocess_text)
# # texts = df['processed_text'].tolist()
# # labels = df['label'].tolist()
# #
# # # 设置超参数
# # vocab_size = 5000  # 词汇表大小
# # embedding_dim = 100  # 词嵌入维度
# # num_filters = 128  # 卷积核数量
# # filter_sizes = [3, 4, 5]  # 卷积核尺寸
# # num_classes = 10  # 类别数量
# # num_epochs = 5  # 训练周期数
# # batch_size = 32
# #
# # # 设置随机搜索的超参数空间
# # param_dist = {
# #     "vocab_size": [5000],
# #     "embedding_dim": [50, 100, 200],
# #     "num_filters": [64, 128, 256],
# #     "filter_sizes": [[3, 4, 5], [4, 5, 6], [3, 4, 5, 6]],
# #     "num_classes": [10],
# #     "num_epochs": [5],
# #     "batch_size": [32]
# # }
# #
# # # 定义F1评分为模型性能的评估指标
# # scorer = make_scorer(f1_score, average='macro')
# #
# # # 创建TextCNNWrapper对象
# # text_cnn = TextCNNWrapper(vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, num_epochs, batch_size)
# #
# # # 创建随机搜索对象
# # random_search = RandomizedSearchCV(
# #     text_cnn,  # 使用TextCNNWrapper作为estimator
# #     param_distributions=param_dist,
# #     scoring=scorer,
# #     cv=5,  # 交叉验证折数
# #     n_iter=10,  # 随机搜索迭代次数，可以根据需要进行调整
# #     verbose=1,  # 控制输出详细程度，可以根据需要进行调整
# #     n_jobs=-1  # 使用所有可用的CPU核心进行计算
# # )
# #
# # # 执行随机搜索
# # random_search.fit(texts, labels)
# #
# # # 打印最佳超参数组合和最佳F1分数
# # print("Best Hyperparameters:")
# # print(random_search.best_params_)
# #