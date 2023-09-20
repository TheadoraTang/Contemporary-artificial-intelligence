# 逻辑回归
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

# 函数用于文本处理
def preprocess_text(text):
    # 去除特殊字符和标点符号，只保留字母和空格
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

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

# 文本向量化，使用TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

for train_index, val_index in kf.split(df['raw'], df['label']): # 将数据集分为K个子集，其中K-1个子集用于训练模型，剩余1个子集用于验证模型性能。

    # 划分训练集(train_df)和验证集(val_df)
    train_df, val_df = df.iloc[train_index], df.iloc[val_index]

    # 利用preprocessed_text去除特殊字符和标点符号
    X_train = [preprocess_text(text) for text in train_df['raw']]
    X_val = [preprocess_text(text) for text in val_df['raw']]

    # 文本向量化
    X_train = tfidf_vectorizer.fit_transform(train_df['raw'])
    X_val = tfidf_vectorizer.transform(val_df['raw'])

    # 建立LR模型
    model = LogisticRegression()
    model.fit(X_train, train_df['label'])

    # 进行预测
    val_preds = model.predict(X_val)

    # 计算性能指标，并存储在list中
    accuracy = accuracy_score(val_df['label'], val_preds)
    precision = precision_score(val_df['label'], val_preds, average='macro')
    recall = recall_score(val_df['label'], val_preds, average='macro')
    f1 = f1_score(val_df['label'], val_preds, average='macro')

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# 计算平均性能指标
avg_accuracy = sum(accuracy_scores) / num_folds
avg_precision = sum(precision_scores) / num_folds
avg_recall = sum(recall_scores) / num_folds
avg_f1 = sum(f1_scores) / num_folds

print(f"逻辑回归准确率: {avg_accuracy}")
print(f"逻辑回归精确率: {avg_precision}")
print(f"逻辑回归召回率: {avg_recall}")
print(f"逻辑回归F1分数: {avg_f1}")

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

# 文本向量化测试集
X_test = tfidf_vectorizer.transform(text_data)

# 预测测试集
test_preds = model.predict(X_test)
test_preds = test_preds.tolist()
out = open('LR results.txt', 'w', encoding='utf-8')
out.write('id, pred\n')

# 输出预测结果到results.txt中
for number, test in zip(number_data, test_preds):
    out.write(number)
    out.write(', ')
    out.write(str(test))
    out.write('\n')