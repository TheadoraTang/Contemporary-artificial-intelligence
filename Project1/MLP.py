# MLP
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier  # 导入MLP模型
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re  # 导入正则表达式库

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

for train_index, val_index in kf.split(df['raw'], df['label']):
    train_df, val_df = df.iloc[train_index], df.iloc[val_index]

    X_train = [preprocess_text(text) for text in train_df['raw']]
    X_val = [preprocess_text(text) for text in val_df['raw']]

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)

    # 使用MLP模型
    model = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000)  # MLP模型
    model.fit(X_train, train_df['label'])

    val_preds = model.predict(X_val)

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

print(f"MLP准确率: {avg_accuracy}")
print(f"MLP精确率: {avg_precision}")
print(f"MLP召回率: {avg_recall}")
print(f"MLP F1分数: {avg_f1}")

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
out = open('MLP results.txt', 'w', encoding='utf-8')
out.write('id, pred\n')

# 输出预测结果到results.txt中
for number, test in zip(number_data, test_preds):
    out.write(number)
    out.write(', ')
    out.write(str(test))
    out.write('\n')


# import json
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neural_network import MLPClassifier  # 导入MLP模型
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import re  # 导入正则表达式库
#
# # 函数用于文本处理
# def preprocess_text(text):
#     # 去除特殊字符和标点符号，只保留字母和空格
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     return text
#
# # 读取训练数据
# with open('train_data.txt', 'r') as file:
#     data = [json.loads(line) for line in file]
#
# # 将数据转换为DataFrame
# df = pd.DataFrame(data)
#
# # 设置交叉验证折数
# num_folds = 16
# kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
#
# # 创建列表用于存储每个折叠的性能指标
# accuracy_scores = []
# precision_scores = []
# recall_scores = []
# f1_scores = []
#
# # 文本向量化，使用TF-IDF
# tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
#
# for train_index, val_index in kf.split(df['raw'], df['label']):
#     train_df, val_df = df.iloc[train_index], df.iloc[val_index]
#
#     X_train = [preprocess_text(text) for text in train_df['raw']]
#     X_val = [preprocess_text(text) for text in val_df['raw']]
#
#     X_train = tfidf_vectorizer.fit_transform(X_train)
#     X_val = tfidf_vectorizer.transform(X_val)
#
#     # 使用MLP模型
#     param_grid = {
#         'hidden_layer_sizes': [(50,), (100,), (150,)],  # 可调节隐藏层大小
#         'max_iter': [500, 1000, 1500],  # 可调节最大迭代次数
#         'alpha': [0.0001, 0.001, 0.01]  # 可调节正则化参数
#     }
#
#     model = MLPClassifier()
#     grid_search = GridSearchCV(model, param_grid, cv=5)  # 使用5折交叉验证作为评估指标
#     grid_search.fit(X_train, train_df['label'])
#
#     best_model = grid_search.best_estimator_
#
#     val_preds = best_model.predict(X_val)
#
#     accuracy = accuracy_score(val_df['label'], val_preds)
#     precision = precision_score(val_df['label'], val_preds, average='macro')
#     recall = recall_score(val_df['label'], val_preds, average='macro')
#     f1 = f1_score(val_df['label'], val_preds, average='macro')
#
#     accuracy_scores.append(accuracy)
#     precision_scores.append(precision)
#     recall_scores.append(recall)
#     f1_scores.append(f1)
#
# # 计算平均性能指标
# avg_accuracy = sum(accuracy_scores) / num_folds
# avg_precision = sum(precision_scores) / num_folds
# avg_recall = sum(recall_scores) / num_folds
# avg_f1 = sum(f1_scores) / num_folds
#
# print(f"MLP准确率: {avg_accuracy}")
# print(f"MLP精确率: {avg_precision}")
# print(f"MLP召回率: {avg_recall}")
# print(f"MLP F1分数: {avg_f1}")
#
# with open('test.txt', 'r', encoding='utf-8') as file:
#     file.readline()
#     lines = file.readlines()
#
# # 创建一个列表用于存储文本数据,一个列表存储序号
# text_data = []
# number_data = []
# # 提取每行文本中第一个逗号后面的部分，并进行预处理
# for line in lines:
#     parts = line.strip().split(',', 1)  # 使用逗号分割，最多分割一次
#     if len(parts) > 1:
#         number_data.append(parts[0])
#         text_data.append(preprocess_text(parts[1]))  # 对测试集文本进行预处理
#
# # 文本向量化测试集
# X_test = tfidf_vectorizer.transform(text_data)
#
# # 预测测试集
# test_preds = best_model.predict(X_test)
# test_preds = test_preds.tolist()
# out = open('MLP results.txt', 'w', encoding='utf-8')
# out.write('id, pred\n')
#
# # 输出预测结果到results.txt中
# for number, test in zip(number_data, test_preds):
#     out.write(number)
#     out.write(', ')
#     out.write(str(test))
#     out.write('\n')
