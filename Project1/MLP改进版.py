import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score
import re

# 函数用于文本处理
def preprocess_text(text):
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

# 文本向量化，使用TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# 创建MLP模型
mlp = MLPClassifier(max_iter=1000, random_state=42)

# 定义超参数搜索范围
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100), (100, 100, 100, 100)],
    'max_iter': [500, 1000, 2000],  # 不同的epoch数量
}

# 使用GridSearchCV来寻找最佳超参数
f1_scorer = make_scorer(f1_score, average='macro')
grid_search = GridSearchCV(mlp, param_grid, cv=kf, scoring=f1_scorer)
grid_search.fit(tfidf_vectorizer.fit_transform(df['raw']), df['label'])

# 打印最佳超参数
best_params = grid_search.best_params_
print("最佳超参数：", best_params)

# # 使用最佳超参数训练模型
# best_mlp = MLPClassifier(**best_params, random_state=42)
# best_mlp.fit(tfidf_vectorizer.fit_transform(df['raw']), df['label'])
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
# test_preds = best_mlp.predict(X_test)
# test_preds = test_preds.tolist()
# out = open('results.txt', 'w', encoding='utf-8')
# out.write('id, pred\n')
#
# # 输出预测结果到results.txt中
# for number, test in zip(number_data, test_preds):
#     out.write(number)
#     out.write(', ')
#     out.write(str(test))
#     out.write('\n')
