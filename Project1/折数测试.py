import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 读取训练数据
with open('train_data.txt', 'r') as file:
    data = [json.loads(line) for line in file]

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 定义折数的范围
num_folds_range = range(2, 21)

# 创建列表用于存储不同折数下的性能指标
avg_accuracy_scores = []
avg_precision_scores = []
avg_recall_scores = []
avg_f1_scores = []

# 文本向量化，使用TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

for num_folds in num_folds_range:
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, val_index in kf.split(df['raw'], df['label']):
        train_df, val_df = df.iloc[train_index], df.iloc[val_index]

        X_train = tfidf_vectorizer.fit_transform(train_df['raw'])
        X_val = tfidf_vectorizer.transform(val_df['raw'])

        model = LogisticRegression()
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

    avg_accuracy_scores.append(avg_accuracy)
    avg_precision_scores.append(avg_precision)
    avg_recall_scores.append(avg_recall)
    avg_f1_scores.append(avg_f1)

# 绘制变化曲线
plt.figure(figsize=(10, 6))
plt.plot(num_folds_range, avg_accuracy_scores, marker='o', label='Average Accuracy')
plt.plot(num_folds_range, avg_precision_scores, marker='o', label='Average Precision')
plt.plot(num_folds_range, avg_recall_scores, marker='o', label='Average Recall')
plt.plot(num_folds_range, avg_f1_scores, marker='o', label='Average F1 Score')
plt.xlabel('Number of Folds')
plt.ylabel('Score')
plt.legend()
plt.title('Performance vs. Number of Folds')
plt.grid()
plt.show()
