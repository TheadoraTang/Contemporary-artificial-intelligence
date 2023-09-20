import json
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 函数用于文本处理
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# 读取训练数据
with open('train_data.txt', 'r') as file:
    data = [json.loads(line) for line in file]

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 设置不同的训练集和验证集划分比例
test_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for test_ratio in test_ratios:
    # 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=test_ratio, random_state=42)

    # 文本向量化，使用TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    # 对训练集和验证集的文本进行预处理
    X_train = [preprocess_text(text) for text in train_df['raw']]
    X_val = [preprocess_text(text) for text in val_df['raw']]

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)

    # 训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, train_df['label'])

    # 预测验证集
    val_preds = model.predict(X_val)

    # 评估性能
    accuracy = accuracy_score(val_df['label'], val_preds)
    precision = precision_score(val_df['label'], val_preds, average='macro')
    recall = recall_score(val_df['label'], val_preds, average='macro')
    f1 = f1_score(val_df['label'], val_preds, average='macro')

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# 绘制变化曲线
plt.figure(figsize=(10, 6))
plt.plot(test_ratios, accuracy_scores, label='Accuracy', marker='o')
plt.plot(test_ratios, precision_scores, label='Precision', marker='o')
plt.plot(test_ratios, recall_scores, label='Recall', marker='o')
plt.plot(test_ratios, f1_scores, label='F1 Score', marker='o')
plt.xlabel('Test Ratio')
plt.ylabel('Score')
plt.title('Performance vs. Test Ratio')
plt.legend()
plt.grid()
plt.show()
