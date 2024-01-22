import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 数据准备
training_data = pd.read_csv("train.csv")
input_data = training_data["description"].tolist()
target_data = training_data["diagnosis"].tolist()

# 划分训练集和验证集
train_input, valid_input, train_target, valid_target = train_test_split(input_data, target_data, test_size=0.2, random_state=911)

# 模型和分词器
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)

# 数据向量化和准备
train_encodings = tokenizer(train_input, max_length=128, truncation=True, padding=True)
valid_encodings = tokenizer(valid_input, max_length=128, truncation=True, padding=True)

train_input_ids = torch.tensor(train_encodings["input_ids"]).to(device)
train_attention_mask = torch.tensor(train_encodings["attention_mask"]).to(device)
train_target_ids = torch.tensor(tokenizer(train_target, truncation=True, padding=True)["input_ids"]).to(device)

valid_input_ids = torch.tensor(valid_encodings["input_ids"]).to(device)
valid_attention_mask = torch.tensor(valid_encodings["attention_mask"]).to(device)
valid_target_ids = torch.tensor(tokenizer(valid_target, truncation=True, padding=True)["input_ids"]).to(device)

# 数据集和数据加载器
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_target_ids)
valid_dataset = TensorDataset(valid_input_ids, valid_attention_mask, valid_target_ids)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# BART模型训练
optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

model.train()
epochs = 20

for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        target_ids = batch[2].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=target_ids[:, :-1])
        logits = outputs.logits

        optimizer.zero_grad()

        # 计算交叉熵损失
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids[:, 1:].contiguous().view(-1))

        loss.backward()
        optimizer.step()

    # 在验证集上计算BLEU分数
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            target_ids = batch[2].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)

            predictions.extend(outputs.tolist())
            targets.extend(target_ids[:, 1:].tolist())

    # 将预测和目标转换为字符串形式
    predicted_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    target_texts = tokenizer.batch_decode(targets, skip_special_tokens=True)

   # 计算BLEU分数
    smoothing_function = SmoothingFunction().method1
    bleu1_score = corpus_bleu([[text.split()] for text in target_texts], [text.split() for text in predicted_texts], weights=(1.0, 0, 0, 0), smoothing_function=smoothing_function)
    bleu2_score = corpus_bleu([[text.split()] for text in target_texts], [text.split() for text in predicted_texts], weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
    bleu3_score = corpus_bleu([[text.split()] for text in target_texts], [text.split() for text in predicted_texts], weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
    bleu4_score = corpus_bleu([[text.split()] for text in target_texts], [text.split() for text in predicted_texts], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
    bleu_avg = (bleu1_score + bleu2_score + bleu3_score + bleu4_score) / 4.0

    print("Epoch:", epoch+1, "Loss:", loss.item(), "BLEU-1 Score:", bleu1_score, "BLEU-2 Score:", bleu2_score, "BLEU-3 Score:", bleu3_score, "BLEU-4 Score:", bleu4_score, "BLEU Avg Score:", bleu_avg)

print("Training completed.")

test_data = pd.read_csv("test.csv")
input_data = test_data["description"].tolist()

batch_size = 64
num_samples = len(input_data)
num_batches = int(np.ceil(num_samples / batch_size))
model.eval()
predictions = []
for i in range(num_batches):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, num_samples)

    batch_input_data = input_data[start_index:end_index]
    batch_encodings = tokenizer(batch_input_data, max_length=128, truncation=True, padding=True)
    batch_input_ids = torch.tensor(batch_encodings["input_ids"]).to(device)
    batch_attention_mask = torch.tensor(batch_encodings["attention_mask"]).to(device)

    with torch.no_grad():
        batch_outputs = model.generate(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        batch_predictions = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

    predictions.extend(batch_predictions)

test_data["diagnosis"] = predictions
test_data.to_csv("BART_large_predict.csv", index=False)