import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, GRU, Dense
from keras.optimizers import Adam  # 添加优化器
from nltk.translate.bleu_score import corpus_bleu

# 读取数据集
df = pd.read_csv('train.csv')

# 划分训练集和验证集
train_data, val_data = train_test_split(df, test_size=0.1, random_state=42)

# 处理文本数据
tokenizer_desc = Tokenizer()
tokenizer_diag = Tokenizer()

tokenizer_desc.fit_on_texts(train_data['description'])
tokenizer_diag.fit_on_texts(train_data['diagnosis'])

vocab_size_desc = len(tokenizer_desc.word_index) + 1
vocab_size_diag = len(tokenizer_diag.word_index) + 1

max_len_desc = max(len(description.split()) for description in train_data['description'])
max_len_diag = max(len(diagnosis.split()) for diagnosis in train_data['diagnosis'])


# 将文本转换为序列
train_seq_desc = tokenizer_desc.texts_to_sequences(train_data['description'])
train_seq_diag = tokenizer_diag.texts_to_sequences(train_data['diagnosis'])

val_seq_desc = tokenizer_desc.texts_to_sequences(val_data['description'])
val_seq_diag = tokenizer_diag.texts_to_sequences(val_data['diagnosis'])

# 填充序列
train_pad_desc = pad_sequences(train_seq_desc, maxlen=max_len_desc, padding='post')
train_pad_diag = pad_sequences(train_seq_diag, maxlen=max_len_diag, padding='post')

# 调整train_pad_diag的形状
train_pad_diag = np.concatenate([train_pad_diag, np.zeros((len(train_pad_diag), 1))], axis=1)

val_pad_desc = pad_sequences(val_seq_desc, maxlen=max_len_desc, padding='post')
val_pad_diag = pad_sequences(val_seq_diag, maxlen=max_len_diag, padding='post')
val_pad_diag = np.concatenate([val_pad_diag, np.zeros((len(val_pad_diag), 1))], axis=1)

# 构建Encoder模型
encoder_input = Input(shape=(max_len_desc,))
embedding_layer = Embedding(input_dim=vocab_size_desc, output_dim=100, input_length=max_len_desc)(encoder_input)
encoder_gru = GRU(128, return_state=True)
_, encoder_state = encoder_gru(embedding_layer)

# 构建Decoder模型
decoder_input = Input(shape=(max_len_diag,))
decoder_embedding = Embedding(input_dim=vocab_size_diag, output_dim=100, input_length=max_len_diag)(decoder_input)
decoder_gru = GRU(128, return_sequences=True, return_state=True)
decoder_output, _ = decoder_gru(decoder_embedding, initial_state=encoder_state)
decoder_dense = Dense(vocab_size_diag, activation='softmax')
decoder_output = decoder_dense(decoder_output)

# 编译模型
learning_rate = 0.0001  # 自行设定学习率
optimizer = Adam(lr=learning_rate)  # 使用Adam优化器，并指定学习率
model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# 训练模型
epochs = 50  # 您可以自行设定epoch的个数
history = {'loss': [], 'val_loss': [], 'bleu4': []}

def predict_sequence(model, tokenizer, input_sequence):
    initial_sequence = train_pad_diag[np.random.randint(0, len(train_pad_diag)), :-1]
    prediction = initial_sequence.tolist()

    for _ in range(max_len_diag - len(initial_sequence)):
        output_probs = model.predict([input_sequence, np.array([prediction])])
        predicted_token_index = np.argmax(output_probs[0, -1, :])
        prediction.append(predicted_token_index)
        if predicted_token_index == tokenizer.word_index['<end>']:
            break
    # 移除所有零值
    prediction = [token for token in prediction if token != 0]
    prediction = list(map(int, prediction))
    return prediction


print(f'Training data shapes - Description: {train_pad_desc.shape}, Diagnosis: {train_pad_diag[:, :-1].shape}')
print(f'Validation data shapes - Description: {val_pad_desc.shape}, Diagnosis: {val_pad_diag[:, :-1].shape}')
for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')

    # 训练模型
    history_epoch = model.fit(
        [train_pad_desc, train_pad_diag[:, :-1]],
        train_pad_diag[:, :-1],
        batch_size=64,
        epochs=1,
        validation_data=([val_pad_desc, val_pad_diag[:, :-1]], val_pad_diag[:, :-1])
    )

    # 保存训练过程中的loss
    history['loss'].append(history_epoch.history['loss'][0])
    history['val_loss'].append(history_epoch.history['val_loss'][0])

    # 预测并计算BLEU-4得分
    val_predictions = []

    for i, seq in enumerate(val_pad_desc):
        pred_seq = predict_sequence(model, tokenizer_diag, np.array([seq]))
        val_predictions.append(pred_seq)

        # 打印一些样本的真实和预测序列
        if i < 3:  # 仅打印前3个样本
            print(f'\nExample {i + 1}:')
            print(f'Description: {val_data["description"].iloc[i]}')
            print(f'Reference: {val_data["diagnosis"].iloc[i]}')
            print(f'Prediction: {" ".join(map(str, pred_seq))}')

    references = [val_data['diagnosis'].values.tolist()]
    candidates = [val_predictions]

    val_predictions_str = [' '.join(map(str, seq)).rstrip(' 0') for seq in val_predictions]

    references_str = val_data['diagnosis'].values.tolist()

    # 计算 BLEU 分数
    bleu_score = corpus_bleu([references_str], [val_predictions_str], weights=(0.25, 0.25, 0.25, 0.25))

    history['bleu4'].append(bleu_score)

    # 打印loss和BLEU-4分数
    print(f'Training Loss: {history_epoch.history["loss"][0]}')
    print(f'Validation Loss: {history_epoch.history["val_loss"][0]}')
    print(f'BLEU-4 Score: {bleu_score:.4f}')

# 绘制训练曲线
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# 绘制BLEU-4曲线
plt.plot(history['bleu4'], label='BLEU-4 Score')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('BLEU-4')
plt.title('BLEU-4 Score')
plt.show()