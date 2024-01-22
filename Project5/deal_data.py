import json
import math
import argparse
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split

data_directory = './dataset/'
train_txt_path = data_directory + 'train.txt'
test_txt_path = data_directory + 'test_without_label.txt'
data_path = data_directory + 'data/'

parser = argparse.ArgumentParser()
parser.add_argument('-train_file', '--train_file',
                    type=str, default='./dataset/train.json', help='路径：训练文件')
parser.add_argument('-test_file', '--test_file',
                    type=str, default='./dataset/test.json', help='路径：测试文件')
parser.add_argument('-dev_file', '--dev_file',
                    type=str, default='./dataset/dev.json', help='路径：验证文件')
parser.add_argument('-dev_size', '--dev_size',
                    type=float, default=0.1, help='验证集大小')
arguments = parser.parse_args()


train_dev_df = pd.read_csv(train_txt_path)
test_df = pd.read_csv(test_txt_path)
train_df, dev_df = train_test_split(train_dev_df, test_size=arguments.dev_size)

def read_text_file(file, encoding):
    text = ''
    with open(file, encoding=encoding) as fp:
        for line in fp.readlines():
            line = line.strip('\n')
            text += line
    return text


def transform_data(data_values):
    dataset = []
    for i in range(len(data_values)):
        guid = str(int(data_values[i][0]))
        label = data_values[i][1]
        if type(label) != str and math.isnan(label):
            label = None

        file_path = data_path + guid + '.txt'
        with open(file_path, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
            if encoding == "GB2312":
                encoding = "GBK"

        text = ''
        try:
            text = read_text_file(file_path, encoding)
        except UnicodeDecodeError:
            try:
                text = read_text_file(file_path, 'ANSI')
            except UnicodeDecodeError:
                print('UnicodeDecodeError')
        dataset.append({
            'guid': guid,
            'text': text,
            'label': label,
            'image_path': data_path + guid + '.jpg',
        })
    return dataset


train_data = transform_data(train_df.values)
dev_data = transform_data(dev_df.values)
test_data = transform_data(test_df.values)

with open(arguments.train_file, 'w', encoding="utf-8") as f:
    json.dump(train_data, f)

with open(arguments.dev_file, 'w', encoding="utf-8") as f:
    json.dump(dev_data, f)

with open(arguments.test_file, "w", encoding="utf-8") as f:
    json.dump(test_data, f)
