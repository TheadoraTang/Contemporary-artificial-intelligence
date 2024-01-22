# import os
# os.environ["http_proxy"] = "http://127.0.0.1:33210"
# os.environ["https_proxy"] = "http://127.0.0.1:33210"
from torchvision.models import convnext
from transformers import RobertaModel
import json
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import RobertaTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self, args, data, transform=None):
        self.args = args
        self.data = data
        self.transform = transform
        self.tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model)
        self.label_dict_number = {
            'negative': 0,
            'neutral': 1,
            'positive': 2,
        }
        self.label_dict_str = {
            0: 'negative',
            1: 'neutral',
            2: 'positive',
        }

    def __getitem__(self, index):
        return self.tokenize(self.data[index])

    def __len__(self):
        return len(self.data)

    def tokenize(self, item):
        item_id = item['id']
        text = item['text']
        image_path = item['image_path']
        label = item['label']

        text_token = self.tokenizer(text, return_tensors="pt", max_length=self.args.text_size,
                                    padding='max_length', truncation=True)
        text_token['input_ids'] = text_token['input_ids'].squeeze()
        text_token['attention_mask'] = text_token['attention_mask'].squeeze()

        img_token = self.transform(image_path) if self.transform else torch.tensor(image_path)

        label_token = self.label_dict_number[label] if label in self.label_dict_number else -1
        return item_id, text_token, img_token, label_token


def load_json(file):
    data_list = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = json.load(f)
        for line in lines:
            item = {
                'image_path': np.array(Image.open(line['image_path'])),
                'text': line['text'],
                'label': line['label'],
                'id': line['guid']
            }
            data_list.append(item)
    return data_list


def load_data(args):
    img_size = (args.img_size, args.img_size)
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(img_size),
         transforms.Normalize([0.5], [0.5])]
    )
    data_list = {
        'train': load_json(args.train_file),
        'dev': load_json(args.dev_file),
        'test': args.test_file and load_json(args.test_file),
    }
    data_set = {
        'train': MyDataset(args, data_list['train'], transform=data_transform),
        'dev': MyDataset(args, data_list['dev'], transform=data_transform),
        'test': args.test_file and MyDataset(args, data_list['test'], transform=data_transform),
    }

    return data_set[args.mode], data_set['dev']


def save_data(file, predict_list):
    with open(file, 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for pred in predict_list:
            f.write(f"{pred['guid']},{pred['tag']}\n")


class TextOnly(nn.Module):
    def __init__(self, args):
        super(TextOnly, self).__init__()
        self.encoder = RobertaModel.from_pretrained(args.pretrained_model)
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.transform = nn.Sequential(
            nn.Linear(768, 1000),
            nn.ReLU(),
        )

    def forward(self, encoded_input):
        encoder_output = self.encoder(**encoded_input)
        hidden_state = encoder_output['last_hidden_state']
        pooler_output = encoder_output['pooler_output']
        output = self.transform(pooler_output)
        return hidden_state, output


class ImgOnly(nn.Module):
    def __init__(self, args):
        super(ImgOnly, self).__init__()
        self.encoder = convnext.convnext_base(weights=convnext.ConvNeXt_Base_Weights.DEFAULT)
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.encoder(x)
        return x


class MultiModal(nn.Module):
    def __init__(self, args):
        super(MultiModal, self).__init__()
        self.TextModule_ = TextOnly(args)
        self.ImgModule_ = ImgOnly(args)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=1000, num_heads=2, batch_first=True)
        self.linear_text_k1 = nn.Linear(1000, 1000)
        self.linear_text_v1 = nn.Linear(1000, 1000)
        self.linear_img_k2 = nn.Linear(1000, 1000)
        self.linear_img_v2 = nn.Linear(1000, 1000)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.classifier_img = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )
        self.classifier_text = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )
        self.classifier_multi = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )

    def forward(self, text=None, image=None):
        if text is not None and image is None:
            _, text_out = self.TextModule_(text)
            text_out = self.classifier_text(text_out)
            return text_out

        if text is None and image is not None:
            img_out = self.ImgModule_(image)
            img_out = self.classifier_img(img_out)
            return img_out

        _, text_out = self.TextModule_(text)
        img_out = self.ImgModule_(image)

        multi_out = torch.cat((text_out, img_out), 1)

        text_out = self.classifier_text(text_out)
        img_out = self.classifier_img(img_out)
        multi_out = self.classifier_multi(multi_out)

        # Return only one output tensor based on available data
        return text_out if text is not None else img_out if image is not None else multi_out


    def Multihead_self_attention(self, text_out, img_out):

        text_k1 = self.linear_text_k1(text_out)
        text_v1 = self.linear_text_v1(text_out)
        img_k2 = self.linear_img_k2(img_out)
        img_v2 = self.linear_img_v2(img_out)

        key = torch.stack((text_k1, img_k2), dim=1)
        value = torch.stack((text_v1, img_v2), dim=1)
        query = torch.stack((text_out, img_out), dim=1)

        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        return attn_output

    def Transformer_Encoder(self, text_out, img_out):
        multimodal_sequence = torch.stack((text_out, img_out), dim=1)
        return self.transformer_encoder(multimodal_sequence)

def train(args, train_dataloader, dev_dataloader):
    model = MultiModal(args).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        for step, batch in enumerate(train_dataloader):
            ids, text, image, labels = batch
            text = text.to(device=args.device)
            image = image.to(device=args.device)
            labels = labels.to(device=args.device)
            optimizer.zero_grad()
            # Forward pass
            text_out, img_out, multi_out = model(text=text, image=image)
            # Choose the appropriate output based on the available data
            output = text_out if text is not None else img_out if image is not None else multi_out
            # Calculate loss
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += torch.sum(torch.argmax(output, dim=1) == labels).item()
            total_samples += labels.size(0)

        train_loss /= total_samples
        train_accuracy = train_correct / total_samples

        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        evaluate(args, model, dev_dataloader, epoch)



def evaluate(args, model, dev_dataloader, epoch=None):
    model.eval()
    dev_loss = 0.0
    dev_correct = 0
    total_samples = 0
    loss_func = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dev_dataloader:
            ids, text, image, labels = batch
            text = text.to(device=args.device)
            image = image.to(device=args.device)
            labels = labels.to(device=args.device)

            outputs = model(text=text, image=image)
            loss = loss_func(outputs, labels)

            dev_loss += loss.item() * labels.size(0)
            dev_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            total_samples += labels.size(0)

    dev_loss /= total_samples
    dev_accuracy = dev_correct / total_samples

    if epoch:
        print(f"  Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f} (Epoch {epoch})")
    else:
        print(f"  Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f}")



def train_and_test(args, train_dataloader, dev_dataloader, test_dataloader):
    model = MultiModal(args).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0
        for step, batch in enumerate(train_dataloader):
          ids, text, image, labels = batch
          text = text.to(device=args.device)
          image = image.to(device=args.device)
          labels = labels.to(device=args.device)
          optimizer.zero_grad()
          output = model(text=text, image=image)
          loss = loss_func(output, labels)
          loss.backward()
          optimizer.step()

          train_loss += loss.item() * labels.size(0)
          train_correct += torch.sum(torch.argmax(output, dim=1) == labels).item()
          total_samples += labels.size(0)

        train_loss /= total_samples
        train_accuracy = train_correct / total_samples

        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        evaluate(args, model, dev_dataloader, epoch)


    # Test the model for different scenarios
    print("\nTesting the model for different scenarios:")

    # Text input only
    print("\nScenario: Text Input Only")
    get_test(args, model, test_dataloader, scenario="text_only")

    # Image input only
    print("\nScenario: Image Input Only")
    get_test(args, model, test_dataloader, scenario="image_only")

    # Text and Image input
    print("\nScenario: Text and Image Input")
    get_test(args, model, test_dataloader, scenario="text_and_image")


def get_test(args, model, test_dataloader, scenario=""):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            ids, text, image, _ = batch
            text = text.to(device=args.device)
            image = image.to(device=args.device)

            outputs = model(text=text, image=image)
            predicted_labels = torch.argmax(outputs, dim=1)

            for i in range(len(ids)):
                item_id = ids[i]
                tag = test_dataloader.dataset.label_dict_str[int(predicted_labels[i])]
                prediction = {
                    'guid': item_id,
                    'tag': tag,
                }
                predictions.append(prediction)

    save_data(args.test_output_file, predictions)

    accuracy = calculate_accuracy(predictions, test_dataloader.dataset.data)
    print(f"  {scenario.capitalize()} Test Accuracy: {accuracy:.4f}")

def calculate_accuracy(predictions, data):
    correct_predictions = 0
    total_samples = len(predictions)

    for pred in predictions:
        try:
            guid_int = int(pred['guid'])
            if guid_int < len(data) and pred['tag'] == data[guid_int]['label']:
                correct_predictions += 1
        except (ValueError, IndexError):
            continue

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    return accuracy

class Config:
    def __init__(self):
        self.do_train = True
        self.do_test = True
        self.test_output_file = './test_with_label.txt'
        self.train_file = './dataset/train.json'
        self.dev_file = './dataset/dev.json'
        self.test_file = './dataset/test.json'
        self.pretrained_model = 'roberta-base'
        self.lr = 1e-6
        self.dropout = 0.1
        self.epochs = 10
        self.batch_size = 4
        self.img_size = 384
        self.text_size = 64
        self.mode = 'train'

args = Config()

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {args.device}')

train_set, dev_set = load_data(args)
train_dataloader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
dev_dataloader = DataLoader(dev_set, shuffle=False, batch_size=args.batch_size)

if args.do_train:
    print('Model training...')
    test_set, _ = load_data(args)
    test_dataloader = DataLoader(test_set, shuffle=False, batch_size=args.batch_size)
    train_and_test(args, train_dataloader, dev_dataloader, test_dataloader)