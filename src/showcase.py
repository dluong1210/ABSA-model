# -*- coding: utf-8 -*-
# preprocessing train-data
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ast
from pprint import pprint
import pandas as pd
from underthesea import word_tokenize
from underthesea import text_normalize

categories = ["GENERAL", "CAMERA", "PERFORMANCE", "DESIGN","BATTERY","FEATURES","SCREEN","STORAGE","PRICE","SER&ACC"]
sentiment = ["Negative", "Positive", "Neutral"]

def upsampling_df(df):
    """
    thông qua EDA, ta nhận thấy rằng nhãn STORAGE trong tập train bị quá ít
    so với những nhãn còn lại
    -> solution: với mỗi sample mà chứa nhãn STORAGE, ta nhân x10 lần sample đó
    lên để chống model bị bias về những nhãn kia mà bỏ quên nhãn STORAGE
    """
    unbalance_df = pd.DataFrame(columns=df.columns)
    num_sample = 10
    for i in range(len(df)):
        if "STORAGE" in df.iloc[i].label:
            for j in range(num_sample):
                unbalance_df = pd.concat([unbalance_df, pd.DataFrame([df.iloc[i]])], ignore_index=True)

    # merge with df
    df = pd.concat([df, unbalance_df], ignore_index=True)
    return df

def get_data_from_raw(file_path, test=False):
    df = pd.read_csv(file_path)
    df = df.drop(['index', 'n_star', 'date_time'], axis=1)
    # if not test:
    #     df = upsampling_df(df)

    def to_json(s):
        s = s[1:-1].split('#')
        return s

    def preprocessing(row):
        sen = row['comment']
        label = row['label'][:-1]
        labels_list = label.split(';')
        labels_list = [to_json(x) for x in labels_list]
        # convert to json
        dict_res = {x[0]: (x[1] if len(x) > 1 and x[1] != '' else 'OTHER') for x in labels_list if len(x) > 0}
        return dict_res

    df['LABELS'] = df.apply(preprocessing, axis=1)
    x1 = df['comment'].tolist()
    y1 = []
    x2 = []
    y2 = []

    for idx, row in df.iterrows():
        comment = row['comment']
        labels = row['LABELS']

        comment = text_normalize(comment)
        comment = word_tokenize(comment, format='text')
        label_aspect = [0] * len(categories)

        for category, value in labels.items():
            if category in categories:
                if category != "OTHERS":
                    text_with_aspect = f"[CLS] {comment} [SEP] {category} [SEP]"
                    label_sentiment = sentiment.index(value)
                    x2.append(text_with_aspect); y2.append(label_sentiment)

                index = categories.index(category)
                label_aspect[index] = 1

        y1.append(label_aspect)

    return x1, y1, x2, y2

print('starting load raw datasets .....')
x1, y1, x2, y2 = get_data_from_raw('/root/.cache/kagglehub/datasets/tdluong1210/uit-visfd/versions/1/Train.csv')
len(x1), len(y1), len(x2), len(y2)
x1_val, y1_val, x2_val, y2_val = get_data_from_raw('/root/.cache/kagglehub/datasets/tdluong1210/uit-visfd/versions/1/Dev.csv')
# len(x1_val), len(y1_val), len(x2_val), len(y2_val)
x1_test, y1_test, x2_test, y2_test = get_data_from_raw('/root/.cache/kagglehub/datasets/tdluong1210/uit-visfd/versions/1/Test.csv', test=True)
# len(x1_test), len(y1_test)
print('done load raw datasets !!!')

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, x, y, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

        encodings = tokenizer.batch_encode_plus(
            x,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors="pt"
        )

        # Lưu lại các encoding và label đã lọc
        self.data = [
            (encodings["input_ids"][i], encodings["attention_mask"][i], label)
            for i, label in enumerate(y)
            if len(tokenizer.encode(x[i], truncation=False)) <= max_len
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, attention_mask, label = self.data[idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long if isinstance(label, int) else torch.float)
        }

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
max_len = 256
batch_size = 32

print(f'starting load datatset into dataloader ....')
# Dataset cho Multi-label
aspect_train_dataset = CustomDataset(x1, y1, tokenizer, max_len)
aspect_train_loader = DataLoader(aspect_train_dataset, batch_size=batch_size, shuffle=True)

aspect_val_dataset = CustomDataset(x1_val, y1_val, tokenizer, max_len)
aspect_val_loader = DataLoader(aspect_val_dataset, batch_size=batch_size, shuffle=True)

aspect_test_dataset = CustomDataset(x1_test, y1_test, tokenizer, max_len)
aspect_test_loader = DataLoader(aspect_test_dataset, batch_size=batch_size, shuffle=False)

# Dataset cho Multi-class
sentiment_train_dataset = CustomDataset(x2, y2, tokenizer, max_len)
sentiment_train_loader = DataLoader(sentiment_train_dataset, batch_size=batch_size, shuffle=True)

sentiment_val_dataset = CustomDataset(x2_val, y2_val, tokenizer, max_len)
sentiment_val_loader = DataLoader(sentiment_val_dataset, batch_size=batch_size, shuffle=True)

sentiment_test_dataset = CustomDataset(x2_test, y2_test, tokenizer, max_len)
sentiment_test_loader = DataLoader(sentiment_test_dataset, batch_size=batch_size, shuffle=False)

print(f'done load dataset into dataloader !!!')

"""# Implement model"""

import torch
import torch.nn as nn
from transformers import BertModel
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.samplers import MPerClassSampler

infonce_criterion = nn.CrossEntropyLoss()

def cl_loss_function(a, b, temp=0.2):
    a = nn.functional.normalize(a, dim=-1)
    b = nn.functional.normalize(b, dim=-1)
    logits = torch.mm(a, b.T)
    logits /= temp
    labels = torch.arange(a.shape[0]).to(a.device)
    return infonce_criterion(logits, labels)

def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)

class AspectDetection(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_aspects, conf):
        super(AspectDetection, self).__init__()
        self.bert = bert_model
        self.aspect_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1), # :D drop out nhieu the: [0,1 -> 0.3]
            nn.Linear(hidden_dim, num_aspects)
        )

        self.conf = conf
        # contrastive learning
        self.augmentation = conf['augment']
        self.noise_weight = conf['noise_weight']
        self.cl_alpha = conf['cl_alpha']
        self.classification_dim = conf['classification_dim']
        self.dropout = nn.Dropout(p=conf['dropout'])
        self.cl_projector = nn.Linear(self.classification_dim, self.classification_dim)
        init(self.cl_projector)
        self.temp = conf['temp']

    def get_bert(self):
        return self.bert

    def forward(self, input_ids, attention_mask, target, test=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token
        # print(f"pooled_output shape: {pooled_output.shape}") # [bs, 768]

        logits = self.aspect_head(pooled_output)

        if test:
            return logits

        criterion = nn.BCEWithLogitsLoss()

        BCE_loss = criterion(logits, target)

        # generate 2 views for contrastive learning
        # pooled_output: original view
        # ....... : augment view
        contrastive_loss = torch.tensor(0)
        if self.augmentation == "FN":
            sub1 = self.cl_projector(
            self.noise_weight*torch.rand_like(pooled_output) + pooled_output
            )

            sub2 = self.cl_projector(
                self.noise_weight*torch.rand_like(pooled_output) + pooled_output
            )

            # contrastive loss
            contrastive_loss = self.cl_alpha*cl_loss_function(
                sub1.view(-1, self.classification_dim),
                sub2.view(-1, self.classification_dim),
                self.temp # temperature
            )

        if self.augmentation == "FD":
            sub1 = self.cl_projector(self.dropout(pooled_output))
            sub2 = self.cl_projector(self.dropout(pooled_output))
            contrastive_loss = self.cl_alpha*cl_loss_function(
                sub1.view(-1, self.classification_dim),
                sub2.view(-1, self.classification_dim),
                self.temp # temperature
            )

        loss = {
            'bce_loss': BCE_loss,
            'contrastive_loss': contrastive_loss
        }

        return loss

"""# Train & Evaluate model"""

def train_model(model, criterion, optimizer, data_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader_multi_class:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            targets = batch['targets']

            probs = model(input_ids, attention_mask)
            loss = criterion(probs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import pickle


def train_model(model, criterion, optimizer, data_loader_train,
                data_loader_valid, epochs, device, task='detection', path_log='test.txt'):
    model.to(device)  # Đưa mô hình vào GPU/CPU

    f1_best = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Train
        pbar = tqdm(data_loader_train, total=len(data_loader_train))
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['label'].to(device)

            # Forward pass
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                loss_dict = model(input_ids, attention_mask, targets, test=False)
                loss = loss_dict['bce_loss'] + loss_dict['contrastive_loss']
            else:
                loss_dict = model(input_ids, attention_mask, targets, test=False)
                loss = loss_dict['BCE_loss'] + loss_dict['contrastive_loss']

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if True:
                pbar.set_description("epoch: %d, " % (epoch) +
                                 ", ".join([
                                     "%s: %.5f" % (l, loss_dict[l].detach()) for l in loss_dict
                                 ]))

        epoch_traning_loss = f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(data_loader_train)}"
        print(epoch_traning_loss)
        # save log 
        log = open(path_log, 'a')
        log.write(epoch_traning_loss + "\n")

        # Evaluate on validation set
        model.eval()
        valid_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(data_loader_valid, total=len(data_loader_valid))
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['label'].to(device)

                # Forward pass
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    probs = model(input_ids, attention_mask, targets, test=True)
                else:
                    probs = model(input_ids, attention_mask, targets, test=True)

                loss = criterion(probs, targets)
                valid_loss += loss.item()

                # Lấy dự đoán
                if isinstance(criterion, torch.nn.CrossEntropyLoss):
                    preds = torch.argmax(probs, dim=1)  # Multi-class
                else:
                    preds = (torch.sigmoid(probs) >= 0.5).int()  # Multi-label

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        # Tính toán các chỉ số đánh giá
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            accuracy = accuracy_score(all_targets, all_preds)
            accuracy_log = f"Validation Accuracy: {accuracy:.4f}"
            print(accuracy_log)
            log.write(accuracy_log + "\n")
        else:
            f1 = f1_score(all_targets, all_preds, average='macro')
            f1_log = f"Validation F1 Score: {f1:.4f}"
            print(f1_log)
            log.write(f1_log + "\n")
        
        validation_loss = f"Validation Loss: {valid_loss / len(data_loader_valid)}"
        
        print(validation_loss)
        log.write(validation_loss + "\n")

        # compare to save best result 
        f1 = f1_score(all_targets, all_preds, average='macro')
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        if f1 > f1_best:
            f1_best = f1
            best_log = f'best in epoch: {epoch+1}/{epochs} - best f1 score: {f1_best}'
            print(best_log)
            log.write(best_log + "\n")

            # write precision, recall, f1 for best epoch 
            log.write(f'f1: {f1} - recall: {recall} - precision: {precision}' + "\n")

            log.write('save best model .... \n')
            # save best model
            with open('model_aspect.pkl', 'wb') as f:
                pickle.dump(model_aspect_detection, f)

        # evaluate on test set
        if task == 'detection':
            evaluate_model(model, aspect_test_loader, device, short_format=True, log_path=path_log)
        if task == 'classification':
            evaluate_model(model, sentiment_test_loader, device, short_format=True, log_path=path_log)
        log.close()

import torch
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def evaluate_model(model, data_loader, device, short_format=False, log_path='test.txt'):
    model.eval()
    all_preds = []
    all_labels = []

    log = open(log_path, 'a') 
    

    with torch.no_grad():
        pbar = tqdm(data_loader, total=len(data_loader))
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            if isinstance(model, AspectDetection):
                probs = model(input_ids, attention_mask, labels, test=True)
                preds = (torch.sigmoid(probs) >= 0.5).int()  # Multi-label

            else:
                probs = model(input_ids, attention_mask, labels, test=True)
                preds = torch.argmax(probs, dim=1)  # Multi-class

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    f1 = f1_score(all_labels, all_preds, average='macro')  # F1 score
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # micro score
    f1_micro = f1_score(all_labels, all_preds, average='micro')  # F1 score
    precision_micro = precision_score(all_labels, all_preds, average='micro')
    recall_micro = recall_score(all_labels, all_preds, average='micro')

    if short_format:
        print(f"Macro score - F1 : {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"Micro score - F1: {f1_micro:.4f}, Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}")
        macro_score = f"Macro Score - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        micro_score = f"Micro score - F1: {f1_micro:.4f}, Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}"

        log.write(macro_score + "\n")
        log.write(micro_score + "\n")
    log.close()

    if not short_format:
        print(classification_report(all_labels, all_preds))

    return f1, precision, recall

from transformers import AutoModel, AutoConfig
from collections import defaultdict
# handle zerodivison in classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def get_number_parameters_model(model):
    num_params = sum(p.numel() for p in model.parameters())
    # print(f"number of parameters: {num_params}")
    return num_params

def init_bert(conf, init_style='default'):
    print('starting init bert')
    if init_style == 'autoconfig':
        new_num_layers = conf['num_hidden_layers_bert']
        config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
    
        # save orginal layer 
        original_num_layers = config.num_hidden_layers
        config.num_hidden_layers = new_num_layers
        
        # original model 
        phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        
        # get current list transformer layer 
        transformer_layers = phobert.encoder.layer
        
        if new_num_layers < original_num_layers:
            phobert.encoder.layer = nn.ModuleList(transformer_layers[:new_num_layers])
            print(f"Reduced number of layers to {new_num_layers}.")
        
        elif new_num_layers > original_num_layers:
            additional_layers = []
            for i in range(new_num_layers - original_num_layers):
                # copy last layer -> new layer
                additional_layers.append(transformer_layers[-1].__class__(config))
            phobert.encoder.layer.extend(additional_layers)
            print(f"Increased number of layers to {new_num_layers}. New layers initialized randomly.")
        
        else:
            print(f'Number of layers remains unchanged at {original_num_layers}.')
        

    if init_style == 'default':
        print(f"Default init")
        phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")

    print("done init bert")

    return phobert

import argparse 

def get_cmd():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-epochs', '--epochs', default=8, type=int)
    parser.add_argument('-layer_bert', '--num_hidden_layers_bert', default=12, type=int)
    parser.add_argument('-cl_alpha', '--cl_alpha', default=0.2, type=float) # hyper weight for contrastive loss 
    parser.add_argument('-log_path', '--log_path', default="", type=str)

    args = parser.parse_args()
    return args



# collect hyperparams to dict: conf ~ config
conf = defaultdict(list)

paras = get_cmd().__dict__
print(f'paras: {paras}')
for p in paras: 
    conf[p] = paras[p]

conf['lr'] = 2e-5
# conf['epochs'] = 8
# conf['epochs'] = 8
conf['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# conf['num_hidden_layers_bert'] = 12
conf['num_hidden_size_bert'] = 768
conf['hidden_dim'] = 256
# contrastive params
conf['augment'] = 'FN'
conf['noise_weight'] = 0.05
# conf['cl_alpha'] = 0
conf['classification_dim'] = 768
conf['dropout'] = 0.3
conf['temp'] = 0.2

phobert = init_bert(init_style='autoconfig', conf=conf)
model_aspect_detection = AspectDetection(phobert, conf['hidden_dim'], len(categories), conf)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params = model_aspect_detection.parameters(), lr=conf['lr'])



train_model(
    model=model_aspect_detection,
    criterion=criterion, # loss function
    optimizer=optimizer,
    data_loader_train=aspect_train_loader,
    data_loader_valid=aspect_val_loader,
    epochs=conf['epochs'],
    device=conf['device'], 
    path_log=conf['log_path']
)

evaluate_model(model_aspect_detection, aspect_test_loader, conf['device'])

# """# Stage 2: aspect classification (3 label)"""

from pytorch_metric_learning.losses import NTXentLoss, SupConLoss

class SentimentClassification(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_sentiments, freeze_bert=False):
        super(SentimentClassification, self).__init__()
        self.bert = bert_model
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, num_sentiments)
        self.freeze_bert = freeze_bert
        # freeze params of bert
        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.cl_alpha = 0

    def forward(self, input_ids, attention_mask, targets, test=False):
        if self.freeze_bert:
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token
        logits = self.sentiment_head(pooled_output)

        if test:
            return logits

        # probs = torch.softmax(logits, dim=-1)  # Multi-class

        criterion = nn.CrossEntropyLoss()
        BCE_loss = criterion(logits, targets)

        # contrastive loss
        # loss_temp = NTXentLoss(temperature=0.2)
        loss_temp = SupConLoss(temperature=0.2)
        contrastive_loss = self.cl_alpha * loss_temp(pooled_output, targets)

        loss = {
            'BCE_loss': BCE_loss,
            'contrastive_loss': contrastive_loss
        }

        return loss

import copy

print('start get bert from stage 1 ...')
bert_stage_1 = model_aspect_detection.get_bert()
print('end get bert from stage 1')
print('-'*60)
print('start copy bert to stage 2')
bert_stage_2 = copy.deepcopy(bert_stage_1)
print('end copy bert to stage 2')


    # from transformers import AutoModel

    # bert_stage_2 = AutoModel.from_pretrained("vinai/phobert-base-v2")

model_sentiment = SentimentClassification(
    bert_model=bert_stage_2,
    hidden_dim=256,
    num_sentiments=len(sentiment),
    freeze_bert=False
)


critertion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model_sentiment.parameters(), lr=2e-5)
epochs = 5

train_model(
        model=model_sentiment,
        criterion=critertion,
        optimizer=optimizer,
        data_loader_train=sentiment_train_loader,
        data_loader_valid=sentiment_val_loader,
        epochs=epochs,
        device=conf['device'],
        task='classification',
        path_log=conf['log_path']
    )

evaluate_model(model_sentiment, sentiment_test_loader, conf['device'])

"""# Evaluate Test Set

# Showcase - Case Study
"""

# predict label for 1 example

class show_case:
    def __init__(self):
        self.categories_dict = {
            k: v for k,v in enumerate(categories)
        }
        self.device = conf['device']

    def show_case_from_test_set(self, index_order_test_set=0):
        input = [x1_test[index_order_test_set]]
        label = [y1_test[index_order_test_set]]
        print(f'input: {input}')
        print(f'label: {label}')
        # categories_dict = {k: v for k, v in enumerate(categories)}

        test_set = CustomDataset(input, label, tokenizer, max_len)
        test_set = DataLoader(test_set, batch_size=1)

        for data in test_set:
            output = model_aspect_detection(
                input_ids=data['input_ids'].to(self.device),
                attention_mask=data['attention_mask'].to(self.device),
                target=None,
                test=True
            )
            output_norm = (output >= 0.5).int().detach().cpu()[0]
            true_label = [categories[i] for i,v in enumerate(label[0]) if v == 1]
            pred_label = [categories[i] for i, v in enumerate(output_norm) if v == 1]
            print(f'true lable: {true_label}')
            print(f'pred label: {pred_label}')

    def show_case_from_string(self, input=""):
            input = [input]
            label = [[0]*10]
            print(f'input: {input}')
            # categories_dict = {k: v for k, v in enumerate(categories)}

            test_set = CustomDataset(input, label, tokenizer, max_len)
            test_set = DataLoader(test_set, batch_size=1)

            for data in test_set:
                output = model_aspect_detection(
                    input_ids=data['input_ids'].to(self.device),
                    attention_mask=data['attention_mask'].to(self.device),
                    target=None,
                    test=True
                )
                output_norm = (output >= 0.5).int().detach().cpu()[0]
                pred_label = [categories[i] for i, v in enumerate(output_norm) if v == 1]
                print(f'pred label: {pred_label}')


showcase = show_case()
showcase.show_case_from_test_set(0)

print("-"*60)

test_input = "Điện thoải ổn. Facelock cực nhanh, vân tay ôk , màn hình lớn, pin trâu ( liên quân , Zalo, YouTube ) một ngày mất khoảng 45 % ) tuy chỉ chip 439 nhưng rất mượt. Đa nhiệm khá ổn"
showcase.show_case_from_string(test_input)


