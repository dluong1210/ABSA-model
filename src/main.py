import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import copy
import argparse 

from transformers import AutoTokenizer, AutoModel, AutoConfig
from dataset import get_data_from_raw, CustomDataset
from model import AspectDetection, SentimentClassification
from train import train_model
from eval import evaluate_model
from torch.utils.data import DataLoader
import warnings
from sklearn.exceptions import UndefinedMetricWarning

def get_cmd():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-epochs', '--epochs', default=10, type=int)
    parser.add_argument('-layer_bert', '--num_hidden_layers_bert', default=6, type=int)
    parser.add_argument('-cl_alpha', '--cl_alpha', default=0.2, type=float) # hyper weight for contrastive loss 
    parser.add_argument('-log_path', '--log_path', default="", type=str)

    args = parser.parse_args()
    return args

# Disable warnings from sklearn
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

categories = ["GENERAL", "CAMERA", "PERFORMANCE", "DESIGN","BATTERY","FEATURES","SCREEN","STORAGE","PRICE","SER&ACC"]
sentiment = ["Negative", "Positive", "Neutral"]

conf = defaultdict(list)

# collect hyperparams to dict: conf ~ config
conf = defaultdict(list)

paras = get_cmd().__dict__
print(f'paras: {paras}')
for p in paras: 
    conf[p] = paras[p]

conf['max_len'] = 256
conf['batch_size'] = 32    
conf['lr'] = 2e-5

# conf['epochs'] = 8
# conf['epochs'] = 8
conf['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# conf['num_hidden_layers_bert'] = 6
conf['num_hidden_size_bert'] = 768
conf['hidden_dim'] = 256

# contrastive params
conf['augment'] = 'FN'
conf['noise_weight'] = 0.05
# conf['cl_alpha'] = 0.2
conf['classification_dim'] = 768
conf['dropout'] = 0.3
conf['temp'] = 0.2

def init_bert(conf, init_style='default'):
    print('starting init bert')
    if init_style == 'autoconfig': 
        # phobert_config = AutoConfig.from_pretrained("vinai/phobert-base-v2")
        # # default config in phobert 
        # print(f"default num_hidden_layers: {phobert_config.num_hidden_layers}")
        # print(f"default num_attention_heads: {phobert_config.num_attention_heads}")
        # print(f"default hidden_size: {phobert_config.hidden_size}")
        # print("-"*60) # clean 
        # # change config
        # phobert_config.num_hidden_layers = conf['num_hidden_layers_bert']
        # phobert_config.hidden_size = conf['num_hidden_size_bert']
        # phobert = AutoModel.from_config(phobert_config)

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
        
    if init_style == 'default':
        phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")

    print("done init bert")

    return phobert



def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    # Load data
    print("Loading data...")
    x1, y1, x2, y2 = get_data_from_raw('/kaggle/input/uit-visfd/Train.csv')
    x1_val, y1_val, x2_val, y2_val = get_data_from_raw('/kaggle/input/uit-visfd/Dev.csv')
    x1_test, y1_test, x2_test, y2_test = get_data_from_raw('/kaggle/input/uit-visfd/Test.csv', test=True)

    # Prepare datasets and dataloaders
    max_len = conf['max_len']
    batch_size = conf['batch_size']

    print("Preparing datasets and dataloaders...")
    #Dataloader cho Aspect Detection model
    aspect_train_dataset = CustomDataset(x1, y1, tokenizer, max_len)
    aspect_val_dataset = CustomDataset(x1_val, y1_val, tokenizer, max_len)
    aspect_test_dataset = CustomDataset(x1_test, y1_test, tokenizer, max_len)

    aspect_train_loader = DataLoader(aspect_train_dataset, batch_size=batch_size, shuffle=True)
    aspect_val_loader = DataLoader(aspect_val_dataset, batch_size=batch_size, shuffle=True)
    aspect_test_loader = DataLoader(aspect_test_dataset, batch_size=batch_size, shuffle=False)

    #Dataloader cho Sentiment Classification model
    sentiment_train_dataset = CustomDataset(x2, y2, tokenizer, max_len)
    sentiment_val_dataset = CustomDataset(x2_val, y2_val, tokenizer, max_len)
    sentiment_test_dataset = CustomDataset(x2_test, y2_test, tokenizer, max_len)

    sentiment_train_loader = DataLoader(sentiment_train_dataset, batch_size=batch_size, shuffle=True)
    sentiment_val_loader = DataLoader(sentiment_val_dataset, batch_size=batch_size, shuffle=True)
    sentiment_test_loader = DataLoader(sentiment_test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    print("Initializing Aspect Detection model...")
    phobert = init_bert(init_style='autoconfig', conf=conf)
    model_aspect_detection = AspectDetection(phobert, conf['hidden_dim'], len(categories), conf)

    # Set up training components
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model_aspect_detection.parameters(), lr=conf['lr'])

    # Train the model
    print("Training Aspect Detection model...")
    train_model(
        model=model_aspect_detection,
        criterion=criterion,
        optimizer=optimizer,
        data_loader_train=aspect_train_loader,
        data_loader_valid=aspect_val_loader,
        epochs=conf['epochs'],
        device=conf['device'], 
        path_log=conf['log_path']
    )

    # Evaluate the model
    print("Evaluating Aspect Detection model...")
    evaluate_model(model_aspect_detection, aspect_test_loader, conf['device'])

    print('-' * 60)
    print('start get bert from stage 1 ...')
    bert_stage_1 = model_aspect_detection.get_bert()
    print('end get bert from stage 1')
    print('-'*60)
    print('start copy bert to stage 2')
    bert_stage_2 = copy.deepcopy(bert_stage_1)
    print('end copy bert to stage 2')

    model_sentiment = SentimentClassification(
        bert_model=bert_stage_2, 
        hidden_dim=conf['hidden_dim'], 
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

    print('-' * 60)
    print("Successfully train and eval 2 model...")

if __name__ == "__main__":
    main()
