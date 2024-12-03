import torch
import torch.nn as nn
from transformers import BertModel
from pytorch_metric_learning.losses import NTXentLoss, SupConLoss
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

        self.cl_alpha = 0.05
    
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
