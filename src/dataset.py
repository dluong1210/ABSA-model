import ast 
import pandas as pd
from underthesea import word_tokenize
from underthesea import text_normalize

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import Dataset

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
    if not test:
        df = upsampling_df(df)

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