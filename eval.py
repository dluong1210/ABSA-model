import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def evaluate_model(model, data_loader, device, short_format=False):
    model.eval()
    all_preds = []
    all_labels = []
    
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
        print(f"Macro F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"Micro F1 Score: {f1_micro:.4f}, Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}")

    if not short_format:
        print(classification_report(all_labels, all_preds))

    return f1, precision, recall
