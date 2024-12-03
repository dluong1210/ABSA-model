from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import pickle


def train_model(model, criterion, optimizer, data_loader_train, 
                data_loader_valid, epochs, device, task='detection'):
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

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(data_loader_train)}")

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
            print(f"Validation Accuracy: {accuracy:.4f}")
        else:
            f1 = f1_score(all_targets, all_preds, average='micro')
            print(f"Validation F1 Score: {f1:.4f}")

        print(f"Validation Loss: {valid_loss / len(data_loader_valid)}")

        # get result 
        f1 = f1_score(all_targets, all_preds, average='micro')
        if f1 > f1_best: 
            f1_best = f1
            print(f'best in epoch: {epoch+1}/{epochs} - best f1 score: {f1_best}')
            # save best model
            name_model_pkl = 'model_aspect.pkl' if isinstance(criterion, nn.BCEWithLogitsLoss) else 'model_sentiment.pkl'
            with open(name_model_pkl, 'wb') as f:
                pickle.dump(model, f)

