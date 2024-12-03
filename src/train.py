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

