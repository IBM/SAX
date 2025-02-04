import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report
import os
from datetime import datetime


class IMDBDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {}
        for key in self.dataset.column_names:
            val = self.dataset[key][idx]
            if isinstance(val, list) or isinstance(val, int):
                item[key] = torch.tensor(val)
            else:
                item[key] = val
        return item


# Define the BERT-based classifier
class SentimentClassifier(nn.Module):
    def __init__(self, num_labels, max_length, freeze_bert=False):
        super(SentimentClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels, output_hidden_states=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, max_length)

        # Freeze BERT parameters if specified :
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        last_hidden_state = outputs.hidden_states[-1]  # Get the last hidden layer from BERT
        fc_output = torch.sigmoid(self.fc(last_hidden_state[:, 0, :]))  # Take the [CLS] token's representation
        return outputs, fc_output

class DoubleBert(nn.Module):
    def __init__(self, num_labels, max_length):
        super(DoubleBert, self).__init__()
        self.bert = SentimentClassifier(num_labels, max_length)
        self.complementry = None
        self.threshold = 0

    def forward(self, input_ids, attention_mask, labels=None):
        outputs, sufficient = self.bert(input_ids, attention_mask=attention_mask, labels=labels)

        #sufficient[attention_mask==0] = 0   # zero padded tokens
        mask = sufficient.ge(self.threshold)
        mask2 = sufficient.lt(self.threshold)
        if self.complementry == 'fixed':
            new_x = (input_ids * mask) + (torch.full(input_ids.size(), 103).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) * mask2) # 103 is the [MASK] token
        if self.complementry == 'noise':
            new_x = (input_ids * mask) + (torch.randint(0, 30522, input_ids.size()).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) * mask2)
        outputs2, _ = self.bert(new_x, attention_mask=attention_mask, labels=torch.argmax(outputs.logits, axis=1))
        return outputs, sufficient, outputs2


# Tokenization function
def tokenize_function(example):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

def train_model(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability):

    multiple_gpus = False
    # Load the IMDB dataset and save to a folder named "data" :
    cache_dir = "./sufficient_xai/data_sg"
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    dataset = load_dataset("imdb", cache_dir=cache_dir)

    # Tokenize the dataset :
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Split into training and testing
    dataset = IMDBDataset(tokenized_datasets["train"])
    test_dataset = IMDBDataset(tokenized_datasets["test"])
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=16)

    # Define model, loss, optimizer, and scheduler :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_length = 128

    model= DoubleBert(max_length=max_length, num_labels=2).to(device)
    model.complementry = complementry
    model.threshold = threshold
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    num_epochs = num_of_epochs
    min_val_loss = float('inf')
    count_val_loss = 0
    for epoch in tqdm(range(num_epochs), position=0, leave=True):
        model.train()
        total_loss = 0
        counter=0
        for batch in tqdm(train_loader, position=0, leave=True):
            counter+=1
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs, sufficient, outputs2 = model(input_ids, attention_mask=attention_mask, labels=labels)
            cardinality_loss = gamma_value * torch.norm(sufficient, p=1) / (input_ids.shape[0] * input_ids.shape[1])

            loss = outputs.loss + outputs2.loss + cardinality_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"imdb.py::train_model: {dt_string} - Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}", flush=True)

        # Evaluation
        model.eval()

        predictions = []
        predictions_explanations = []
        true_labels = []

        total_val_loss = 0.0
        total_regularization = 0.0
        correct_explanation = 0.0
        counter = 0
        total_val = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, position=0, leave=True):
                counter+=1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs, sufficient, outputs2 = model(input_ids, attention_mask=attention_mask, labels=labels)
                cardinality_loss = gamma_value * torch.norm(sufficient, p=1) / (input_ids.shape[0] * input_ids.shape[1])

                loss = outputs.loss + outputs2.loss + cardinality_loss
                if multiple_gpus:
                    total_val_loss += loss.sum().item()
                else:
                    total_val_loss += loss.item()

                prediction = outputs.logits.argmax(dim=1).cpu().numpy()
                prediction_explanation = outputs2.logits.argmax(dim=1).cpu().numpy()

                predictions.extend(prediction)
                predictions_explanations.extend(prediction_explanation)
                true_labels.extend(labels.cpu().numpy())
                
                correct_explanation += outputs.logits.argmax(dim=1).eq(outputs2.logits.argmax(dim=1)).sum().item()
                total_val += labels.size(0)

                num_positive = (outputs2.logits >= threshold).sum().item()
                total_positive = num_positive / outputs2.logits.numel()
                regularization_term = torch.tensor(total_positive)
                total_regularization += regularization_term


        total_val_loss = total_val_loss/total_val
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Validation Loss: {total_val_loss:.4f}", flush=True)
        print(f"Validation Accuracy: {accuracy*100:.4f}", flush=True)
        print(f"Validation Fidelity: {correct_explanation / total_val * 100:.4f}", flush=True)
        print(f"Percentage Sufficiency (Validation): {total_regularization / counter *100:.4f}", flush=True)

        if total_val_loss <= min_val_loss:
            min_val_loss = total_val_loss
            torch.save(model.state_dict(), f'./sufficient_xai/methodological/trained_models/imdb/imdb_{complementry}_lr_{learning_rate}_g_{gamma_value}_e_{epsilon_pgd}')
            count_val_loss = 0
        else:
            count_val_loss += 1
        if count_val_loss == 5:
            break


def run_main_code_imdb(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability):
    train_model(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability)


