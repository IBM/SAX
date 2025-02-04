import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from tqdm import tqdm


# Load the SNLI dataset (you need to have the SNLI dataset in a suitable format)
# You can download it from here: https://nlp.stanford.edu/projects/snli/
# For this example, let's assume you have two CSV files "train.csv" and "dev.csv" containing your data.

# Define a custom dataset class
class SNLIDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2num = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise = self.data.loc[idx, 'sentence1']
        hypothesis = self.data.loc[idx, 'sentence2']
        label = self.label2num[max(Counter(self.data.loc[idx, 'annotator_labels']), key=Counter(self.data.loc[idx, 'annotator_labels']).get)]

        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Custom module class that includes BERT with a fully connected layer
class BERTWithFC(nn.Module):
    def __init__(self, num_labels, max_length):
        super(BERTWithFC, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels, output_hidden_states=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, max_length)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, labels=labels)
        last_hidden_state = outputs.hidden_states[-1]  # Get the last hidden layer from BERT
        fc_output = torch.sigmoid(self.fc(last_hidden_state[:, 0, :]))  # Take the [CLS] token's representation
        return outputs, fc_output

class DoubleBert(nn.Module):
    def __init__(self, num_labels, max_length):
        super(DoubleBert, self).__init__()
        self.bert = BERTWithFC(num_labels, max_length)
        self.complementry = None
        self.threshold = 0

    def forward(self, input_ids, attention_mask, labels=None):
        outputs, sufficient = self.bert(input_ids, attention_mask=attention_mask, labels=labels)

        mask = sufficient.ge(self.threshold)
        mask2 = sufficient.lt(self.threshold)
        if self.complementry == 'fixed':
            new_x = (input_ids * mask) + (torch.full(input_ids.size(), 103).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) * mask2) # 103 is the [MASK] token
        if self.complementry == 'noise':
            new_x = (input_ids * mask) + (torch.randint(0, 30522, input_ids.size()).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) * mask2)
        outputs2, _ = self.bert(new_x, attention_mask=attention_mask, labels=torch.argmax(outputs.logits, axis=1))
        return outputs, sufficient, outputs2

def train_model(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability):

    train_data = pd.read_json("./sufficient_xai/data/snli_1.0/snli_1.0_train.jsonl", lines=True)
    dev_data = pd.read_json("./sufficient_xai/data/snli_1.0/snli_1.0_dev.jsonl", lines=True)

    print('gpu' if torch.cuda.is_available() else 'cpu', flush=True)

    # Initialize the BERT tokenizer and custom model:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets and data loaders:
    max_length = 128
    train_dataset = SNLIDataset(train_data, tokenizer, max_length)
    dev_dataset = SNLIDataset(dev_data, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size,num_workers=16, pin_memory=True)

    multiple_gpus = False

    model = DoubleBert(num_labels=3, max_length = max_length)  # 3 classes (contradiction, neutral, entailment), hidden size of FC layer = 128
    model.complementry = complementry
    model.threshold = threshold
    if multiple_gpus:
        model = nn.DataParallel(model)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Define training parameters:
    num_epochs = num_of_epochs
    weight_decay = 0.01

    # Define loss function and optimizer:
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    min_val_loss = float('inf')
    count_val_loss = 0

    # Training loop
    for epoch in tqdm(range(num_epochs), position=0, leave=True):
        model.train()
        total_loss = 0.0
        counter= 0

        for batch in tqdm(train_loader, position=0, leave=True):
            counter+=1
            input_ids = batch['input_ids'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            attention_mask = batch['attention_mask'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            labels = batch['labels'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            optimizer.zero_grad()
            outputs, sufficient, outputs2 = model(input_ids, attention_mask=attention_mask, labels=labels)
            cardinality_loss = gamma_value * torch.norm(sufficient, p=1) / (input_ids.shape[0] * input_ids.shape[1])

            loss = outputs.loss + outputs2.loss + cardinality_loss
            if multiple_gpus:
                loss.sum().backward()
            else:
                loss.backward()
            optimizer.step()
            if multiple_gpus:
                total_loss += loss.sum().item()
            else:
                total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}", flush=True)

        # Evaluation:
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
            for batch in dev_loader:
                counter+=1
                input_ids = batch['input_ids'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                attention_mask = batch['attention_mask'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                labels = batch['labels'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                optimizer.zero_grad()
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
            torch.save(model.state_dict(), f'./sufficient_xai/methodological/trained_models/snli/snli_{complementry}_lr_{learning_rate}_g_{gamma_value}_e_{epsilon_pgd}')
            count_val_loss = 0
        else:
            count_val_loss += 1
        if count_val_loss == 5:
            break


def run_main_code_snli(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability):
    train_model(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability)
