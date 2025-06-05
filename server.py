import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random

os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Force offline mode


# Map status labels to numeric values
def map_status_to_label(status):
    status_map = {
        "Accepted": 0,
        "Interview Scheduled": 1,
        "Shortlisted": 2,
        "On Hold": 3,
        "Rejected": 4,
    }
    return status_map.get(status, 0)  # Default to 0 if status not found


# Process fake data to create text pairs and labels
def process_fake_data(data):
    text_pairs = []
    labels = []

    for job in data:
        jd = job["jd"]
        for cv_item in job["cvs"]:
            cv = cv_item["cv"]
            status = cv_item["status"]

            # Add JD and CV as a text pair
            text_pairs.append((jd, cv))

            # Convert status to numeric label
            label = map_status_to_label(status)
            labels.append(label)

    return text_pairs, labels


class TextPairDataset(Dataset):
    def __init__(self, text_pairs, labels, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.text_pairs = text_pairs
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        print("I AM IN HERE")
        return len(self.labels)

    def __getitem__(self, idx):
        text1, text2 = self.text_pairs[idx]

        encoding = self.tokenizer(
            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        token_type_ids = encoding["token_type_ids"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BertForTextPairClassification(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_rate=0.1):
        super(BertForTextPairClassification, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Feed-forward classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Use the [CLS] token representation
        pooled_output = outputs.pooler_output

        # Pass through classifier
        logits = self.classifier(pooled_output)

        return logits


def train_model(
    model, train_dataloader, val_dataloader, optimizer, criterion, epochs, device
):
    model.to(device)
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, token_type_ids)  #Forward Pass 
            loss = criterion(outputs, labels)  # This function is to calculate the loss between labels and outputs like output is score for each class and label which is coming from input

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

        train_accuracy = correct_predictions / total_predictions
        train_loss = train_loss / len(train_dataloader)

        # Validation
        val_accuracy, val_loss = evaluate_model(
            model, val_dataloader, criterion, device
        )

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pt")

    return model

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    average_loss = val_loss / len(dataloader)

    return accuracy, average_loss


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-downloaded model and tokenizer
    bert_model_name = "./local_bert_model"  # Path to locally stored model
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # Process fake data to create text pairs and labels
    text_pairs, labels = process_fake_data(fake_data)
    print(f"Loaded {len(text_pairs)} text pairs with labels")

    # Create dataset
    dataset = TextPairDataset(text_pairs, labels, tokenizer, max_length=256)

    # # # Split into train and validation
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # Use random_split with generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    print(f"Training on {train_size} samples, validating on {val_size} samples")

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    # # Initialize model
    num_classes = 5  # Five possible statuses: Accepted, Interview Scheduled, Shortlisted, On Hold, Rejected
    model = BertForTextPairClassification(bert_model_name, num_classes)

    # # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    print("Starting training...")
    model = train_model(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=5, device=device)

    print("Training complete!")

    # Save the final model
    torch.save(model.state_dict(), 'final_model.pt')
    print("Final model saved to final_model.pt")


if __name__ == "__main__":
    main()
