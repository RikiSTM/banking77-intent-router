import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# 1. Load raw data and tokenizer
ROOT = Path(__file__).resolve().parents[2]
train_df = pd.read_csv(ROOT / "data/raw/train.csv")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. Create the PyTorch Dataset bridge
class BankingDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        # Map string labels ("card_arrival") to integers (0, 1, 2...)
        self.label_map = {label: i for i, label in enumerate(df["category"].unique())}
        self.labels = [self.label_map[label] for label in df["category"]]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize one row on-the-fly
        text = self.texts[idx]
        encoded = tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=64
        )
        
        return {
            "input_ids": torch.tensor(encoded["input_ids"]),
            "attention_mask": torch.tensor(encoded["attention_mask"]),
            "label": torch.tensor(self.labels[idx])
        }

# 3. Initialize and inspect
train_dataset = BankingDataset(train_df)
print(f"Total Rows: {len(train_dataset)}")
print(f"Item 0 Keys: {train_dataset[0].keys()}")
print(f"Input IDs Shape: {train_dataset[0]['input_ids'].shape}")
print(f"Integer Label: {train_dataset[0]['label']}")

# 4. Create the DataLoader (the delivery truck)
train_loader = DataLoader(
    train_dataset,   # The warehouse (our custom Dataset)
    batch_size=16,   # Fetch 16 packages per trip
    shuffle=True     # Randomize order every epoch
)

# 5. Inspect the first batch manually
batch = next(iter(train_loader))
print(f"input_ids shape: {batch['input_ids'].shape}")
print(f"label shape: {batch['label'].shape}")