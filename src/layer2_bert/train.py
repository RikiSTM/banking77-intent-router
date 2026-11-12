import importlib
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel
import dataset as ds
from pathlib import Path


# 1. Wrap BERT + head into one nn.Module
class IntentRouter(nn.Module):
    def __init__(self, num_classes: int = 77):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.head = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        # Extract the [CLS] vector from BERT's last hidden state
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_vector)

# 2. Setup device, model, loss, optimizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)
model = IntentRouter().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 3. Rebuild dataset + loader from previous module
train_dataset = ds.BankingDataset(ds.train_df)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 4. Training loop: the 5-step cycle
for epoch in range(4):
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()                        # Reset old gradients
        logits = model(input_ids, attention_mask)    # Forward
        loss = criterion(logits, labels)             # Loss
        loss.backward()                              # Backward
        optimizer.step()                             # Update weights

        total_loss += loss.item()
    print(f"Epoch done. Avg loss: {total_loss / len(train_loader):.4f}")
    
    

def find_project_root():
    # Walk up from this file until we find the .git folder
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Root project (.git) not found!")

# 1. Find Root & Create 'models' folder safely
ROOT = find_project_root()
save_dir = ROOT / "models"
save_dir.mkdir(parents=True, exist_ok=True)

# 2. Save the model to the exact root project
save_path = save_dir / "bert_banking77_latest.pt"
torch.save(model.state_dict(), save_path)
print(f"SUCCESS! Model saved to Root: {save_path}")