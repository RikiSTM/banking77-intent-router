import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel
from torch.utils.data import DataLoader
import dataset as ds  # use the same import style as train.py
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

class IntentRouter(nn.Module):
    """Blueprint must be IDENTICAL to the one used during training."""
    def __init__(self, num_classes: int = 77):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.head = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_vector)


def find_project_root():
    # Walk up until the .git marker is found
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Project root (.git) not found!")


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Rebuild the empty skeleton
model = IntentRouter().to(device)

# 2. Inject the trained weights from disk
weight_path = find_project_root() / "models" / "bert_banking77_latest.pt"
model.load_state_dict(torch.load(weight_path, map_location=device))

# 3. Switch to exam mode (turns off Dropout)
model.eval()
print("Model loaded. Ready for evaluation.")


# Rebuild the ORIGINAL train mapping, then force it onto test
train_ds = ds.BankingDataset(ds.train_df)
test_ds = ds.BankingDataset(ds.test_df, train_ds.label_map)

test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
print(f"Test samples: {len(test_ds)}")

# Inspect the actual contents of the checkpoint
state = torch.load(weight_path, map_location="cpu")
keys = list(state.keys())
print(f"Total tensors: {len(keys)}")
print("First :", keys[0], tuple(state[keys[0]].shape))
print("Last  :", keys[-1], tuple(state[keys[-1]].shape))

# Eval start
correct = 0
total = 0
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        preds = logits.argmax(dim=1)

        # Move back to CPU before converting to plain Python lists
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# Invert label_map so the report shows intent names, not indices
idx_to_label = {i: name for name, i in train_ds.label_map.items()}
target_names = [idx_to_label[i] for i in range(len(idx_to_label))]

acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)


# Build the report as a string instead of flooding the terminal
report = classification_report(
    all_labels, all_preds,
    labels=list(range(len(target_names))),
    target_names=target_names,
    digits=3,
    zero_division=0,
)

# Save to reports/ at project root for eval_report.txt 
report_dir = find_project_root() / "reports"
report_dir.mkdir(parents=True, exist_ok=True)
report_path = report_dir / "eval_report.txt"
report_path.write_text(f"Accuracy: {acc:.4f}\n\n{report}")

print(f"Accuracy: {acc:.4f}")
print(f"Report saved to {report_path}")


# create confusion pair report
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(target_names))))

support = cm.sum(axis=1)   # row sum = actual questions per class
claims = cm.sum(axis=0)    # column sum = how often model claimed each class

# Isolate errors only; diagonal holds correct predictions
errors = cm.copy()
np.fill_diagonal(errors, 0)

flat_order = np.argsort(errors, axis=None)[::-1]
rows, cols = np.unravel_index(flat_order, errors.shape)

lines = ["TOP CONFUSION PAIRS", "=" * 100]
for r, c in zip(rows[:20], cols[:20]):
    if errors[r, c] == 0:
        break
    lines.append(f"{target_names[r]:<48} -> {target_names[c]:<42} {errors[r, c]}")

# Expose the claim counts hidden by classification_report
lines += ["", "MOST OVER-CLAIMED INTENTS", "=" * 100]
for i in np.argsort(claims - support)[::-1][:8]:
    lines.append(
        f"{target_names[i]:<48} claims={claims[i]:<5} actual={support[i]:<5} excess={claims[i] - support[i]}"
    )


cm_path = report_dir / "confusion_matrix.txt"
cm_path.write_text("\n".join(lines))
print(f"Confusion matrix saved to {cm_path}")