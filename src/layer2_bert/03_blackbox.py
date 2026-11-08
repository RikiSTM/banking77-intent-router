import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# 1. Load the pre-trained Black Box (BERT Transformer)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

# 2. Raw input and tokenization
text = "I lost my credit card"
encoded = tokenizer(text, return_tensors="pt")

# 3. Pass tokens through the Transformer Black Box
# This outputs the 'contextualized' vectors for all tokens
outputs = bert_model(**encoded)
hidden_states = outputs.last_hidden_state

# 4. Extract the [CLS] token vector (The Summary Box)
# Shape is [batch_size, sequence_length, hidden_dim] -> [1, 7, 768]
# We slice [:, 0, :] to grab the 0-th token (CLS) across all batches
cls_vector = hidden_states[:, 0, :] 

# 5. Feed the summary vector into our custom Classification Head
# Banking77 has 77 classes. Input dim MUST match BERT's output (768)
classification_head = nn.Linear(in_features=768, out_features=77)

# 6. Get the final raw predictions (logits)
logits = classification_head(cls_vector)

print(f"CLS Vector Shape: {cls_vector.shape}")
print(f"Final Logits Shape: {logits.shape}")

predicted_class_index = torch.argmax(logits, dim=1).item()

# Dummy label mapping for demonstration only
# In production, we will load the real 77 labels from the Banking77 dataset
id2label = {
    0: "card_arrival",
    1: "card_linking",
    2: "card_payment_not_recognised",
    3: "exchange_rate",
    4: "transfer_not_received_by_recipient",
    # ... all 77 labels will be added later
}

# Convert integer index to string label
predicted_label = id2label.get(predicted_class_index, "unknown_label")

print(f"Predicted Label: {predicted_label}")