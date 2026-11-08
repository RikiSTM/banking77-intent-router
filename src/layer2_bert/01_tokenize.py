from transformers import AutoTokenizer
import torch

# 1. Load pre-trained BERT tokenizer (contains 30,522 vocab items)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. Raw text from banking customer
text = "I lost my credit card"

# 3. Encode text into PyTorch tensors
# 'pt' stands for PyTorch format
encoded = tokenizer(text, return_tensors="pt")

# 4. Print the actual ID sequence
print(f"Input IDs: {encoded['input_ids']}")

# 5. Prove how BERT breaks words into sub-words (WordPiece)
pieces = tokenizer.tokenize(text)
print(f"Sub-words: {pieces}")