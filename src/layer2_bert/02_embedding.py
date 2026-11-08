import torch
from torch import nn
from transformers import AutoTokenizer


def main():
    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Raw banking text
    text = "I lost my credit card"

    # Convert text into token IDs
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]

    # Create an embedding lookup table
    # Each token ID will be mapped to a 768-dimensional vector
    embedding = nn.Embedding(
        num_embeddings=tokenizer.vocab_size,
        embedding_dim=768
    )

    # Convert token IDs into token vectors
    token_vectors = embedding(input_ids)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Token vectors shape: {token_vectors.shape}")


if __name__ == "__main__":
    main()