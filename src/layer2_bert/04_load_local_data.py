import pandas as pd
from pathlib import Path

# 1. Define raw data paths
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

# 2. Load raw datasets directly
train_df = pd.read_csv(RAW_DIR / "train.csv")
test_df = pd.read_csv(RAW_DIR / "test.csv")

# 3. Light-cleaning function for BERT (Preserve grammar & context)
def light_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Lowercase and strip leading/trailing whitespaces
    return text.lower().strip()

# 4. Apply light-cleaning to the text column
train_df["clean_text"] = train_df["text"].apply(light_clean)
test_df["clean_text"] = test_df["text"].apply(light_clean)

# 5. Inspect the difference visually
print("--- RAW TEXT ---")
print(train_df["text"].iloc[0])
print("\n--- BERT CLEAN TEXT ---")
print(train_df["clean_text"].iloc[0])

print(f"\nTrain Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")