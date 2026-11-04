import torch
import torch.nn as nn

"""# 1. Define the "Space Warping" layers
# Input (1) -> Hidden 16D -> Hidden 32D -> Output (1)
layer1 = nn.Linear(in_features=1, out_features=16)
layer2 = nn.Linear(in_features=16, out_features=32)
layer_out = nn.Linear(in_features=32, out_features=1)

# 2. Dummy input (1 sample, 1 feature)
x = torch.tensor([[3.0]])

# 3. Forward pass (chaining the layers sequentially)
h1 = layer1(x)
h2 = layer2(h1)
final_pred = layer_out(h2)

# 4. Print the shape transformation
print(f"Input: {x.shape} -> H1: {h1.shape} -> H2: {h2.shape} -> Output: {final_pred.shape}")


# 1. Define the Activation Function
relu = nn.ReLU()

# 2. Dummy raw output from a Linear layer
# Notice we have negative and positive signal values
raw_signals = torch.tensor([[-5.0, 2.0, -10.0, 8.0]])

# 3. Pass signals through the threshold
# Rule: If positive, pass it. If negative, crush it to 0.
activated_signals = relu(raw_signals)

print(f"Raw signals: {raw_signals}")
print(f"Activated signals: {activated_signals}")"""



import torch
import torch.nn as nn

# 1. Factory 1 (Expands 1 feature to 4 features)
layer1 = nn.Linear(in_features=1, out_features=4)

# 2. The Security Guard (The Activation Function)
relu = nn.ReLU()

# 3. Factory 2 (Takes 4 cleaned features, squashes to 1)
layer2 = nn.Linear(in_features=4, out_features=1)

# 4. Dummy input
x = torch.tensor([[3.0]])

# 5. Forward pass: Factory -> Guard -> Factory
# Step A: Linear outputs raw numbers (can be negative!)
raw_output = layer1(x)

# Step B: ReLU crushes the negative numbers to 0.0
cleaned_output = relu(raw_output)

# Step C: Pass the clean data to the next layer
final_pred = layer2(cleaned_output)

print(f"1. Raw from Linear:\n{raw_output}")
print(f"\n2. Cleaned by ReLU:\n{cleaned_output}")
print(f"\n3. Final Prediction: {final_pred.item():.3f}")