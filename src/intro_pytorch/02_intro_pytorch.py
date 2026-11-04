import torch
import torch.nn as nn

# 1. The Conveyor Belt (Assembly Line)
# Data flows from top to bottom automatically
model = nn.Sequential(
    nn.Linear(in_features=1, out_features=4),  # Station 1: Factory
    nn.ReLU(),                                 # Station 2: Security Guard
    nn.Linear(in_features=4, out_features=1)   # Station 3: Final Factory
)

# 2. Dummy input
x = torch.tensor([[3.0]])

# 3. Forward pass (Just 1 line!)
# 'x' enters the belt and comes out as 'final_pred'
final_pred = model(x)

print(f"Final Prediction: {final_pred.item():.3f}")

# 4. Print the blueprint of the assembly line
print(f"\nModel Architecture:\n{model}")