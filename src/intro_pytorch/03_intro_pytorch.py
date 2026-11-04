import torch
import torch.nn as nn

# Create a custom Neural Network blueprint
class SimpleNet(nn.Module):
    def __init__(self):
        # Mandatory: Register this class as a PyTorch module
        super().__init__()
        
        # Define the parts of our factory (Layers)
        self.layer1 = nn.Linear(in_features=1, out_features=4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=4, out_features=1)

    def forward(self, x):
        # Define exactly how data flows through the parts
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Build the actual model from the blueprint
model = SimpleNet()

# Dummy input
x = torch.tensor([[3.0]])

# Forward pass
prediction = model(x)

print(f"Prediction: {prediction.item():.3f}")
print(f"\nModel Architecture:\n{model}")