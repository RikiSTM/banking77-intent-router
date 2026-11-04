import torch

device = torch.device("mps")

'''
# 2. Create tensors DIRECTLY on the GPU (Pro-tip!)
# We add device=device so it's born in the GPU, skipping CPU
a = torch.tensor([1.0, 2.0, 3.0], device=device)
b = torch.tensor([4.0, 5.0, 6.0], device=device)

# 3. Element-wise multiplication (Multiplies matching pairs)
# Math: [1*4, 2*5, 3*6]
c = a * b

# 4. Dot product (Multiplies pairs, then SUMS them all up)
# Math: (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
d = torch.dot(a, b)

print(f"Element-wise result: {c}")
print(f"Dot product result: {d}")

# Prove: SHAPE (dimension) and DTYPE (data type) are DIFFERENT things
print(f"c -> shape: {c.shape}, dtype: {c.dtype}")
print(f"d -> shape: {d.shape}, dtype: {d.dtype}")

# Force integer dtype to make the .0 disappear
d_int = d.to(torch.int64)
print(f"d as integer: {d_int}")

# Element-wise: same positions multiplied, shapes must match
a = torch.tensor([[1., 2.],
                  [3., 4.]])
b = torch.tensor([[5., 6.],
                  [7., 8.]])
hadamard = a * b   # [[1*5, 2*6], [3*7, 4*8]]

# Matrix mult: row of A dotted with column of B
# Shape rule: (2,2) @ (2,2) -> (2,2)
matmul = a @ b

print(f"Hadamard:\n{hadamard}")
print(f"Matmul:\n{matmul}")
'''

'''# This tensor WILL be tracked (it's a weight)
weight = torch.tensor([2.0], requires_grad=True)

# This tensor will NOT be tracked (it's just input data)
input_data = torch.tensor([3.0])

# Forward pass
y = weight * input_data

# Backward pass
y.backward()

# Check who got the gradient
print(f"Weight gradient: {weight.grad}")           # tensor([3.])
print(f"Input gradient: {input_data.grad}")         # None (not tracked!)'''


# The knob (weight) we want to tune
w = torch.tensor([2.0], requires_grad=True)

# Fixed input data and our goal
x = torch.tensor([3.0])
y_true = torch.tensor([12.0])

# The optimizer: the hand that turns the knob
# lr = learning rate = how big each turn is
optimizer = torch.optim.SGD([w], lr=0.05)


for step in range(5):
    optimizer.zero_grad()      # wipe old gradient (prevent accumulation)
    y_pred = w * x                  # forward pass (prediction)
    loss = (y_pred - y_true) ** 2   # how wrong are we?
    loss.backward()            # compute the compass
    print(f"Step {step}: GRAD = {w.grad.item():.3f}")  # the compass
    optimizer.step()
    print(f"Step {step}: WEIGHT = {w.item():.3f}, loss = {loss.item():.3f}")