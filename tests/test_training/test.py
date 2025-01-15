import numpy as np
import torch

print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
# Converting NumPy to PyTorch
arr = np.array([1, 2, 3])
tensor = torch.from_numpy(arr)
print("Converted to PyTorch Tensor:", tensor)

# Converting PyTorch to NumPy
new_arr = tensor.numpy()
print("Converted back to NumPy Array:", new_arr)
