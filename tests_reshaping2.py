import numpy as np
import torch

np_arr = torch.Tensor(np.zeros(shape=(32, 370, 1)))

print(np_arr.shape)
result = torch.mean(np_arr, dim=(0, 1))

print(result.shape)
print(result.tolist())
