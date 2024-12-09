


import torch

# Create a simple tensor to illustrate the concept
# Let's create a 3x6 tensor for easy understanding
mx = torch.zeros((3, 6))

# Assume the following time domain indices to be masked (for example purpose)
# This is similar to the result of calling `nonzero()` on a condition
time_domain_masked_indices = torch.tensor([
    [0, 1], 
    [1, 2], 
    [2, 4]
    ])

# Transpose the indices to get them ready for advanced indexing
indices_for_advanced_indexing = tuple(time_domain_masked_indices.t())
print(indices_for_advanced_indexing)

# Create random noise for each index to be masked
masking_noise = torch.rand(time_domain_masked_indices.shape[0])

# Assign the masking noise to the specified indices in mx
mx[indices_for_advanced_indexing] = masking_noise

print("Masked tensor:")
print(mx)