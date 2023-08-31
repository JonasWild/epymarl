def generate_subsets(indices):
    subsets = []
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            subsets.append((indices[i], indices[j]))
    return subsets

# Example usage:
indices = [1, 4, 3]
print(generate_subsets(indices))