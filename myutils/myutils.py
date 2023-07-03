def get_zero_index_from_num_array(array):
    """
    Returns the index of the first occurrence of zero in the given array.

    Args:
        array (list): A list of numbers.

    Returns:
        int: The index of the first occurrence of zero in the array,
             or -1 if there are no zero elements or multiple zero elements.
    """
    index_array = [index for (index, value) in enumerate(array) if value == 0]
    return index_array[0] if len(index_array) == 1 else -1


def is_single_significantly_below(array):
    """
    Checks if there is exactly one number that is significantly below all the other values in the given array of floats.

    Args:
        array (list): A list of floats representing the values to be checked.

    Returns:
        bool: True if there is exactly one number significantly below all the others, False otherwise.

    """
    minimum = min(array)
    maximum = max(array)
    threshold = (maximum - minimum) * 0.1  # Adjust the multiplier (0.1) as needed

    count = 0

    for value in array:
        if value < minimum - threshold:
            count += 1

    return count == 1


def find_same_indices(numbers):
    """
    Finds the indices of the same numbers in a given integer array.

    Args:
        numbers (list[int]): The input array of integers.

    Returns:
        dict: A dictionary where keys are the numbers that appear more than once, and values are lists of indices where those numbers appear.

    Example:
        actions = [1, 2, 3, 2, 4, 1, 5, 5, 6, 7, 6, 1]
        same_indices = find_same_indices(actions)
        # Output: {1: [0, 5, 11], 2: [1, 3], 5: [6, 7], 6: [8, 10]}
    """
    indices = {}  # Dictionary to store numbers as keys and indices as values
    for i, num in enumerate(numbers):
        if num in indices:
            indices[num].append(i)  # Append the index to the existing list for the number
        else:
            indices[num] = [i]  # Create a new list for the number and store the index

    # Remove entries with only one index
    indices = {num: index_list for num, index_list in indices.items() if len(index_list) > 1}

    return indices
