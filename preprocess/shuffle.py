import pickle
import random

def load_dataset(dataset_name):
    with open(f'{dataset_name}.txt', 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def flatten_dataset(dataset):
    # Unpack the dataset tuple
    list_of_lists, targets = dataset
    
    # Ensure that the lengths of list_of_lists and targets match
    if len(list_of_lists) != len(targets):
        raise ValueError("Length of list_of_lists and targets must be the same.")
    
    # Combine each list from list_of_lists with the corresponding target
    flattened_dataset = [lst + [target] for lst, target in zip(list_of_lists, targets)]
    
    return flattened_dataset

def unflatten_dataset(flattened_dataset):
    # Check if flattened_dataset is empty
    if not flattened_dataset:
        return [], []
    
    # Extract the targets (last element of each sublist)
    targets = [lst[-1] for lst in flattened_dataset]
    
    # Extract the list_of_lists (all elements except the last one in each sublist)
    list_of_lists = [lst[:-1] for lst in flattened_dataset]
    
    return list_of_lists, targets

def full_shuffle(dataset, keep_first=True, keep_last=True):
    shuffled_dataset = []
    
    # Determine indices based on flags; these are constant for the whole dataset
    start_index = 1 if keep_first else 0
    end_index = -1 if keep_last else None

    for sequence in dataset:
        # Extract elements based on the flags
        first_element = sequence[0] if keep_first else None
        last_element = sequence[-1] if keep_last else None
        shuffle_section = sequence[start_index:end_index]

        # Include the original sequence
        shuffled_dataset.append(sequence)
        
        # Precompute the prefix and suffix based on flags
        prefix = [first_element] if keep_first else []
        suffix = [last_element] if keep_last else []

        # Shuffle and construct new sequences
        num_shuffles = len(shuffle_section)
        for _ in range(num_shuffles):
            random.shuffle(shuffle_section)
            # Combine prefix, shuffled middle section, and suffix
            shuffled_sequence = prefix + shuffle_section + suffix
            shuffled_dataset.append(shuffled_sequence)

    return shuffled_dataset

def slide_shuffle(dataset, window_size=2, keep_first=True, keep_last=True):
    shuffled_dataset = []

    # Determine indices based on flags
    start_index = 1 if keep_first else 0
    end_index = -1 if keep_last else None
    
    for sequence in dataset:
        first_element = sequence[0] if keep_first else None
        last_element = sequence[-1] if keep_last else None
        shuffle_section = sequence[start_index:end_index]

        shuffled_dataset.append(sequence)
        
        prefix = [first_element] if keep_first else []
        suffix = [last_element] if keep_last else []

        if len(shuffle_section) < window_size:
            continue

        for i in range(len(shuffle_section) - window_size + 1):
            window = shuffle_section[i:i + window_size]
            random.shuffle(window)


            pre_window = shuffle_section[:i]
            post_window = shuffle_section[i + window_size:]

            shuffled_sequence = prefix + pre_window + window + post_window + suffix

            shuffled_dataset.append(shuffled_sequence)

    return shuffled_dataset