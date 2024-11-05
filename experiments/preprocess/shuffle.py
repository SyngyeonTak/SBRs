import pickle
import random

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