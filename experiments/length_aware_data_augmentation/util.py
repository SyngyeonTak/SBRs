import pickle
import math
import random
import pandas as pd
from collections import Counter
import visual as vi
#from node2vec import Node2Vec

import similarity as sim
import json



# Function to load the dataset from pickle file
def load_dataset(dataset_path):
    with open(f'{dataset_path}.txt', 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def flatten_dataset(dataset):
    list_of_lists, targets = dataset
    
    if len(list_of_lists) != len(targets):
        raise ValueError("Length of list_of_lists and targets must be the same.")

    flattened_dataset = [lst + [target] for lst, target in zip(list_of_lists, targets)]
    
    return flattened_dataset

def get_last_item_occurance(dataset):
    last_items = [sequence[-1] for sequence in dataset]

    # Count the occurrences of each last item
    occurrences = Counter(last_items)

    # Display the occurrences
    return occurrences

def count_last_item_frequencies(dataset, all_item_ids):
    # Initialize a Counter to keep track of last item occurrences
    last_item_frequencies = Counter()

    # Iterate through each sequence in the dataset
    for sequence in dataset:
        last_item = sequence[-1]  # Get the last item of the sequence
        last_item_frequencies[last_item] += 1  # Increment the count for this last item

    for item_id in all_item_ids:
        if item_id not in last_item_frequencies:
            last_item_frequencies[item_id] = 0  # Set frequency to 0 for missing items

    return last_item_frequencies


def get_all_item_occurance(dataset):
    all_items = [item for sequence in dataset for item in sequence]

    # Count the occurrences of each item
    occurrences = Counter(all_items)

    # Display the occurrences
    return occurrences

def get_dataset(dataset_name, length_type = 'all', if_augmented=False):
    base_path = f'./datasets/{dataset_name}/'
    if_flatten = False
    if length_type == 'all':
        dataset_path = f'{base_path}{"all_" if not if_augmented else ""}train{"_seq" if not if_augmented else ""}'

        if if_augmented:
                if_flatten = True
    
    else:
        dataset_path = f'{base_path}{length_type}_train{"_seq" if not if_augmented else ""}'

    dataset = load_dataset(dataset_path)

    if if_flatten:
        dataset = flatten_dataset(dataset)

    return dataset


def get_label_frequencies(args):
    length_types = ['all']

    all_training_frequencies = []
    all_augmented_frequencies = []
    all_swapped_frequencies = []

    all_item_ids = set()

    for dataset_name in args.datasets:
        for length_type in length_types:
            training_dataset = get_dataset(dataset_name, length_type, if_augmented=False)
            augmented_dataset = get_dataset(dataset_name, length_type, if_augmented=True)
            swapped_dataset = label_swap(training_dataset)

            #training_dataset = training_dataset[:1000]
            #augmented_dataset = augmented_dataset[:1000]
            #swapped_dataset = swapped_dataset[:1000]

            training_dataset = dummy_augment(training_dataset)
            #print(len(dummy_dataset))

            for sequence in training_dataset:
                all_item_ids.update(sequence)

            training_label_frequencies = count_last_item_frequencies(training_dataset, all_item_ids)
            augmented_label_frequencies = count_last_item_frequencies(augmented_dataset, all_item_ids)
            swapped_label_frequencies = count_last_item_frequencies(swapped_dataset, all_item_ids)

            all_training_frequencies.append(training_label_frequencies)
            all_augmented_frequencies.append(augmented_label_frequencies)
            all_swapped_frequencies.append(swapped_label_frequencies)

    #print(all_training_frequencies[0][:5])
    #first_dict = all_training_frequencies[0]

    all_frequencies = [all_training_frequencies, all_augmented_frequencies, all_swapped_frequencies]

    # Print first five items from the dictionary
    #print(len(all_augmented_frequencies[0]))
    #print(len(all_swapped_frequencies[0]))
    vi.visual_frequency_distribution(args.datasets, all_frequencies)



def compare_datasets(training_dataset, modified_dataset):
    for i, (original, modified) in enumerate(zip(training_dataset, modified_dataset)):
        if original != modified:
            print(f"Difference found in sequence {i}:")
            print("Original:", original)
            print("Modified:", modified)
        else:
            print(f"Sequence {i} is identical in both datasets.")

def dummy_augment(training_dataset):
    modified_dataset = []

    for i, sequence in enumerate(training_dataset):
        for _ in range(len(sequence) - 2):
            modified_dataset.append(sequence)
    combined_dataset = modified_dataset + training_dataset

    return combined_dataset        


def label_swap(training_dataset):
    label_swap_dataset = []
    length_dataset = [len(sequence) for sequence in training_dataset]
    average_length = sum(length_dataset) / len(length_dataset)
    windowsize = math.floor(average_length / 2)

    for sequence in training_dataset:
        modified_sequence = sequence.copy()
        effective_window_size = min(windowsize, len(sequence))  # Adjust for short sequences
        window_items = sequence[-effective_window_size:]

        # Ensure there's more than one item to swap
        if len(window_items) > 1:
            # Choose a random item from the window to swap with the last item
            random_item = random.choice(window_items[:-1])  # Exclude the last item in the window
            last_item_index = len(sequence) - 1
            random_item_index = sequence.index(random_item, len(sequence) - effective_window_size)
            
            # Swap the chosen item with the last item
            modified_sequence[last_item_index], modified_sequence[random_item_index] = (
                modified_sequence[random_item_index],
                modified_sequence[last_item_index],
            )

        # Append the modified sequence to the dataset
        label_swap_dataset.append(modified_sequence)

    # Combine the original dataset with the modified dataset
    #combined_dataset = training_dataset + modified_dataset
    return label_swap_dataset

def prefix_cropping(training_dataset):

    prefix_cropping_dataset = []

    for sequence in training_dataset:
        loop_range = len(sequence) - 2
 

        for idx in range(0, loop_range):
            modified_sequence = sequence[:-(idx+1)] # idx = 0 -> 0 ~ -1
            prefix_cropping_dataset.append(modified_sequence)

    return prefix_cropping_dataset

def select_random_items(session, max_items, min_items = 1):
    # Ignore the last item (target)
    input_length = len(session) - 1
    
    # Calculate the number of items to select based on the session length
    num_items = min(max_items, min_items + math.isqrt(input_length))
    
    # Get indices for selection, excluding the last item
    indices = list(range(input_length))
    
    # Randomly select indices based on the calculated number
    selected_indices = random.sample(indices, num_items)
    
    return selected_indices

def create_augmented_session(session, similarity_pairs, max_items=10, augment_type="substitute"):
    # Ignore the last item in the session (target)
    input_length = len(session)

    num_items = min(max_items, math.isqrt(input_length))
    selected_indices = []
    available_indices = set(range(input_length - 1))
    
    # Counter to ensure we don't loop infinitely
    attempts = 0
    max_attempts = len(available_indices)  # Only as many attempts as there are indices


    while len(selected_indices) < num_items and attempts < max_attempts:
        #print('selected_indices: ', selected_indices)
        #print('attempts: ', attempts)
        #print('num_items: ', num_items)
        #print('max_attempts: ', max_attempts)
        #print('available_indices: ', available_indices)
        #print('session: ', session)
        index = random.choice(list(available_indices))
        original_item = session[index]
        
        # Check if the original item has similar items in the pool
        if similarity_pairs.get(str(original_item)):
            selected_indices.append(index)
        else:
            # Increment attempts if no similar items found
            attempts += 1
        
        # Remove the index from the pool of available indices
        available_indices.remove(index)

    new_session = session[:]
    offset = 0

    for index in selected_indices:
        original_item = session[index]
        similar_items = similarity_pairs.get(str(original_item))
        if similar_items:
            # Randomly choose one similar item from the pool
            substitute_item = random.choice(similar_items)[0]
            
            if augment_type == "substitute":
                # Substitute the original item with the similar item
                new_session[index] = substitute_item
            elif augment_type == "insert":
                # Insert the similar item right after the selected item
                new_session.insert(index + 1 + offset, substitute_item)
                # Increment offset to adjust for the next insertion
                offset += 1
            else:
                raise ValueError("augment_type must be either 'substitute' or 'insert'")
            
    # if input_length > 7:
    #     print('session: ', session)
    #     print('new_session: ', new_session)
    #     print('selected_indices: ', selected_indices)
    #     return

    return new_session

def test_augmented_session():
    session = [0, 1, 2, 3, 4, "v_last"]
    similarity_pairs = sim.load_similarity_pairs('experiments/length_aware_data_augmentation/results/similarity/node2vec/diginetica_similarity_pairs_entire.txt')

    augmented_session = create_augmented_session(session, similarity_pairs, max_items = 10, augment_type="insert")

    print("Original Session:", session)
    print("Augmented Session:", augmented_session)


def similarity_based_augment(dataset, dataset_name, similarity_pool, similarity_type, augment_type, iteration_k = 1):
    similarity_pairs = sim.load_similarity_pairs(f'experiments/length_aware_data_augmentation/results/similarity/{similarity_type}/{dataset_name}_{similarity_pool}_similarity_pairs.txt')
    
    augmented_dataset = []

    if_highest_similarity = True
    if_similarity_threshold = True


    if if_highest_similarity:
        for key, pairs in similarity_pairs.items():
            # Find the pair with the highest value (second element of each pair)
            if not pairs:
                similarity_pairs[key] = []  # Keep it as an empty list
                continue

            highest_pair = max(pairs, key=lambda x: x[1])
            similarity_pairs[key] = [highest_pair]

    #print(similarity_pairs['9629'])

    if if_similarity_threshold:
        all_similarity = [pair[1] for pairs in similarity_pairs.values() for pair in pairs]

        threshold = sum(all_similarity) / len(all_similarity) if all_similarity else 0

        #print(threshold)


        for key, pairs in similarity_pairs.items():
            if not pairs:
                similarity_pairs[key] = []  # Keep it as an empty list
                continue
            similarity_pairs[key] = [pair for pair in pairs if pair[1] >= threshold]

    for i, session in enumerate(dataset):
        #print('original_sessiom: ', session)
        for _ in range(iteration_k):
            # Generate an augmented session with insertion
            augmented_session = create_augmented_session(session, similarity_pairs, max_items=1, augment_type= augment_type)
            augmented_dataset.append(augmented_session)


    augmented_key = f'{similarity_pool}_{augment_type}'

    return augmented_key, augmented_dataset

def similarity_based_augment_random(dataset, dataset_name, similarity_pool, similarity_type, augment_type, target_augmented_count):
    # Load similarity pairs
    similarity_pairs = sim.load_similarity_pairs(
        f'experiments/length_aware_data_augmentation/results/similarity/{similarity_type}/{dataset_name}_{similarity_pool}_similarity_pairs.txt'
    )
    
    augmented_dataset = []

    if_highest_similarity = True
    if_similarity_threshold = True


    if if_highest_similarity:
        for key, pairs in similarity_pairs.items():
            # Find the pair with the highest value (second element of each pair)
            if not pairs:
                similarity_pairs[key] = []  # Keep it as an empty list
                continue

            highest_pair = max(pairs, key=lambda x: x[1])
            similarity_pairs[key] = [highest_pair]

    #print(similarity_pairs['9629'])

    if if_similarity_threshold:
        all_similarity = [pair[1] for pairs in similarity_pairs.values() for pair in pairs]

        threshold = sum(all_similarity) / len(all_similarity) if all_similarity else 0

        #print(threshold)


        for key, pairs in similarity_pairs.items():
            if not pairs:
                similarity_pairs[key] = []  # Keep it as an empty list
                continue
            similarity_pairs[key] = [pair for pair in pairs if pair[1] >= threshold]

    while len(augmented_dataset) < target_augmented_count:
        # Randomly select a session index (duplicates allowed)
        session_index = random.randint(0, len(dataset) - 1)
        session = dataset[session_index]
        
        # Generate an augmented session
        augmented_session = create_augmented_session(
            session, similarity_pairs, max_items = 1, augment_type=augment_type
        )
        augmented_dataset.append(augmented_session)
 
        # Stop if the target count is reached
        if len(augmented_dataset) >= target_augmented_count:
            break

    augmented_key = f'{similarity_pool}_{augment_type}_random'
    
    return augmented_key, augmented_dataset

def get_solo_similarity_based_augment(args):
    length_types = ['all']
    iteration_range = range(1, 11)  # Loop from 1 to 5
    last_iteration_k = 1
    for dataset_name in args.datasets:
        for length_type in length_types:
            density_values = []
            edge_counts = []
            labels = []

            training_dataset = get_dataset(dataset_name, length_type, if_augmented=False)

            # Loop over iterations
            for iteration_k in iteration_range:
                # Perform similarity-based augmentation with different iteration_k
                #unseen_insert_dataset = similarity_based_augment(training_dataset, dataset_name, similarity_type='unseen', augment_type='insert', iteration_k=iteration_k)
                #label_swap_unseen_insert_dataset = label_swap(unseen_insert_dataset)
                entire_insert_dataset = similarity_based_augment(training_dataset, dataset_name, similarity_type='entire', augment_type='insert', iteration_k=iteration_k)
                label_swap_entire_insert_dataset = label_swap(entire_insert_dataset)


                # Create a dictionary to hold datasets for current iteration_k
                datasets = {
                    #'label_swap_unseen_insert_G': label_swap_unseen_insert_dataset
                    'label_swap_entire_insert_G': label_swap_entire_insert_dataset
                }

                # Loop over each dataset in the current iteration and calculate graph properties
                for name, dataset in datasets.items():
                    density, num_edges = sim.calculate_graph_properties(dataset)
                    density_values.append(density)
                    edge_counts.append(num_edges)
                    labels.append(f"{name}_k{iteration_k}")  # Include iteration_k in the label

                last_iteration_k = iteration_k
                # Visualize the graph properties and save the images
        vi.visual_density_edge_number(density_values, edge_counts, labels, dataset_name, last_iteration_k, similarity_type='entire')

def get_original_learnable_dataset(args):
    for dataset_name in args.datasets:
        dataset_path = f'./datasets/{dataset_name}/train'

        learnable_dataset = load_dataset(dataset_path)

        print(learnable_dataset[0][80:85])
        print(learnable_dataset[1][80:85])
        print(len(learnable_dataset))
        print(type(learnable_dataset))

def make_learnable_dataset(dataset):
    input_sequences = [sublist[:-1] for sublist in dataset]  # All elements except the last in each sublist
    target_items = [sublist[-1] for sublist in dataset]  
    result = (input_sequences, target_items)

    return result

def get_augmented_dataset(params):
    dataset_name = params['dataset_name']
    similarity_type = params['similarity_type']
    augment_type = params['augment_type']
    iteration_k = params['iteration_k']
    if_label_swap = params['if_label_swap']

    training_dataset = get_dataset(dataset_name, if_augmented=False)

    # Perform similarity-based augmentation with different iteration_k
    augmented_dataset = similarity_based_augment(training_dataset
                                                 , dataset_name
                                                 , similarity_type= similarity_type
                                                 , augment_type=augment_type
                                                 , iteration_k=iteration_k)
    
    if if_label_swap:
        augmented_dataset = label_swap(training_dataset)


    return augmented_dataset

def save_hop_relationships(dataset_name, similarity_type, similarity_pool):

    dataset_path = f'datasets/{dataset_name}/all_train_seq'

    dataset = load_dataset(dataset_path)

    G = sim.get_undirected_graph(dataset)

    similarity_path = f"experiments/length_aware_data_augmentation/results/similarity/{similarity_type}/{dataset_name}_{similarity_pool}_similarity_pairs"

    with open(f"{similarity_path}.txt", "r") as f:
        similarity_pairs = json.load(f)

    hop_relationships = sim.find_hop_relationships(G, similarity_pairs)

    with open(f"experiments/length_aware_data_augmentation/results/similarity/{similarity_type}/{dataset_name}_{similarity_pool}_hop_relationships.txt", "w") as f:
        json.dump(hop_relationships, f, indent=4)

    print("Hop relationships saved to 'hop_relationships.json'")

def get_direct_item_list(dataset_name, item_id):

    dataset = get_dataset(dataset_name, length_type = 'all', if_augmented=False)

    adjacent_items = set()
    
    # Iterate over each session in the dataset
    for session in dataset:
        # Check if the item_id is in the current session
        if item_id in session:
            # Add all other items in the session to the adjacent_items set
            adjacent_items.update(session)
    
    # Remove the item_id itself from the set of adjacent items (if present)
    adjacent_items.discard(item_id)
    
    print(adjacent_items)

    return adjacent_items