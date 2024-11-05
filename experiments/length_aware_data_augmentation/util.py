import pickle
import math
import random
import pandas as pd
from collections import Counter
import visual as vi
from node2vec import Node2Vec
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

def get_undirected_graph(dataset):
    G = nx.Graph()  # Use an undirected graph
    unique_nodes = set()  # To store unique nodes

    for sublist in dataset:
        for i in range(len(sublist) - 1):
            source = sublist[i]
            target = sublist[i + 1]
            unique_nodes.add(source)
            unique_nodes.add(target)
    
            # Add an unweighted edge between source and target
            G.add_edge(source, target)

    # Remove self-loops if any
    self_loops = list(nx.selfloop_edges(G))
    if len(self_loops) > 0:
        G.remove_edges_from(self_loops)
        print(f"Removed {len(self_loops)} self-loops from the graph.")

    return G

def get_node2vec_similarity(G):
    G = nx.relabel_nodes(G, lambda x: str(x))

    # Define a custom temporary directory with an ASCII-only path
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Initialize Node2Vec model with the custom temp_folder path
        node2vec = Node2Vec(G, dimensions=100, walk_length=80, num_walks=10, p=1, q=1, workers=8, temp_folder=tmpdirname)
        
        # Generate embeddings
        model = node2vec.fit(window=10, min_count=1, batch_words=4, epochs=5)

    # Get embeddings as vectors
    node_embeddings = [model.wv[str(node)] for node in G.nodes()]

    # Compute cosine similarity between nodes
    cosine_sim = cosine_similarity(node_embeddings)

    # Print similarity between node 0 and node 1 (if there are at least 2 nodes)
    if len(G.nodes()) > 1:
        print("Cosine similarity between node 0 and node 1:", cosine_sim[0, 1])
    else:
        print("Not enough nodes to calculate similarity.")

    return cosine_sim  # Return the cosine similarity matrix if needed



# Function to load the dataset from pickle file
def load_dataset(dataset_name):
    with open(f'{dataset_name}.txt', 'rb') as f:
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

def get_bulk_visualization(args):
    #length_types = ['short', 'medium', 'long', 'all']
    length_types = ['all']
    for dataset_name in args.datasets:
        print('---------------------------------------------------')
        for length_type in length_types:
            dataset = get_dataset(dataset_name, length_type, if_augmented=False)

            occurrences = get_last_item_occurance(dataset)

            vi.visualize_bin_occurrence(occurrences)

def get_bulk_similarity(args):
    for dataset_name in args.datasets:
        dataset = get_dataset(dataset_name, length_type = 'all', if_augmented=False)
        session_G = get_undirected_graph(dataset)
        similarity = get_node2vec_similarity(session_G)

        print(similarity)

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




def get_bulk_statistics(args):
    length_types = ['all']
    training_ratios = []
    augmented_ratios = []
    swapped_ratios = []

    dataset_names = []

    for dataset_name in args.datasets:
        for length_type in length_types:
            training_dataset = get_dataset(dataset_name, length_type, if_augmented=False)
            augmented_dataset = get_dataset(dataset_name, length_type, if_augmented=True)
            swapped_dataset = label_swap(training_dataset)

            # Check for differences
            #half_index = len(swapped_dataset) // 2
            #original_dataset = swapped_dataset[:half_index]
            #modified_dataset = swapped_dataset[half_index:]
            # compare_datasets(original_dataset, modified_dataset)


            training_all_occurrences = get_all_item_occurance(training_dataset)
            training_label_occurrences = get_last_item_occurance(training_dataset)
            
            augmented_all_occurances = get_all_item_occurance(augmented_dataset)
            augmented_label_occurances = get_last_item_occurance(augmented_dataset)
            
            swapped_all_occurrences = get_all_item_occurance(swapped_dataset)
            swapped_label_occurrences = get_last_item_occurance(swapped_dataset)


            # Calculate ratios
            training_ratio = len(training_label_occurrences) / len(training_all_occurrences)
            augmented_ratio = len(augmented_label_occurances) / len(augmented_all_occurances)
            swapped_ratio = len(swapped_label_occurrences) / len(swapped_all_occurrences)

            # Store ratios for plotting
            training_ratios.append(training_ratio)
            augmented_ratios.append(augmented_ratio)
            swapped_ratios.append(swapped_ratio)

            dataset_names.append(dataset_name)

            print('training session size: ', len(training_dataset))
            print('training all size: ', len(training_all_occurrences))
            print('training label size: ', len(training_label_occurrences))

            print('augmented_data session size: ', len(augmented_dataset))
            print('augmented_data all size: ', len(augmented_all_occurances))
            print('augmented_data label size: ', len(augmented_label_occurances))

            print('swapped_data session size: ', len(swapped_dataset))
            print('swapped_data all size: ', len(swapped_all_occurrences))
            print('swapped_data label size: ', len(swapped_label_occurrences))

    # Plotting the bar plot
    vi.visualize_dataset_label_ratio(training_ratios, augmented_ratios, swapped_ratios, dataset_names)



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
    modified_dataset = []
    length_dataset = [len(sequence) for sequence in training_dataset]
    average_length = sum(length_dataset) / len(length_dataset)
    windowsize = math.floor(average_length / 2)
    
    for i, sequence in enumerate(training_dataset):
        # Create multiple augmented sequences for each input sequence
        for swap_count in range(len(sequence) - 2):  # Total swaps equal to length of sequence - 1
            modified_sequence = sequence.copy()
            effective_window_size = min(windowsize, len(sequence))  # Adjust for short sequences
            window_items = sequence[-effective_window_size:]

            # Ensure there's more than one item to swap
            if len(window_items) > 1:
                # Choose an item to swap with the last item
                random_item = window_items[swap_count % (effective_window_size - 1)]
                
                # Find indices in the original sequence
                last_item_index = len(sequence) - 1
                random_item_index = sequence.index(random_item, len(sequence) - effective_window_size)
                
                # Swap the items
                modified_sequence[last_item_index], modified_sequence[random_item_index] = (
                    modified_sequence[random_item_index],
                    modified_sequence[last_item_index],
                )
            
            # Append each modified sequence to the dataset
            modified_dataset.append(modified_sequence)

    combined_dataset = training_dataset + modified_dataset
    return combined_dataset

        



