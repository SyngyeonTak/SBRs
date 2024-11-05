import argparse
import pickle
from collections import Counter
import numpy as np
import visualization as vi
import nodecentrality as nc
import matplotlib.pyplot as plt
import math
import pandas as pd

exclusive_rates = {}
discrepency_rates = {}

def save_dataset(filename, data):
    with open(f'{filename}.txt', 'wb') as f:
        pickle.dump(data, f)

# Function to load the dataset from pickle file
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

def calculate_click_statistics(data):
    # Assuming 'data' is the extracted list from the tuple
    # Flatten the list of lists into a single list of item IDs
    #flattened_data = [item for sublist in data for item in sublist]
    item_counts = Counter(data)
    items, counts = zip(*item_counts.items())

    # Sort items and counts by counts (frequency)
    items_sorted_by_counts, counts_sorted = zip(*sorted(zip(items, counts), key=lambda x: x[1], reverse=True))

    max_item = np.max(items_sorted_by_counts)
    max_count = np.max(counts)

    print('max_item: ', max_item)
    print('max_count: ', max_count)

    return items_sorted_by_counts, counts_sorted

def get_basic_statistics(dataset_name, length_type ,if_augmented = False):
    
    type_str = 'original'

    if length_type == 'all':
        if if_augmented:
            dataset = load_dataset(f'./datasets/{dataset_name}/train')
            dataset = flatten_dataset(dataset)
            type_str = 'augmented'
        else:
            dataset = load_dataset(f'./datasets/{dataset_name}/{length_type}_train_seq')
    else:
        if if_augmented:
            dataset = load_dataset(f'./datasets/{dataset_name}/{length_type}_train')
            type_str = 'augmented'
        else:
            dataset = load_dataset(f'./datasets/{dataset_name}/{length_type}_train_seq')
    

    dataset_length = len(dataset)

    lengths = [len(sublist) for sublist in dataset]

    # Calculate the average length
    dataset_length_average = sum(lengths) / len(dataset)

    variance = sum((length - dataset_length_average) ** 2 for length in lengths) / len(dataset)

    # Calculate the standard deviation
    standard_deviation = math.sqrt(variance)

    dataset_name
    #print('dataset_name: ', dataset_name)
    #print('dataset_type: ', type_str)
    #print('length_type: ', length_type)
    #print('size: ', dataset_length)
    #print('average length: ', round(dataset_length_average, 2))
    #print('length std: ', round(standard_deviation, 2))
    
    return {
        'dataset_name': dataset_name,
        'dataset_type': type_str,
        'length_type': length_type,
        'size': dataset_length,
        'average_length': round(dataset_length_average, 2),
        'length_std': round(standard_deviation, 2)
    }



def calculate_session_groups(data):
    # Count the number of items in each session
    session_lengths = [len(sublist) for sublist in data]

    # Initialize counters for the groups
    group1_count = 0
    group2_count = 0
    group3_count = 0

    # Count sessions in each group
    for length in session_lengths:
        if length > 8:
            group1_count += 1
        if length > 5 and length <= 8:
            group2_count += 1
        elif length < 5:
            group3_count += 1

    return group1_count, group2_count, group3_count, session_lengths


# Function to calculate basic statistics
def calculate_basic_statistics(counts):
    mean_count = np.mean(counts)
    max_count = np.max(counts)
    std_count = np.std(counts)
    return mean_count, max_count, std_count

def count_high_nodes_in_sessions(data, high_degree_nodes, high_coreness_nodes):
    session_high_degree_counts = []
    session_high_coreness_counts = []

    for session in data:
        high_degree_count = len([node for node in session if node in high_degree_nodes])
        high_coreness_count = len([node for node in session if node in high_coreness_nodes])

        session_high_degree_counts.append(high_degree_count)
        session_high_coreness_counts.append(high_coreness_count)

    return session_high_degree_counts, session_high_coreness_counts



def get_exclusive_rate(original_G, augmented_G, original_percent, augmented_percent, node_centrality, dataset_name, data_type):
    print('data_type: ', data_type)

    number_of_nodes = original_G.number_of_nodes()
    original_percent_nodes = int(original_percent * number_of_nodes)
    augmented_percent_nodes = int(augmented_percent * number_of_nodes)

    original_nodecentrality = nc.detect_cal_node_centrality(node_centrality, original_G)
    augmented_nodecentrality = nc.detect_cal_node_centrality(node_centrality, augmented_G)


    sorted_original_nodecentrality = sorted(list(original_nodecentrality.items()), key=lambda x: x[1], reverse=True)
    sorted_augmented_nodecentrality = sorted(list(augmented_nodecentrality.items()), key=lambda x: x[1], reverse=True)

    top_original_nodecentrality = sorted_original_nodecentrality[:original_percent_nodes]
    top_augmented_nodecentrality = sorted_augmented_nodecentrality[:augmented_percent_nodes]

    #print('top_original_nodecentrality: ', top_original_nodecentrality)
    #print('top_augmented_nodecentrality: ', top_augmented_nodecentrality)

    top_original_item = [item[0] for item in top_original_nodecentrality]
    top_augmented_item = [item[0] for item in top_augmented_nodecentrality]

    difference = [item for item in top_original_item if item not in top_augmented_item]

    number_of_discrepency = len(difference)
    discrepency_rate = round(float((number_of_discrepency) / original_percent_nodes), 2)

    if original_percent not in exclusive_rates:
        exclusive_rates[original_percent] = {}

    if dataset_name not in exclusive_rates[original_percent]:
        exclusive_rates[original_percent][dataset_name] = {}

    if data_type not in exclusive_rates[original_percent][dataset_name]:
        exclusive_rates[original_percent][dataset_name][data_type] = {}    

    exclusive_rates[original_percent][dataset_name][data_type][node_centrality] = discrepency_rate


def augment_list(original_dataset):
    augmented_list = []

    for sequence in original_dataset:
        # Add the original sequence first
        augmented_list.append(sequence)
        
        # Add sequences with prefixes longer than 2
        for i in range(2, len(sequence)):
            augmented_list.append(sequence[:i])
    
    return augmented_list

def filter_sequences(original_dataset, min_length=2, max_length=float('inf')):
    # Ensure the input is a list of lists
    if not all(isinstance(sequence, list) for sequence in original_dataset):
        raise ValueError("original_dataset must be a list of lists.")
    
    # Filter based on sequence length
    filtered_list = [
        i for i, sequence in enumerate(original_dataset)
        if len(sequence) >= min_length and len(sequence) <= max_length
    ]
    return filtered_list

def flatten(dataset):
    first_list = dataset[0]
    second_list = dataset[1]

    # Append elements from the second list to the corresponding sublists in the first list
    augmented_all_dataset = [sublist + [item] for sublist, item in zip(first_list, second_list)]

    return augmented_all_dataset


def process_and_plot_dataset(dataset_name, dataset_type, ax, node_centrality):
    """
    Process datasets of a specific type and plot the node centrality rankings.
    
    Args:
        dataset_name (str): Name of the dataset.
        dataset_type (str): Type of the dataset (short, medium, long, all).
        ax (matplotlib.axes.Axes): The axes on which to plot.
    """
    print(f'Processing dataset type: {dataset_type}')

    # Load original and augmented datasets
    if dataset_type == 'all':
        original_dataset = load_dataset(f'./datasets/{dataset_name}/{dataset_type}_train_seq')
        augmented_dataset = load_dataset(f'./datasets/{dataset_name}/train')
        augmented_dataset = flatten_dataset(augmented_dataset)
    else:
        original_dataset = load_dataset(f'./datasets/{dataset_name}/{dataset_type}_train_seq')
        augmented_dataset = load_dataset(f'./datasets/{dataset_name}/{dataset_type}_train')

    # Build graphs
    original_G = nc.build_graph(original_dataset)
    augmented_G = nc.build_graph(augmented_dataset)
    
    # Calculate node centrality for the original graph
    original_node_centrality = nc.detect_cal_node_centrality(node_centrality, original_G)
    augmented_node_centrality = nc.detect_cal_node_centrality(node_centrality, augmented_G)
    
    # Get node centrality ranking
    ori_ori_ranked_dataset = nc.get_nodecentrality_ranking(original_dataset, original_node_centrality)
    aug_aug_ranked_dataset = nc.get_nodecentrality_ranking(augmented_dataset, augmented_node_centrality)
    

    print('original_dataset: ', len(original_dataset))
    print('augmented_dataset: ',len(augmented_dataset))
    print('ori_ori_ranked_dataset: ',len(ori_ori_ranked_dataset))
    print('aug_aug_ranked_dataset: ',len(aug_aug_ranked_dataset))

    # Plot for the given dataset type
    vi.analyze_and_plot_last_item_rankings(ori_ori_ranked_dataset, f'Original {dataset_type.capitalize()} {node_centrality} Rankings', ax[0], color='skyblue')
    vi.analyze_and_plot_last_item_rankings(aug_aug_ranked_dataset, f'Augmented {dataset_type.capitalize()} {node_centrality} Rankings', ax[1], color='lightgreen')


def do_excursive_rate(args):
    for dataset_name in args.datasets:
        print('---------------------------------------------------')
        
        print(f'Processing dataset: {dataset_name}')

        #-------------------------------------------------------------

        original_short_dataset = load_dataset(f'./datasets/{dataset_name}/short_train_seq')
        augmented_short_dataset = load_dataset(f'./datasets/{dataset_name}/short_train')

        original_short_G = nc.build_graph(original_short_dataset)
        augmented_short_G = nc.build_graph(augmented_short_dataset)

        #-------------------------------------------------------------

        original_medium_dataset = load_dataset(f'./datasets/{dataset_name}/medium_train_seq')
        augmented_medium_dataset = load_dataset(f'./datasets/{dataset_name}/medium_train')

        original_medium_G = nc.build_graph(original_medium_dataset)
        augmented_medium_G = nc.build_graph(augmented_medium_dataset)

        #-------------------------------------------------------------

        original_long_dataset = load_dataset(f'./datasets/{dataset_name}/long_train_seq')
        augmented_long_dataset = load_dataset(f'./datasets/{dataset_name}/long_train')

        original_long_G = nc.build_graph(original_long_dataset)
        augmented_long_G = nc.build_graph(augmented_long_dataset)

        #-------------------------------------------------------------
        original_all_dataset = load_dataset(f'./datasets/{dataset_name}/all_train_seq')
        augmented_all_dataset = load_dataset(f'./datasets/{dataset_name}/train')
        augmented_all_dataset = flatten_dataset(augmented_all_dataset)

        original_all_G = nc.build_graph(original_all_dataset)
        augmented_all_G = nc.build_graph(augmented_all_dataset)

        #-------------------------------------------------------------

        for original_percent in args.original_percents:
            print(f'in the scope of original_percent: {original_percent}')
            for taget_node_centrality in args.taget_node_centralities:
                print(f'in the scope of target node_centrality: {taget_node_centrality}')
                get_exclusive_rate(original_short_G, augmented_short_G, original_percent, 0.1, taget_node_centrality, dataset_name, 'short')
                get_exclusive_rate(original_medium_G, augmented_medium_G, original_percent, 0.1, taget_node_centrality, dataset_name, 'medium')
                get_exclusive_rate(original_long_G, augmented_long_G, original_percent, 0.1, taget_node_centrality, dataset_name, 'long')
                get_exclusive_rate(original_all_G, augmented_all_G, original_percent, 0.1, taget_node_centrality, dataset_name, 'all')

    #print(exclusive_rates)

    vi.plot_mismatch_rate(exclusive_rates)   

def do_split_dataset(args): 
    for dataset_name in args.datasets:

        original_dataset = load_dataset(f'./datasets/{dataset_name}/all_train_seq')

        short_indice = filter_sequences(original_dataset, min_length=2, max_length = 4)
        medium_indice = filter_sequences(original_dataset, min_length=5, max_length = 8)
        long_indice = filter_sequences(original_dataset, min_length=9)
        
        short_sequence = [original_dataset[i] for i in short_indice]
        medium_sequence = [original_dataset[i] for i in medium_indice]
        long_sequence = [original_dataset[i] for i in long_indice]

        augmented_short_sequence = augment_list(short_sequence)
        augmented_medium_sequence = augment_list(medium_sequence)
        augmented_long_sequence = augment_list(long_sequence)

        save_dataset(f'./datasets/{dataset_name}/short_train_seq', short_sequence)
        save_dataset(f'./datasets/{dataset_name}/medium_train_seq', medium_sequence)
        save_dataset(f'./datasets/{dataset_name}/long_train_seq', long_sequence)

        save_dataset(f'./datasets/{dataset_name}/short_train', augmented_short_sequence)
        save_dataset(f'./datasets/{dataset_name}/medium_train', augmented_medium_sequence)
        save_dataset(f'./datasets/{dataset_name}/long_train', augmented_long_sequence)

def do_make_long_dataset(args): 
    for dataset_name in args.datasets:

        original_dataset = load_dataset(f'./datasets/{dataset_name}/all_train_seq')

        long_indice = filter_sequences(original_dataset, min_length=9, max_length=19)
        
        long_sequence = [original_dataset[i] for i in long_indice]

        augmented_long_sequence = augment_list(long_sequence)
        save_dataset(f'./datasets/{dataset_name}/longt_train_seq', long_sequence)
        save_dataset(f'./datasets/{dataset_name}/longt_train', augmented_long_sequence)

def do_last_item_ranking(args):
    #dataset_types = ['short', 'medium', 'long', 'longt','all']
    dataset_types = ['longt']

    for dataset_name in args.datasets:
        print('---------------------------------------------------')
        print(f'Processing dataset: {dataset_name}')

        #-------------------------------------------------------------

        for dtype in dataset_types:
            for taget_node_centrality in args.taget_node_centralities:
                fig, ax = plt.subplots(1, 2, figsize=(15, 6))
                process_and_plot_dataset(dataset_name, dtype, ax, taget_node_centrality)
                
                # Save the plot for each dataset type
                plt.tight_layout()
                plt.savefig(f'./practice/noise_aware/images/last_ranking_diff/{dataset_name}/{taget_node_centrality}/ranking_comparison_{dataset_name}_{taget_node_centrality}_{dtype}.png', format='png')
                #plt.savefig(f'ranking_comparison_{dataset_name}_{taget_node_centrality}_{dtype}.png', format='png')

                plt.close(fig)
        

def do_basic_statistics(args):
    length_types = ['short', 'medium', 'long', 'longt','all']
    statistics = []

    for dataset_name in args.datasets:
        print('---------------------------------------------------')
        for length_type in length_types:
            stats_augmented  = get_basic_statistics(dataset_name, length_type, if_augmented=True)
            stats_original = get_basic_statistics(dataset_name, length_type, if_augmented=False)

            statistics.append(stats_augmented)
            statistics.append(stats_original)

    df = pd.DataFrame(statistics)
    
    # Group by 'dataset_name', 'dataset_type', and 'length_type'
    grouped_df = df.groupby(['dataset_name', 'length_type', 'dataset_type']).agg(
        size=('size', 'sum'),
        average_length=('average_length', 'mean'),
        length_std=('length_std', 'mean')
    ).reset_index()

    # Display the grouped DataFrame
    print(grouped_df)

    grouped_df.to_csv('session_statiscitcs.csv', encoding= "utf-8")