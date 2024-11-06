import pickle
import math
import random
import pandas as pd
from collections import Counter
import visual as vi
#from node2vec import Node2Vec

import util as utill
import similarity as sim

def get_bulk_visualization(args):
    #length_types = ['short', 'medium', 'long', 'all']
    length_types = ['all']
    for dataset_name in args.datasets:
        print('---------------------------------------------------')
        for length_type in length_types:
            dataset = utill.get_dataset(dataset_name, length_type, if_augmented=False)

            occurrences = utill.get_last_item_occurance(dataset)

            vi.visualize_bin_occurrence(occurrences)

def get_bulk_similarity(args):

    for filter_connected in args.filter_connected_list:
        for dataset_name in args.datasets:
            dataset = utill.get_dataset(dataset_name, length_type = 'all', if_augmented=False)
        
            #cosine_sim, similarity_pairs = sim.get_node2vec_similarity(dataset)
            sim.get_node2vec_similarity(dataset, dataset_name, filter_connected)

            print('similarity done')

def get_bulk_statistics(args):
    length_types = ['all']
    training_ratios = []
    augmented_ratios = []
    swapped_ratios = []

    dataset_names = []

    for dataset_name in args.datasets:
        for length_type in length_types:
            training_dataset = utill.get_dataset(dataset_name, length_type, if_augmented=False)
            augmented_dataset = utill.get_dataset(dataset_name, length_type, if_augmented=True)
            swapped_dataset = utill.label_swap(training_dataset)

            # Check for differences
            #half_index = len(swapped_dataset) // 2
            #original_dataset = swapped_dataset[:half_index]
            #modified_dataset = swapped_dataset[half_index:]
            # compare_datasets(original_dataset, modified_dataset)


            training_all_occurrences = utill.get_all_item_occurance(training_dataset)
            training_label_occurrences = utill.get_last_item_occurance(training_dataset)
            
            augmented_all_occurances = utill.get_all_item_occurance(augmented_dataset)
            augmented_label_occurances = utill.get_last_item_occurance(augmented_dataset)
            
            swapped_all_occurrences = utill.get_all_item_occurance(swapped_dataset)
            swapped_label_occurrences = utill.get_last_item_occurance(swapped_dataset)


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

def get_bulk_similarity_based_augment(args):
    length_types = ['all']
    iteration_range = range(1, 6)  # Loop from 1 to 5

    for dataset_name in args.datasets:
        for length_type in length_types:
            density_values = []
            edge_counts = []
            labels = []

            training_dataset = utill.get_dataset(dataset_name, length_type, if_augmented=False)

            # Loop over iterations
            for iteration_k in iteration_range:
                # Perform similarity-based augmentation with different iteration_k
                entire_substitute_dataset = utill.similarity_based_augment(training_dataset, dataset_name, similarity_type='entire', augment_type='substitute', iteration_k=iteration_k)
                entire_insert_dataset = utill.similarity_based_augment(training_dataset, dataset_name, similarity_type='entire', augment_type='insert', iteration_k=iteration_k)
                unseen_substitute_dataset = utill.similarity_based_augment(training_dataset, dataset_name, similarity_type='unseen', augment_type='substitute', iteration_k=iteration_k)
                unseen_insert_dataset = utill.similarity_based_augment(training_dataset, dataset_name, similarity_type='unseen', augment_type='insert', iteration_k=iteration_k)

                label_swap_entire_substitute_dataset = utill.label_swap(entire_substitute_dataset)
                label_swap_entire_insert_dataset = utill.label_swap(entire_insert_dataset)
                label_swap_unseen_substitute_dataset = utill.label_swap(unseen_substitute_dataset)
                label_swap_unseen_insert_dataset = utill.label_swap(unseen_insert_dataset)

                # Create a dictionary to hold datasets for current iteration_k
                datasets = {
                    'entire_substitute_G': entire_substitute_dataset,
                    'entire_insert_G': entire_insert_dataset,
                    'unseen_substitute_G': unseen_substitute_dataset,
                    'unseen_insert_G': unseen_insert_dataset,
                    'label_swap_entire_substitute_G': label_swap_entire_substitute_dataset,
                    'label_swap_entire_insert_G': label_swap_entire_insert_dataset,
                    'label_swap_unseen_substitute_G': label_swap_unseen_substitute_dataset,
                    'label_swap_unseen_insert_G': label_swap_unseen_insert_dataset
                }

                # Loop over each dataset in the current iteration and calculate graph properties
                for name, dataset in datasets.items():
                    density, num_edges = sim.calculate_graph_properties(dataset)
                    density_values.append(density)
                    edge_counts.append(num_edges)
                    labels.append(f"{name}_k{iteration_k}")  # Include iteration_k in the label

                # Visualize the graph properties and save the images
                vi.visual_density_edge_number(density_values, edge_counts, labels, dataset_name, iteration_k)

def get_bulk_similarity_ranking(args):
    length_types = ['all']
    similarity_type_list = ['unseen', 'entire']
    #similarity_type_list = ['unseen']

    for dataset_name in args.datasets:
        for length_type in length_types:
            training_dataset = utill.get_dataset(dataset_name, length_type, if_augmented=False)
            for similarity_type in similarity_type_list:
                head_highest_similarity, tail_highest_similarity = sim.get_similarity_ranking(training_dataset, dataset_name, similarity_type)

                params_head = {
                    'highest_similarity': head_highest_similarity,
                    'dataset_name': dataset_name,
                    'similarity_type': similarity_type,
                    'item_type': 'head'
                }

                params_tail = {
                    'highest_similarity': tail_highest_similarity,
                    'dataset_name': dataset_name,
                    'similarity_type': similarity_type,
                    'item_type': 'tail'
                }


                vi.visual_similarity_ranking_plot(params_head)
                vi.visual_similarity_ranking_plot(params_tail)