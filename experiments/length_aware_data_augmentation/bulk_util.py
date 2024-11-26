import pickle
import math
import random
import pandas as pd
from collections import Counter
import visual as vi
#from node2vec import Node2Vec

import util
import similarity as sim

def get_bulk_visualization(args):
    #length_types = ['short', 'medium', 'long', 'all']
    length_types = ['all']
    for dataset_name in args.datasets:
        print('---------------------------------------------------')
        for length_type in length_types:
            dataset = util.get_dataset(dataset_name, length_type, if_augmented=False)

            occurrences = util.get_last_item_occurance(dataset)

            vi.visualize_bin_occurrence(occurrences)

def get_bulk_similarity(args):

    for dataset_name in args.datasets:
        for similarity_type in args.similarity_type_list:
            for filter_connected in args.filter_connected_list:            
                dataset = util.get_dataset(dataset_name, length_type = 'all', if_augmented=False)
                sim.get_similarity(dataset, dataset_name, filter_connected, similarity_type)
                
                print('similarity done')

def get_bulk_statistics(args):
    length_types = ['all']
    training_ratios = []
    augmented_ratios = []
    swapped_ratios = []

    dataset_names = []

    for dataset_name in args.datasets:
        for length_type in length_types:
            training_dataset = util.get_dataset(dataset_name, length_type, if_augmented=False)
            augmented_dataset = util.get_dataset(dataset_name, length_type, if_augmented=True)
            swapped_dataset = util.label_swap(training_dataset)

            training_all_occurrences = util.get_all_item_occurance(training_dataset)
            training_label_occurrences = util.get_last_item_occurance(training_dataset)
            
            augmented_all_occurances = util.get_all_item_occurance(augmented_dataset)
            augmented_label_occurances = util.get_last_item_occurance(augmented_dataset)
            
            swapped_all_occurrences = util.get_all_item_occurance(swapped_dataset)
            swapped_label_occurrences = util.get_last_item_occurance(swapped_dataset)


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
    iteration_range = range(1, 2)  # Loop from 1 to 5

    for dataset_name in args.datasets:
        training_dataset = util.get_dataset(dataset_name, length_type = 'all', if_augmented=False)

        # Loop over iterations
        for iteration_k in iteration_range:
            density_values = []
            edge_counts = []
            labels = []

            datasets = {'training_dataset': training_dataset
                        , 'training_dataset_label_swap': util.label_swap(training_dataset)
                        }

            # Perform similarity-based augmentation with different iteration_k
            for similarity_type in args.similarity_type_list:
                for similarity_pool in args.similarity_pool_list:
                    for augmented_type in args.augmented_type_list:
                        for if_random in args.if_random_list:
                            if if_random:
                                training_dataset_size = len(training_dataset)
                                prifix_dataset = util.get_dataset(dataset_name, length_type = 'all', if_augmented=True)
                                target_dataset_size = len(prifix_dataset)
                                similarity_augmented_key, similarity_augmented_dataset = util.similarity_based_augment_random(training_dataset, dataset_name, similarity_pool, similarity_type, augmented_type, target_augmented_count = (target_dataset_size - training_dataset_size))
                            else:
                                similarity_augmented_key, similarity_augmented_dataset = util.similarity_based_augment(training_dataset, dataset_name, similarity_pool, similarity_type, augmented_type, iteration_k=iteration_k)
                            similarity_augmented_dataset = similarity_augmented_dataset + training_dataset
                            datasets[similarity_augmented_key] = similarity_augmented_dataset

                            label_swap_dataset = util.label_swap(similarity_augmented_dataset)
                            label_swap_key = similarity_augmented_key + '_label_swap'
                            datasets[label_swap_key] = label_swap_dataset
   
            # Loop over each dataset in the current iteration and calculate graph properties
            for name, dataset in datasets.items():
                print(len(dataset))
                density, num_edges = sim.calculate_graph_properties(dataset)
                density_values.append(density)
                edge_counts.append(num_edges)
                labels.append(f"{name}")  # Include iteration_k in the label

            # Visualize the graph properties and save the images
            vi.visual_density_edge_number(density_values, labels, dataset_name, similarity_type, iteration_k, if_random= if_random)

def get_bulk_similarity_ranking(args):
    length_types = ['all']

    for dataset_name in args.datasets:
        for length_type in length_types:
            training_dataset = util.get_dataset(dataset_name, length_type, if_augmented=False)
            for similarity_pool in args.similarity_pool_list:
                head_highest_similarity, tail_highest_similarity = sim.get_similarity_ranking(training_dataset, dataset_name, similarity_pool)


                params_head = {
                    'highest_similarity': head_highest_similarity,
                    'dataset_name': dataset_name,
                    'similarity_pool': similarity_pool,
                    'item_type': 'head'
                }

                params_tail = {
                    'highest_similarity': tail_highest_similarity,
                    'dataset_name': dataset_name,
                    'similarity_pool': similarity_pool,
                    'item_type': 'tail'
                }


                vi.visual_similarity_ranking_plot(params_head)
                vi.visual_similarity_ranking_plot(params_tail)

      
def get_bulk_augmented_dataset(args):
    for dataset_name in args.datasets:
        params = {
            'dataset_name': dataset_name,
            'similarity_type': 'unseen',
            'augment_type': 'insert',
            'iteration_k': 2,
            'if_label_swap': False,
        }

        augment_dataset = util.get_augmented_dataset(params)

        print(len(augment_dataset))
        print(type(augment_dataset))

        # with open(f'./datasets/{params['dataset_name']}/{params['similarity_type']}_{params['augment_type']}_{params['iteration_k']}_{params['if_label_swap']}_.txt', 'wb') as file:
        #     pickle.dump(augment_dataset, file)

def save_bulk_save_hop_relationships(args):
    
    for dataset in args.datasets:
        for similarity_type in args.similarity_type_list:
            for similarity_pool in args.similarity_pool_list:
                util.save_hop_relationships(dataset, similarity_type, similarity_pool)

def clean_bulk_similarity_pool(args):
    
    for dataset in args.datasets:
        for similarity_type in args.similarity_type_list:
            for similarity_pool in args.similarity_pool_list:
                similarity_pairs = sim.load_similarity_pairs(f'experiments/length_aware_data_augmentation/results/similarity/{similarity_type}/{dataset}_{similarity_pool}_similarity_pairs.txt')

                similarity_pairs = sim.clean_similarity_pool(similarity_pairs, dataset, similarity_type, similarity_pool)

                #print(similarity_pairs['21922'])

def get_bulk_similarity_based_augment_random(args):
    iteration_range = range(1, 2)  # Loop from 1 to 5

    for dataset_name in args.datasets:
        training_dataset = util.get_dataset(dataset_name, length_type = 'all', if_augmented=False)

        # Loop over iterations
        for iteration_k in iteration_range:
            density_values = []
            edge_counts = []
            labels = []

            datasets = {'training_dataset': training_dataset
                        , 'training_dataset_label_swap': util.label_swap(training_dataset)
                        }

            # Perform similarity-based augmentation with different iteration_k
            for similarity_type in args.similarity_type_list:
                for similarity_pool in args.similarity_pool_list:
                    for augmented_type in args.augmented_type_list:
                        for if_random in args.if_random_list:
                            similarity_augmented_key, similarity_augmented_dataset = util.similarity_based_augment_random(training_dataset, dataset_name, similarity_pool, similarity_type, augmented_type, target_augmented_count = (719470 - len(training_dataset)))
                            similarity_augmented_dataset = similarity_augmented_dataset + training_dataset
                            datasets[similarity_augmented_key] = similarity_augmented_dataset

                            label_swap_dataset = util.label_swap(similarity_augmented_dataset)
                            label_swap_key = similarity_augmented_key + '_label_swap'
                            datasets[label_swap_key] = label_swap_dataset
   
            # Loop over each dataset in the current iteration and calculate graph properties
            for name, dataset in datasets.items():
                #print(len(dataset))
                density, num_edges = sim.calculate_graph_properties(dataset)
                density_values.append(density)
                edge_counts.append(num_edges)
                labels.append(f"{name}")  # Include iteration_k in the label

            # Visualize the graph properties and save the images
            #vi.visual_density_edge_number(density_values, labels, dataset_name, similarity_type, iteration_k, if_random = True)