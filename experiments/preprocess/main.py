import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shuffle as sf
import random
import pickle
import substitute as st
from tqdm import tqdm

def load_dataset(dataset_name):
    with open(f'{dataset_name}.txt', 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def load_communties(dataset_name):
    with open(f'./experiments/community_detection/results/{dataset_name}_community_to_nodes.txt', 'rb') as file:
        community_to_nodes = pickle.load(file)

    return community_to_nodes

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


def make_slide_shuffle_dataset(dataset, dataset_name, keep_first = False, keep_last = False):
    slide_shuffled_dataset = sf.slide_shuffle(dataset, keep_first = keep_first, keep_last = keep_last)
    print('length slide shuffled_dataset: ', len(slide_shuffled_dataset))
    

    with open(f'./datasets/{dataset_name}/shuffle_slide_train.txt', 'wb') as file:
        pickle.dump(slide_shuffled_dataset, file)

def make_community_substitute_dataset(dataset, dataset_name):
    partitions = load_communties(dataset_name)

    #st.validate_community_partitions(partitions)

    #st.get_item_community(partitions, 366)
    #st.get_items_for_community(partitions, 0)

    #print(dataset)
    print(len(dataset))

    #fixed_number_community_augmented = st.augment_sessions_fixed_number_community(dataset, partitions)
    #length_aware_community_augmented = st.augment_sessions_length_aware_community(dataset, partitions)
    length_aware_random_augmented = st.augment_sessions_length_aware_random(dataset)

    #print(fixed_number_community_augmented)
    #print(len(fixed_number_community_augmented))

    #print(length_aware_community_augmented)
    #print(len(length_aware_community_augmented))

    #print(length_aware_random_augmented)
    #print(len(length_aware_random_augmented))

    # with open(f'./datasets/{dataset_name}/fixed_community_train.txt', 'wb') as file:
    #     pickle.dump(fixed_number_community_augmented, file)

    # with open(f'./datasets/{dataset_name}/length_community_train.txt', 'wb') as file:
    #     pickle.dump(length_aware_community_augmented, file)

    with open(f'./datasets/{dataset_name}/length_random_train.txt', 'wb') as file:
        pickle.dump(length_aware_random_augmented, file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Tmall',
                        help='Dataset name: yoochoose1_64, yoochoose1_4, diginetica, Tmall, Nowplaying, Retailrocket')
    # parser.add_argument('--datasets', nargs='+', default= ['yoochoose1_64', 'yoochoose1_4', 'diginetica', 'Tmall', 'Nowplaying', 'Retailrocket'],
    #                    help='List of dataset names')
    parser.add_argument('--datasets', nargs='+', default= [ 'diginetica', 'Tmall', 'Retailrocket'],
                       help='List of dataset names')

    # parser.add_argument('--datasets', nargs='+', default= ['Tmall'],
    #                     help='List of dataset names')
    parser.add_argument('--length_type', nargs='+', default= ['short', 'medium', 'long', 'longt','all'],
                        help='List of length type')

    args = parser.parse_args()
    # ---------------------------------------------------------------------------
    
    for dataset_name in tqdm(args.datasets, 'datasets'):
        print('---------------------------------------------------')
        print(f'Processing dataset: {dataset_name}')

        #-------------------------------------------------------------
        dataset = load_dataset(f'./datasets/{dataset_name}/all_train_seq')
        # make_slide_shuffle_dataset(dataset, dataset_name)
        # make_community_substitute_dataset(dataset, dataset_name)


        


if __name__ == "__main__":
    main()