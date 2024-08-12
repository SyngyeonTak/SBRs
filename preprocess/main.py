import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shuffle as sf
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Tmall',
                        help='Dataset name: yoochoose1_64, yoochoose1_4, diginetica, Tmall, Nowplaying, Retailrocket')
    #parser.add_argument('--datasets', nargs='+', default= ['yoochoose1_64', 'yoochoose1_4', 'diginetica', 'Tmall', 'Nowplaying', 'Retailrocket'],
    #                    help='List of dataset names')
    parser.add_argument('--datasets', nargs='+', default= ['Tmall'],
                        help='List of dataset names')
    parser.add_argument('--length_type', nargs='+', default= ['short', 'medium', 'long', 'longt','all'],
                        help='List of length type')

    args = parser.parse_args()
    # ---------------------------------------------------------------------------
    
    for dataset_name in args.datasets:
        print('---------------------------------------------------')
        print(f'Processing dataset: {dataset_name}')

        #-------------------------------------------------------------
        dataset = sf.load_dataset(f'./datasets/{dataset_name}/all_train_seq')
        print('dataset[0]: ', dataset[0:5])
        augmented_dataset = sf.load_dataset(f'./datasets/{dataset_name}/train')
        #augmented_dataset = sf.flatten_dataset(augmented_dataset)
        #filtered_dataset = [item for item in dataset if len(item) > 6]
        #test_dataset = filtered_dataset[:1]
        #print('length dataset: ', len(dataset))
        #print('length dataset: ', len(augmented_dataset))
        dataset = sf.unflatten_dataset(dataset)
        print('dataset[0][0]: ', dataset[0][:5])
        print('dataset[1][0]: ', dataset[1][:5])
        print('type(dataset): ', type(dataset))

        print('augmented_dataset[0][0]: ', augmented_dataset[0][:5])
        print('augmented_dataset[1][0]: ', augmented_dataset[1][:5])
        print('type(augmented_dataset): ', type(augmented_dataset))

        #full_shuffled_dataset = sf.full_shuffle(dataset, keep_first = True, keep_last = True)
        #print('length full_shuffled_dataset: ', len(full_shuffled_dataset))
        #slide_shuffled_dataset = sf.slide_shuffle(dataset, keep_first = True, keep_last = True)
        #print('length slide shuffled_dataset: ', len(slide_shuffled_dataset))
        
if __name__ == "__main__":
    main()