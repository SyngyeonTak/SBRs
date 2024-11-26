import argparse
import pandas as pd
import numpy as np
import util
import bulk_util as bulk
import torch
import torch_geometric
import similarity as sim
import json
def main():

    
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='diginetica',
                            help='Dataset name: yoochoose1_64, yoochoose1_4, diginetica, Tmall, Nowplaying, Retailrocket')
        #parser.add_argument('--datasets', nargs='+', default= ['yoochoose1_64', 'yoochoose1_4', 'diginetica', 'Tmall', 'Nowplaying', 'Retailrocket'],
        #                   help='List of dataset names')
        #parser.add_argument('--datasets', nargs='+', default= ['diginetica', 'Tmall', 'Nowplaying', 'Retailrocket'],
        #                   help='List of dataset names')
        parser.add_argument('--datasets', nargs='+', default= ['diginetica', 'Tmall'],
                            help='List of dataset names')

        parser.add_argument('--original_percents', nargs='+', default= [0.1],
                            help='List of original_percent')
        
        parser.add_argument('--filter_connected_list', nargs='+', default= [True, False],
                            help='filter_connected')
        
        parser.add_argument('--similarity_type_list', nargs='+', default= ['node2vec', 'jaccard'],
                            help='similarity_type')
        
        parser.add_argument('--similarity_pool_list', nargs='+', default= ['unseen', 'entire'],
                            help='similarity_pool')
        
        parser.add_argument('--augmented_type_list', nargs='+', default= ['insert', 'substitute'],
                            help='augmented_type')
        
        parser.add_argument('--if_random_list', nargs='+', default= [False],
                            help='if randomly extract sessions for augmentation')



        args = parser.parse_args()

        #############################################
        #bulk.get_bulk_statistics(args)
        #bulk.get_bulk_visualization(args)
        #util.get_label_frequencies(args)

        #############################################
        #bulk.get_bulk_similarity(args) 
        #bulk.get_bulk_similarity_based_augment(args)
        #bulk.get_bulk_similarity_based_augment_random(args)
        #util.get_solo_similarity_based_augment(args)
        #bulk.get_bulk_similarity_ranking(args)

        #############################################
        #util.get_original_learnable_dataset(args)

        #bulk.get_bulk_augmented_dataset(args)
        
        #############################################
        #bulk.save_bulk_save_hop_relationships(args)
        #util.get_direct_item_list('diginetica', 13)

        #############################################
        #bulk.clean_bulk_similarity_pool(args)
        original_dataset = util.get_dataset(args.dataset, length_type = 'all', if_augmented=False)
        prefix_dataset = util.get_dataset(args.dataset, length_type = 'all', if_augmented=True)
        
        augmented_dataset = util.prefix_cropping(original_dataset)


        print(len(original_dataset))
        print(len(prefix_dataset))
        print(len(prefix_dataset) - len(original_dataset))
        print(len(augmented_dataset))



    #################################
    
    

if __name__ == "__main__":
    main()

