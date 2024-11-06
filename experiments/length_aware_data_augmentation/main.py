import argparse
import pandas as pd
import numpy as np
import util
import bulk_util as bulk
import torch
#import torch_geometric

filter_connected_list = [True, False]

def main():

    
        parser = argparse.ArgumentParser()
        #parser.add_argument('--dataset', type=str, default='Tmall',
        #                    help='Dataset name: yoochoose1_64, yoochoose1_4, diginetica, Tmall, Nowplaying, Retailrocket')
        #parser.add_argument('--datasets', nargs='+', default= ['yoochoose1_64', 'yoochoose1_4', 'diginetica', 'Tmall', 'Nowplaying', 'Retailrocket'],
        #                   help='List of dataset names')
        #parser.add_argument('--datasets', nargs='+', default= ['diginetica', 'Tmall', 'Nowplaying', 'Retailrocket'],
        #                   help='List of dataset names')
        parser.add_argument('--datasets', nargs='+', default= ['diginetica', 'Tmall'],
                            help='List of dataset names')

        parser.add_argument('--original_percents', nargs='+', default= [0.1],
                            help='List of original_percent')
        parser.add_argument('--filter_connected_list', nargs='+', default= [False],
                            help='filter_connected')


        args = parser.parse_args()

        #############################################
        #bulk.get_bulk_statistics(args)
        #bulk.get_bulk_visualization(args)
        #util.get_label_frequencies(args)

        #############################################
        #bulk.get_bulk_similarity(args) 
        #bulk.get_bulk_similarity_based_augment(args)
        util.get_solo_similarity_based_augment(args)
        #bulk.get_bulk_similarity_ranking(args)

    #################################
    
    

if __name__ == "__main__":
    main()

