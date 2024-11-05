import argparse
import pandas as pd
import numpy as np
import util



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Tmall',
                        help='Dataset name: yoochoose1_64, yoochoose1_4, diginetica, Tmall, Nowplaying, Retailrocket')
    #parser.add_argument('--datasets', nargs='+', default= ['yoochoose1_64', 'yoochoose1_4', 'diginetica', 'Tmall', 'Nowplaying', 'Retailrocket'],
    #                   help='List of dataset names')
    parser.add_argument('--datasets', nargs='+', default= ['Tmall'],
                        help='List of dataset names')

    parser.add_argument('--original_percents', nargs='+', default= [0.1],
                        help='List of original_percent')

    args = parser.parse_args()

    #util.get_bulk_statistics(args)
    #util.get_bulk_visualization(args)
    #util.get_label_frequencies(args)
    util.get_bulk_similarity(args)

    #################################
    
    

if __name__ == "__main__":
    main()

