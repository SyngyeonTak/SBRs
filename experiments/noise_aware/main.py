import argparse
import get_statistics as gs
import nodecentrality as nc
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Tmall',
                        help='Dataset name: yoochoose1_64, yoochoose1_4, diginetica, Tmall, Nowplaying, Retailrocket')
    #parser.add_argument('--datasets', nargs='+', default= ['yoochoose1_64', 'yoochoose1_4', 'diginetica', 'Tmall', 'Nowplaying', 'Retailrocket'],
    #                    help='List of dataset names')
    parser.add_argument('--datasets', nargs='+', default= ['Tmall'],
                        help='List of dataset names')

    parser.add_argument('--original_percents', nargs='+', default= [0.1],
                        help='List of original_percent')
    parser.add_argument('--node_centralities', nargs='+', default= ['weighted_edge', 'degree', 'coreness', 'eigenvector_centrality', 'pagerank'],
                        help='List of node centrality')
    parser.add_argument('--taget_node_centralities', nargs='+', default= ['weighted_edge', 'eigenvector_centrality', 'pagerank'],
                        help='List of target node centrality')

    args = parser.parse_args()
    # ---------------------------------------------------------------------------
    # in the scope of mismatch_rate
    #gs.do_excursive_rate(args)      
    # ---------------------------------------------------------------------------
    # the short, medium, long session indexing
    #gs.do_split_dataset(args)
    #gs.do_make_long_dataset(args)
    # ---------------------------------------------------------------------------
    #gs.do_last_item_ranking(args)
    # ---------------------------------------------------------------------------
    gs.do_basic_statistics(args)

    # ---------------------------------------------------------------------------

    # df = pd.read_csv('session_statiscitcs.csv', encoding= "utf-8")

    # #print(df[df['dataset_name'] == 'diginetica'])

    # # Define custom sort order
    # dataset_order = {'diginetica': 0, 'Tmall': 1, 'Nowplaying': 2, 'Retailrocket': 3, 'yoochoose1_4': 4, 'yoochoose1_64': 5}
    # type_order = {'original': 0, 'augmented': 1}
    # length_order = {'all': 0, 'short': 1, 'medium': 2, 'long': 3}

    # # Apply custom sort order to dataset_type
    # df['dataset_order'] = df['dataset_name'].map(dataset_order)
    # df['dataset_type_order'] = df['dataset_type'].map(type_order)
    # df['length_type_order'] = df['length_type'].map(length_order)

    # df = df.sort_values(by=['dataset_order', 'length_type_order', 'dataset_type_order'])

    # # Drop the auxiliary columns used for sorting if not needed anymore
    # df = df.drop(columns=['dataset_order', 'dataset_type_order', 'length_type_order'])

    # # Print the sorted DataFrame
    # #print(df)

    # ## sort by order
    
    # stacked_df = df.set_index(['dataset_name', 'length_type', 'dataset_type']).stack().unstack(level = [0, 1, 2])

    # print(stacked_df)

    # stacked_df.to_csv('stacked_session_statistics.csv', encoding= 'utf-8')

    # ---------------------------------------------------------------------------

    

if __name__ == "__main__":
    main()