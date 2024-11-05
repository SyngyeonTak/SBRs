import networkx as nx
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain
import sessiongraph as sg
import numpy as np
from collections import Counter
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Tmall',
                    help='Dataset name: yoochoose1_64, yoochoose1_4, diginetica, Tmall, Nowplaying, Retailrocket')
parser.add_argument('--datasets', nargs='+', default= ['yoochoose1_64', 'yoochoose1_4', 'diginetica', 'Tmall', 'Nowplaying', 'Retailrocket'],
                    help='List of dataset names')

# parser.add_argument('--datasets', nargs='+', default= ['Tmall'],
#                     help='List of dataset names')

args = parser.parse_args()

for dataset_name in args.datasets:
    with open(f'./experiments/community_detection/results/{dataset_name}_community_to_nodes.txt', 'rb') as file:
        partition = pickle.load(file)

    # Count the number of nodes in each community
    community_sizes = Counter(partition.values())

    # Extract basic statistics
    min_membership = min(community_sizes.values())
    max_membership = max(community_sizes.values())
    avg_membership = sum(community_sizes.values()) / len(community_sizes)

    print(f"Dataset: {dataset_name}")
    print(f"Minimum community size: {min_membership}")
    print(f"Maximum community size: {max_membership}")
    print(f"Average community size: {avg_membership:.2f}")

    # Plot the barplot
    sorted_community_sizes = dict(sorted(community_sizes.items(), key=lambda item: item[1], reverse=True))

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_community_sizes)), list(sorted_community_sizes.values()), color='skyblue')
    plt.xlabel('Community ID')
    plt.ylabel('Number of Members')
    plt.title(f'Community Sizes for {dataset_name}')
    #plt.xticks(range(len(sorted_community_sizes)), sorted_community_sizes.keys(), rotation=90)
    plt.tight_layout()

    # Save and/or show the plot
    plt.savefig(f'./experiments/community_detection/results/{dataset_name}_community_summary_image.png')
    #plt.show()