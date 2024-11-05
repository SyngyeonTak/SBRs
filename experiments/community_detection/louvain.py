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
# parser.add_argument('--datasets', nargs='+', default= ['yoochoose1_64', 'yoochoose1_4', 'diginetica', 'Tmall', 'Nowplaying', 'Retailrocket'],
#                     help='List of dataset names')

parser.add_argument('--datasets', nargs='+', default= ['yoochoose1_64', 'diginetica'],
                    help='List of dataset names')

args = parser.parse_args()

for dataset_name in args.datasets:

    # Load dataset and build graph
    dataset_path = f'./datasets/{dataset_name}/all_train_seq'
    dataset = sg.load_dataset(dataset_path)
    G, len_zero_degree_node = sg.build_graph(dataset)

    print(len_zero_degree_node)

    # Compute the best partition using the Louvain algorithm
    partition = community_louvain.best_partition(G, weight='weight', resolution= 2)
    modularity = community_louvain.modularity(partition, G, weight='weight')
    
    with open(f'./experiments/community_detection/results/{dataset_name}_community_to_nodes.txt', 'wb') as file:
        pickle.dump(partition, file)

    # Create a new graph with each community as a node
    community_graph = nx.Graph()

    # Create a mapping from community ID to nodes
    community_to_nodes = {}
    for node, comm in partition.items():
        if comm not in community_to_nodes:
            community_to_nodes[comm] = []
        community_to_nodes[comm].append(node)

    for community in community_to_nodes:
        community_graph.add_node(community)

    # Add edges between community nodes if there are intra-community edges in the original graph
    for comm1, nodes1 in community_to_nodes.items():
        for comm2, nodes2 in community_to_nodes.items():
            if comm1 < comm2:  # Ensure each edge is checked only once
                # Check if there's at least one edge between any nodes of comm1 and comm2
                if any(G.has_edge(n1, n2) for n1 in nodes1 for n2 in nodes2):
                    community_graph.add_edge(comm1, comm2)
                    break  # Stop checking other edges between comm1 and comm2

    # Get connected components
    connected_components = list(nx.connected_components(community_graph))

    color_map = {}
    for component in connected_components:
        color = 'lightgreen'  # Color for connected components
        for node in component:
            color_map[node] = color

    # Draw the community graph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(community_graph, seed=42, k=0.3)  # Decrease 'k' to make the graph more compact

    nx.draw(
        community_graph,
        pos,
        with_labels=True,
        node_color=[color_map[node] for node in community_graph.nodes()],
        edge_color='gray',
        node_size=300,  # Adjust node size for visibility
        font_size=10,   # Adjust font size for labels
        font_color='black'
    )


    # Set title and display the plot
    plt.title('Community Graph with Intra-Community Edges')
    plt.savefig(f'./experiments/community_detection/results/{dataset_name}_communitygraph_resoultion_2.png')
    #plt.show()


    # Count the number of edges in the community graph
    num_edges = community_graph.number_of_edges()
    num_communities = community_graph.number_of_nodes()
    
    # Find isolated nodes
    isolated_nodes = list(nx.isolates(community_graph))
    num_isolated_nodes = len(isolated_nodes)

    # Find connected components
    connected_components = list(nx.connected_components(community_graph))
    num_connected_components = len(connected_components)

    # Find nodes in connected components
    connected_nodes = set()
    for component in connected_components:
        connected_nodes.update(component)

    num_connected_nodes = len(connected_nodes)

    with open(f'./experiments/community_detection/results/{dataset_name}_communitysummary_resolytion_2.txt', 'w') as file:
        file.write(f"removed nodes: {len_zero_degree_node}\n")
        file.write(f"Final optimized modularity (Q): {round(modularity, 3)}\n")
        file.write(f"Number of edges in the community graph: {num_edges}\n")
        file.write(f"Number of num_communities in the community graph: {num_communities}\n")
        file.write(f"Number of connected commuities: {num_connected_nodes}\n")
        file.write(f"Number of isolated commuities: {num_isolated_nodes}")