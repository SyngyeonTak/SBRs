import matplotlib.pyplot as plt
import networkx as nx
import hypernetx as hnx
from tqdm import tqdm
import get_statistics as stat
import nodecentrality as nc
import numpy as np
from collections import Counter

num_centrality = 3

def visualize_item_click_counts(items, counts):
    plt.figure(figsize=(15, 6))
    plt.bar(items, counts)
    plt.title('Item Click Counts')
    plt.xlabel('Item ID')
    plt.ylabel('Click Count')
    plt.xticks(rotation=90)
    plt.show()

# # Function to visualize item click distribution
def visualize_item_click_distribution(items, counts):
    plt.figure(figsize=(15, 6))
    plt.plot(range(len(items)), counts, alpha=0.6)
    plt.title('Distribution of Item Click Frequencies')
    plt.xlabel('Item ID (Index)')
    plt.ylabel('Click Frequency')
    plt.show()

def visualize_graph(G):
    # Compute layout positions with progress bar
    print("Computing layout positions...")
    with tqdm(total=len(G.nodes())) as pbar:
        pos = nx.spring_layout(G)
        pbar.update(len(G.nodes()))

    # Compute edge weights with progress bar
    print("Computing edge weights...")
    with tqdm(total=len(G.edges())) as pbar:
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        pbar.update(len(G.edges()))

    # Draw the graph
    nx.draw(G, pos, edges=G.edges(), node_size=100, width=weights, edge_color=weights, edge_cmap=plt.cm.Blues, with_labels=True)
    plt.show()

def visualize_hypergraph(data):
    H = hnx.Hypergraph(data)
    hnx.draw(H)
    plt.show()

def plot_distributions(G):
    # Calculate node degrees
    node_degrees = nc.cal_degree(G)
    degree_values = list(node_degrees.values())

    # Calculate node coreness
    kcore = nc.cal_kcore(G)
    coreness_values = list(kcore.values())

    # Extract the largest eigenvector centrality values
    eigenvector_large_centrality = nc.calculate_largest_eigenvector_centrality(G)
    eigenvector_large_values = list(eigenvector_large_centrality.values())

    # Extract the smallest eigenvector centrality values
    #eigenvector_small_centrality = nc.calculate_smallest_eigenvector_centrality(G)
    #eigenvector_small_values = list(eigenvector_small_centrality.values())

    pagerank = nc.calculate_pagerank(G)
    pagerank_values = list(pagerank.values())

    

    # Plot degree distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, num_centrality, 1)
    plt.hist(degree_values, bins=20, color='skyblue', edgecolor='black')
    plt.title('Degree')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')

    # Plot coreness distribution
    plt.subplot(1, num_centrality, 2)
    plt.hist(coreness_values, bins=20, color='salmon', edgecolor='black')
    plt.title('Coreness')
    plt.xlabel('Coreness')
    plt.ylabel('Frequency')

    
    # Plot the largest eigenvector centrality distribution
    #plt.subplot(1, num_centrality, 3)
    #plt.hist(eigenvector_large_values, bins=20, color='lightgreen', edgecolor='black')
    #plt.title('The Largest EC')
    #plt.xlabel('Eigenvector Centrality')
    #plt.ylabel('Frequency')

    # Plot the smallest eigenvector centrality distribution
    #plt.subplot(1, num_centrality, 4)
    #plt.hist(eigenvector_small_values, bins=20, color='lightpink', edgecolor='black')
    #plt.title('The Smallest EC')
    #plt.xlabel('Eigenvector Centrality')
    #plt.ylabel('Frequency')

     # Plot the Pagerank distribution
    #plt.subplot(1, num_centrality, 3)
    #plt.hist(pagerank_values, bins=20, color='dimgray', edgecolor='black')
    #plt.title('Pagerank')
    #plt.xlabel('Pagerank')
    #plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_session_groups(group1_count, group2_count, group3_count, session_lengths, dataset):
    
    total_sessions = len(session_lengths)
    group1_percentage = (group1_count / total_sessions)
    group2_percentage = (group2_count / total_sessions)
    group3_percentage = (group3_count / total_sessions)

    plt.figure(figsize=(8, 4))

    # Plot counts of session groups
    bars = plt.bar(
        ['Long', 'Medium', 'Short'], 
        [group1_percentage, group2_percentage, group3_percentage], 
        color=['coral', 'lightblue', 'lightgrey']
    )
    
    # Adding the legend by associating each bar with a label
    bars[0].set_label('Long: 9 ~')
    bars[1].set_label('Medium: 5 ~ 8')
    bars[2].set_label('Short: ~ 4')

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    plt.ylabel('Percentage of Sessions')
    plt.title(f'in {dataset}')
    
    # Show the legend
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_distances(distances_high_degree, distances_high_coreness):
    # Extract distances from tuples

    distances_high_degree_values = [distance for distance, _ in distances_high_degree]
    distances_high_coreness_values = [distance for distance, _ in distances_high_coreness]
    
    # Convert to numpy arrays for consistency
    distances_high_degree_values = np.array(distances_high_degree_values)
    distances_high_coreness_values = np.array(distances_high_coreness_values)
    
    # Plot the histograms
    plt.figure(figsize=(14, 7))
    
    # High-Degree Nodes
    plt.subplot(1, 2, 1)
    plt.hist(distances_high_degree_values, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distances of High-Degree Nodes from Last Node')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    
    # High-Coreness Nodes
    plt.subplot(1, 2, 2)
    plt.hist(distances_high_coreness_values, bins=20, color='lightgreen', edgecolor='black')
    plt.title('Distances of High-Coreness Nodes from Last Node')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def plot_value_frequencies(distances_high_degree, distances_high_coreness, total_sessions, degree_stats, kcore_stats):
    """
    This function takes two datasets of dictionaries, counts the occurrences of each position in the dictionaries,
    and plots histograms for both datasets side-by-side for comparison. It colors bars differently for position 0,
    and annotates the total number of sessions and other statistics on the plots.
    """
    # Extract the positions and counts from the dictionaries
    degree_positions = list(distances_high_degree.keys())
    degree_frequencies = list(distances_high_degree.values())
    
    coreness_positions = list(distances_high_coreness.keys())
    coreness_frequencies = list(distances_high_coreness.values())
    
    # Color customization: specify colors for position 0
    degree_colors = ['red' if position == 0 else 'skyblue' for position in degree_positions]
    coreness_colors = ['red' if position == 0 else 'lightgreen' for position in coreness_positions]
    
    # Create a figure and axes for side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Plot histogram for degree distances
    bars_degree = axs[0].bar(degree_positions, degree_frequencies, color=degree_colors, edgecolor='black')
    axs[0].set_xlabel('Position')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Frequency of Degree Distances')
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)

    axs[0].set_xlim(-1, 10)

    # Annotate the total number of sessions and stats
    axs[0].text(0.95, 0.95, f'Total Sessions: {total_sessions}\n'
                            f"Max: {degree_stats['max']}\n"
                            f"Mean: {degree_stats['mean']}\n"
                            f"Median: {degree_stats['median']}\n"
                            f"Min: {degree_stats['min']}\n"
                            f"Top 10%: {degree_stats['threshold']}",
                horizontalalignment='right', verticalalignment='top',
                transform=axs[0].transAxes, fontsize=10,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))



    # Plot histogram for coreness distances
    bars_coreness = axs[1].bar(coreness_positions, coreness_frequencies, color=coreness_colors, edgecolor='black')
    axs[1].set_xlabel('Position')
    axs[1].set_title('Frequency of Coreness Distances')
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    axs[1].set_xlim(-1, 10)

    # Annotate the total number of sessions and stats
    axs[1].text(0.95, 0.95, f'Total Sessions: {total_sessions}\n'
                             f"Max: {kcore_stats['max']}\n"
                             f"Mean: {kcore_stats['mean']}\n"
                             f"Median: {kcore_stats['median']}\n"
                             f"Min: {kcore_stats['min']}\n"
                             f"Top 10%: {kcore_stats['threshold']}",
                horizontalalignment='right', verticalalignment='top',
                transform=axs[1].transAxes, fontsize=10,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def analyze_and_plot_last_item_rankings(dataset, title, ax, color):
    """
    Compute the ranking of the last item in each session and plot the ranking frequencies.
    
    Args:
        dataset (list of lists): A list of sessions where each session is a list of tuples (item, degree, rank).
        title (str): Title of the plot.
        ax (matplotlib.axes.Axes): The axes on which to plot.
    """
    # Calculate the rankings of the last item in each session
    last_item_rankings = []

    for session in dataset:
        # Get the last item in the session and its rank
        last_item = session[-1]
        last_item_rank = last_item[-1]
        
        # Append the last item's rank to the list
        last_item_rankings.append(last_item_rank)


    # Count the frequency of each ranking
    ranking_counts = Counter(last_item_rankings)
    
    # Separate keys and values for plotting
    rankings = list(ranking_counts.keys())
    frequencies = list(ranking_counts.values())
    
    print('title: ', title)
    print('ranking_counts: ', ranking_counts)

    # Create a bar plot on the specified axes
    ax.bar(rankings, frequencies, color= color)
    
    # Add labels and title
    ax.set_xlabel('Ranking')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.set_xticks(rankings)  # Ensure all rankings are shown on the x-axis
    #ax.set_xticks(range(1, 20))  # Show ticks from 1 to 19
    ax.set_xlim(0, 19)  # Set x-axis limits to 0 and 19



def plot_mismatch_rate(exclusive_rates):
    for original_percent, dataset_mismatch_rates in exclusive_rates.items():
        datasets = list(dataset_mismatch_rates.keys())
        data_types = list(dataset_mismatch_rates[datasets[0]].keys())
        centrality_types = list(dataset_mismatch_rates[datasets[0]][data_types[0]].keys())
        
        for rate_type in centrality_types:
            # Define bar width and positions
            bar_width = 0.2
            num_data_types = len(data_types)
            indices = np.arange(len(datasets))
            
            plt.figure(figsize=(14, 8))
            
            for i, data_type in enumerate(data_types):
                rates = [dataset_mismatch_rates[dataset][data_type].get(rate_type, 0) for dataset in datasets]
                plt.bar(indices + i * bar_width, rates, bar_width, label=data_type)
            
            plt.xlabel('Datasets')
            plt.ylabel(f'{rate_type.replace("_", " ").title()} Rate')
            plt.title(f'{rate_type.replace("_", " ").title()} Rates for Original Percent {original_percent}')
            plt.xticks(indices + (num_data_types - 1) * bar_width / 2, datasets, rotation=45, ha='right')
            plt.legend()
            
            # Save plot to file
            plt.tight_layout()
            plt.savefig(f'Exclusive_rate_{rate_type}.png')
            plt.close()