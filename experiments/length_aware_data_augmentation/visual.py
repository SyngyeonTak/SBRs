import matplotlib.pyplot as plt
import numpy as np

def visualize_occurrence(occurrences):
    # Sort the occurrences by count in descending order
    sorted_occurrences = sorted(occurrences.items(), key=lambda x: x[1], reverse=True)
    
    # Unpack the sorted occurrences into items and counts
    items, counts = zip(*sorted_occurrences)

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    #plt.bar(items, counts, color='skyblue')
    plt.bar(range(len(counts)), counts, color='skyblue')  # X-axis is now the index (rank)
    
    
    # Add labels and title
    plt.xlabel('Last Item')
    plt.ylabel('Occurrences')
    plt.title('Occurrences of Last Items in Dataset (Sorted by Count)')

    # Rotate x-axis labels for readability if there are many items
    plt.xlim(0, 1000)
    plt.xticks([])

    # Display the plot
    plt.show()

def visualize_bin_occurrence(occurrences, bin_size=5):
    # Create bins
    max_count = max(occurrences.values())
    bins = range(0, max_count + bin_size, bin_size)
    #bin_labels = [f"{b+1}-{b+bin_size}" for b in bins[:-1]]  # Create labels for the bins
    bin_labels = [f"{i+1}" for i in range(len(bins) - 1)]  # Create labels like "Bin 1", "Bin 2", etc.
    bin_counts = [0] * (len(bins) - 1)  # Initialize counts for each bin

    # Count occurrences in each bin
    for count in occurrences.values():
        for i in range(len(bins) - 1):
            if bins[i] < count <= bins[i + 1]:
                bin_counts[i] += 1
                break

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_labels, bin_counts, color='skyblue')

    # Add labels and title
    plt.xlabel('Occurrences')
    plt.xticks([])
    plt.ylabel('Number of IDs')
    plt.title('Number of IDs by Occurrence')

    # Display the plot
    plt.tight_layout()
    plt.show()

def visualize_dataset_label_ratio(training_ratios, augmented_ratios, swapped_ratios, dataset_names):
    x = np.arange(len(dataset_names))  # the label locations
    width = 0.30  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, training_ratios, width, label='Training', color='skyblue')
    rects2 = ax.bar(x, augmented_ratios, width, label='Prefix Aug', color='lightgreen')
    rects3 = ax.bar(x + width, swapped_ratios, width, label='Swapped Aug', color='pink')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Dataset Names')
    ax.set_ylabel('Ratio (All Occurrences / Label Occurrences)')
    ax.set_title('Ratio of All Occurrences to Label Occurrences by Dataset and Type')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.legend()

    fig.tight_layout()

    plt.savefig(f'experiments/length_aware_data_augmentation/images/visualize_dataset_label_ratio.png', format = 'png', dpi = 700)

    plt.show()

def visual_frequency_distribution(dataset_names, all_frequencies, bin_size=5, max_bin=60):
    distribution_labels = ['Training', 'Augmented', 'Swapped']
    colors = ['skyblue', 'lightgreen', 'salmon']  # Define colors for each distribution
    num_distributions = len(all_frequencies)
    
    for j, dataset_name in enumerate(dataset_names):
        fig, axes = plt.subplots(1, num_distributions, figsize=(15, 6), sharey=True)
        fig.suptitle(f'Frequency Distribution for {dataset_name}', fontsize=16)

        for i, (freqs, label) in enumerate(zip(all_frequencies, distribution_labels)):
            ax = axes[i] if num_distributions > 1 else axes  # Adjust for single subplot case

            # Get sorted frequencies for the current dataset and distribution type
            sorted_freqs = sorted(freqs[j].items(), key=lambda x: x[1], reverse=True)

            # Unzip the sorted items
            #items, counts = zip(*sorted_freqs) if sorted_freqs else ([], [])
            items, counts = zip(*[(item, count) for item, count in sorted_freqs if count > 0]) if sorted_freqs else ([], [])

            # Cap counts at max_bin for the histogram
            capped_counts = [min(count, max_bin) for count in counts]

            # Create bins up to the specified maximum value
            bins = np.arange(0, max_bin + bin_size, bin_size)

            # Create a histogram of the frequencies
            counts, bin_edges = np.histogram(capped_counts, bins=bins)

            # Use the right edges of bins for x-axis
            bin_right_edges = bin_edges[1:]

            # Plot the histogram with a specific color
            ax.bar(bin_right_edges, counts, width=bin_size, label=label, alpha=0.7, color=colors[i])

            # Set labels and title for each subplot
            ax.set_xlabel('Frequency Bins')
            ax.set_ylabel('Number of Items in Each Bin')
            ax.set_title(f'{label} Distribution')
            ax.set_xticks(bin_right_edges)
            ax.legend()

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout with space for the title

        plt.savefig(f'experiments/length_aware_data_augmentation/images/visual_frequency_distribution_{dataset_name}.png', format = 'png', dpi = 700)
        #plt.show()

def visual_density_edge_number(density_values, edge_counts, labels, dataset_name, iteration_k, similarity_type = 'entire'):
    # Plotting graph density
    plt.figure(figsize=(10, 5))
    plt.bar(labels, density_values, color='skyblue')
    plt.xlabel('Graphs')
    plt.ylabel('Density')
    plt.title(f'{dataset_name} Graph Density Comparison (Iteration {iteration_k})')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'experiments/length_aware_data_augmentation/images/similarity/node2vec/visual_density_values_{dataset_name}_k{iteration_k}_{similarity_type}.png', format='png', dpi=700)
    plt.close()

    # Plotting edge count
    plt.figure(figsize=(10, 5))
    plt.bar(labels, edge_counts, color='salmon')
    plt.xlabel('Graphs')
    plt.ylabel('Number of Edges')
    plt.title(f'{dataset_name} Edge Count Comparison (Iteration {iteration_k})')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'experiments/length_aware_data_augmentation/images/similarity/node2vec/visual_edge_number_{dataset_name}_k{iteration_k}_{similarity_type}.png', format='png', dpi=700)
    plt.close()

def visual_similarity_ranking_plot(params):
    # Unpack the dictionary
    highest_similarity = params['highest_similarity']
    dataset_name = params['dataset_name']
    similarity_type = params['similarity_type']
    item_type = params['item_type']

    # Choose color based on item_type
    color = 'skyblue' if item_type == 'head' else 'lightcoral'  # You can choose your own colors
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.hist(highest_similarity, bins=10, edgecolor='black', color=color)
    plt.title(f"Histogram of Similarity Scores for {item_type.capitalize()} Items")
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    
    # Save the plot with a dynamic filename
    plt.savefig(f'experiments/length_aware_data_augmentation/images/similarity/node2vec/ranking/visual_similarity_ranking_{item_type}_{similarity_type}_{dataset_name}.png', format='png', dpi=700)
    plt.close()