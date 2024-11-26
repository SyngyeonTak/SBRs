import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx
import torch_cluster
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm import tqdm
import json

gpu_index = 0
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
directory_path = 'experiments/length_aware_data_augmentation/results/similarity'

def get_undirected_graph(dataset):
    G = nx.Graph()  # Use an undirected graph
    unique_nodes = set()  # To store unique nodes

    for sublist in dataset:
        for i in range(len(sublist) - 1):
            source = sublist[i]
            target = sublist[i + 1]
            unique_nodes.add(source)
            unique_nodes.add(target)
    
            # Add an unweighted edge between source and target
            G.add_edge(source, target)

    # Remove self-loops if any
    self_loops = list(nx.selfloop_edges(G))
    if len(self_loops) > 0:
        G.remove_edges_from(self_loops)
        print(f"Removed {len(self_loops)} self-loops from the graph.")

    return G

def calculate_graph_properties(dataset):
    # Generate the undirected graph from the dataset
    G = get_undirected_graph(dataset)
    
    # Calculate density
    density = nx.density(G)
    
    # Get the number of edges
    num_edges = G.number_of_edges()
    
    #print(f"Graph density: {density}")
    #print(f"Number of edges: {num_edges}")
    
    return density, num_edges

def calculate_jaccard_similarity(G):
    """
    Calculate the Jaccard similarity for all pairs of nodes in an undirected graph G.
    
    Args:
    G (networkx.Graph): An undirected graph.
    
    Returns:
    np.ndarray: A square matrix of Jaccard similarities for all pairs of nodes.
    """
    # Initialize a matrix to store the Jaccard similarities
    num_nodes = len(G.nodes)
    jaccard_sim = np.zeros((num_nodes, num_nodes))
    
    # Get all pairs of nodes in the graph
    node_list = list(G.nodes)
    
    # Calculate Jaccard similarity for each pair of nodes
    for i in tqdm(range(num_nodes), desc="Calculating Jaccard Similarity", mininterval=10.0):
        for j in range(i, num_nodes):  # Only compute upper triangle to save time
            node_i, node_j = node_list[i], node_list[j]
            
            # Get the neighbor sets for nodes i and j
            neighbors_i = set(G.neighbors(node_i))
            neighbors_j = set(G.neighbors(node_j))
            
            # Calculate intersection and union
            intersection = len(neighbors_i.intersection(neighbors_j))
            union = len(neighbors_i.union(neighbors_j))
            
            # Compute the Jaccard similarity
            jaccard_sim[i, j] = intersection / union if union > 0 else 0
            jaccard_sim[j, i] = jaccard_sim[i, j]  # Symmetric matrix
    
    return jaccard_sim

# def calculate_jaccard_similarity(G):
#     """
#     Calculate the Jaccard similarity for all pairs of nodes in an undirected graph G using PyTorch on GPU.
    
#     Args:
#     G (networkx.Graph): An undirected graph.
    
#     Returns:
#     torch.Tensor: A square matrix of Jaccard similarities for all pairs of nodes.
#     """
#     # Check if CUDA is available and move tensor to GPU
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print(f"Current CUDA device: {torch.cuda.current_device()}")
#         print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2} MB")
#         print(f"Memory cached: {torch.cuda.memory_reserved() / 1024**2} MB")
#     else:
#         device = torch.device("cpu")
#         print("CUDA is not available. Using CPU.")

#     # Get the number of nodes in the graph
#     num_nodes = len(G.nodes)
    
#     # Initialize a sparse tensor to store Jaccard similarities (on the GPU if available)
#     jaccard_sim = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    
#     # Create a mapping from node to index
#     node_list = list(G.nodes)
#     node_to_index = {node: idx for idx, node in enumerate(node_list)}
    
#     # Create a sparse tensor of neighbor sets for each node
#     indices = []
#     values = []

#     for node in G.nodes:
#         node_idx = node_to_index[node]
#         neighbors = set(G.neighbors(node))
#         for neighbor in neighbors:
#             neighbor_idx = node_to_index[neighbor]
#             indices.append([node_idx, neighbor_idx])
#             values.append(1)

#     indices = torch.tensor(indices, dtype=torch.long, device=device).t()
#     values = torch.tensor(values, dtype=torch.float32, device=device)
    
#     neighbors_tensor = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device)

#     # Convert the sparse tensor to a dense tensor just before calculating intersections
#     dense_neighbors_tensor = neighbors_tensor.to_dense()

#     # Now calculate Jaccard similarity using the dense neighbor tensor on the GPU
#     for i in tqdm(range(num_nodes), desc="Calculating Jaccard Similarity (rows)", ncols=100):
#         for j in range(i, num_nodes):  # Only compute upper triangle to save time
#             # Convert to boolean before bitwise operations for intersection and union
#             i_neighbors = neighbors_tensor[i].to_dense().bool()  # Convert to boolean
#             j_neighbors = neighbors_tensor[j].to_dense().bool()  # Convert to boolean
            
#             # Perform intersection (AND) and union (OR) using bitwise operations
#             intersection = torch.sum(i_neighbors & j_neighbors)  # Bitwise AND for intersection
#             union = torch.sum(i_neighbors | j_neighbors)  # Bitwise OR for union

#             # Compute the Jaccard similarity
#             jaccard_sim[i, j] = intersection.float() / union.float() if union > 0 else 0
#             jaccard_sim[j, i] = jaccard_sim[i, j]  # Symmetric matrix
    
#     print(jaccard_sim)

#     return jaccard_sim




def get_node2vec_embeddings(G, gpu_index=0):
    # Convert NetworkX graph to PyTorch Geometric format
    pyg_graph = from_networkx(G)

    print(pyg_graph.edge_index)

    # Initialize Node2Vec model
    node2vec = Node2Vec(
        pyg_graph.edge_index,
        embedding_dim=100,
        walk_length=80,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1
    )

    # Move node2vec model to the specified device
    node2vec = node2vec.to(device)

    # Generate embeddings
    optimizer = torch.optim.Adam(node2vec.parameters(), lr=0.01)

    # Create a DataLoader from node2vec
    loader = node2vec.loader(batch_size=128, shuffle=True)

    # Training loop using the DataLoader
    for epoch in range(5):
        node2vec.train()
        total_loss = 0

        # Loop over loader without tqdm
        for batch_idx, (pos_rw, neg_rw) in enumerate(loader):
            pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Print progress every few batches, or adjust as needed
            if batch_idx % 10 == 0:  # Adjust the frequency here
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(loader)}, Loss: {loss.item()}")

        print(f'Epoch {epoch + 1}, Avg Loss: {total_loss / len(loader)}')

    # Get embeddings as vectors
    #print(f"Type of node2vec: {type(node2vec)}")
    #print(f"Type of pyg_graph.num_nodes: {type(pyg_graph.num_nodes)}")
    

    node_embeddings = node2vec(torch.arange(pyg_graph.num_nodes, device=device))
    return node_embeddings.cpu().detach().numpy()

"""
def calculate_cosine_similarity(embeddings):
    # Convert embeddings to a NumPy array if they are not already
    embeddings = np.array(embeddings)
    
    # Initialize the cosine similarity matrix
    num_embeddings = embeddings.shape[0]
    cosine_sim = np.zeros((num_embeddings, num_embeddings))
    
    # Calculate cosine similarity for each pair of embeddings
    for i in tqdm(range(num_embeddings), desc="Calculating Cosine Similarity"):
    #for i in range(num_embeddings):
        cosine_sim[i] = cosine_similarity(embeddings[i].reshape(1, -1), embeddings).flatten()
    
    return cosine_sim
"""

def calculate_cosine_similarity(embeddings, batch_size=1000):
    # Convert embeddings to a PyTorch tensor and move to GPU
    embeddings = torch.tensor(embeddings, device='cuda')
    
    # Normalize embeddings to compute cosine similarity via dot product
    embeddings = F.normalize(embeddings)
    
    # Calculate cosine similarity using matrix multiplication
    num_rows = embeddings.shape[0]

    # Initialize an empty tensor for the similarity matrix
    cosine_sim = torch.zeros((num_rows, num_rows), device='cuda')

    for i in tqdm(range(0, num_rows, batch_size), desc="Calculating Similarity"):
        batch_embeddings = embeddings[i : i + batch_size]
        cosine_sim[i : i + batch_size] = torch.mm(batch_embeddings, embeddings.T)
    
    return cosine_sim.cpu().numpy()  # Convert back to NumPy array if needed

def save_similarity_pairs(similarity_pairs, filename, dataset_name, similarity_pool, similarity_type):

    print(dataset_name)

    with open(f'{directory_path}/{similarity_type}/{dataset_name}_{similarity_pool}_{filename}', 'w') as f:
        json.dump(similarity_pairs, f, indent=4)

def get_top_similarity_pairs(cosine_sim, pool_size=3, G=None, filter_connected= False):
    similarity_pairs = {}
    similarity_type = 'entire'
    for i in range(len(cosine_sim)):
        # Get indices of the top `pool_size` similarities excluding itself
        if len(cosine_sim[i]) > pool_size:  # Check if there are enough nodes
            #similar_indices = np.argsort(cosine_sim[i])[-(pool_size + 1):-1]  # Exclude the node itself

            similar_indices = np.argsort(cosine_sim[i])[:-1]

            if filter_connected:
                similar_indices = [sim_idx for sim_idx in similar_indices if not G.has_edge(i + 1, int(sim_idx + 1))]
                similarity_type = 'unseen'

            similar_indices = similar_indices[-(pool_size):]

        similar_indices = [idx.item() if isinstance(idx, torch.Tensor) else idx for idx in similar_indices]

        similar_scores = cosine_sim[i][similar_indices]

        similarity_pairs[str(i + 1)] = [(int(sim_idx + 1), float(sim_score)) for sim_idx, sim_score in zip(similar_indices, similar_scores)]
    
    return similarity_pairs, similarity_type


def get_similarity(dataset, dataset_name, filter_connected, similarity_type):

    G = get_undirected_graph(dataset)

    if similarity_type == 'node2vec':
        node_embeddings = get_node2vec_embeddings(G)
        similarity_matrix = calculate_cosine_similarity(node_embeddings)

    if similarity_type == 'jaccard':    
        similarity_matrix = calculate_jaccard_similarity(G)
    
    similarity_pairs, similarity_pool = get_top_similarity_pairs(similarity_matrix, pool_size=3, G = G, filter_connected = filter_connected)

    save_similarity_pairs(similarity_pairs, 'similarity_pairs.txt', dataset_name, similarity_pool, similarity_type)

def load_similarity_pairs(file_path):
    # Load the similarity pairs from the JSON file
    with open(file_path, 'r') as f:
        similarity_pairs = json.load(f)
    return similarity_pairs

def separate_head_tail_by_pareto(node_degree_info):
    sorted_nodes = sorted(node_degree_info.items(), key=lambda x: x[1], reverse=True)
    top_20_percent_count = int(0.2 * len(sorted_nodes))
    
    head_items = [node for node, degree in sorted_nodes[:top_20_percent_count]]
    tail_items = [node for node, degree in sorted_nodes[top_20_percent_count:]]
    
    head_items_degree = [degree for node, degree in sorted_nodes[:top_20_percent_count]]
    tail_items_degree = [degree for node, degree in sorted_nodes[top_20_percent_count:]]

    return head_items, tail_items

def get_highest_similarity(similarity_pairs, nodes_list):
    highest_similarity = []  # List to store nodes with the highest similarity

    for node in nodes_list:
        # Ensure the node is in the similarity_pairs dictionary
        if str(node) in similarity_pairs:
            pairs = similarity_pairs[str(node)]
            if not pairs:
                # If empty, add a default similarity (node, 0) to indicate no similarity found
                highest_similarity.append(0)
                #highest_similarity.append((node, 0))
            else:
                # Find the pair with the highest similarity score
                max_pair = max(pairs, key=lambda x: x[1])
                max_pair_similarity = max_pair[1]
                highest_similarity.append(max_pair_similarity)
                #highest_similarity.append((node, max_pair[1]))

    #print("Before sorting:", highest_similarity)
    #highest_similarity = sorted(highest_similarity, key= lambda x : x[1], reverse=True)
    highest_similarity = sorted(highest_similarity, reverse=True)
    
    return highest_similarity

def get_similarity_ranking(dataset, dataset_name, simiarity_type):
    similarity_pairs = load_similarity_pairs(f'experiments/length_aware_data_augmentation/results/similarity/node2vec/{dataset_name}_similarity_pairs_{simiarity_type}.txt')
    G = get_undirected_graph(dataset)
    node_degree_info = dict(G.degree())
    head_list, tail_list = separate_head_tail_by_pareto(node_degree_info)

    head_highest_similarity = get_highest_similarity(similarity_pairs, head_list)
    tail_highest_similarity = get_highest_similarity(similarity_pairs, tail_list)

    #print(head_highest_similarity[-30:])
    #print(tail_highest_similarity[-30:])

    return head_highest_similarity, tail_highest_similarity

def find_hop_relationships(G, similarity_pairs):
    hop_relationships = {}

    for key, similar_items in similarity_pairs.items():
        key = int(key)  # Convert key to integer if needed
        hop_relationships[key] = []

        for item_id, similarity in similar_items:
            # Find shortest path length (hop distance) between key and item_id
            if G.has_node(key) and G.has_node(item_id):
                try:
                    hop_distance = nx.shortest_path_length(G, source=key, target=item_id)
                except nx.NetworkXNoPath:
                    hop_distance = None  # No path exists
            else:
                hop_distance = None  # Node not in graph

            hop_relationships[key].append((item_id, similarity, hop_distance))

    return hop_relationships

def clean_similarity_pool(similarity_pairs, dataset_name, similarity_type, similarity_pool):
    # Iterate over each key-value pair in the dictionary
    cleaned_pairs = {
        key: [item for item in value if item[1] != 0.0]
        for key, value in similarity_pairs.items()
    }

    save_similarity_pairs(cleaned_pairs, 'similarity_pairs.txt', dataset_name, similarity_pool, similarity_type)

    return cleaned_pairs