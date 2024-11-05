import pickle
import networkx as nx

def load_dataset(dataset_name):
    with open(f'{dataset_name}.txt', 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def build_graph(dataset):
    G = nx.Graph()
    num_sessions = 0
    num_edges = 0
    unique_nodes = set()  # To store unique nodes

    for sublist in dataset:
        num_sessions += 1
        
        for i in range(len(sublist) - 1):
            num_edges += 1
            source = sublist[i]
            target = sublist[i+1]
            unique_nodes.add(source)
            unique_nodes.add(target)
    
            if not G.has_edge(source, target):
                # Add a new edge with weight 1
                G.add_edge(source, target, weight=1)

    numnodes = len(unique_nodes)  # Number of unique nodes

    self_loops = list(nx.selfloop_edges(G))
    if len(self_loops) > 0:
        G.remove_edges_from(self_loops)
        print(f"Removed {len(self_loops)} self-loops from the graph.")
    

    #print(f"Number of sessions: {num_sessions}")
    #print(f"Number of edges: {num_edges}")
    #print(f"Number of nodes: {numnodes}")

    zero_degree_nodes = [node for node in G.nodes if G.degree(node) == 0]
    G.remove_nodes_from(zero_degree_nodes)   

    print(f"Removed {len(zero_degree_nodes)} isolated nodes.")

    return G, len(zero_degree_nodes)