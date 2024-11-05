import networkx as nx

def build_graph(dataset):
    G = nx.DiGraph()
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
    
            if G.has_edge(source, target):
                # Increment the weight of the existing edge
                G[source][target]['weight'] += 1
            else:
                # Add a new edge with initial weight 1
                G.add_edge(source, target, weight=1)

    numnodes = len(unique_nodes)  # Number of unique nodes

    #print(f"Number of sessions: {num_sessions}")
    #print(f"Number of edges: {num_edges}")
    #print(f"Number of nodes: {numnodes}")

    self_loops = list(nx.selfloop_edges(G))
    if len(self_loops) > 0:
        G.remove_edges_from(self_loops)
        print(f"Removed {len(self_loops)} self-loops from the graph.")

    return G


def cal_degree(G):
    node_degrees = dict(G.degree())
    return node_degrees    

def cal_kcore(G):
    kcore = nx.core_number(G)
    return kcore

def cal_weighted_edge(G):
    weighted_edge = {node: sum(weight for _, _, weight in G.edges(node, data='weight')) for node in G.nodes()}

    return weighted_edge

def cal_eigenvector_centrality(G, weight_flag):
    if weight_flag: 
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, weight = 'weight')
    else:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    return eigenvector_centrality

def cal_pagerank(G, weight_flag, alpha=0.85, max_iter=10000, tol=1e-6):
    if weight_flag: 
        pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol, weight = 'weight')
    else:
        pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)
    
    return pagerank


def detect_cal_node_centrality(node_centrality, G):
    if node_centrality == 'weighted_edge':
        nodecentrality_G = cal_weighted_edge(G)
    elif node_centrality == 'pagerank':
        nodecentrality_G = cal_pagerank(G, weight_flag = True)
    elif node_centrality == 'eigenvector_centrality':
        nodecentrality_G = cal_eigenvector_centrality(G, weight_flag = True)

    else:    
        print('Node Centrality value seems wrong')

    return nodecentrality_G

def get_nodecentrality_ranking(dataset, node_centrality, fixed_value=1, alpha=1):
    dataset_ranked = []

    for session in dataset:  # Iterate over the first 10 sessions
        length = len(session)
        session_degrees = [(item, node_centrality.get(item, 0)) for item in session]
        
        # Sort by degree in descending order to determine ranking
        session_degrees_sorted = sorted(session_degrees, key=lambda x: x[1], reverse=True)
        
        # Create a mapping from degree to rank, handling ties
        degree_to_rank = {}
        current_rank = 1
        for i, (item, degree) in enumerate(session_degrees_sorted):
            if degree not in degree_to_rank:
                degree_to_rank[degree] = current_rank
            current_rank += 1

        # Append the rank to each item in the original order
        session_degrees_ranked  = [(item, degree, degree_to_rank[degree]) for item, degree in session_degrees]

        dataset_ranked.append(session_degrees_ranked)


    return dataset_ranked