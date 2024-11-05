from node2vec import Node2Vec
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile

G = nx.karate_club_graph()

# Initialize Node2Vec model
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, p=1, q=1, workers=4)

# Generate embeddings
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get embeddings as vectors
node_embeddings = [model.wv[str(node)] for node in G.nodes()]

# Compute cosine similarity between nodes
cosine_sim = cosine_similarity(node_embeddings)

# Print similarity between node 0 and node 1
print("Cosine similarity between node 0 and node 1:", cosine_sim[0, 1])


"""
# Get the temporary directory path
temp_path = os.path.expanduser('~')  # For example, use the user's home directory

# Check if there are any non-ASCII characters in the path
if any(ord(char) > 127 for char in temp_path):
    print("The path contains non-ASCII characters:", temp_path)
else:
    print("The path is ASCII-only:", temp_path)
"""

"""
temp_path = tempfile.gettempdir()
print("Temporary directory path:", temp_path)

if any(ord(char) > 127 for char in temp_path):
    print("The temporary path contains non-ASCII characters.")
else:
    print("The temporary path is ASCII-only.")
"""