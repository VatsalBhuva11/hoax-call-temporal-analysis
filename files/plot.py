import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

# Load your data
df = pd.read_csv('final_data_with_apostrophe.csv', sep=',')

# Create an undirected graph
G = nx.Graph()

# Edge color mapping
edge_colors = []

# Process each row
for _, row in df.iterrows():
    label = row['label']  # Extract label (fraud or normal)
    
    # Extract words after the 2nd column (ignoring 'label' and 'year')
    # print(row['message'])
    words = [str(word).strip() for word in row["message"].split(',') if isinstance(word, str) and len(word.strip()) > 3]
    # if _ < 2:
    #     print(words)
    # Create edges between all pairs of words in this row
    for u, v in combinations(words, 2):
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1  # Increase weight if edge exists
        else:
            G.add_edge(u, v, weight=1)  # Create new edge
        
        # Assign color based on fraud or normal
        edge_colors.append("red" if label.lower() == "fraud" else "blue")

# Draw the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.5)  # Positioning for better visualization

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_size=700)
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.0, alpha=0.5)  # Apply color

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

plt.title("Word Co-occurrence Network (Fraud = Red, Normal = Blue)")
plt.axis('off')
plt.tight_layout()
plt.savefig("word_network.png", dpi=300)


# Save the graph for further analysis
nx.write_gexf(G, "word_network.gexf")
