import osmnx as ox
import networkx as nx
import geopandas as gpd
from sklearn.cluster import KMeans
import numpy as np

# 1. Download the drivable road network for a city (e.g., Manhattan)
place_name = "Manhattan, New York, USA"
G = ox.graph_from_place(place_name, network_type="drive")

# 2. Convert the graph to GeoDataFrames (nodes + edges)
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
node_ids = list(nodes_gdf.index)   # list of node IDs used by the graph

# 3. Cluster the node coordinates into 30 clusters
num_clusters = 30
coords = nodes_gdf[['x', 'y']].values  # shape: (N, 2), where N is number of nodes

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(coords)       # cluster label for each node
centroids = kmeans.cluster_centers_       # shape: (30, 2) => x, y for each cluster center

# 4. Normalize the 30 centroid coordinates to fit in bounding box [0,10] x [0,10]
x_min, x_max = centroids[:, 0].min(), centroids[:, 0].max()
y_min, y_max = centroids[:, 1].min(), centroids[:, 1].max()

# Avoid division by zero if data is degenerate (unlikely with real city data)
x_range = x_max - x_min if x_max != x_min else 1
y_range = y_max - y_min if y_max != y_min else 1

norm_centroids = []
for (cx, cy) in centroids:
    # Scale each dimension to [0, 10]
    norm_x = 10.0 * (cx - x_min) / x_range
    norm_y = 10.0 * (cy - y_min) / y_range
    norm_centroids.append((norm_x, norm_y))

# 5. Create a "super-graph" with 30 nodes, each storing its normalized centroid
super_G = nx.Graph()
for c in range(num_clusters):
    nx_c, ny_c = norm_centroids[c]
    super_G.add_node(c, x=nx_c, y=ny_c)

# 6. Add edges between these super-nodes (clusters) if original G had edges 
#    connecting nodes in different clusters.
id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
for u, v in G.edges():
    u_cluster = labels[id_to_idx[u]]
    v_cluster = labels[id_to_idx[v]]
    if u_cluster != v_cluster:
        super_G.add_edge(u_cluster, v_cluster)

# 7. Print out results
print("Super-graph nodes (30 clusters), normalized to [0,10] x [0,10]:")
for cluster_id in sorted(super_G.nodes()):
    x_c = super_G.nodes[cluster_id]['x']
    y_c = super_G.nodes[cluster_id]['y']
    print(f"Cluster {cluster_id}: x={x_c:.3f}, y={y_c:.3f}")

print("\nSuper-graph edges (showing adjacency between clusters):")
for (c1, c2) in sorted(super_G.edges()):
    print(f"Edge: {c1} -- {c2}")
