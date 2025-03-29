import networkx as nx
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.applications import Maxcut 
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator 
from qiskit import transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Create a custom graph
# -------------------------------
custom_graph = nx.Graph()

# Define nodes for two clusters
cluster1 = [0, 1, 2, 3]
cluster2 = [4, 5, 6, 7]
custom_graph.add_nodes_from(cluster1)
custom_graph.add_nodes_from(cluster2)

# Add internal edges for cluster1 (fully connected, low weight)
for i in range(len(cluster1)):
    for j in range(i+1, len(cluster1)):
        custom_graph.add_edge(cluster1[i], cluster1[j], weight=1)

# Add internal edges for cluster2 (fully connected, low weight)
for i in range(len(cluster2)):
    for j in range(i+1, len(cluster2)):
        custom_graph.add_edge(cluster2[i], cluster2[j], weight=1)

# Add connecting edges between clusters with high weights
custom_graph.add_edge(2, 5, weight=10)
custom_graph.add_edge(3, 4, weight=10)

# -------------------------------
# Steps 2-4: QUBO and QAOA Setup
# -------------------------------
# Convert graph to QUBO using Maxcut class
maxcut = Maxcut(custom_graph)
qubo = maxcut.to_quadratic_program()

# Convert QUBO to Ising Hamiltonian
ising, ising_offset = qubo.to_ising()

# Create QAOA Circuit
qaoa_reps = 2
qaoa_ansatz = QAOAAnsatz(cost_operator=ising, reps=qaoa_reps, name='qaoa')
qaoa_ansatz.measure_active()

# Assign random parameters
betas = np.random.uniform(0, np.pi, qaoa_reps)
gammas = np.random.uniform(0, 2*np.pi, qaoa_reps)
parameter_values = [*betas, *gammas]
qaoa_with_parameters = qaoa_ansatz.assign_parameters(
    dict(zip(qaoa_ansatz.parameters, parameter_values))
)

# Execute on Simulator
aer_simulator = AerSimulator()
compiled_qaoa = transpile(qaoa_with_parameters, aer_simulator)
result = aer_simulator.run(compiled_qaoa, shots=10**5).result()
counts = result.get_counts()

# Discard trivial solutions (all 0's or all 1's)
def is_trivial_solution(binary_string):
    return binary_string == '0' * len(binary_string) or binary_string == '1' * len(binary_string)

filtered_counts = {binary_string: count for binary_string, count in counts.items() 
                   if not is_trivial_solution(binary_string)}

# -------------------------------
# New Section: Plot Histogram of the Cuts
# -------------------------------
plt.figure(figsize=(8, 6))
plot_histogram(filtered_counts)
plt.title("Histogram of QAOA Measurement Results (Non-Trivial Cuts)")
plt.show()

# -------------------------------
# Step 5: Calculate the Max-Cut Partition
# -------------------------------
def calculate_max_cut(graph):
    max_cuts = 0
    max_weight = 0 
    best_partition = None
    num_nodes = len(graph.nodes)
    for i in range(1, 1 << num_nodes):
        set_a = [node for node in range(num_nodes) if (i & (1 << node)) > 0]
        set_b = [node for node in range(num_nodes) if (i & (1 << node)) == 0]
        cut_edges = []
        cut_weight = 0 
        for edge in graph.edges:
            if (edge[0] in set_a and edge[1] in set_b) or (edge[0] in set_b and edge[1] in set_a):
                cut_edges.append(edge)
                cut_weight += graph[edge[0]][edge[1]].get("weight", 1)
        if cut_weight > max_weight:
            max_weight = cut_weight
            max_cuts = len(cut_edges)
            best_partition = (set_a, set_b, cut_edges, cut_weight)
    return max_cuts, max_weight, best_partition

max_cuts, max_weight, best_partition = calculate_max_cut(custom_graph)

# -------------------------------
# Step 6: Plotting Function with Minimum Path Overlay
# -------------------------------
def plot_max_cut_with_min_path(graph, best_partition):
    set_a, set_b, cut_edges, cut_weight = best_partition
    pos = nx.spring_layout(graph)
    
    # Draw nodes: color them by partition
    nx.draw_networkx_nodes(graph, pos, nodelist=set_a, node_color='lightblue', edgecolors='black', label="Set A")
    nx.draw_networkx_nodes(graph, pos, nodelist=set_b, node_color='lightgreen', edgecolors='black', label="Set B")
    
    # Draw all non-cut edges in gray
    all_edges = list(graph.edges)
    cut_edge_set = set(cut_edges)
    nx.draw_networkx_edges(graph, pos, edgelist=[e for e in all_edges if e not in cut_edge_set], edge_color="gray")
    
    # Draw cut edges in blue (dashed)
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color="blue", width=2.5, style="dashed", label="Cut Edges")
    
    # Calculate and draw the minimum path from the first to the last node
    source_node = 0
    target_node = max(graph.nodes())
    try:
        min_path = nx.shortest_path(graph, source=source_node, target=target_node, weight='weight')
        min_path_edges = list(zip(min_path, min_path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=min_path_edges, edge_color='red', width=3, label="Min Path")
    except nx.NetworkXNoPath:
        print("No path exists between node 0 and node", target_node)
    
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold")
    
    # Draw edge weights
    edge_labels = {(u, v): round(graph[u][v].get("weight", 1), 2) for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title(f"Max Cut (Weight = {cut_weight:.2f}) with Minimum Path Overlay")
    plt.legend()
    plt.show()

print(f"Maximum number of cuts: {max_cuts}")
print(f"Best partition: Set A: {best_partition[0]}, Set B: {best_partition[1]}")
print(f"Cut edges: {best_partition[2]}")

# Optionally, run the balanced_max_cut if needed (prints a warning when imbalance is detected)
def balanced_max_cut(graph, threshold=0):
    best_cut_value = -1
    best_partition = None
    best_cut_weight = 0
    num_nodes = len(graph.nodes)
    for i in range(1, 1 << num_nodes):
        set_a = [node for node in range(num_nodes) if (i & (1 << node)) > 0]
        set_b = [node for node in range(num_nodes) if (i & (1 << node)) == 0]
        cut_edges = []
        cut_weight = 0
        for edge in graph.edges:
            if (edge[0] in set_a and edge[1] in set_b) or (edge[0] in set_b and edge[1] in set_a):
                cut_edges.append(edge)
                cut_weight += graph[edge[0]][edge[1]].get("weight", 1)
        internal_a = sum(graph[u][v].get("weight", 1) for u, v in graph.edges if u in set_a and v in set_a)
        internal_b = sum(graph[u][v].get("weight", 1) for u, v in graph.edges if u in set_b and v in set_b)
        if abs(internal_a - internal_b) <= threshold:
            if cut_weight > best_cut_weight:
                best_cut_value = len(cut_edges)
                best_cut_weight = cut_weight
                best_partition = (set_a, set_b, cut_edges, cut_weight)
        else:
            print("🚨 ALARM TRIPPED: Police alert triggered due to imbalance!")
    return best_cut_value, best_partition

best_cut_val, best_partition_ = balanced_max_cut(custom_graph)

# -------------------------------
# Step 7: Plot the final graph with the minimum path overlaid
# -------------------------------
plot_max_cut_with_min_path(custom_graph, best_partition)
