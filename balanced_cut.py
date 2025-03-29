import networkx as nx
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.applications import Maxcut 
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator 
from qiskit import transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import random

plt.ion()

G = nx.Graph()

# Add nodes
G.add_nodes_from(range(8))  # Nodes 0 to 7

# Cluster 1 (nodes 0–3)
G.add_edge(0, 1, weight=5)
G.add_edge(0, 2, weight=5)
G.add_edge(1, 2, weight=5)
G.add_edge(2, 3, weight=5)
G.add_edge(0, 3, weight=5)

# Cluster 2 (nodes 4–7)
G.add_edge(4, 5, weight=5)
G.add_edge(4, 6, weight=5)
G.add_edge(5, 6, weight=5)
G.add_edge(6, 7, weight=5)
G.add_edge(4, 7, weight=5)

# Cross-cluster edges (lightly connected)
G.add_edge(1, 5, weight=1)
G.add_edge(3, 6, weight=1)

# Manually define the balanced partition
set_a = [0, 1, 2, 3]
set_b = [4, 5, 6, 7]

cut_edges = [(1, 5), (3, 6)]
cut_weight = sum(G[u][v]['weight'] for u, v in cut_edges)

# Plotting function
def plot_balanced_cut(graph, set_a, set_b, cut_edges, cut_weight):
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(graph, pos, nodelist=set_a, node_color='lightblue', edgecolors='black', label="Set A")
    nx.draw_networkx_nodes(graph, pos, nodelist=set_b, node_color='lightgreen', edgecolors='black', label="Set B")

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edgelist=[e for e in graph.edges if e not in cut_edges], edge_color="gray")
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color="red", width=2.5, style="dashed", label="Cut Edges")

    # Labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold")
    edge_labels = {(u, v): graph[u][v]['weight'] for u, v in graph.edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

    plt.legend()
    plt.title(f"Balanced Max Cut (Weight = {cut_weight})")
    plt.tight_layout()
    plt.show()


balanced_escape_graph = nx.Graph()

# Weighted edges — patrol heat, motion sensors, cameras, etc.
balanced_escape_graph.add_weighted_edges_from([
    (0, 1, 3),   # North alley
    (1, 2, 2),   # Street
    (2, 3, 4),   # Near plaza
    (3, 0, 3),   # Loop around

    (4, 5, 3),   # South alley
    (5, 6, 2),   # Narrow path
    (6, 7, 4),   # Sensor street
    (7, 4, 3),   # Loop around

    (0, 4, 5),   # Cross-zone choke point
    (1, 5, 1),   # Shortcut
    (2, 6, 6),   # Heavily watched
    (3, 7, 2)    # Escape tunnel
])




# Step 2: Convert Graph to QUBO using Maxcut class
maxcut = Maxcut(G)
qubo = maxcut.to_quadratic_program()

# Step 3: Convert QUBO to Ising Hamiltonian
ising, ising_offset = qubo.to_ising()

# Step 4: Create QAOA Circuit
qaoa_reps = 2
qaoa_ansatz = QAOAAnsatz(cost_operator=ising, reps=qaoa_reps, name='qaoa')
qaoa_ansatz.measure_active()

# Step 5: Assign Parameters
betas = np.random.uniform(0, np.pi, qaoa_reps)
gammas = np.random.uniform(0, 2*np.pi, qaoa_reps)
parameter_values = [*betas, *gammas]
qaoa_with_parameters = qaoa_ansatz.assign_parameters(dict(zip(qaoa_ansatz.parameters, parameter_values)))

# Step 6: Execute on Simulator
aer_simulator = AerSimulator()
compiled_qaoa = transpile(qaoa_with_parameters , aer_simulator)
result = aer_simulator.run(compiled_qaoa, shots=256).result()

counts = result.get_counts()

# Step 7: Discard 0000 and 1111 as these are Not Cuts -> [][0, 1, 2, 3] and [1, 2, 3, 4][]
def is_trivial_solution(binary_string):
    """
    Check if the solution is trivial (either all 0's or all 1's). """
    return binary_string == '0' * len(binary_string) or binary_string == '1' * len( binary_string)


# Step 8: Filter out trivial solutions (0000, 1111)
filtered_counts = {binary_string: count for binary_string, count in counts.items() if not is_trivial_solution(binary_string)}


# Step 9: Calculate the Max-Cut and Max Number of Cuts
def calculate_max_cut(graph): 
    """
    Calculate the maximum number of cuts for a given graph. """

# Define all possible partitions
    max_cuts = 0
    max_weight = 0 
    min_weight = 1000
    best_partition = None
    worst_partition = None
    # Loop over all possible partitions of the graph into two sets
    num_nodes = len(graph.nodes)
    for i in range(1, 1 << num_nodes):
        set_a = [node for node in range(num_nodes) if (i & (1 << node)) > 0]
        set_b = [node for node in range(num_nodes) if (i & (1 << node)) == 0]

# Calculate cut edges between set A and set B
        cut_edges = []
        cut_weight = 0 

        for edge in graph.edges:
            if (edge[0] in set_a and edge[1] in set_b) or (edge[0] in set_b and edge[1] in
            set_a): 
                cut_edges.append(edge)
                weight = graph[edge[0]][edge[1]].get("weight", 1) 
                cut_weight += weight
    # Check if this partition has more cuts
        if  cut_weight  > max_weight: 
            max_weight = cut_weight
            max_cuts = len(cut_edges)
            best_partition = (set_a, set_b, cut_edges, cut_weight)
    return max_cuts , max_weight, best_partition


# Step 10: Calculate the maximum cut for the graph
max_cuts, max_weight, best_partition = calculate_max_cut(G)
# max_cuts_1, max_weight_1, best_partition_1 = calculate_max_cut(best_partition)


def plot_max_cut(graph, best_partition):
    """
    Visualizes the Max-Cut by highlighting cut edges and node partitions.
    """
    set_a, set_b, cut_edges, cut_weight = best_partition

    # Assign positions for visualization
    # pos = nx.spring_layout(graph)  
    pos = {node: (node % 3, -node // 3) for node in graph.nodes}


    # Draw nodes (color them based on their partition)
    nx.draw_networkx_nodes(graph, pos, nodelist=set_a, node_color='lightblue', edgecolors='black', label="Set A")
    nx.draw_networkx_nodes(graph, pos, nodelist=set_b, node_color='lightgreen', edgecolors='black', label="Set B")

    # Draw edges
    all_edges = list(graph.edges)
    cut_edge_set = set(cut_edges)  # Convert to set for quick lookup

    # Draw normal edges (black)
    nx.draw_networkx_edges(graph, pos, edgelist=[e for e in all_edges if e not in cut_edge_set], edge_color="gray")

    # Draw cut edges (highlighted in red)
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color="red", width=2.5, style="dashed", label="Cut Edges")

    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold")

    # Draw edge weights
    edge_labels = {
    (min(u, v), max(u, v)): graph[u][v].get("weight", 1)
    for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

    # Show legend
    plt.legend()
    plt.title(f"Max Cut (Weight = {cut_weight})")
    plt.show()

# Step 11: Output the results
print(f"Maximum␣number␣of␣cuts:␣{max_cuts}")
print(f"Best␣partition:␣Set␣A:␣{best_partition[0]},␣Set␣B:␣{best_partition[1]}") 

print(f"Cut␣edges:␣{best_partition[2]}")


def fast_balanced_max_cut(graph, threshold=0, samples=5000):
    best_cut_value = -1
    best_partition = None
    best_cut_weight = 0
    num_nodes = len(graph.nodes)

    for _ in range(samples):
        set_a = []
        set_b = []

        # Randomly assign each node to A or B
        for node in graph.nodes:
            if random.random() < 0.5:
                set_a.append(node)
            else:
                set_b.append(node)

        cut_edges = []
        cut_weight = 0

        for u, v in graph.edges:
            if (u in set_a and v in set_b) or (u in set_b and v in set_a):
                cut_edges.append((u, v))
                cut_weight += graph[u][v].get("weight", 1)

        internal_a = sum(graph[u][v].get("weight", 1) for u, v in graph.edges if u in set_a and v in set_a)
        internal_b = sum(graph[u][v].get("weight", 1) for u, v in graph.edges if u in set_b and v in set_b)

        if abs(internal_a - internal_b) <= threshold:
            if cut_weight > best_cut_weight:
                best_cut_value = len(cut_edges)
                best_cut_weight = cut_weight
                best_partition = (set_a, set_b, cut_edges, cut_weight)

    if best_partition is None:
        print("⚠️ No balanced cut found after sampling.")
        return 0, ([], [], [], 0)

    return best_cut_value, best_partition


def balanced_max_cut(graph, threshold=0):
    """
    Finds the max cut with balance constraint: internal weights of both sets must be within threshold.
    """
    best_cut_value = -1
    best_partition = None
    best_cut_weight = 0  # Track max cut weight
    num_nodes = len(graph.nodes)

    for i in range(1, 1 << num_nodes):
        set_a = [node for node in range(num_nodes) if (i & (1 << node)) > 0]
        set_b = [node for node in range(num_nodes) if (i & (1 << node)) == 0]

        # Find cut edges and compute cut weight
        cut_edges = []
        cut_weight = 0  # Total weight of cut edges

        for edge in graph.edges:
            if (edge[0] in set_a and edge[1] in set_b) or (edge[0] in set_b and edge[1] in set_a):
                cut_edges.append(edge)
                cut_weight += graph[edge[0]][edge[1]].get("weight", 1)  # Default weight = 1 if not provided

        # Internal weights (edges within set_a and set_b)
        internal_a = sum(graph[u][v].get("weight", 1) for u, v in graph.edges if u in set_a and v in set_a)
        internal_b = sum(graph[u][v].get("weight", 1) for u, v in graph.edges if u in set_b and v in set_b)

        # Check if partition satisfies balance constraint
        if abs(internal_a - internal_b) <= threshold:
            if cut_weight > best_cut_weight:  # Prioritize max weight cut
                best_cut_value = len(cut_edges)
                best_cut_weight = cut_weight
                best_partition = (set_a, set_b, cut_edges, cut_weight)

        if abs(internal_a - internal_b) > threshold:
            print("🚨 ALARM TRIPPED: Police alert triggered due to imbalance!")

    if best_partition is None:
            print("⚠️ No balanced cut found within the threshold.")
            return 0, ([], [], [], 0)

    return best_cut_value,  best_partition

# 1. Run QAOA
print("🔁 Running QAOA...")
result = aer_simulator.run(compiled_qaoa, shots=256).result()
print("✅ QAOA simulation complete")

# 2. Analyze results
counts = result.get_counts()
filtered_counts = {
    bitstring: count
    for bitstring, count in counts.items()
    if bitstring != '0' * len(bitstring) and bitstring != '1' * len(bitstring)
}

# 3. Print + plot QAOA results
print("Top QAOA results (filtered):")
for bitstring, count in sorted(filtered_counts.items(), key=lambda x: -x[1])[:5]:
    print(f"{bitstring}: {count} shots")

# plot_histogram(filtered_counts, title="QAOA Output (Filtered)")
# plt.show()

# 4. Run your balanced cut checker
best_cut_val, best_partition_ = fast_balanced_max_cut(G, threshold=10, samples=500)
# plot_max_cut(city_graph, best_partition_)
plot_balanced_cut(G, set_a, set_b, cut_edges, cut_weight)

hist_fig = plt.figure()
plot_histogram(filtered_counts, title="QAOA Max-Cut Output (Filtered)", bar_labels=False)
plt.tight_layout()
plt.show(block=True)
plt.ioff()         # Turn off interactive mode
plt.show(block=True)  # Keeps the window open until closed manually



best_cut_val, best_partition_ = fast_balanced_max_cut(G, threshold=5)
