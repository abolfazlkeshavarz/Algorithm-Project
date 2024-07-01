import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

def Floyd_Warshal(graph, D, P):
    for i in range(len(graph)):
        for j in range(len(graph)):
            D[i][j] = graph[i][j]
            P[i][j] = -1
    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                if D[i][j] > D[i][k] + D[k][j]:
                    D[i][j] = D[i][k] + D[k][j]
                    P[i][j] = k

def plot_graph(graph, D):
    G = nx.DiGraph()
    num_nodes = len(graph)

    for i in range(num_nodes):
        G.add_node(i)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if D[i][j] != float('inf') and i != j:
                G.add_edge(i, j, weight=D[i][j])

    pos = nx.spring_layout(G)

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrows=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')

    plt.title('Graph with Shortest Paths')
    plt.show()

dim = int(input('Enter Nodes Count: '))
graph = np.zeros((dim, dim), dtype=float)

print('Enter adjacency matrix (use 9999 for no direct path):')
for i in range(dim):
    row = list(map(int, input().split()))
    for j in range(dim):
        if row[j] == 9999:
            graph[i][j] = float('inf')
        else:
            graph[i][j] = float(row[j])

D = np.zeros((dim, dim), dtype=float)
P = np.zeros((dim, dim), dtype=int)

Floyd_Warshal(graph, D, P)
print('Shortest Path Distance Matrix:')
print(D)
print(P)
plot_graph(graph, D)



# 0 3 8 9999 -4  
# 9999 0 9999 1 7
# 9999 4 0 9999 9999
# 2 9999 -5 0 99999
# 9999 9999 9999 6 0