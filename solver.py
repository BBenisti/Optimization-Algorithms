
class Solver:
    """
    A solver class. 

    Attributes: 
    -----------
    grid: Grid
        The grid
    pairs: list[tuple[tuple[int]]]
        A list of pairs, each being a tuple ((i1, j1), (i2, j2))
    """

    def __init__(self, grid):
        """
        Initializes the solver.

        Parameters: 
        -----------
        grid: Grid
            The grid
        """
        self.grid = grid
        self.pairs = list()

    def score(self):
        """
        Computes the score based on selected pairs and remaining unpaired cells.
        Output:
        -------
        int
            The total cost computed as the sum of absolute differences of values in pairs,
            plus the sum of values of unpaired non-black cells.
        """
        l = [ (i, j) for i in range(self.grid.n) for j in range(self.grid.m) if self.grid.color[i][j] != 4 ] 
        """
        Computes the score of the list of pairs in self.pairs
        Does not consider the cells that are not black
        """
        tot= 0
        # adds the score of each pair that has been considered during the loop
        for couple in self.pairs : 
            tot += self.grid.cost(couple)
            p1 , p2 = couple
            l.remove(p1)
            l.remove(p2)
        # adds the score of the remaining cells
        for p in l : 
            a,b = p
            tot+=self.grid.value[a][b]
        return tot


class SolverEmpty(Solver):
    def run(self):
        pass


class SolverGreedy(Solver):
    def diff(self):
        """
        Sorts all possible pairs based on their cost (absolute difference between values).
        """
        dif = []
        L = self.grid.all_pairs()
        for i in range(len(L)):
            dif.append((self.grid.cost(L[i]), i))  # Associate each pair with its cost
        dif.sort(key=lambda x: x[0])  # Sort by increasing cost
        return dif

    def sum_diff(self):
        """
        Uses a greedy algorithm to select pairs that minimize the sum of differences.
        Then adds the values of the remaining unpaired cells.
        """
        sum_cost = 0 
        cases_taken = set()  # Use a set for faster lookup
        dif = self.diff()  
        selected_pairs = []  # Stores the selected pairs

        # Pair selection process
        for cost, index in dif:
            pair = self.grid.all_pairs()[index]
            c1, c2 = pair

            # Ensure the cells are not already paired and are not forbidden
            if c1 not in cases_taken and c2 not in cases_taken and not self.grid.is_forbidden(c1[0], c1[1]) and not self.grid.is_forbidden(c2[0], c2[1]):
                sum_cost += cost
                cases_taken.add(c1)
                cases_taken.add(c2)
                selected_pairs.append(pair)

        # Store the selected pairs
        self.pairs = selected_pairs

        # Add values of unpaired cells (excluding forbidden ones)
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                if (i, j) not in cases_taken and not self.grid.is_forbidden(i, j):
                    sum_cost += self.grid.value[i][j]

        return sum_cost

"""
The complexity of this algorithm is O((nm)**2 * log(nm))
"""
import numpy as np
from collections import deque
import numpy as np


class Graph:
    def __init__(self, graph):
        """
        Initializes the graph with an adjacency matrix.
        Parameters:
        -----------
        graph: list[list[int]]
            The adjacency matrix representation of the graph.
        """
        self.graph = graph
        self.row = len(graph)
        self.collum = len(graph[0])

    # Using BFS as a searching algorithm 
    def searching_algo_BFS(self, s, t, parent):
        """
        Implements BFS to find an augmenting path in the residual graph.
        Parameters:
        -----------
        s: int
            The source node.
        t: int
            The sink node.
        parent: list[int]
            An array to store the path found by BFS.
        Output:
        -------
        bool
            True if a path from source to sink exists, False otherwise.
        """
        visited = [False] * self.row
        queue = []
        queue.append(s)
        visited[s] = True
        while queue:
            u = queue.pop(0)
            for ind, val in enumerate(self.graph[u]):
                if not visited[ind] and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
        return visited[t]

    # Applies the Ford-Fulkerson algorithm
    def ford_fulkerson(self, source, sink):
        """
        Implements the Ford-Fulkerson algorithm to find the maximum flow.
        """
        parent = [-1] * self.row
        max_flow = 0
        flow_edges = []
        while self.searching_algo_BFS(source, sink, parent):
            path_flow = float("Inf")
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, self.graph[u][v])
                v = parent[v]
            max_flow += path_flow
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                flow_edges.append((u, v, path_flow))
                v = parent[v]
        return max_flow, flow_edges

class SolverFulkerson(Solver):
    """
    Question 6:
    Ce solver trouve la solution optimale pour les grilles ne possédant que des 1 comme valeurs.
    """
    def build_adjacency_matrix(self, grid):
        """
        Constructs an adjacency matrix from the given grid representation.
        """
        rows, cols = len(grid.value), len(grid.value[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        node_index = {}
        index = 0
        for i in range(rows):
            for j in range(cols):
                node_index[(i, j)] = index
                index += 1
        N = len(node_index)
        adjacency_matrix = [[0 for _ in range(N + 2)] for _ in range(N + 2)]
        for (i, j), u in node_index.items():
            if (i + j) % 2 == 0:
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if (ni, nj) in node_index and self.grid.compatible_color((ni, nj), (i, j)):
                        v = node_index[(ni, nj)]
                        adjacency_matrix[u][v] = 1
                adjacency_matrix[N][u] = 1
            else:
                adjacency_matrix[u][N + 1] = 1
        return adjacency_matrix, node_index

    def run(self):
        graph, dico = self.build_adjacency_matrix(self.grid)
        graph1 = Graph(graph)
        source, sink = len(graph) - 2, len(graph) - 1
        _, flow_edges = graph1.ford_fulkerson(source, sink)
        for u, v, flow in flow_edges:
            if flow > 0 and u < source and v < sink:
                for k, val in dico.items():
                    if val == u:
                        case1 = k
                    elif val == v:
                        case2 = k
                self.pairs.append((case1, case2))




"""     

Comparing to SolverGreedy's complexity, we can foresee the fact that, the more
the grid is big, the more Greedy will tend to be useless compare to Fulkerson.

This prediction can be confirmed by using the function in main to compare both methods
on the different grids available in input.
"""

import networkx as nx

class SolverGeneral:
    def __init__(self, grid):
        """
        Initializes the class with a given grid.
        Creates a flow graph with a source and a sink.
        """
        self.grid = grid
        self.n = grid.n
        self.m = grid.m
        self.size = self.n * self.m
        self.graph = np.zeros((self.size + 2, self.size + 2))  # Graph including source and sink
        self.source = self.size  # Index of the source
        self.sink = self.size + 1  # Index of the sink
        self.build_graph()

    def index(self, i, j):
        return i * self.m + j
    
    def build_graph(self):
        for i in range(self.n):
            for j in range(self.m):
                if self.grid.color[i][j] == 4:
                    continue
                node = self.index(i, j)
                if (i + j) % 2 == 0:
                    self.graph[self.source][node] = 1
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.n and 0 <= nj < self.m:
                        if self.grid.color[ni][nj] == 4:
                            continue
                        if self.grid.color[i][j] == 3 and self.grid.color[ni][nj] not in [0, 3]:
                            continue
                        if self.grid.color[i][j] in [1, 2] and self.grid.color[ni][nj] not in [0, 1, 2]:
                            continue
                        neighbor = self.index(ni, nj)
                        self.graph[node][neighbor] = 1
                if (i + j) % 2 == 1:
                    self.graph[node][self.sink] = 1

    def solve(self):
        """
        Creates the bipartite graph for non-black cells and adds edges between adjacent cells
        that can be paired according to the rules.
        """
        # Create the graph
        G = nx.Graph()
        non_black = []

        # Collect all non-black cells
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                if not self.grid.is_forbidden(i, j):  # The cell is not black
                    non_black.append((i, j))

        # Partition cells into two sets based on parity (bipartite)
        partitionA = []
        partitionB = []
        for cell in non_black:
            if (cell[0] + cell[1]) % 2 == 0:
                partitionA.append(cell)
            else:
                partitionB.append(cell)

        # Add nodes to the graph
        for cell in partitionA:
            G.add_node(cell, bipartite=0)
        for cell in partitionB:
            G.add_node(cell, bipartite=1)

        # Add edges using all_pairs()
        for (i, j), (ni, nj) in self.grid.all_pairs():
            # Edge weight = 2 * min(v1, v2)
            v1 = self.grid.value[i][j]
            v2 = self.grid.value[ni][nj]
            weight = v1 + v2 + abs(v1 - v2)
            G.add_edge((i, j), (ni, nj), weight=weight)

        # Calculate the maximum weight matching
        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight='weight')

        # Calculate the total benefit of the matchings
        matching_benefit = 0
        for cell1, cell2 in matching:
            v1 = self.grid.value[cell1[0]][cell1[1]]
            v2 = self.grid.value[cell2[0]][cell2[1]]
            matching_benefit += abs(v1 - v2)  # Calculating the cost as the absolute difference

        # Base score: sum of values of non-black cells that are not in the matching
        unpaired_value = 0
        paired_cells = set()
        for cell1, cell2 in matching:
            paired_cells.add(cell1)
            paired_cells.add(cell2)

        # Calculate unpaired cells' value
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                if (i, j) not in paired_cells and not self.grid.is_forbidden(i, j):
                    unpaired_value += self.grid.value[i][j]

        # Total score
        total_score = matching_benefit + unpaired_value

        self.matching = matching
        self.score = total_score
        return matching, total_score
 