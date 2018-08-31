from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
import sys

__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']

class Node:
    """
    Class to store one node information in association graph.
    """
    __slots__ = ('id', 'edge_w', 'edge_l')

    def __init__(self, identity):
        self.id = identity
        self.edge_w = {}
        self.edge_l = {}

    def add_edge(self, n2, w, l=''):
        self.edge_w[n2] = w
        self.edge_l[n2] = l


class Graph:
    """
    Class to store graph.
    """
    __slots__ = ('nodes', 'number_of_nodes')

    def __init__(self, n):
        self.nodes = []
        self.number_of_nodes = n

    def add_node(self, identity):
        if identity not in self.nodes:
            self.nodes.append(Node(identity))

    def get_node(self, identity):
        for n in self.nodes:
            if n.id == identity:
                return n
        return None

    def add_edge(self, n1, n2, weight, label='', directed=False):
        self.get_node(n1).add_edge(n2, weight, label)
        if not directed:
            self.get_node(n2).add_edge(n1, weight, label)

    def get_adjacency_list(self):
        al = []
        id_to_index = {}
        index_to_id = {}
        i = 0
        for no in self.nodes:
            id_to_index[no.id] = i
            index_to_id[i] = no.id
            i = i + 1

        for no in self.nodes:
            aln = [0 for i in range(self.number_of_nodes)]
            n1 = id_to_index[no.id]
            for ed in no.edge_w.keys():
                n2 = id_to_index[ed]
                aln[n2] = no.edge_w[ed]
            al.append(aln)
        return [al, index_to_id]


def get_mst(g):
    """
    Compute MST from a graph object.
    """
    adj_list, index_to_id = g.get_adjacency_list()
    x = csr_matrix(adj_list)
    mst = minimum_spanning_tree(x)
    mst_adj_list = mst.toarray().astype(float)
    mst_graph = Graph(len(mst_adj_list))
    for no in range(len(mst_adj_list)):
        mst_graph.add_node(index_to_id[no])
    for ni in range(len(mst_adj_list)):
        for nj in range(len(mst_adj_list)):
            if mst_adj_list[ni][nj] > 0:
                mst_graph.add_edge(index_to_id[ni], index_to_id[nj],
                                   mst_adj_list[ni][nj])
                mst_graph.add_edge(index_to_id[nj], index_to_id[ni],
                                   mst_adj_list[ni][nj])
                mst_adj_list[nj][ni] = mst_adj_list[ni][nj]

    return [mst_adj_list, index_to_id]

def add_dummy_node(g):
    g1 = [[0 for i in range(len(g)+1)] for _ in range(len(g)+1)]
    for r in range(len(g1)):
        for c in range(1, len(g1[r])):
            if r == 0:
                g1[r][c] = -0.01
            else:
                g1[r][c] = g[r-1][c-1]

    return g1

def build_incoming_adj_list(g):
    ing = [[0 for i in range(len(g))] for _ in range(len(g))]

    for row in range(len(g)):
        for col in range(len(g)):
            if g[row][col] != 0:
                ing[col][row] = g[row][col]

    return ing


def get_possible_tree(g):
    g1 = [[0 for i in range(len(g))] for _ in range(len(g))]

    for row in range(len(g)):
        mine = 1000000
        mini = 0
        for col in range(len(g)):
            if g[row][col] != 0 and g[row][col] < mine:
                mine = g[row][col]
                mini = col
        if mine < 1000000:
            g1[mini][row] = mine

    return g1


def is_cyclic_util(v, visited, stack, g):
    visited[v] = True
    stack[v] = True

    for neighbour in [i for i in range(len(g[v])) if g[v][i] != 0]:
        if visited[neighbour] == False:
            res = is_cyclic_util(neighbour, visited, stack, g)
            if res > -1:
                return neighbour
        elif stack[neighbour] == True:
            return neighbour

    stack[v] = False
    return -1


# Returns true if graph is cyclic else false
def is_cyclic(g):
    visited = [False] * len(g)
    stack = [False] * len(g)
    for node in range(len(g)):
        if visited[node] == False:
            val = is_cyclic_util(node, visited, stack, g)
            if val > -1:
                return val
    return -1

def contract_cycle(g, start_index):
    nodes_in_cycle = []
    path = {}

    q = [start_index]
    current = start_index
    path[current] = -1
    visi = [0 for _ in g]
    breaks = False
    while len(q) > 0 and not breaks:
        prev = current
        current = q.pop()
        if current != start_index:
            path[current] = prev
        visi[current] = 1
        for col in range(len(g[current])):
            if g[current][col] != 0:
                if visi[col] == 1:
                    breaks = True
                    break
                else:
                    q.append(col)
    while current != -1:
        nodes_in_cycle.append(current)
        current = path[current]

    new_n = len(g) - len(nodes_in_cycle) + 1
    g1 = [[1000000 for _ in range(new_n)] for i in range(new_n)]
    mappings = [[0 for _ in range(new_n)] for i in range(new_n)]

    old_to_new = {}
    index = 0
    for i in range(len(g)):
        if i in nodes_in_cycle:
            old_to_new[i] = new_n-1
        else:
            old_to_new[i] = index
            index += 1


    for r in range(len(g)):
        for c in range(len(g)):
            if g[r][c] != 0:
                if r in nodes_in_cycle and c not in nodes_in_cycle:
                    if g1[old_to_new[r]][old_to_new[c]] > g[r][c]:
                        g1[old_to_new[r]][old_to_new[c]] = g[r][c]
                        mappings[old_to_new[r]][old_to_new[c]] = [r,c]
                elif r not in nodes_in_cycle and c in nodes_in_cycle:
                    if g1[old_to_new[r]][old_to_new[c]] > g[r][c]:
                        g1[old_to_new[r]][old_to_new[c]] = g[r][c]
                        mappings[old_to_new[r]][old_to_new[c]] = [r, c]
                elif r not in nodes_in_cycle and c not in nodes_in_cycle:
                    g1[old_to_new[r]][old_to_new[c]] = g[r][c]

    for r in range(len(g1)):
        for c in range(len(g1[r])):
            if g1[r][c] == 1000000:
                g1[r][c]=0

    return [g1, nodes_in_cycle, old_to_new, mappings]

def construct_full_mst(g, no, o2n, map, og):
    full_n = len(g)+len(no)-1
    mst = [[0 for _ in range(full_n)] for i in range(full_n)]
    n2o = {}
    for i in list(o2n.keys()):
        k = o2n[i]
        v = i
        if k in n2o:
            n2o[k].append(v)
        else:
            n2o[k] = [v]

    ni = []

    for r in range(len(g)):
        for c in range(len(g[r])):
            if r < len(g)-1 and c == len(g)-1 and g[r][c] != 0 :
                mst[n2o[r][0]][map[r][c][1]] = g[r][c]
                ni.append(map[r][c][1])
                no.remove(map[r][c][1])
            elif r == len(g)-1 and c < len(g)-1 and g[r][c] != 0:
                mst[map[r][c][0]][n2o[c][0]] = g[r][c]
            elif r < len(g)-1 and c < len(g)-1:
                mst[n2o[r][0]][n2o[c][0]] = g[r][c]

    if len(ni) == 0:
        mst[0][no[0]] = -0.01
        ni.append(no[0])
        no.remove(no[0])

    while len(no) > 0:
        for cuno in no:
            minw = 10000000
            mini = 0
            for cur in ni:
                if og[cur][cuno] < minw and og[cur][cuno] != 0:
                    minw = og[cur][cuno]
                    mini = cur
            if minw < 10000000:
                mst[mini][cuno] = minw
                ni.append(cuno)
                no.remove(cuno)
    return mst

def edmonds_mst(g, add_dummy):
    if add_dummy:
        g_dummy_node_out = add_dummy_node(g)
    else:
        g_dummy_node_out = g
    g_dummy_node_in = build_incoming_adj_list(g_dummy_node_out)
    g_min_incom = get_possible_tree(g_dummy_node_in)
    has_cycle = is_cyclic(g_min_incom)
    if has_cycle > -1:
        g_cycle_contracted, nodes_in_cycle, o2n, mappings = contract_cycle(
            g_min_incom,has_cycle)
        contracted_mst = edmonds_mst(g_cycle_contracted, False)
        g_mst = construct_full_mst(contracted_mst, nodes_in_cycle, o2n,
                                  mappings, g_min_incom)
        if add_dummy:
            g_ret = []
            for g in g_mst[1:]:
                g_ret.append(g[1:])
            return g_ret
        else:
            return g_mst

    else:
        if add_dummy:
            g_ret = []
            for g in g_min_incom[1:]:
                g_ret.append(g[1:])
            return g_ret
        else:
            return g_min_incom

def test_graph():
    g = Graph(4)
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    g.add_node(4)
    g.add_edge(3, 4, -0.6, directed=True)
    g.add_edge(2, 3, -0.2, directed=True)
    g.add_edge(4, 2, -0.9, directed=True)
    g.add_edge(1, 2, -0.8, directed=True)
    g.add_edge(1, 4, -0.3, directed=True)
    ga, i2i = g.get_adjacency_list()
    print(ga)
    print(get_mst(g))

    # g = [[0, 11, 1, 0], [2, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 0]]
    # a = is_cyclic(g)
    # print(a)
    # g1, n, o2n, m = contract_cycle(g, a)
    # print(g1, o2n, m)
    # mst = construct_full_mst(g1, n, o2n, m, g)
    # print(mst)


if __name__ == '__main__':
    test_graph()
