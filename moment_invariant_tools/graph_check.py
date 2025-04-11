import collections

import networkx
from networkx.algorithms.isomorphism import categorical_node_match, categorical_edge_match, is_isomorphic


def draw_graph(g, pos=None, draw_node_kwargs=None, draw_edge_kwargs=None):
    import networkx as nx
    import matplotlib.pyplot as plt

    if draw_node_kwargs is None:
        draw_node_kwargs = {}
    if draw_edge_kwargs is None:
        draw_edge_kwargs = {}

    if pos is None:
        pos = nx.spring_layout(g)

    # Draw Nodes
    nx.draw_networkx_nodes(g, pos, node_color='black', **draw_node_kwargs)

    # Draw Edges (excluding self-loops)
    nx.draw_networkx_edges(g, pos, edgelist=[(u, v) for u, v in g.edges() if u != v], **draw_edge_kwargs)

    # Draw Labels
    nx.draw_networkx_labels(g, pos, labels={k: v['rank'] for k, v in g.nodes.items()}, font_color='white', font_weight='bold')

    # Draw Edge Labels (excluding self-loops)
    edge_labels = {(u, v): d['weight'] for u, v, d in g.edges(data=True) if u != v}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

    # Equal Aspect Ratio
    plt.axis('equal')

# def graph_from_contraction(contraction):
#     g = networkx.Graph()
#     g.add_nodes_from(range(len(contraction)))
#     edges = collections.defaultdict(set)
#     for node_i, ind in enumerate(contraction):
#         g.add_node(node_i, rank=len(ind))
#         for edge in ind:
#             edges[edge].add(node_i)
#
#     edge_weights = collections.Counter(tuple(e) for e in edges.values() if len(e) == 2)  # if not 2, dangling edges
#     edge_weights = [(*k, v) for k, v in edge_weights.items()]
#     g.add_weighted_edges_from(edge_weights)
#     return g


def graph_from_contraction(contraction):
    g = networkx.Graph()
    g.add_nodes_from(range(len(contraction)))
    edges = collections.defaultdict(list)
    for node_i, ind in enumerate(contraction):
        g.add_node(node_i, rank=len(ind))
        for edge in ind:
            edges[edge].append(node_i)

    edge_weights = collections.Counter(tuple(e) for e in edges.values() if len(e) == 2)  # if not 2, dangling edges
    edge_weights = [(*k, v) for k, v in edge_weights.items()]
    g.add_weighted_edges_from(edge_weights)

    # Add self-loops where a node is connected to itself
    for node, neighbors in edges.items():
        if len(neighbors) == 1 and next(iter(neighbors)) == node:
            g.add_edge(node, node, weight=1)

    return g


class GraphFilter:
    def __init__(self):
        self.hashes = {}

    def __call__(self, contraction):
        graph = graph_from_contraction(contraction)

        hash = networkx.weisfeiler_lehman_graph_hash(graph, edge_attr="weight", node_attr="rank", iterations=5, digest_size=32)

        hash_set = self.hashes.get(hash, [])
        if not hash_set:
            self.hashes[hash] = [graph]
            return True

        node_match = categorical_node_match("rank", None)
        edge_match = categorical_edge_match("weight", None)

        for other_graph in hash_set:
            if is_isomorphic(graph, other_graph, node_match=node_match, edge_match=edge_match):
                return False
        else:
            hash_set.append(graph)
            self.hashes[hash] = hash_set
        return True


