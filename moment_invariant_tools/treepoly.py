"""
Experimental strategies for fast sparse polynomial evaluation.
"""
import torch
import networkx
import itertools


def uniqify_ind_coeff(ind, vals):
    print("uniqifying!")
    indsort = ind.sort(axis=0)
    uniq = torch.unique(indsort.values, dim=1, return_inverse=True, return_counts=True)
    uniques, inverses, counts = uniq
    uniq_ind = uniques
    out_coeffs = torch.zeros(uniques.shape[1], dtype=vals.dtype)
    out_coeffs = out_coeffs.index_add(0, inverses, vals)
    uniq_coeff = out_coeffs
    where_coeff_0 = uniq_coeff == 0
    uniq_coeff = uniq_coeff[~where_coeff_0]
    uniq_ind = uniq_ind[:, ~where_coeff_0]
    print("Reduced from", ind.shape[1], "to", uniq_ind.shape[1])
    return uniq_ind, uniq_coeff


def remove_index_factor(term_indices, best_term):
    dtype = term_indices.dtype
    dev = term_indices.device
    list_vals = term_indices.T.tolist()

    new_list = []
    for term in list_vals:
        removed = False
        new_term = []
        for i in term:
            if not removed:
                if i == best_term:
                    removed = True
                    continue
            new_term.append(i)
        if not removed:
            raise ValueError(f"No value {best_term} to remove from {term}.")
        new_list.append(new_term)
    new_ind = torch.as_tensor(new_list, dtype=dtype, device=dev).T
    return new_ind


def get_factor_counts(indices):
    equal_ind = indices == torch.arange(indices.max() + 1).unsqueeze(1).unsqueeze(1)
    any_across_multiindex = equal_ind.any(axis=1)
    total_nonzero_counts = any_across_multiindex.sum(axis=1)
    return total_nonzero_counts


def get_new_term(indices, coefficients):
    # print(indices)
    ndim = indices.shape[0]
    counts = get_factor_counts(indices)
    best_term = counts.max(axis=0).indices
    # print("got best factor",best_term,counts)

    where_term = (indices == best_term).any(axis=0)

    term_coeff = coefficients[where_term].sum()  # bad math, no math this way.
    term_indices = indices[:, where_term]

    term_indices = remove_index_factor(term_indices, best_term)

    new_term = (
        best_term,
        None,  #
        term_indices,
        coefficients[where_term],
    )
    # torch.ones(term_indices.shape[1],dtype=term_indices.dtype))

    where_not_term = ~where_term

    remaining_indices = indices[:, where_not_term]
    remaining_coeffs = coefficients[where_not_term]

    return new_term, remaining_indices, remaining_coeffs


def make_subterm(indices, coeffs):
    if indices.shape[0] < 2:
        return Monomial(indices, coeffs)
    return TreePoly(indices, coeffs)


class TreePoly:
    def __init__(self, indices, coefficients):
        if indices.shape[0] < 2:
            raise ValueError("TreePoly is not for linear terms")
        print("NEW TREEPOLY")
        print("COEFFICIENTS", coefficients)
        print("INDICES", indices)

        current_indices = indices
        current_coeff = coefficients

        terms = []
        while current_indices.shape[1] > 0:
            print("iteration", current_indices, current_coeff)
            term, remaining_indices, remaining_coeff = get_new_term(current_indices, current_coeff)
            print("identified component", term[0], len(current_coeff) - len(remaining_coeff), len(current_coeff))
            terms.append(term)
            current_indices = remaining_indices
            current_coeff = remaining_coeff

        self.terms = []
        for t in terms:
            # create the terms themselves.
            lead_factor, lead_coeff, subterm_indices, subterm_coeffs = t
            # print("subindices",subterm_indices,subterm_coeffs)

            subterm = make_subterm(subterm_indices, subterm_coeffs)
            t = lead_factor, lead_coeff, subterm
            self.terms.append(t)
        self.children = [t[-1] for t in self.terms]
        self.indices = torch.stack([t[0] for t in self.terms])
        # self.terms = terms

    def execute(self, inputs):
        term_vals = []
        # print('in terms:',self.terms)
        for factor, coeff, subterm in self.terms:
            # print("factor",factor,coeff,subterm)
            subterm_value = subterm.execute(inputs)
            # print("subterm value",subterm_value.shape)
            term_value = inputs[:, factor] * subterm_value  # .squeeze()#*coeff
            # print("Term value",term_value.shape)
            term_vals.append(term_value)
        return sum(term_vals)

    def rebuild_indices(self):
        indices = []
        coeffs = []
        for factor, coeff, subterm in self.terms:
            subind, subcoeff = subterm.rebuild_indices()
            expfact = factor.expand((1, subind.shape[1]))
            subind = torch.cat([expfact, subind])

            indices.append(subind)
            coeffs.append(subcoeff)
        indices = torch.cat(indices, dim=1)
        coeff = torch.cat(coeffs)
        return indices, coeff

    def add_to_graph(self, g, parent, node_counter):
        self_count = next(node_counter)
        g.add_node(self_count, label=f"P{self_count}")
        if parent is not None:
            g.add_edge(parent, self_count)
        for f, c, subterm in self.terms:
            subterm.add_to_graph(g, self_count, node_counter)

    def make_graph(self):
        g = networkx.Graph()
        counter = iter(itertools.count())
        self.add_to_graph(g, None, counter)
        return g

    def collect_indices(self):

        node_lists = [[(0, self)]]
        needs_next = True
        while needs_next:
            print("Rounds")

            last_level = node_lists[-1]
            next_level = [(i, c) for i, (ii, n) in enumerate(last_level) for c in n.children]
            node_lists.append(next_level)
            if isinstance(next_level[0][1], Monomial):
                needs_next = False

        label_lists = [[num for num, node in level_list] for level_list in node_lists]

        last_labels = [i for i, (jjj, m) in enumerate(node_lists[-1]) for _ in range(len(m.coefficients))]
        label_lists.append(last_labels)

        index_lists = [[n.indices for i, n in this_level] for this_level in node_lists]
        # print("index lengths",[len(x) for x in index_lists])
        indices = [torch.cat(ind) for ind in index_lists]

        #         for i, ind in enumerate(index_lists):
        #             print('index list',i)
        #             torch.cat(ind)
        #         #last_indices = torch.cat([m.indices] for i,m in node_lists[-1])

        coeffs = torch.cat([m.coefficients for i, m in node_lists[-1]])
        return label_lists, coeffs, indices


#  1  procedure BFS(G, root) is # Breadth first search
#  2      let Q be a queue
#  3      label root as explored
#  4      Q.enqueue(root)
#  5      while Q is not empty do
#  6          v := Q.dequeue()
#  7          if v is the goal then
#  8              return v
#  9          for all edges from v to w in G.adjacentEdges(v) do
# 10              if w is not labeled as explored then
# 11                  label w as explored
# 12                  w.parent := v
# 13                  Q.enqueue(w)


class Monomial:
    def __init__(self, indices, coefficients):
        self.indices = indices.squeeze(0)
        self.coefficients = coefficients.to(torch.float64)
        print("MONOMIAL indices, coefficients", indices, coefficients)

    def execute(self, inputs):
        out = inputs[:, self.indices] @ self.coefficients
        # print("monomial out",out.shape)
        return out

    def add_to_graph(self, g, parent, node_counter):
        self_count = next(node_counter)
        g.add_node(self_count, label=f"M{len(self.coefficients)}")
        if parent is not None:
            g.add_edge(parent, self_count)

    def rebuild_indices(self):
        return self.indices, self.coefficients
