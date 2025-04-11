import copy
import itertools
import torch

from .perform_contractions import perform_contraction_grad
from .construct_tensors import sample_reduced_tensor, sample_solid_tensor
from .construct_tensors import cartesian_irreducible_mapping, cartesian_solid_mapping
from .construct_contractions import iter_indices
from .construct_terms import iter_homogenous_terms, iter_mixed_terms


def expected_hom_invariants(rank):
    if rank in [0, 1]:
        return 1
    else:
        return 2 * rank - 2


def expected_mixed_invariants(rank1, rank2):
    rank1, rank2 = map(lambda x: x(rank1, rank2), [min, max])  # ha. ha. ha.
    if rank1 == 1:
        return 2
    if rank1 >= 1:
        return 3
    raise ValueError(f"Something went wrong with ranks: {rank1},{rank2}")


def test_invariants(grads, grad_map, rank_set):
    if len(grad_map) == 0:  # No invariants in the grad_map yet!
        return 1
    grad_matrix = build_grad_matrix(grad_map, rank_set)
    new_row = build_grad_row(grads, rank_set)
    grad_matrix = torch.cat([grad_matrix, new_row.unsqueeze(0)], dim=0)
    tol = None
    total_invariants = torch.linalg.matrix_rank(grad_matrix, atol=tol, rtol=tol).item()
    return total_invariants


def get_current_invariants(grad_map, rank_set):
    if not grad_map:
        return 0
    grad_matrix = build_grad_matrix(grad_map, rank_set)
    total_invariants = torch.linalg.matrix_rank(grad_matrix).item()
    return total_invariants


def build_grad_row(grads, rank_set):
    row = torch.cat([grads[r] for r in rank_set])
    return row


def build_grad_matrix(grad_map, rank_set):
    rows = [build_grad_row(g, rank_set) for g in grad_map.values()]
    grad_matrix = torch.stack(rows, dim=0)  # only works for this tensor order 2
    return grad_matrix


def setup_search(rank_set, spherical=True):

    if spherical:
        mapping = cartesian_irreducible_mapping
        sampler = sample_reduced_tensor
    else:
        mapping = cartesian_solid_mapping
        sampler = sample_solid_tensor

    grad_map = {}

    mapper_set = {r: mapping(r).to(torch.float64) for r in rank_set}

    tensor_set = {r: sampler(r).requires_grad_(True) for r in rank_set}
    return grad_map, mapper_set, tensor_set


def search_homogenous_invariants(
    grad_map, rank_set, max_order, tensor_set, mapper_set, one_invariant_per_term=True, skip_if_hom_found=True
):

    current_invariants = 0
    for r in rank_set:
        expected = expected_hom_invariants(r)
        start_invariants = get_current_invariants(grad_map, rank_set)

        print("Finding homogenous for", r, "Expect", expected)
        for term in iter_homogenous_terms(r, max_order):
            print("Start Term", term)

            for contraction in iter_indices(term):

                grads = perform_contraction_grad(term, contraction, tensor_set, mapper_set)

                new_total = test_invariants(grads, grad_map, rank_set)

                if new_total > current_invariants:
                    current_invariants = new_total
                    grad_map[term, contraction] = grads
                    print("Found one!", term, contraction)
                    if one_invariant_per_term:
                        break
                    if skip_if_hom_found and current_invariants - start_invariants == expected:
                        break

            if skip_if_hom_found and current_invariants - start_invariants == expected:
                break


def search_inhomogenous_invariants(
    grad_map, rank_set, tensor_set, mapper_set, max_order, one_invariant_per_term=True, skip_if_mixed_found=True
):
    current_invariants = get_current_invariants(grad_map, rank_set)
    for rank_pair in itertools.combinations(rank_set, 2):
        print("Searching mixed:", rank_pair)
        cur_mixed_invariants = 0
        num_expected = expected_mixed_invariants(*rank_pair)

        for term in iter_mixed_terms(*rank_pair, max_order):
            print("Start Term", term)

            for contraction in iter_indices(term):
                grads = perform_contraction_grad(term, contraction, tensor_set, mapper_set)

                new_total = test_invariants(grads, grad_map, rank_set)

                if new_total > current_invariants:
                    current_invariants = new_total
                    grad_map[term, contraction] = grads
                    print("Found one!", term, contraction)
                    cur_mixed_invariants += 1
                    if one_invariant_per_term:
                        break
            if skip_if_mixed_found and cur_mixed_invariants == num_expected:
                break


def report_results(grad_map, rank_set, max_order):
    row_indices = {k: i for i, k in enumerate(grad_map.keys())}
    contract_from_row = {v: k for k, v in row_indices.items()}

    grad_matrix = torch.stack(
        [torch.cat([v[r] for r in rank_set]) for v in grad_map.values()], axis=0
    )  # only works for this tensor order 2

    print("input orders:", rank_set)
    print("maximum factors:", max_order)

    prev_rank = 0
    this_rank = 0
    for i in range(grad_matrix.shape[0]):
        this_rank = torch.linalg.matrix_rank(grad_matrix[: i + 1]).item()
        ranks, indices = contract_from_row[i]
        independent = this_rank > prev_rank

        if len(set(ranks)) == 1:
            type_ = f"hom. {ranks[0]}"
        else:
            type_ = f"mixed: {tuple(set(ranks))}"

        contract_sig = " ".join(["".join([str(y) for y in x]) for x in indices])

        print(i, independent, type_, ranks, contract_sig)  # because [:1] means using 0 only
        prev_rank = this_rank
    print("Found total independent invariants:", this_rank)


def perform_search(rank_set, max_order, one_invariant_per_term=True, skip_if_mixed_found=True, skip_if_hom_found=True):

    grad_map, mapper_set, tensor_set = setup_search(rank_set)

    search_homogenous_invariants(
        grad_map,
        rank_set,
        max_order,
        tensor_set,
        mapper_set,
        one_invariant_per_term=one_invariant_per_term,
        skip_if_hom_found=skip_if_hom_found,
    )
    search_inhomogenous_invariants(
        grad_map,
        rank_set,
        tensor_set,
        mapper_set,
        max_order,
        one_invariant_per_term=one_invariant_per_term,
        skip_if_mixed_found=skip_if_mixed_found,
    )

    report_results(grad_map, rank_set, max_order)
    return grad_map


def perform_independent_searches(rank_set, max_order, one_invariant_per_term=True, skip_if_mixed_found=True, skip_if_hom_found=True):

    full_grad_map, mapper_set, tensor_set = setup_search(rank_set)

    for r in rank_set:
        grad_map = {}
        search_homogenous_invariants(
            grad_map,
            {r},
            max_order,
            tensor_set,
            mapper_set,
            one_invariant_per_term=one_invariant_per_term,
            skip_if_hom_found=skip_if_hom_found,
        )
        full_grad_map.update(grad_map)

    for rank_pair in itertools.combinations(rank_set, 2):
        rank_pair = set(rank_pair)
        grad_map = {}
        search_inhomogenous_invariants(
            grad_map,
            rank_pair,
            tensor_set,
            mapper_set,
            max_order,
            one_invariant_per_term=one_invariant_per_term,
            skip_if_mixed_found=skip_if_mixed_found,
        )
        full_grad_map.update(grad_map)

    report_results(full_grad_map, rank_set, max_order)
    return full_grad_map


def perform_overcomplete_search(rank_set, max_order):

    full_grad_map, mapper_set, tensor_set = setup_search(rank_set)

    rank_grad_maps = {}
    for r in rank_set:
        grad_map = {}
        search_homogenous_invariants(grad_map, {r}, max_order, tensor_set, mapper_set, one_invariant_per_term=False, skip_if_hom_found=True)
        full_grad_map.update(grad_map)
        rank_grad_maps[r] = grad_map

    for rank_pair in itertools.combinations(rank_set, 2):
        rank_pair = set(rank_pair)
        r0, r1 = rank_pair
        grad_map = {**rank_grad_maps[r0], **rank_grad_maps[r1]}
        search_inhomogenous_invariants(
            grad_map, rank_pair, tensor_set, mapper_set, max_order, one_invariant_per_term=False, skip_if_mixed_found=True
        )
        full_grad_map.update(grad_map)

    report_results(full_grad_map, rank_set, max_order)
    return full_grad_map


def search_term_iterable(terms):

    terms = list(terms)
    rank_set = set()
    max_order = 0
    for t in terms:
        rank_set.update(t)
        max_order = max(max_order, len(t))

    current_invariants = 0
    grad_map, mapper_set, tensor_set = setup_search(rank_set)
    for term in terms:
        print("Searching", term)

        for contraction in iter_indices(term):
            grads = perform_contraction_grad(term, contraction, tensor_set, mapper_set)

            new_total = test_invariants(grads, grad_map, rank_set)

            if new_total > current_invariants:
                current_invariants = new_total
                grad_map[term, contraction] = grads
                print("Found one!", term, contraction)

    report_results(grad_map, rank_set, max_order)
    return grad_map


def test_independence(contractions, spherical=True):
    all_ranks = set(len(x) for c in contractions for x in c)

    grad_map, mapper_set, tensor_set = setup_search(all_ranks, spherical=spherical)

    for c in contractions:
        this_ranks = tuple(len(t) for t in c)
        grads = perform_contraction_grad(this_ranks, c, tensor_set, mapper_set)
        grad_map[this_ranks, c] = grads

    report_results(grad_map, all_ranks, max_order="Not given")
    return grad_map
