import torch

from .construct_tensors import cartesian_irreducible_mapping


def contract(tensors, indices):

    args = list(zip(tensors, indices))
    args = [x for y in args for x in y]

    return torch.einsum(*args)


def contraction_mapping(mappers, indices):
    ranks = [len(m.shape) - 1 for m in mappers]
    start_rank_indexes = sum(ranks)  # conservatively, should start 1/2 lower and be fine.
    new_indices = [(*ind, r) for r, ind in enumerate(indices, start=start_rank_indexes)]
    out_indices = [ind[-1] for ind in new_indices]

    args = [(m, ind) for m, ind in zip(mappers, new_indices)]
    args = [i for t in args for i in t]

    return torch.einsum(*args, out_indices)


def make_contraction_map(indices):
    ranks = [len(i) for i in indices]
    rank_set = set(ranks)
    mappers = {r: cartesian_irreducible_mapping(r) for r in rank_set}

    return contraction_mapping([mappers[r] for r in ranks], indices)


def perform_contraction_grad(ranks, contraction, tensor_set, mapper_set):

    input_tensors = [mapper_set[r] @ tensor_set[r] for r in ranks]

    output = contract(input_tensors, contraction)

    contraction_grad_values = torch.autograd.grad(output, tensor_set.values(), allow_unused=True)

    contraction_grad_dict = {k: v for k, v in zip(tensor_set.keys(), contraction_grad_values)}
    # Substitute none for zeros.
    for k, v in contraction_grad_dict.items():
        if v is None:
            contraction_grad_dict[k] = torch.zeros_like(tensor_set[k])
    return contraction_grad_dict
