import itertools
import collections
import torch

ndim = 3  # number of spatial dimensions


def trace(tensor, axis_1, axis_2):
    return torch.diagonal(tensor, 0, axis_1, axis_2).sum(dim=-1)


def tpl(int_tensor):
    return tuple(x.item() for x in int_tensor.unbind())


def build_full_mapping(ind_order):
    # First, construct the map from the full cartesian space to a symmetric basis,
    # mapping each element from the cartesian space to the sorted version of its indices.
    input_ind = []
    output_ind = []
    for ind in itertools.product(range(ndim), repeat=ind_order):
        input_ind.append(ind)
        output_ind.append(tuple(sorted(ind)))
    out_ind = torch.as_tensor(output_ind)
    in_ind = torch.as_tensor(input_ind)

    # List of symmetric indices only.
    needed_ind = torch.unique(out_ind, dim=0)

    # Figure out numbering for nonsymmetric indices within the symmetric ones.
    # # This step is of somewhat large size, could maybe be reduced with better algorithm.
    equal = (needed_ind == out_ind.unsqueeze(1)).all(dim=2)
    out_order = torch.where(equal)[1]

    # build sparse matrix -> all values are 1
    ind_comb = torch.stack(
        (torch.arange(len(out_order)), out_order),
    )
    xform = torch.sparse_coo_tensor(ind_comb, torch.ones(len(out_order)), dtype=torch.int64).to_dense()
    # This maps symmetric space to d^order space
    # shape (d^o, t(o)) where t(o) is the triangular number.

    # Now we need to build the map from traceless symmetric tensors to symmetric ones.

    # Which indices are constrainted by tracelessness
    # Constrained elements are ones that end with 2,2 in the full cartesian space.
    # # Again, may use more memory than needed here.
    constrained_elements = (needed_ind[:, -2:] == torch.as_tensor([2, 2])).all(dim=-1)
    constrained_pos = torch.where(constrained_elements)[0]
    # Set of indices which are constrained by trace.
    constrained_ind = needed_ind[constrained_elements]
    # print("constrained_ind",constrained_ind)

    # Which indices are not constrained
    free_elements = ~constrained_elements
    free_pos = torch.where(free_elements)[0]

    if ind_order == 1:
        # Exception to above which is wrong when only one dimension
        constrained_ind = []
        free_elements = [True, True, True]

    # Now we use regular python b/c it is easier to construct these loops.

    # Map from the index space to the symmetric basis number.
    bare_sym_map = {tpl(x): i for i, x in enumerate(needed_ind.unbind(0))}
    # print("bare sym map",bare_sym_map)
    # Map from the index space to traceless symmetric baseless number, but only for values
    # that are in both (i.e. not affected by tracelessness)
    bare_traceless_map = {tpl(x): i for i, x in enumerate(needed_ind[free_elements].unbind(0))}

    # Map from symetric indices to traceless ones, but onyl for values that are in both.
    sym_traceless_map = {bare_sym_map[x]: j for x, j in bare_traceless_map.items()}

    # Initialize full map in "csr" form based on unconstrained elements.
    sym_traceless_csr = collections.defaultdict(list)
    for k, v in sym_traceless_map.items():
        sym_traceless_csr[k].append((v, 1))

    # Extend map for trace-constrained elements
    for x in map(tpl, constrained_ind):
        x_sym = bare_sym_map[x]
        # print(x,x_sym)
        new_sym_vals = []
        # Get the two other components x...00 and x_...11 (constraint is on x...22)
        for k in (0, 1):
            a = tuple(sorted(x[:-2] + (k, k)))
            aa = bare_sym_map[a]
            new_sym_vals.append(aa)

        # Writ this is in terms of
        new_traceless_vals = collections.Counter()
        for aa in new_sym_vals:
            for aaa, v in sym_traceless_csr[aa]:
                new_traceless_vals[aaa] += -v
        new_traceless_vals = [(k, v) for k, v in new_traceless_vals.items()]
        sym_traceless_csr[x_sym].extend(new_traceless_vals)

    # Rewrite in coo form
    sym_traceless_coo = collections.Counter()
    for i, row in sym_traceless_csr.items():
        for j, val in row:
            sym_traceless_coo[i, j] += val

    # Convert into pytorch tensor
    ind, vals = zip(*list(sym_traceless_coo.items()))
    ind = torch.as_tensor(ind).T
    vals = torch.as_tensor(vals)
    xform2 = torch.sparse_coo_tensor(ind, vals, size=(xform.shape[1], 2 * ind_order + 1))

    # The full map is simply the map (cartesian index, symmetric) @ (symmetric, sym. traceless)
    full_xform = xform @ xform2

    # unflatten cartesian set.
    full_xform = full_xform.reshape(*((ndim,) * ind_order), -1)
    xform = xform.reshape(*((ndim,) * ind_order), -1)
    return full_xform, xform, xform2


def cartesian_irreducible_mapping(ind_order):

    full_xform, cartesian_symmetric, symmetric_traceless = build_full_mapping(ind_order)

    return full_xform


def cartesian_solid_mapping(ind_order):

    full_xform, cartesian_symmetric, symmetric_traceless = build_full_mapping(ind_order)

    return cartesian_symmetric


def get_order(tensor):
    return len(tensor.shape)


def check_traceless(tensor):
    tensor_order = get_order(tensor)

    for i, j in itertools.combinations(range(tensor_order), r=2):
        close = torch.allclose(trace(tensor, 0, 1), torch.as_tensor(0, dtype=tensor.dtype))
        if not close:
            return False
    # This else statement is a joke, it is not needed and confuses the way this function works.
    else:
        return True


def check_symmetric(tensor):
    tensor_order = get_order(tensor)
    for p in itertools.permutations(range(tensor_order), r=tensor_order):
        close = torch.allclose(tensor.permute(p), tensor)
        if not close:
            return False
    # This else statement is a joke, it is not needed and confuses the way this function works.
    else:
        return True


def sample_reduced_tensor(rank, dtype=torch.float64):
    rand_vals = torch.rand(2 * rank + 1, dtype=dtype)
    return rand_vals


def sample_solid_tensor(rank, dtype=torch.float64):
    size = (rank + 1) * (rank + 2) / 2
    size = int(size)
    rand_vals = torch.rand(size, dtype=dtype)
    return rand_vals


def sample_tensor(map_, dtype=torch.float64):
    n_components = map_.shape[-1]
    rand_vals = torch.rand(n_components, dtype=dtype)
    return map_.to(dtype) @ rand_vals
