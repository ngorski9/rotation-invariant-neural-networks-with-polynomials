import itertools


def is_odd(x):
    "ha. ha."
    return x % 2 == 1


def odd_parity(tensor_orders):
    return is_odd(sum(tensor_orders))


def iter_tensor_terms(max_tensor_order, poly_order):
    tensor_orders = list(range(1, max_tensor_order + 1))

    for porder in range(2, poly_order + 1):
        for tensor_order_set in itertools.combinations_with_replacement(tensor_orders, r=porder):
            if odd_parity(tensor_order_set):
                continue
            yield tensor_order_set
    return


def iter_homogenous_terms(order, max_order=4):
    i = 2
    term = [order]
    while True:
        candidate = tuple(term * i)
        if not odd_parity(candidate):
            yield candidate
        if i == max_order:
            return
        i += 1


def iter_mixed_terms(o1, o2, max_order=4):
    o1, o2 = min(o1, o2), max(o1, o2)

    total_order = 2
    while True:
        for j in range(total_order - 1, 0, -1):
            candidate = tuple([o1] * j + [o2] * (total_order - j))
            if not odd_parity(candidate):
                yield candidate
        if total_order == max_order:
            return
        total_order += 1
