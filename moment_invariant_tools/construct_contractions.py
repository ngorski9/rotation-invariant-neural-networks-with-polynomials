import collections
import functools
import itertools
from .graph_check import GraphFilter


@functools.lru_cache
def is_closed_set(*indices_so_far):
    c = collections.Counter(indices_so_far)
    return set(c.values()) == set((2,))


def is_closed_pool(counter):
    # Takes advantage of counters only being 0,1,2.
    for k in counter.values():
        if k == 1:
            return False
    return True


def iter_valid_index_partition(index_pool, r):
    """
    Returns splits of first r values vs n-r values using combinations.
    Remembers not to repeat splits that are equal;
    Treats input as a multiset (order does not matter).
    First values must be valid indices for tensor.
    """

    seen_first = set()
    seen_twice = 0

    allowed_indices = [k for k, v in index_pool.items() if v == 1] + [k for k, v in index_pool.items() if v == 2]
    for first_vals in itertools.combinations(allowed_indices, r):
        first_vals = tuple(sorted(first_vals))
        if first_vals not in seen_first:
            seen_first.add(first_vals)
            for i in first_vals:
                index_pool[i] -= 1

            yield first_vals, index_pool
            for i in first_vals:
                index_pool[i] += 1
        else:
            seen_twice += 1
    if seen_twice:
        print("Seen twice val", seen_twice)


def get_next_indices(indices_so_far, index_pool, remaining_orders, gfilter):
    global full_count

    if not remaining_orders:
        yield indices_so_far
        print("got somewhere unexpected")
        return

    rank = remaining_orders[0]
    if indices_so_far:
        last_indices = indices_so_far[-1]
        last_rank = len(last_indices)
    else:
        last_rank = 0
        last_indices = ()  # empty tuple

    next_orders = remaining_orders[1:]
    for this_indices, next_pool in iter_valid_index_partition(index_pool, rank):
        # Note: next pool IS the index pool, iter_valid_index_partition operates in-place on it.
        if rank == last_rank:  # assumes rank-ordered term.
            if this_indices < last_indices:
                # when less, not lexicographic.
                # when equal, we have a closed set
                continue
        next_indices = indices_so_far + (this_indices,)

        if next_orders:
            if is_closed_pool(next_pool):
                continue

            if gfilter(next_indices):
                yield from get_next_indices(next_indices, next_pool, next_orders, gfilter)
        else:
            full_count += 1

            yield next_indices


def iter_ordered_index_sets(index_pool, ranks):
    # first term must start with 012...
    r0, *ranks = ranks

    if not all(index_pool[i] > 0 in index_pool for i in range(r0)):
        # In this case, the first thing has too high rank.
        # TODO: Add this constraint to term generators
        print("Unsatisfiable", r0, ranks)
        return
    for i in range(r0):
        index_pool[i] -= 1
    indices_so_far = (tuple(range(r0)),)

    gfilter = GraphFilter()

    yield from get_next_indices(indices_so_far, index_pool, ranks, gfilter)


def relabel_index_order_deep(index_order):
    """Relabel the indices so that new indices appear from 0...n"""
    index_map = {}
    counter = iter(itertools.count())
    result = []
    for group in index_order:
        group_result = []
        for i in group:
            if i not in index_map:
                # Number them as they come up
                index_map[i] = next(counter)
            group_result.append(index_map[i])
        result.append(tuple(sorted(group_result)))

    return tuple(result)


def has_noncanonical_order_deep(contraction):
    orders = [len(indices) for indices in contraction]
    i1 = contraction[0]
    o1 = len(i1)
    for i2 in contraction[1:]:
        o2 = len(i2)
        if tuple(sorted(i2)) != i2:
            print("broken within")
            return True
        if (o1 == o2) and (i2 < i1):
            print("not canonical between", i1, i2)
            return True
        o1, i1 = o2, i2

    else:
        return False


def is_disconnected(index_set):
    for r in range(len(index_set)):
        for combo in itertools.combinations(index_set, r=r):
            indices = [i for c in combo for i in c]
            if is_closed_set(*indices):
                return True
    else:
        return False


full_count = 0


def iter_indices(ranks):
    ranks = tuple(reversed(sorted(ranks)))
    max_index = (sum(ranks) + 1) // 2
    index_pool = collections.Counter()
    for k in range(max_index):
        index_pool[k] = 2
    global full_count
    full_count = 0

    visited = set()
    seen = 0
    yielded = 0

    for index_set in iter_ordered_index_sets(index_pool, ranks):

        index_set = relabel_index_order_deep(index_set)

        seen += 1
        if index_set in visited:
            continue
        visited.add(index_set)

        if is_disconnected(index_set):
            continue

        yielded += 1
        yield relabel_index_order_deep(tuple(reversed(index_set)))

    print("Term exhausted")
    print("yielded", yielded, "visited", len(visited), "seen", seen, "full count", full_count)
