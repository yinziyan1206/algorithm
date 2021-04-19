"""
    basic sort method
"""
from copy import deepcopy


def to_tree(data: list, root=-1, sid='sid', pid='pid'):
    """
        make list to tree structure
            >>> data = [{'sid': 1, 'pid': -1}, {'sid': 2, 'pid': 1}]
            >>> to_tree(data)
            <list: [{'sid': 1, 'pid': -1, 'child': [{'sid': 2, 'pid': 1}]}
    """
    res = list()

    def get_leaf(item, parent, heap):
        item['child'] = list()
        local = deepcopy(heap)
        for h in heap:
            if h['pid'] == parent:
                local.remove(h)
                get_leaf(h, h[sid], local)
                item['child'].append(h)
        del local

    temp = deepcopy(data)
    for node in data:
        if node[pid] == root:
            temp.remove(node)
            node['child'] = list()
            get_leaf(node, node[sid], temp)
            res.append(node)
    del temp

    return res


def filter_leaf_path(data: list, foot: list, sid='sid', pid='pid'):
    """
        make list to tree structure
            >>> data = [{'sid': 1, 'pid': -1}, {'sid': 2, 'pid': 1}]
            >>> to_tree(data)
            <list: [{'sid': 1, 'pid': -1, 'child': [{'sid': 2, 'pid': 1}]}
    """
    res = list()

    def get_parent(item, leaf, heap):
        for h in heap:
            if h[sid] == leaf:
                if 'child' not in h.keys():
                    h['child'] = list()
                if item not in h['child']:
                    h['child'].append(item)
                out = get_parent(h, h[pid], heap)
                return out
        return item

    temp = deepcopy(data)
    for node in data:
        if node[sid] in foot:
            temp.remove(node)
            node['child'] = list()
            r = get_parent(node, node[pid], temp)
            if r not in res:
                res.append(r)
    del temp
    return res


def gaussian(left, right):
    for s in range(len(right) - 1):
        left, right = __gaussian_step(left, right, s)
    return left, right


def __gaussian_step(left, right, start):
    for i in range(len(left) - 1 - start):
        alpha = left[i][start] / left[i + 1][start]
        for j in range(len(left[i])):
            left[i][j] = left[i][j] - left[i + 1][j] * alpha
        right[i] = right[i] - right[i + 1] * alpha
    return left, right
