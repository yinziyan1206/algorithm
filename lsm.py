from typing import List

import internal


def __create_arr(group: List, times=0):
    res = []
    for i in group:
        t = 1
        for j in range(0, times):
            t *= i
        res.append(t)
    return res


def __sum(group: List, const: List, times=0):
    total = 0
    for i in range(len(group)):
        g = group[i]
        t = const[i]
        for j in range(times):
            t *= g
        total += t
    return total


def cal_result(left: List, right: List):
    result = []
    for i in range(len(left)):
        total = 0
        for r in range(len(result)):
            total += result[r] * left[i][len(left) - r - 1]
        try:
            t = (right[i] - total) / left[i][len(left) - i - 1]
        except ZeroDivisionError:
            raise ZeroDivisionError('delta is zero')
        result.append(t)
    return result


def calculate(x, y, times=1):
    """
        least square method main function
        >>> calculate([[1, 2, 3], [4, 5, 6]], [5, 7, 12], times=1))
    """
    is_equal = True
    if len(x) > 0:
        for x_t in x:
            if len(x_t) != len(x[0]):
                is_equal = False
                break
        if len(x[0]) != len(y):
            is_equal = False

        if is_equal:
            m = len(x[0])
            t = len(x)
            const = [1] * m
            const_x = [const]
            for i in range(0, t):
                for ti in range(times):
                    const_x.append(__create_arr(x[i], ti + 1))

            left = []
            for i in range(len(const_x)):
                left_temp = [__sum(const_x[i], const, 1)]
                for j in range(t):
                    left_temp.extend([__sum(x[j], const_x[i], ti) for ti in range(1, times + 1)])
                left.append(left_temp)

            right = [__sum(const_x[k], y, 1) for k in range(len(const_x))]
            left, right = internal.gaussian(left, right)
            res = cal_result(left, right)
            return res
    return []
