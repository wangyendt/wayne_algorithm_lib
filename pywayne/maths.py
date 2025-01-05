# !/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: Wang Ye (Wayne)
@file: maths.py
@time: 2022/03/01
@contact: wangye@oppo.com
@site: 
@software: PyCharm
# code is far away from bugs.
"""

import functools


def get_all_factors(n: int) -> list:
    """
    Return all factors of positive integer n.

    Author:   wangye
    Datetime: 2019/7/16 16:00

    :param n: A positive number
    :return: a list which contains all factors of number n
    """
    return list(set(functools.reduce(list.__add__, ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0))))


def digitCount(n, k):
    """
    Count the number of occurrences of digit k from 1 to n.
    Author:   wangye
    Datetime: 2019/7/18 14:49

    :param n:
    :param k:
    :return: The count.
    """
    N, ret, dig = n, 0, 1
    while n >= 1:
        m, r = divmod(n, 10)
        if r > k:
            ret += (m + 1) * dig
        elif r < k:
            ret += m * dig
        elif r == k:
            ret += m * dig + (N - n * dig + 1)
        n //= 10
        dig *= 10
    if k == 0:
        if N == 0:
            return 1
        else:
            return ret - dig // 10
    return ret


def karatsuba_multiplication(x, y):
    """Multiply two integers using Karatsuba's algorithm."""

    # convert to strings for easy access to digits
    def zero_pad(number_string, zeros, left=True):
        """Return the string with zeros added to the left or right."""
        for i in range(zeros):
            if left:
                number_string = '0' + number_string
            else:
                number_string = number_string + '0'
        return number_string

    x = str(x)
    y = str(y)
    # base case for recursion
    if len(x) == 1 and len(y) == 1:
        return int(x) * int(y)
    if len(x) < len(y):
        x = zero_pad(x, len(y) - len(x))
    elif len(y) < len(x):
        y = zero_pad(y, len(x) - len(y))
    n = len(x)
    j = n // 2
    # for odd digit integers
    if (n % 2) != 0:
        j += 1
    BZeroPadding = n - j
    AZeroPadding = BZeroPadding * 2
    a = int(x[:j])
    b = int(x[j:])
    c = int(y[:j])
    d = int(y[j:])
    # recursively calculate
    ac = karatsuba_multiplication(a, c)
    bd = karatsuba_multiplication(b, d)
    k = karatsuba_multiplication(a + b, c + d)
    A = int(zero_pad(str(ac), AZeroPadding, False))
    B = int(zero_pad(str(k - ac - bd), BZeroPadding, False))
    return A + B + bd
