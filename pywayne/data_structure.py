# !/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: Wang Ye (Wayne)
@file: data_structure.py
@time: 2022/03/01
@contact: wangye@oppo.com
@site: 
@software: PyCharm
# code is far away from bugs.
"""

import xml.etree.ElementTree as ET


class ConditionTree:
    """
    逻辑条件树，用来存储逻辑关系的树

    Author:   wangye
    Datetime: 2019/7/31 16:17
    """

    def __init__(self, tag: str):
        self.tag = tag
        self.children = []
        self.attribute = dict()
        self.text = ''

    def append_by_path(self, path: list):
        """
        从路径导入
        :param path:
        :return:
        """
        if not path: return
        p = path[0]
        for ch in self.children:
            if p['tag'] == ch.tag:
                ch.append_by_path(path[1:])
                return
        child = ConditionTree(p['tag'])
        child.attribute = p['attrib']
        child.text = p['text']
        self.children.append(child)
        self.children[-1].append_by_path(path[1:])

    def find(self, nick_name: str) -> 'ConditionTree':
        if not self.children: return None
        for ch in self.children:
            if ch.tag == nick_name:
                return ch
            res = ch.find(nick_name)
            if res: return res
        return None

    def find_by_path(self, path: list) -> 'ConditionTree':
        cur = self
        while path:
            p = path.pop(0)
            for ch in cur.children:
                if p == ch.tag:
                    cur = ch
                    if not path:
                        return cur
                    break
            else:
                return None

    def print_path(self):
        def helper(_tree):
            if not _tree: return []
            if not _tree.children and _tree.text:
                return [[_tree.tag + ': ' + _tree.text]]
            return [[_tree.tag] + res for ch in _tree.children for res in helper(ch)]

        print('-*-*-*-*-*-*- start print tree -*-*-*-*-*-*-')
        [print(' -> '.join(h)) for h in helper(self)]
        print('-*-*-*-*-*-*- end print tree -*-*-*-*-*-*-')


class UnionFind:
    """An implementation of union find data structure.
    It uses weighted quick union by rank with path compression.

    This module implements an union find or disjoint set data structure.

    An union find data structure can keep track of a set of elements into a number
    of disjoint (non overlapping) subsets. That is why it is also known as the
    disjoint set data structure. Mainly two useful operations on such a data
    structure can be performed. A *find* operation determines which subset a
    particular element is in. This can be used for determining if two
    elements are in the same subset. An *union* Join two subsets into a
    single subset.

    The complexity of these two operations depend on the particular implementation.
    It is possible to achieve constant time (O(1)) for any one of those operations
    while the operation is penalized. A balance between the complexities of these
    two operations is desirable and achievable following two enhancements:

    1.  Using union by rank -- always attach the smaller tree to the root of the
        larger tree.
    2.  Using path compression -- flattening the structure of the tree whenever
        find is used on it.
    """

    def __init__(self, N):
        """Initialize an empty union find object with N items.

        Args:
            N: Number of items in the union find object.
        """

        self._id = list(range(N))
        self._count = N
        self._rank = [0] * N

    def find(self, p):
        """Find the set identifier for the item p."""

        id = self._id
        while p != id[p]:
            p = id[p] = id[id[p]]  # Path compression using halving.
        return p

    def count(self):
        """Return the number of items."""

        return self._count

    def connected(self, p, q):
        """Check if the items p and q are on the same set or not."""

        return self.find(p) == self.find(q)

    def union(self, p, q):
        """Combine sets containing p and q into a single set."""

        id = self._id
        rank = self._rank

        i = self.find(p)
        j = self.find(q)
        if i == j:
            return

        self._count -= 1
        if rank[i] < rank[j]:
            id[i] = j
        elif rank[i] > rank[j]:
            id[j] = i
        else:
            id[j] = i
            rank[i] += 1

    def __str__(self):
        """String representation of the union find object."""
        return " ".join([str(x) for x in self._id])

    def __repr__(self):
        """Representation of the union find object."""
        return "UF(" + str(self) + ")"


class XmlIO:
    """
    用于xml文件的读取，返回ConditionTree数据结构

    Author:   wangye
    Datetime: 2019/7/31 16:24
    """

    def __init__(self, file_read='', file_write=''):
        self.file_read = file_read
        self.file_write = file_write

    def read(self) -> ConditionTree:
        tree = ET.parse(self.file_read)
        root = tree.getroot()

        def helper(_tree: ET.Element, _is_root=True):
            if not _tree: return [[{'tag': _tree.tag, 'attrib': _tree.attrib, 'text': _tree.text.strip('\t\n') if _tree.text else ''}]]
            ret = [] if _is_root else [{'tag': _tree.tag, 'attrib': _tree.attrib, 'text': _tree.text.strip('\t\n') if _tree.text else ''}]
            return [ret + res for ch in _tree for res in helper(ch, False)]

        c_tree = ConditionTree(root.tag)
        c_tree.attribute = root.attrib
        c_tree.text = root.text.strip('\t\n')
        [c_tree.append_by_path(p) for p in helper(root)]
        # c_tree.print_path()
        return c_tree

    def write(self, root_name, tree: ConditionTree):
        _tree = ET.ElementTree()
        _root = ET.Element(root_name)
        _tree._setroot(_root)

        def helper(etree, c_tree):
            if not c_tree.children and c_tree.attribute:
                for k, v in c_tree.attribute.items():
                    ET.SubElement(etree, k).text = v
            else:
                for ch in c_tree.children:
                    son = ET.SubElement(etree, ch.tag)
                    helper(son, ch)

        helper(_root, tree)
        self._indent(_root)
        _tree.write(self.file_write, 'utf-8', True)

    def _indent(self, elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
