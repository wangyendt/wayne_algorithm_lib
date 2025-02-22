数据结构 (data_structure)
==========================

本模块提供了一些基础数据结构的实现，包括逻辑条件树 (ConditionTree)、并查集 (UnionFind) 以及 XML 文件读写辅助工具 (XmlIO)。这些数据结构在处理复杂条件、集合合并及 XML 数据管理时具有广泛应用。

ConditionTree 类
------------------

.. py:class:: ConditionTree(tag: str)

   逻辑条件树，用于存储和管理条件表达式的树形结构。该树结构适用于表达复杂的逻辑关系，支持通过路径添加节点、查找指定节点以及打印整棵树的路径。
   
   **方法**:

   - **__init__(self, tag: str)**
     
     初始化条件树节点，并为节点指定标签。
     
     **参数**:
       - **tag**: 节点的标签（字符串）。
     
     **示例**::

        tree = ConditionTree("root")

   - **append_by_path(self, path: list)**
     
     根据给定的路径列表添加节点。路径中每个元素应为包含 'tag'、'attrib' 和 'text' 的字典。
     
     **参数**:
       - **path**: 节点路径列表（列表）。
     
     **应用场景**:
       用于动态构建带有嵌套逻辑关系的树结构。
     
     **示例**::

        path = [{'tag': 'if', 'attrib': {}, 'text': '条件1'}]
        tree.append_by_path(path)

   - **find(self, nick_name: str) -> ConditionTree**
     
     在树中查找标签名称为 *nick_name* 的节点，返回找到的节点；若找不到，则返回 None。
     
     **参数**:
       - **nick_name**: 要查找的标签名称（字符串）。
     
     **示例**::

        result = tree.find("if")

   - **find_by_path(self, path: list) -> ConditionTree**
     
     根据给定的标签名称列表，沿路径逐层查找节点。
     
     **参数**:
       - **path**: 标签名称列表（列表）。
     
     **示例**::

        node = tree.find_by_path(["root", "if"])

   - **print_path(self)**
     
     打印树中所有从根到叶子的路径。
     
     **示例**::

        tree.print_path()

UnionFind 类
-------------

.. py:class:: UnionFind(N)

   并查集（Union-Find）数据结构实现，支持按秩合并与路径压缩。
   适用于管理不相交集合的问题，如网络连接、最小生成树等。
   
   **方法**:

   - **__init__(self, N)**
     
     初始化一个含有 N 个元素的并查集。
     
     **参数**:
       - **N**: 集合中元素的数量（整数）。
     
     **示例**::

        uf = UnionFind(10)

   - **find(self, p) -> int**
     
     查找元素 p 所在集合的代表。
     
     **参数**:
       - **p**: 待查询元素的索引。
     
     **返回**:
       - 元素 p 所在集合的标识（整数）。
     
     **示例**::

        rep = uf.find(3)

   - **count(self) -> int**
     
     返回当前并查集中独立集合的数量。
     
     **示例**::

        num = uf.count()

   - **connected(self, p, q) -> bool**
     
     判断两个元素 p 和 q 是否位于同一集合中。
     
     **参数**:
       - **p**, **q**: 元素的索引。
     
     **示例**::

        if uf.connected(1, 2):
            print("相连")

   - **union(self, p, q)**
     
     合并包含元素 p 和 q 的两个集合。
     
     **参数**:
       - **p**, **q**: 元素的索引。
     
     **示例**::

        uf.union(1, 2)

   - **__str__(self)**
     
     返回并查集的字符串表示。

   - **__repr__(self)**
     
     返回并查集的详细表示。

XmlIO 类
----------

.. py:class:: XmlIO(file_read: str = '', file_write: str = '')

   用于 XML 文件的读写，将 XML 文件转换为 ConditionTree 结构，或将 ConditionTree 输出为 XML 文件。
   
   **方法**:

   - **__init__(self, file_read: str = '', file_write: str = '')**
     
     初始化 XmlIO 实例，指定待读取和写入的文件路径。
     
     **参数**:
       - **file_read**: 待读取的 XML 文件路径（字符串）。
       - **file_write**: 输出的 XML 文件路径（字符串）。
     
     **示例**::

        xml_io = XmlIO("config.xml", "output.xml")

   - **read(self) -> ConditionTree**
     
     读取 XML 文件并解析为 ConditionTree 对象。
     
     **返回**:
       - 解析得到的 ConditionTree 实例。
     
     **示例**::

        tree = xml_io.read()

   - **write(self, root_name, tree: ConditionTree)**
     
     将给定的 ConditionTree 写入 XML 文件，根节点名称为 *root_name*。
     
     **参数**:
       - **root_name**: 根节点的名称（字符串）。
       - **tree**: ConditionTree 对象。
     
     **示例**::

        xml_io.write("root", tree)

   - **_indent(self, elem, level=0)**
     
     内部方法，用于对 XML 元素进行缩进，以美化输出格式。

示例代码
----------

下面的示例展示了如何使用 data_structure 模块中的各个类：

.. code-block:: python

   from pywayne.data_structure import ConditionTree, UnionFind, XmlIO
   
   # 构建一个简单的逻辑条件树
   tree = ConditionTree("root")
   tree.append_by_path([{'tag': 'if', 'attrib': {}, 'text': '条件1'}])
   print("打印条件树路径:")
   tree.print_path()
   
   # 使用并查集管理元素
   uf = UnionFind(10)
   uf.union(1, 2)
   uf.union(2, 3)
   print("1 和 3 是否相连:", uf.connected(1, 3))
   
   # XML 文件读写示例
   xml_io = XmlIO("input.xml", "output.xml")
   condition_tree = xml_io.read()
   xml_io.write("root", condition_tree)

--------------------------------------------------

通过上述示例，可以快速掌握 data_structure 模块中各数据结构的使用方法，适用于复杂条件处理、集合管理以及 XML 数据操作等场景。 