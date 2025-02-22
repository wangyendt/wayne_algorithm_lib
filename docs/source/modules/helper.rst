辅助工具 (helper)
===================

本模块提供了常用辅助功能，主要用于项目配置管理和获取项目信息。通过 Helper 类，用户可以方便地进行以下操作：
- 定位项目根目录
- 获取配置文件路径
- 设置、获取以及删除配置值

Helper 类详细说明
------------------

.. py:class:: Helper(project_root: str = '', config_file_name: str = 'common_info.yaml')

   Helper 类封装了项目配置管理功能。当未指定 project_root 时，会自动根据调用者所在目录确定项目根目录；同时，配置文件名称默认为 "common_info.yaml"。

   **方法**:

   - **__init__(self, project_root: str = '', config_file_name: str = 'common_info.yaml')**
     
     初始化 Helper 实例。
     
     **参数**:
     
     - **project_root**: 项目根目录的路径。如果为空，则自动根据调用者所在文件的目录确定项目根目录。
     - **config_file_name**: 配置文件名称，默认为 "common_info.yaml"。
     
     **应用场景**:
     
     用于项目启动阶段的配置管理，确保项目根目录和配置文件正确定位。
     
     **示例**::

        helper = Helper()

   - **get_proj_root(self) -> str**

     获取项目根目录的名称。

     **返回**:
     
     - 项目根目录的名称（字符串）。
     
     **示例**::

        root = helper.get_proj_root()
        print("项目根目录:", root)
        
   - **get_config_path(self) -> str**

     获取配置文件的名称或路径。

     **返回**:
     
     - 配置文件的名称（字符串）。
     
     **示例**::

        config_name = helper.get_config_path()
        print("配置文件:", config_name)

   - **set_module_value(self, *keys, value)**

     设置配置文件中嵌套键的值。通过可变参数指定键的层级，最终将 value 存入相应的位置。

     **参数**:
     
     - **keys**: 配置项的键，按照嵌套层级排列。
     - **value**: 要设置的值。

     **应用场景**:
     
     用于更新或保存项目的动态配置，例如数据库连接参数或其他模块的配置项。

     **示例**::

        helper.set_module_value('database', 'host', value='127.0.0.1')
        
   - **get_module_value(self, *keys, max_waiting_time: float = 0.0, debug: bool = True)**

     从配置文件中获取指定嵌套键的值，并可设置最大等待时间以便在配置尚未生成时轮询获取。

     **参数**:
     
     - **keys**: 待查询的配置键，按照嵌套层级排列。
     - **max_waiting_time**: 最大等待时间（单位为秒），若大于0，则在该时间内反复查询配置文件。
     - **debug**: 是否启用调试信息输出，默认为 True。

     **返回**:
     
     - 获取到的配置值，如果未找到则返回 None。

     **应用场景**:
     
     用于动态读取配置值，如启动时读取数据库配置或外部服务参数，支持等待机制确保读取到配置。

     **示例**::

        db_host = helper.get_module_value('database', 'host', max_waiting_time=5)
        if db_host is None:
            print("未获取到数据库配置。")
        else:
            print("数据库主机:", db_host)
        
   - **delete_module_value(self, *keys)**

     删除配置文件中指定键的配置项。

     **参数**:
     
     - **keys**: 待删除的配置项键，按照嵌套层级排列。

     **应用场景**:
     
     用于移除不再需要的配置项，或者实现配置重置操作。

     **示例**::

        helper.delete_module_value('database', 'host')

使用示例
----------

下面的示例展示了如何使用 Helper 类进行配置管理：

.. code-block:: python

   from pywayne.helper import Helper
   
   # 初始化 Helper 对象（使用当前目录作为项目根目录）
   helper = Helper('./')
   
   # 设置多个配置项
   helper.set_module_value('a', 'b', value=2)
   helper.set_module_value('a', 'c', value=1)
   helper.set_module_value('d', 'e', 'f', value=3)
   
   # 删除部分配置项
   helper.delete_module_value('a', 'c')
   helper.delete_module_value('a')
   helper.delete_module_value('d')

--------------------------------------------------

以上详细介绍了 helper 模块中 Helper 类各个方法的功能、参数、应用场景和示例代码，帮助用户快速掌握项目配置管理和辅助工具的使用。 