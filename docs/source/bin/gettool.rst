gettool: 获取/安装 C++ 工具集
===============================

gettool 是安装 pywayne 后提供的命令行脚本，用于从指定的 cpp_tools 仓库按需拉取第三方库源码，支持稀疏检出、选择性构建与可选安装。其逻辑包含：仓库稀疏克隆、子模块完整克隆、可选构建、清理式拷贝、安装脚本执行、URL 管理等。

基本用法
--------

.. code-block:: bash

   gettool <name> [-t TARGET] [-b] [-c] [-v VERSION] [-i] [--global-install-flag true|false]
   gettool -l                         # 列出当前支持的工具
   gettool --get-url                  # 显示当前使用的仓库 URL
   gettool --set-url <URL>            # 设置仓库 URL
   gettool --reset-url                # 重置仓库 URL 为默认

参数说明
--------

- **name**: 工具名（来自仓库根目录的 name_to_path_map.yaml）
- **-t/--target_path**: 目标输出路径（默认根据映射在当前目录展开）
- **-b/--build**: 构建产生的 lib 输出（若可构建）
- **-c/--clean**: 仅拷贝 src 与 include（若存在）
- **-v/--version**: 指定分支/标签（仅当对应项为子模块时生效）
- **-i/--install**: 拉取后执行安装脚本（若安装信息存在）
- **--global-install-flag**: 是否使用 sudo make install（字符串 true/false）

高级说明
--------

- 稀疏克隆：默认通过 git clone --sparse 拉取仓库，并按需 sparse-checkout 指定路径，避免下载整个仓库。
- 子模块判断：若 name 对应路径是子模块，则直接完整克隆子模块至目标目录；此时 --version 可用于检出分支/标签。
- 构建逻辑：当 -b/--build 指定且该工具可构建时，会在源内创建 build 目录并运行 CMake + make，最后优先复制 source/lib 到目标位置。
- 清理拷贝：当 -c/--clean 指定时，若存在 src/include，仅复制这两个目录；否则退化为全量拷贝。
- 安装脚本：当 -i/--install 指定时，读取 name_to_path_map.yaml 中 installation/install_script 并执行；可选 --global-install-flag 控制是否 sudo 安装。
- URL 管理：--set-url/--reset-url/--get-url 操控配置文件 bin/config.yaml，持久化 cpp_tools 仓库地址。

工作流程（简述）
----------------

1. 使用 git clone --sparse 拉取配置与映射
2. 若名称对应子模块：直接完整克隆到目标目录；可选检出版本
3. 否则使用 sparse-checkout set 精确拉取路径并拷贝到目标目录
4. 按需构建或清理拷贝；可选执行安装脚本

示例
----

.. code-block:: bash

   # 拉取并编译 apriltag_detection 到默认路径
   gettool apriltag_detection -b

   # 只拷贝源码到指定路径且执行安装脚本（不使用 sudo）
   gettool eigen -t third_party/eigen -c -i --global-install-flag false

参数组合示例
------------

.. code-block:: bash

   # 仅拉取并放到默认路径（按 name 映射）
   gettool opencv

   # 拉取并放入指定路径
   gettool opencv -t third_party/opencv

   # 拉取并构建（若可构建），构建结果复制到目标（优先 lib 目录）
   gettool apriltag_detection -b -t build/apriltag

   # 拉取特定版本（仅子模块有效）
   gettool fmt -v 9.1.0

   # 拉取后执行安装脚本（脚本位置由配置决定）
   gettool pcl -i

   # 拉取后安装且允许使用 sudo make install
   gettool pcl -i --global-install-flag true

   # 查看支持的工具名清单
   gettool -l

   # 管理仓库 URL
   gettool --get-url
   gettool --set-url https://github.com/wangyendt/cpp_tools
   gettool --reset-url

常见问题
--------

- 子模块与非子模块的区别：子模块通过独立 git clone 获取完整仓库；非子模块通过 sparse-checkout 获取仓库中的子路径。
- 构建失败：请先确保工具具备 CMakeLists.txt 且本机工具链可用；日志中会详细打印 CMake 与 make 的返回。
- 没有生成 lib：若构建成功但未生成 lib 目录，脚本会回退为拷贝整个源树，或按 -c 清理拷贝。


