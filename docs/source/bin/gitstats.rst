gitstats: 统计 Git 提交时间分布
================================

``gitstats`` 是安装 ``pywayne`` 后提供的命令行工具。进入任意 Git 仓库目录后，运行该命令即可统计提交时间分布并生成图表。

安装与环境
------------

- 需要已安装 ``git``
- Python 依赖：``pandas``、``matplotlib``（随 ``pywayne`` 一并安装）

基本用法
--------

.. code-block:: bash

   gitstats -h                        # 查看帮助
   gitstats .. --since "2024-01-01"   # 指定仓库路径与起始时间

命令行参数
----------

.. code-block:: text

   usage: gitstats [repo] [--since SINCE] [--until UNTIL] [--tz TZ]
                   [--branch BRANCH] [--all] [--save SAVE]
                   [-p | --show-plot | --show_plot]

   位置参数:
     repo                  git 仓库路径；省略时为当前目录

   可选参数:
     --since SINCE         起始时间；如 2024-01-01、2024-01-01T00:00:00、"90 days ago"、"1 year ago"
     --until UNTIL         终止时间；同上格式
     --tz TZ               时区；如 Asia/Shanghai、UTC；默认 Asia/Shanghai
     --branch BRANCH       指定单一分支；示例：main、develop、origin/main
     --all                 统计所有分支（与 --branch 互斥，指定该项则忽略 --branch）
     --save SAVE           输出图片路径；默认 git_time_distribution.png
     -p, --show-plot       弹窗展示图表（不保存文件）
         --show_plot       同 --show-plot

选项说明与注意事项
------------------

- 分支选择优先级：指定 --all 时会忽略 --branch；二者都未指定时默认使用当前 HEAD 所在分支。
- 时间范围：可单独指定 --since 或 --until，也可二者同时指定；二者都未指定时统计全历史。
- 时区：内部以 UTC 解析提交时间，再转换到 --tz；若要原始 UTC，请设置 --tz UTC。
- 输出：文件若存在将被覆盖；支持绝对与相对路径。
- 大仓库性能：启用 --all 可能带来较长扫描时间。
- 默认行为：无任何参数时显示帮助并退出；若需仅展示图而不保存，使用 -p/--show-plot。

输出
----

命令运行成功后，会在 ``--save`` 指定的路径生成一张 3×2 子图布局的统计图：

- 每日提交数量折线图
- 不同小时提交数量柱状图
- 星期分布柱状图（Mon–Sun）
- 星期 × 小时 热力图（使用 ``parula`` 配色）

示例
----

.. code-block:: bash

   # 统计当前仓库最近一年所有分支的提交分布并保存
   gitstats --since "1 year ago" --all --save out.png

   # 统计当前仓库 main 分支近 90 天的提交，时区为 UTC
   gitstats --branch main --since "90 days ago" --tz UTC

   # 统计上级目录仓库 develop 分支，指定起止日期
   gitstats .. --branch develop --since 2024-01-01 --until 2024-06-30

   # 仅指定终止时间（统计至某日为止）
   gitstats --until 2025-01-01

   # 指定远程分支（如 origin/main）
   gitstats --branch origin/main

   # 自定义输出文件名与目录
   gitstats --all --save results/commit_stats.png

   # 仅弹窗展示（不写入文件）
   gitstats --since "30 days ago" -p

常见问题
--------

- 提示“没有读到提交记录”：检查 repo 路径、分支名是否存在、时间过滤是否过严。
- 提示 git 调用失败：确认系统已安装 git，且当前目录或传入路径为有效仓库。


