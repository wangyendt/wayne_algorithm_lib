gitstats: 统计 Git 提交时间分布
================================

``gitstats`` 是安装 ``pywayne`` 后提供的命令行工具。进入任意 Git 仓库目录后，运行该命令即可统计提交时间分布并生成图表。

安装与环境
------------

- 需要已安装 ``git``
- Python 依赖：``pandas``、``matplotlib``、``seaborn`` （随 ``pywayne`` 一并安装）

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
- 若需仅展示图而不保存，使用 -p/--show-plot。

输出
----

命令运行成功后，会在 ``--save`` 指定的路径生成一张紧凑的 Seaborn 风格统计图。默认画布为 11×7.2 英寸。脚本会检查字体文件是否真正包含所需中文字形：找到可用字体时显示中文，否则自动改用英文标签；日期始终使用与系统 locale 无关的数字格式。

- 提交趋势图：跨度不超过 180 天时按日、3 年内按周、更长时按月聚合，并标出当前粒度下的峰值
- 小时分布柱状图：每 3 小时显示一个横坐标标签，突出最活跃时段
- 星期分布柱状图（周一至周日），突出最活跃星期
- 星期 × 小时热力图：每 2 小时显示一个横坐标标签

页眉同时显示仓库名、分支、时区、统计日期范围、提交总数和活跃天数。保存图片使用 160 DPI；日期与小时刻度会按固定上限精简，避免标签重合。

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

   # 在当前仓库、当前分支、Asia/Shanghai、保存到默认文件
   gitstats

常见问题
--------

- 提示“没有读到提交记录”：检查 repo 路径、分支名是否存在、时间过滤是否过严。
- 提示 git 调用失败：确认系统已安装 git，且当前目录或传入路径为有效仓库。
- Ubuntu 未安装中文字体：图表会自动改用英文标签并输出安装提示。若希望继续显示中文，可执行 ``sudo apt install fonts-noto-cjk`` 后重新运行。
- 已安装字体但未被旧缓存识别：脚本每次启动都会补充扫描系统字体，并按实际中文字形能力选择字体，不再只依赖固定字体名。
