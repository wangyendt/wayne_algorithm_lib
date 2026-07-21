pywayne: 强大的 Python 工具库
========================================

pywayne 是一款集成多种实用功能的 Python 工具库，专注于提供高效、可靠的开发工具和算法实现。本库涵盖了从基础工具到高级功能的多个领域，包括数据处理、图形界面、机器人控制、计算机视觉以及通信存储等方面。

特点
------------

- **全面的功能支持**：从基础工具到专业算法，满足不同层次的开发需求
- **高度的可扩展性**：模块化设计，便于扩展和定制
- **完善的文档支持**：详细的 API 文档和使用示例
- **可靠的代码质量**：严格的代码审查和测试保证
- **活跃的维护更新**：持续改进和功能扩展

近期重点能力
------------

如果你主要关注消息平台和机器人能力，当前文档已经覆盖这一批高频能力：

- 飞书应用机器人：主动发消息、Markdown 自动路由、引用回复、reaction、置顶、群管理、批量发送、资源上传下载
- 飞书监听器：文本 / 图片 / 文件 / 音频 / 媒体 / post / interactive 监听，消息撤回、已读、reaction、群成员变更、卡片回调
- 飞书自定义机器人：Webhook 文本 / post / 图片 / interactive 卡片推送
- LLM 与消息平台组合：可与 ``pywayne.llm`` 组合实现问答机器人、事件驱动消息流

安装
------------

使用 pip 安装最新版本：

.. code-block:: bash

   pip install -U pywayne

或者从源码安装：

.. code-block:: bash

   git clone https://github.com/wangyendt/wayne_algorithm_lib.git
   cd wayne_algorithm_lib
   pip install -e .

快速开始
------------

以下是一个简单的示例，展示如何使用 pywayne 中的一些基本功能：

.. code-block:: python

   from pywayne.tools import func_timer
   from pywayne.helper import Helper
   
   # 使用函数计时器
   @func_timer
   def process_data():
       # 处理数据...
       pass
   
   # 使用配置助手
   helper = Helper()
   helper.set_module_value('database', 'host', value='localhost')

模块文档
-----------

.. toctree::
   :maxdepth: 2
   :caption: 基础工具:
   :name: basic-tools

   modules/tools
   modules/helper
   modules/data_structure
   modules/maths
   modules/crypto

.. toctree::
   :maxdepth: 2
   :caption: 数据处理:
   :name: data-processing

   modules/dsp
   modules/plot
   modules/statistics

.. toctree::
   :maxdepth: 2
   :caption: 图形界面:
   :name: gui-tools

   modules/gui
   modules/visualization

.. toctree::
   :maxdepth: 2
   :caption: 机器人与视觉:
   :name: robotics-vision

   modules/ahrs
   modules/calibration
   modules/cv
   modules/vio

.. toctree::
   :maxdepth: 2
   :caption: AI 与消息平台:
   :name: ai-messaging

   modules/llm
   modules/lark_bot
   modules/lark_bot_listener
   modules/lark_custom_bot

.. toctree::
   :maxdepth: 2
   :caption: 通信与存储:
   :name: communication-storage

   modules/adb
   modules/aliyun_oss
   modules/cross_comm
   modules/tts

.. toctree::
   :maxdepth: 1
   :caption: 命令行工具:
   :name: cli-tools

   bin/gitstats
   bin/gettool
   bin/cmdlogger
   bin/toolsetup

贡献指南
------------

我们欢迎社区贡献，无论是修复 bug、添加新功能，还是改进文档。请参考以下步骤：

1. Fork 项目仓库
2. 创建功能分支
3. 提交您的更改
4. 确保所有测试通过
5. 提交 Pull Request

问题反馈
------------

如果您在使用过程中遇到任何问题，或有任何建议，请通过以下方式联系我们：

- GitHub Issues：https://github.com/wangyendt/wayne_algorithm_lib/issues
- 邮件联系：wang121ye@hotmail.com

联系方式与社区
----------------

- **GitHub**: `wangyendt@github.com <https://github.com/wangyendt>`_
- **邮箱**: 
    - wang121ye@hotmail.com
    - y-w22@mails.tsinghua.edu.cn
- **个人网站**: 
    - `wangye.xin <http://wangye.xin>`_
    - `cvllm.com <http://cvllm.com>`_
- **LeetCode**: `leetcode.com/wangyehope <https://leetcode.com/wangyehope>`_


许可证
------------

本项目采用 Apache License 2.0 许可证。详细信息请参见 LICENSE 文件。

索引和搜索
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
