聊天机器人接口 (chat_bot)
=========================

.. automodule:: pywayne.llm.chat_bot

该部分主要通过 `LLMChat` 类实现。主要功能包括：

- 使用指定的 `base_url`、`api_key` 及其他参数初始化 LLM 接口。
- 提供 `ask` 与 `chat` 方法, 支持同步及流式返回结果。
- 支持更新系统提示 (`update_system_prompt`) 及清空历史记录 (`clear_history`) 等操作。

此外，`LLMConfig` 类用于构造聊天配置参数，而 `ChatManager` 则负责管理多个 `LLMChat` 实例，实现对话会话的创建和切换。

.. autoclass:: pywayne.llm.chat_bot.LLMConfig
   :members:
   :undoc-members:

.. autoclass:: pywayne.llm.chat_bot.LLMChat
   :members:
   :undoc-members:
   :exclude-members: model, base_url, api_key, system_prompt, history

.. autoclass:: pywayne.llm.chat_bot.ChatManager
   :members:
   :undoc-members:

**示例**::

   >>> from pywayne.llm.chat_bot import LLMChat, LLMConfig, ChatManager
   >>> # 初始化配置
   >>> config = LLMConfig(base_url='https://api.example.com/v1', api_key='your_api_key')
   >>> # 创建聊天实例
   >>> chat = LLMChat(base_url=config.base_url, api_key=config.api_key)
   >>> response = chat.ask('你好，LLM!')
   >>> print(response) 