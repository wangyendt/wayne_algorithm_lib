语言模型 (LLM)
====================

本模块提供了与大型语言模型 (Large Language Model, LLM) 相关的多种功能，旨在实现基于 LLM 的对话接口、配置管理以及图形化聊天窗口等。模块主要包含以下几个部分：

1. 聊天机器人接口（LLMChat、LLMConfig 与 ChatManager）
2. 图形聊天窗口（ChatWindow）
3. 基于 Ollama 的 Gradio 聊天界面（OllamaChatGradio）


聊天机器人接口
--------------------

该部分主要通过位于 pywayne/llm/chat_bot.py 的 LLMChat 类实现。主要功能包括：

- 使用指定的 base_url、api_key 及其他参数初始化 LLM 接口。
- 提供 ask 与 chat 方法, 支持同步及流式返回结果。
- 支持更新系统提示 (update_system_prompt) 及清空历史记录 (clear_history) 等操作。

此外，LLMConfig 类用于构造聊天配置参数，而 ChatManager 则负责管理多个 LLMChat 实例，实现对话会话的创建和切换。

**示例**::

   >>> from pywayne.llm.chat_bot import LLMChat, LLMConfig, ChatManager
   >>> # 初始化配置
   >>> config = LLMConfig(base_url='https://api.example.com/v1', api_key='your_api_key')
   >>> # 创建聊天实例
   >>> chat = LLMChat(base_url=config.base_url, api_key=config.api_key)
   >>> response = chat.ask('你好，LLM!')
   >>> print(response)


图形聊天窗口
--------------

位于 pywayne/llm/chat_window.py 的 ChatWindow 类实现了基于 GUI 的对话窗口。其主要特点包括：

- 使用 ChatConfig 数据类配置窗口参数，比如尺寸、标题和聊天模型等。
- 提供实时对话输入与显示，并支持添加系统消息。
- 可通过 launch 方法快速启动一个独立的聊天窗口。

**示例**::

   >>> from pywayne.llm.chat_window import ChatWindow, ChatConfig
   >>> config = ChatConfig(base_url='https://api.example.com/v1', api_key='your_api_key', model='deepseek-chat', window_title='LLM 聊天')
   >>> ChatWindow.launch(base_url=config.base_url, api_key=config.api_key, model=config.model)


基于 Ollama 的 Gradio 聊天界面
---------------------------------

在 pywayne/llm/chat_ollama_gradio.py 中，OllamaChatGradio 类提供了一个基于 Gradio 的聊天界面，主要功能包括：

- 获取 Ollama 模型列表。
- 初始化和管理聊天会话。
- 提供创建新会话、切换会话、格式化历史记录以及流式聊天等功能。
- 通过 create_demo 与 launch 方法，可以快速启动一个基于浏览器的聊天示例应用。

**示例**::

   >>> from pywayne.llm.chat_ollama_gradio import OllamaChatGradio
   >>> ollama_chat = OllamaChatGradio(base_url='http://localhost:11434/v1', server_port=7870)
   >>> ollama_chat.launch()


模块扩展建议
--------------

未来可以在该模块中增加更多对话模型的适配接口，扩展模型管理、会话记录持久化等功能，从而为用户提供更多定制化的对话机器人解决方案。 