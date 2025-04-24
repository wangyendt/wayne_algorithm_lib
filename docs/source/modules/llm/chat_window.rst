图形聊天窗口 (chat_window)
==========================

.. automodule:: pywayne.llm.chat_window

`ChatWindow` 类实现了基于 GUI 的对话窗口。其主要特点包括：

- 使用 `ChatConfig` 数据类配置窗口参数，比如尺寸、标题和聊天模型等。
- 提供实时对话输入与显示，并支持添加系统消息。
- 可通过 `launch` 方法快速启动一个独立的聊天窗口。

.. autoclass:: pywayne.llm.chat_window.ChatConfig
   :members:
   :undoc-members:

.. autoclass:: pywayne.llm.chat_window.ChatWindow
   :members: launch, add_system_message
   :undoc-members:

**示例**::

   >>> from pywayne.llm.chat_window import ChatWindow, ChatConfig
   >>> config = ChatConfig(base_url='https://api.example.com/v1', api_key='your_api_key', model='deepseek-chat', window_title='LLM 聊天')
   >>> ChatWindow.launch(base_url=config.base_url, api_key=config.api_key, model=config.model) 