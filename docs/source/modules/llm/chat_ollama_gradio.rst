Ollama Gradio 聊天界面 (chat_ollama_gradio)
===========================================

.. automodule:: pywayne.llm.chat_ollama_gradio

`OllamaChatGradio` 类提供了一个基于 Gradio 的聊天界面，主要功能包括：

- 获取 Ollama 模型列表。
- 初始化和管理聊天会话。
- 提供创建新会话、切换会话、格式化历史记录以及流式聊天等功能。
- 通过 `create_demo` 与 `launch` 方法，可以快速启动一个基于浏览器的聊天示例应用。

.. autoclass:: pywayne.llm.chat_ollama_gradio.OllamaChatGradio
   :members:
   :undoc-members:

**示例**::

   >>> from pywayne.llm.chat_ollama_gradio import OllamaChatGradio
   >>> ollama_chat = OllamaChatGradio(base_url='http://localhost:11434/v1', server_port=7870)
   >>> ollama_chat.launch() 