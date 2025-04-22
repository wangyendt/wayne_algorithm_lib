飞书消息监听器 (lark_bot_listener)
====================================

本模块提供了飞书消息监听和处理功能，支持实时接收和处理飞书平台的各类消息，包括文本、图片、文件等。通过异步处理和消息去重机制，确保消息处理的高效性和可靠性。

主要功能
--------

- 支持多种消息类型的监听和处理（文本、图片、文件、富文本等）
- 提供便捷的消息处理装饰器（text_handler、image_handler、file_handler）
- 支持群组消息和私聊消息的分别处理
- 自动消息去重，避免重复处理
- 异步处理机制，提高消息处理效率

LarkBotListener 类
------------------

.. py:class:: LarkBotListener(app_id: str, app_secret: str, message_expiry_time: int = 60)

   飞书消息监听器，用于实时接收和处理飞书消息。支持文本、图片、文件等多种消息类型，并提供消息去重和异步处理功能。

   **参数**:

   - app_id (str): 飞书应用 ID
   - app_secret (str): 飞书应用密钥
   - message_expiry_time (int): 消息去重过期时间（秒），默认 60 秒

   **主要方法**:

   - **listen(message_type: Optional[str] = None, group_only: bool = False, user_only: bool = False)**
     
     消息监听装饰器，用于注册消息处理函数。

     **参数**:

     - message_type: 消息类型（"text"、"image"、"file"、"post"），None 表示所有类型
     - group_only: 是否只监听群组消息
     - user_only: 是否只监听私聊消息

   - **text_handler(group_only: bool = False, user_only: bool = False)**
     
     文本消息处理装饰器，提供更便捷的文本消息处理接口。

     装饰的函数可以接收以下参数（除 text 外都是可选的）：
     
     - text (str): 文本内容（必需）
     - chat_id (str): 会话 ID
     - is_group (bool): 是否群组消息
     - group_name (str): 群组名称
     - user_name (str): 发送消息的用户姓名

   - **image_handler(group_only: bool = False, user_only: bool = False)**
     
     图片消息处理装饰器，自动下载和处理图片文件。

     装饰的函数可以接收以下参数（除 image_path 外都是可选的）：
     
     - image_path (Path): 图片文件路径
     - chat_id (str): 会话 ID
     - is_group (bool): 是否群组消息
     - group_name (str): 群组名称
     - user_name (str): 发送消息的用户姓名

   - **file_handler(group_only: bool = False, user_only: bool = False)**
     
     文件消息处理装饰器，自动下载和处理文件。

     装饰的函数可以接收以下参数（除 file_path 外都是可选的）：
     
     - file_path (Path): 文件路径
     - chat_id (str): 会话 ID
     - is_group (bool): 是否群组消息
     - group_name (str): 群组名称
     - user_name (str): 发送消息的用户姓名

   - **send_message(chat_id: str, content: str)**
     
     发送消息到飞书（使用 Markdown 格式）。

   - **run()**
     
     启动消息监听服务。

使用示例
--------

以下是一个完整的使用示例，展示了如何使用 LarkBotListener 处理各类消息：

.. code-block:: python

   from pywayne.lark_bot_listener import LarkBotListener
   
   # 创建监听器实例
   listener = LarkBotListener(
       app_id="your_app_id",
       app_secret="your_app_secret"
   )
   
   # 处理文本消息
   @listener.text_handler()
   async def handle_text(text: str, chat_id: str, user_name: str):
       print(f"收到来自 {user_name} 的消息: {text}")
       # 回复消息
       listener.send_message(chat_id, f"已收到您的消息：{text}")
   
   # 处理图片消息
   @listener.image_handler()
   async def handle_image(image_path: Path, chat_id: str):
       print(f"收到图片: {image_path}")
       # 处理图片...
   
   # 处理文件消息
   @listener.file_handler()
   async def handle_file(file_path: Path, chat_id: str):
       print(f"收到文件: {file_path}")
       # 处理文件...
   
   # 使用原始监听器处理任意类型消息
   @listener.listen(message_type="post")
   async def handle_post(ctx: MessageContext):
       print(f"收到富文本消息: {ctx.content}")
   
   # 启动监听服务
   listener.run()

注意事项
--------

1. 消息处理：
   
   - 所有处理函数都是异步的，需要使用 async/await 语法
   - 每个消息可以被多个处理函数处理
   - 消息会进行去重，避免重复处理

2. 临时文件：
   
   - 图片和文件会被下载到临时目录
   - 建议在处理完成后及时清理临时文件
   - 临时目录路径：系统临时目录/lark_bot_temp

3. 错误处理：
   
   - 每个处理函数的异常都会被单独捕获，不会影响其他处理函数
   - 建议在处理函数中添加适当的错误处理逻辑

4. 性能考虑：
   
   - 消息去重默认过期时间为 60 秒
   - 可以通过 message_expiry_time 参数调整去重时间
   - 处理函数应尽量避免耗时操作，必要时可以启动新的任务

模块扩展建议
---------------

1. 消息处理增强：
   
   - 添加更多消息类型的专用处理器
   - 支持消息处理的优先级设置
   - 提供消息处理的中间件机制

2. 监控与管理：
   
   - 添加消息处理状态的监控接口
   - 提供处理函数的动态注册和注销功能
   - 支持处理函数的热重载

3. 高级功能：
   
   - 集成更多飞书 API 功能
   - 支持消息的批量处理
   - 添加消息处理的统计和分析功能 