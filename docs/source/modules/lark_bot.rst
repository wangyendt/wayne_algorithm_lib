飞书机器人 (lark_bot)
=====================

本模块实现了与飞书平台进行交互的功能，主要用于发送文本、图片、文件、音频以及可交互消息等消息。该模块包含以下三个主要部分：

1. TextContent 类：提供各种文本格式化工具，用于生成消息中所需的特殊格式（例如加粗、斜体、下划线、@某人及链接等）。
2. PostContent 类：用于构造复杂的消息内容，包括标题、文本、链接、图片和其他格式化内容。
3. LarkBot 类：核心类，管理与飞书平台的交互，支持发送不同类型的消息，并提供用户和群组信息查询等功能。
4. LarkBotListener 类：提供飞书消息监听和处理功能。


TextContent 类
----------------

TextContent 类包含了多种静态方法，用于生成飞书消息中的文本格式。例如：

- :py:func:`make_at_all_pattern()`
  生成@所有人的文本标识。

- :py:func:`make_at_someone_pattern(someone_open_id, username, id_type)`
  生成@指定用户的文本标识。

- :py:func:`make_bold_pattern(content)`
  将文本加粗。

- :py:func:`make_italian_pattern(content)`
  将文本设置为斜体。

- :py:func:`make_underline_pattern(content)`
  为文本添加下划线。

- :py:func:`make_delete_line_pattern(content)`
  为文本添加删除线效果。

- :py:func:`make_url_pattern(url, text)`
  将文本格式化为超链接文本。

这些方法帮助用户在构造消息时快速生成需要的文本格式。


PostContent 类
----------------

PostContent 类用于构造复杂的消息内容，支持设置消息标题和添加多种格式化内容。主要方法包括：

- ``__init__(self, title: str = '')``
  初始化 PostContent 实例，可选择性设置标题。

- ``get_content(self) -> Dict``
  获取构造好的消息内容。

- ``set_title(self, title: str)``
  设置消息的标题。

- ``list_text_styles()``
  列出可用的文本样式。

- ``make_text_content(text, styles=None, unescape=False)``
  构造文本消息内容。

- ``make_link_content(text, link, styles=None)``
  构造带超链接的文本内容。

- ``make_at_content(at_user_id, styles=None)``
  构造 @ 用户 的文本格式。

- ``make_image_content(image_key)``
  构造图片消息内容。

- ``make_media_content(file_key, image_key='')``
  构造多媒体消息内容。

- ``make_emoji_content(emoji_type)``
  构造 emoji 消息内容。

- ``make_hr_content()``
  构造分隔线内容。

- ``make_code_block_content(language, text)``
  构造代码块形式的消息内容。

- ``make_markdown_content(md_text)``
  构造 Markdown 格式的消息内容。

用户可以根据实际需求将这些内容方法组合使用，以构造丰富且格式化的消息。


LarkBot 类
-----------

LarkBot 类为与飞书平台进行交互的核心类，提供了发送各种消息类型的接口。主要功能包括：

- **发送文本消息**：
  - ``send_text_to_user(user_open_id, text)``：向指定用户发送文本消息。
  - ``send_text_to_chat(chat_id, text)``：向群聊发送文本消息。

- **发送图像消息**：
  - ``send_image_to_user(user_open_id, image_key)`` 和 ``send_image_to_chat(chat_id, image_key)``：用于发送图像消息。

- **发送音频与多媒体消息**：
  - 对应接口支持发送音频消息（如 ``send_audio_to_user`` / ``send_audio_to_chat``）以及其他多媒体消息（如 ``send_media_to_user`` / ``send_media_to_chat``）。

- **发送文件**：
  - ``send_file_to_user(user_open_id, file_key)`` 和 ``send_file_to_chat(chat_id, file_key)``：用于文件消息发送。

- **发送交互消息及分享消息**：
  - 包括发送交互式消息（``send_interactive_to_user`` / ``send_interactive_to_chat``）和分享聊天、分享用户信息的接口。

- **用户和群组信息查询**：
  - 包含获取用户信息 (``get_user_info``)、查询群组列表 (``get_group_list``)、通过群名称查找群聊 ID (``get_group_chat_id_by_name``) 和获取群成员信息 (``get_members_in_group_by_group_chat_id``) 等功能。

下面是使用 LarkBot 类的示例：

.. code-block:: python

   from pywayne.lark_bot import LarkBot, TextContent
   
   # 初始化 LarkBot 实例
   bot = LarkBot(app_id="your_app_id", app_secret="your_app_secret")
   
   # 使用 TextContent 构造加粗文本消息
   text_msg = TextContent.make_bold_pattern("Hello, 飞书!")
   
   # 发送消息到指定用户
   response = bot.send_text_to_user(user_open_id="user_open_id_example", text=text_msg)
   print(response)

通过这些接口，用户可以方便地构造并发送各类消息，实现与飞书平台的高效互动。


LarkBotListener 类
------------------

.. py:class:: LarkBotListener(app_id: str, app_secret: str, message_expiry_time: int = 60)
   :noindex:

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

下面是使用 LarkBotListener 类的示例：

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

**注意事项**:

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

未来可以在 LarkBot 模块的基础上扩展更多飞书 API 接口，例如支持更多交互组件、自定义机器人行为以及消息事件的实时处理，以满足更复杂的业务需求。

其他接口和高级功能
--------------------

除了上文介绍的基本发送消息接口外，LarkBot 类还提供了一些额外接口和高级功能，以满足更复杂的应用需求，例如：

- **查询接口**：
  - ``get_user_info(emails, mobiles)``：根据邮箱或手机号获取用户信息。
  - ``get_group_list()``：返回当前飞书账号下的所有群组列表。
  - ``get_group_chat_id_by_name(group_name)``：通过群名称获取对应的群聊 ID 列表。
  - ``get_members_in_group_by_group_chat_id(chat_id)``：获取指定群聊的成员列表。

- **文件和多媒体传输**：
  - ``send_file_to_user(user_open_id, file_key)`` 与 ``send_file_to_chat(chat_id, file_key)``：支持文件消息的发送。
  - ``send_audio_to_user(user_open_id, file_key)`` 与 ``send_audio_to_chat(chat_id, file_key)``：支持音频消息的发送。
  - ``send_media_to_user(user_open_id, file_key)`` 与 ``send_media_to_chat(chat_id, file_key)``：支持其他多媒体消息的发送（如视频）。

- **交互式和共享消息**：
  - ``send_interactive_to_user`` 和 ``send_interactive_to_chat``：发送交互式消息，支持按钮、卡片等富交互组件。
  - ``send_shared_chat_to_user``、``send_shared_chat_to_chat``：发送聊天分享消息。
  - ``send_shared_user_to_user``、``send_shared_user_to_chat``：发送用户分享消息。

下面是一个使用高级功能的例子：

.. code-block:: python

   # 获取并打印当前账号的群组列表
   groups = bot.get_group_list()
   print("当前群组：", groups)
   
   # 根据群名称获取对应的群聊 ID
   chat_ids = bot.get_group_chat_id_by_name("项目讨论组")
   print("项目讨论组的群聊 ID：", chat_ids)

注意事项与最佳实践
----------------------

1. 调试时请确保 API 的有效性，确保 app_id 和 app_secret 配置正确；
2. 在群组消息发送时，请注意飞书平台的消息格式限制，例如文本长度、图片大小等要求；
3. 对于交互式消息，建议提前测试各组件的显示效果，确保在不同客户端上均正常显示；
4. 异常处理：调用 send_* 系列方法时，请关注返回的信息，及时处理可能出现的错误或超时；
5. 为提高消息发送效率，建议在实际应用中添加必要的缓存或限流机制。

总结
----

LarkBot 模块为飞书平台的消息交互提供了灵活且强大的接口，涵盖了文本、图像、音频、文件、交互消息及用户信息查询等多种功能。通过组合这些接口，开发者可以构建功能丰富的飞书机器人系统，并满足各种业务需求。同时，该模块为后续扩展提供了良好的基础，可以根据实际场景添加更多定制化的功能。 