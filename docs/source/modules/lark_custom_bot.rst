飞书自定义机器人 (lark_custom_bot)
==================================

本模块提供了一个飞书自定义机器人的实现，主要用于通过 webhook 与飞书平台进行消息交互。该模块支持文本消息、图片消息、交互消息以及分享消息的发送，同时提供了一些辅助函数，便于快速构造消息内容。

LarkCustomBot 类
------------------

LarkCustomBot 类是该模块的核心，用于管理与飞书平台的交互，其主要功能包括：

- 初始化机器人实例，配置 webhook、secret、bot_app_id 以及 bot_secret。
- 设置日志记录，用于调试与错误跟踪。
- 获取租户访问令牌（tenant access token），用于 API 鉴权。
- 通过文件或 OpenCV 图像上传图片，并返回图片对应的 key。
- 根据时间戳和 secret 生成签名，确保消息安全性。
- 发送 HTTP 请求，与飞书平台进行消息传输。
- 提供发送文本消息、Post 消息、会话分享、图片消息以及交互消息等接口。

主要方法如下：

- __init__(self, webhook: str, secret: str = '', bot_app_id: str = '', bot_secret: str = '')
  初始化自定义机器人实例。

- _setup_logging(self)
  设置日志记录系统，用于调试和错误跟踪。

- _get_tenant_access_token(self) -> str
  获取租户访问令牌，用于对飞书 API 的鉴权。

- upload_image(self, file_path: str) -> str
  上传本地文件作为图片消息，并返回图片的 key。

- upload_image_from_cv2(self, cv2_image: np.ndarray) -> str
  从 OpenCV 图像上传图片消息。

- _generate_signature(self, timestamp: str, secret: str) -> str
  根据时间戳和 secret 生成签名，确保消息传输的安全性。

- _send_request(self, data: Dict) -> None
  发送 HTTP 请求到飞书平台，实现消息的传递。

- send_text(self, text: str, mention_all: bool = False) -> None
  发送文本消息，可设置是否 @所有人。

- send_post(self, content: List[List[Dict]], title: Optional[str] = None) -> None
  发送 Post 类型消息，支持丰富的格式化内容。

- send_share_chat(self, share_chat_id: str) -> None
  发送会话分享消息。

- send_image(self, image_key: str) -> None
  发送图片消息，通过 image_key 指定图片。

- send_interactive(self, card: Dict) -> None
  发送交互式消息，支持按钮、卡片等富交互组件。

辅助函数
---------

模块还提供了一些辅助函数，用于快速构造消息内容，主要包括：

- create_text_content(text: str, unescape: bool = False) -> Dict
  构造文本消息内容字典。

- create_link_content(href: str, text: str) -> Dict
  构造超链接消息内容字典。

- create_at_content(user_id: str, user_name: str) -> Dict
  构造 @ 用户 的消息内容字典。

- create_image_content(image_key: str, width: Optional[int] = None, height: Optional[int] = None) -> Dict
  构造图片消息内容字典，可选设置宽度和高度。

使用示例
----------

下面是一个简单的示例，展示如何使用 LarkCustomBot 类及辅助函数发送消息：

.. code-block:: python

   from pywayne.lark_custom_bot import LarkCustomBot, create_text_content, create_link_content, create_at_content, create_image_content
   
   # 初始化自定义机器人实例
   bot = LarkCustomBot(webhook="your_webhook_url", secret="your_secret", bot_app_id="your_bot_app_id", bot_secret="your_bot_secret")
   
   # 发送文本消息
   bot.send_text("Hello, 飞书自定义机器人!")
   
   # 发送包含链接的文本消息
   content = create_link_content("https://www.example.com", "点击访问")
   bot.send_text(str(content))
   
   # 发送图片消息
   image_key = bot.upload_image("path/to/image.png")
   bot.send_image(image_key)
   
   # 发送交互式消息（卡片示例）
   card = {
       "title": "自定义卡片消息",
       "elements": [
           create_text_content("这是一条交互消息"),
           create_at_content("user_id_example", "用户名")
       ]
   }
   bot.send_interactive(card)

模块扩展建议
---------------

未来可以在 LarkCustomBot 模块的基础上扩展更多功能，例如：

- 支持更多消息类型（如视频、音频等）的发送；
- 增加消息接收与处理功能，实现完整的双向消息交互；
- 自定义日志与异常处理，提升机器人稳定性与安全性；
- 与飞书其他 API 集成，实现更丰富的业务场景功能。

总结
----

LarkCustomBot 模块为飞书自定义机器人提供了简洁而灵活的实现，通过核心类和辅助函数，开发者可以快速构造并发送各种消息，实现与飞书平台的高效交互。该模块具有良好的扩展性，适用于构建多功能、定制化的飞书机器人系统。 