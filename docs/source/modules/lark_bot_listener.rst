飞书消息监听器 (lark_bot_listener)
==================================

``pywayne.lark_bot_listener`` 提供基于飞书 WebSocket 事件流的实时监听能力，适合做自动回复、消息分流、文件落盘、卡片交互回调、群聊事件处理、reaction 联动等场景。

这一层的定位不是“替代 LarkBot”，而是：

- ``LarkBot`` 负责主动调用飞书 OpenAPI 发消息、改消息、回消息、加表情、置顶、群管理。
- ``LarkBotListener`` 负责把飞书推送进来的消息与事件整理成好用的 decorator 和上下文对象。
- 两者组合后，可以实现“收到什么消息，就按什么规则自动处理”的完整闭环。


能力概览
--------

当前 ``LarkBotListener`` 已覆盖以下两类能力：

- 消息监听

  - 文本 ``text``
  - 图片 ``image``
  - 文件 ``file``
  - 音频 ``audio``
  - 媒体 ``media``
  - 贴纸 ``sticker``
  - 富文本 ``post``
  - interactive 卡片消息 ``interactive``
  - 任意消息类型统一入口 ``listen(message_type=None)``

- 事件监听

  - 消息撤回 ``recalled``
  - 消息已读 ``message_read``
  - reaction 新增 / 删除
  - 机器人被拉入群
  - 机器人被移出群
  - 机器人进入某个用户私聊
  - 群成员加入 / 被移除 / 主动退出
  - 群信息变更
  - 群解散
  - 卡片 action HTTP 回调

除此之外，监听器还内置了这些增强点：

- 按 handler 维度的消息去重
- 过期消息清理
- 群聊名 / 用户名自动补齐
- sync/async handler 统一兼容
- 图片 / 文件 / 音频 / 媒体自动下载到临时目录
- 处理函数返回文件路径后自动重新上传回发
- ``MessageContext`` 携带 ``message_id`` / ``thread_id`` / ``root_id`` / ``parent_id`` / ``mentions`` / ``raw_event``


MessageContext
--------------

.. py:class:: MessageContext

   通用消息上下文对象。凡是通过 ``listen(...)`` 进入的消息，都会先被整理为 ``MessageContext`` 再传给你的 handler。

   关键字段如下：

   - ``chat_id``: 会话 ID
   - ``user_id``: 发送人 open_id
   - ``message_type``: 当前消息类型
   - ``content``: 解析后的消息内容
   - ``is_group``: 是否群聊
   - ``chat_type``: 飞书原始 chat type
   - ``message_id``: 消息 ID
   - ``thread_id``: 所在线程 ID
   - ``root_id``: 根消息 ID
   - ``parent_id``: 父消息 ID
   - ``mentions``: 被 @ 列表
   - ``raw_event``: 飞书 SDK 原始事件对象

典型用法：

.. code-block:: python

   from pywayne.lark_bot_listener import LarkBotListener, MessageContext
   from pywayne.tools import wayne_print

   listener = LarkBotListener(app_id="cli_xxx", app_secret="sec_xxx")

   @listener.listen()
   async def dump_all(ctx: MessageContext):
       wayne_print(
           f"{ctx.message_type} {ctx.message_id} {ctx.chat_id} {ctx.thread_id}",
           color="cyan"
       )


LarkBotListener 类
------------------

.. py:class:: LarkBotListener(app_id: str, app_secret: str, message_expiry_time: int = 60)

   创建监听器实例。

   **参数**

   - ``app_id``: 飞书应用 app id
   - ``app_secret``: 飞书应用 app secret
   - ``message_expiry_time``: 去重缓存保留时间，单位秒，默认 ``60``

   **实例属性**

   - ``bot``: 内置 ``LarkBot`` 实例。监听到消息后，如果你要引用回复、加 reaction、撤回、下载资源、更新卡片，都直接复用 ``listener.bot`` 即可。


核心方法
--------

通用消息入口
~~~~~~~~~~~~

.. py:method:: listen(message_type: Optional[str] = None, group_only: bool = False, user_only: bool = False)

   注册通用消息处理函数。

   **参数**

   - ``message_type``: 指定消息类型；为 ``None`` 表示所有消息都进来
   - ``group_only``: 只处理群聊
   - ``user_only``: 只处理私聊

   **适用场景**

   - 你需要完整的 ``MessageContext``
   - 你需要拿 ``message_id`` 做引用回复或 reaction
   - 你要统一处理多种消息类型
   - 你不需要自动下载附件

   示例：统一路由不同消息类型

.. code-block:: python

   @listener.listen()
   async def router(ctx: MessageContext):
       if ctx.message_type == "text":
           listener.bot.reply_message(
               ctx.message_id,
               "text",
               {"text": "收到文本"}
           )
       elif ctx.message_type == "image":
           listener.bot.reply_message(
               ctx.message_id,
               "text",
               {"text": "收到图片"}
           )


发送 Markdown 回复
~~~~~~~~~~~~~~~~~~

.. py:method:: send_message(chat_id: str, content: str)

   一个轻量发送入口。内部使用 post + markdown 内容发送到指定 chat。

   它适合快速调试，不适合复杂业务。更推荐在正式业务中直接使用 ``listener.bot`` 里的完整接口，比如：

   - ``reply_message``
   - ``send_markdown_to_chat``
   - ``send_interactive_to_chat``


启动服务
~~~~~~~~

.. py:method:: run()

   启动飞书 WebSocket 监听。

   一个监听器实例一般在进程中只调用一次。


高级消息 decorator
------------------

text_handler
~~~~~~~~~~~~

.. py:method:: text_handler(group_only: bool = False, user_only: bool = False)

   文本消息快捷入口。传入的不是 ``MessageContext``，而是解好的业务参数。

   处理函数可按需声明以下参数：

   - ``text``
   - ``chat_id``
   - ``is_group``
   - ``group_name``
   - ``user_name``

示例：群聊指令机器人

.. code-block:: python

   @listener.text_handler(group_only=True)
   async def handle_group_cmd(text: str, chat_id: str, user_name: str):
       if text == "/ping":
           listener.bot.send_text_to_chat(chat_id, f"{user_name} pong")


image_handler
~~~~~~~~~~~~~

.. py:method:: image_handler(group_only: bool = False, user_only: bool = False)

   自动下载图片到临时文件。处理完成后，如果你的函数返回新的图片路径，会自动重新上传并回发到当前会话。

   可声明参数：

   - ``image_path``
   - ``chat_id``
   - ``is_group``
   - ``group_name``
   - ``user_name``

示例：收到图片后打水印并自动回传

.. code-block:: python

   import cv2
   import tempfile
   from pathlib import Path

   @listener.image_handler()
   async def add_watermark(image_path: Path) -> Path:
       image = cv2.imread(str(image_path))
       cv2.putText(image, "processed", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
           result = Path(f.name)
       cv2.imwrite(str(result), image)
       return result


file_handler
~~~~~~~~~~~~

.. py:method:: file_handler(group_only: bool = False, user_only: bool = False)

   自动下载文件到临时路径。若函数返回新文件路径，会自动上传并回发。

   可声明参数：

   - ``file_path``
   - ``chat_id``
   - ``is_group``
   - ``group_name``
   - ``user_name``

示例：收到文件后原样回发

.. code-block:: python

   from pathlib import Path

   @listener.file_handler()
   async def bounce_file(file_path: Path) -> Path:
       return file_path


audio_handler
~~~~~~~~~~~~~

.. py:method:: audio_handler(group_only: bool = False, user_only: bool = False)

   自动下载音频消息，默认保存为 ``.opus`` 临时文件；若返回新的音频路径，会按 ``opus`` 上传回发。

   可声明参数：

   - ``audio_path``
   - ``chat_id``
   - ``is_group``
   - ``group_name``
   - ``user_name``
   - ``message_id``
   - ``thread_id``

示例：收到语音后保存一份副本并回一个文字确认

.. code-block:: python

   @listener.audio_handler()
   async def save_audio(audio_path, message_id):
       listener.bot.reply_message(message_id, "text", {"text": "收到音频"})


media_handler
~~~~~~~~~~~~~

.. py:method:: media_handler(group_only: bool = False, user_only: bool = False)

   自动下载视频 / 媒体文件，默认临时扩展名为 ``.mp4``；若返回新路径，会按 ``mp4`` 上传回发。

   可声明参数与 ``audio_handler`` 类似，只是资源参数名为 ``media_path``。


sticker_handler
~~~~~~~~~~~~~~~

.. py:method:: sticker_handler(group_only: bool = False, user_only: bool = False)

   贴纸消息快捷入口。适合做“看到某个贴纸就回复固定文案”的轻量交互。

   可声明参数：

   - ``sticker_content``
   - ``raw_content``
   - ``chat_id``
   - ``is_group``
   - ``group_name``
   - ``user_name``
   - ``message_id``
   - ``thread_id``


mention_handler
~~~~~~~~~~~~~~~

.. py:method:: mention_handler(group_only: bool = False, user_only: bool = False)

   只处理“机器人被 @ 的文本消息”。

   可声明参数：

   - ``text``
   - ``mentions``
   - ``chat_id``
   - ``is_group``
   - ``group_name``
   - ``user_name``
   - ``message_id``
   - ``thread_id``
   - ``root_id``
   - ``parent_id``
   - ``raw_event``

示例：只有被 @ 时才回复

.. code-block:: python

   @listener.mention_handler(group_only=True)
   async def when_mentioned(text: str, message_id: str, user_name: str):
       listener.bot.reply_message(
           message_id,
           "text",
           {"text": f"{user_name}，我在。"}
       )


事件 decorator
--------------

recall_handler
~~~~~~~~~~~~~~

.. py:method:: recall_handler()

   处理消息撤回事件。

   可声明参数：

   - ``message_id``
   - ``chat_id``
   - ``recall_time``
   - ``recall_type``
   - ``raw_event``


message_read_handler
~~~~~~~~~~~~~~~~~~~~

.. py:method:: message_read_handler()

   处理消息已读事件。

   可声明参数：

   - ``reader``
   - ``message_id_list``
   - ``raw_event``


reaction_handler
~~~~~~~~~~~~~~~~

.. py:method:: reaction_handler()

   处理 reaction 新增 / 删除事件。

   可声明参数：

   - ``action``: ``created`` 或 ``deleted``
   - ``message_id``
   - ``emoji_type``
   - ``operator_type``
   - ``user_id``
   - ``app_id``
   - ``action_time``
   - ``raw_event``


bot_added_handler / bot_removed_handler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: bot_added_handler()
.. py:method:: bot_removed_handler()

   处理机器人被拉入群 / 移出群事件。

   常用参数：

   - ``chat_id``
   - ``operator_id``
   - ``external``
   - ``name``
   - ``raw_event``


bot_p2p_chat_entered_handler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: bot_p2p_chat_entered_handler()

   处理机器人首次进入某个用户私聊的事件。

   适合做“首次欢迎语”场景。


member_changed_handler
~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: member_changed_handler()

   将“成员加入 / 被删除 / 主动退出”三类群成员变更事件汇总为一个 decorator。

   常用参数：

   - ``action``: ``added`` / ``deleted`` / ``withdrawn``
   - ``chat_id``
   - ``operator_id``
   - ``users``
   - ``name``
   - ``raw_event``


chat_updated_handler / chat_disbanded_handler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: chat_updated_handler()
.. py:method:: chat_disbanded_handler()

   用于监听群资料变更与群解散。


卡片回调
--------

.. py:method:: card_action_handler(verification_token: str, encrypt_key: str = "")

   注册 interactive 卡片 action 回调处理器。

.. py:method:: get_card_action_handler() -> CardActionHandler

   取回 HTTP handler，挂到 Flask / FastAPI / Django 等 Web 框架路由中。

示例：按钮点击后更新原卡片

.. code-block:: python

   from pywayne.lark_bot_listener import LarkBotListener

   listener = LarkBotListener(app_id="cli_xxx", app_secret="sec_xxx")

   @listener.card_action_handler(verification_token="token_xxx", encrypt_key="")
   def on_card_action(card_event):
       open_message_id = card_event.event.context.open_message_id
       card = {
           "type": "template",
           "data": {
               "template_id": "AAqC5c999",
               "template_variable": {"status": "已处理"}
           }
       }
       listener.bot.update_interactive_card(open_message_id, card)
       return {"toast": {"type": "success", "content": "已处理"}}

   # FastAPI 示例
   # app.post("/feishu/card")(listener.get_card_action_handler())


组合场景示例
------------

场景 1：只监听部分群和某个私聊用户
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ALLOWED_GROUP_NAMES = {"项目群", "值班群"}
   ALLOWED_PRIVATE_USER_NAMES = {"Wayne"}

   def allow(group_name: str, user_name: str, is_group: bool) -> bool:
       if is_group:
           return group_name in ALLOWED_GROUP_NAMES
       return user_name in ALLOWED_PRIVATE_USER_NAMES

   @listener.listen(message_type="text")
   async def handle_text(ctx: MessageContext):
       group_name, user_name = listener.bot.get_chat_and_user_name(ctx.chat_id, ctx.user_id)
       if not allow(group_name, user_name, ctx.is_group):
           return
       listener.bot.reply_message(ctx.message_id, "text", {"text": "命中过滤条件"})


场景 2：先加 reaction，1 秒后引用回复，再取消 reaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio

   async def ack_with_temp_reaction(ctx: MessageContext, text: str):
       reaction = listener.bot.add_reaction(ctx.message_id, "WITTY")
       reaction_id = reaction["reaction_id"]
       try:
           await asyncio.sleep(1)
           listener.bot.reply_message(ctx.message_id, "text", {"text": text})
       finally:
           listener.bot.delete_reaction(ctx.message_id, reaction_id)

   @listener.listen(message_type="text")
   async def on_text(ctx: MessageContext):
       await ack_with_temp_reaction(ctx, "收到文本")

   @listener.listen(message_type="image")
   async def on_image(ctx: MessageContext):
       await ack_with_temp_reaction(ctx, "收到图片")

   @listener.listen(message_type="file")
   async def on_file(ctx: MessageContext):
       await ack_with_temp_reaction(ctx, "收到文件")

   @listener.listen(message_type="audio")
   async def on_audio(ctx: MessageContext):
       await ack_with_temp_reaction(ctx, "收到音频")

   @listener.listen(message_type="post")
   async def on_post(ctx: MessageContext):
       await ack_with_temp_reaction(ctx, "收到post")

   @listener.listen(message_type="interactive")
   async def on_card_message(ctx: MessageContext):
       await ack_with_temp_reaction(ctx, "收到card")


场景 3：线程内回复，而不是回复到主会话
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @listener.listen(message_type="text")
   async def on_thread(ctx: MessageContext):
       if ctx.thread_id:
           listener.bot.reply_message(
               ctx.message_id,
               "text",
               {"text": "在线程里回复"},
               reply_in_thread=True
           )


场景 4：收到图片，识别后回图片；收到文本，只回文本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cv2
   import tempfile
   from pathlib import Path

   @listener.image_handler()
   async def detect_image(image_path: Path) -> Path:
       image = cv2.imread(str(image_path))
       cv2.rectangle(image, (20, 20), (200, 200), (0, 255, 0), 3)
       with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
           out = Path(f.name)
       cv2.imwrite(str(out), image)
       return out

   @listener.text_handler()
   async def echo_text(text: str, chat_id: str):
       listener.bot.send_text_to_chat(chat_id, f"收到文本: {text}")


场景 5：群里被 @ 才响应，否则忽略
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @listener.mention_handler(group_only=True)
   async def mention_only(text: str, user_name: str, message_id: str):
       listener.bot.reply_message(
           message_id,
           "text",
           {"text": f"{user_name}，你刚刚 @ 了我。"}
       )


场景 6：收到文件后解析，再把结果摘要引用回去
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @listener.file_handler()
   async def summarize_file(file_path, message_id):
       size = file_path.stat().st_size
       listener.bot.reply_message(
           message_id,
           "text",
           {"text": f"文件已收到，大小 {size} 字节"}
       )


场景 7：成员变更时自动欢迎 / 记录离群
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @listener.member_changed_handler()
   async def on_member_change(action: str, chat_id: str, users):
       if action == "added":
           listener.bot.send_text_to_chat(chat_id, "欢迎新成员")
       elif action == "withdrawn":
           listener.bot.send_text_to_chat(chat_id, "有人退出了群聊")


场景 8：机器人被拉进群后自动发入群说明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @listener.bot_added_handler()
   async def on_bot_added(chat_id: str, name: str):
       listener.bot.send_markdown_to_chat(
           chat_id,
           md_text=(
               "# 机器人已上线\n\n"
               "- 支持文本/图片/文件自动回复\n"
               "- 支持 reaction 与引用回复\n"
               "- 支持卡片按钮回调"
           ),
           title=f"{name} 使用说明"
       )


场景 9：消息已读后做状态更新
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pywayne.tools import wayne_print

   @listener.message_read_handler()
   async def on_read(message_id_list, reader):
       wayne_print(f"这些消息已读: {message_id_list}", color="cyan")


场景 10：reaction 事件和消息业务联动
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pywayne.tools import wayne_print

   @listener.reaction_handler()
   async def on_reaction(action: str, message_id: str, emoji_type: str):
       if action == "created" and emoji_type == "THUMBSUP":
           wayne_print(f"某条消息被点赞了: {message_id}", color="green", bold=True)


场景 11：群配置变化时同步外部系统
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pywayne.tools import wayne_print

   @listener.chat_updated_handler()
   async def on_chat_updated(chat_id: str, before_change, after_change):
       wayne_print(
           f"群信息变化: {chat_id} {before_change} -> {after_change}",
           color="yellow",
           bold=True
       )


场景 12：通用路由器 + 精细分派
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @listener.listen()
   async def dispatch(ctx: MessageContext):
       if ctx.message_type == "text" and ctx.content.startswith("/pin "):
           msg = listener.bot.reply_message(ctx.message_id, "text", {"text": ctx.content[5:]})
           listener.bot.pin_message(msg["message_id"])
       elif ctx.message_type == "text" and ctx.content.startswith("/card"):
           listener.bot.reply_message(
               ctx.message_id,
               "interactive",
               {
                   "header": {"title": {"content": "快捷卡片", "tag": "plain_text"}},
                   "elements": [{"tag": "markdown", "content": "通过 listen 分派生成"}]
               }
           )


场景 13：消息审核链路，先加表情，再查正文，再转发给值班人
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   DUTY_USER_OPEN_ID = "ou_xxx"

   @listener.listen(message_type="text", group_only=True)
   async def moderation(ctx: MessageContext):
       reaction = listener.bot.add_reaction(ctx.message_id, "OK")
       try:
           detail = listener.bot.get_message(ctx.message_id)
           text = detail["body"]["content"]
           if "紧急" in text:
               listener.bot.forward_message(
                   ctx.message_id,
                   DUTY_USER_OPEN_ID,
                   receive_id_type="open_id"
               )
       finally:
           listener.bot.delete_reaction(ctx.message_id, reaction["reaction_id"])


场景 14：从消息上下文里继续下载资源
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @listener.listen(message_type="image")
   async def save_original(ctx: MessageContext):
       content = json.loads(ctx.content)
       listener.bot.download_message_resource(
           ctx.message_id,
           "image",
           "/tmp/original.png",
           content["image_key"]
       )


场景 15：交互卡片按钮 + 原卡片原位更新
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @listener.card_action_handler(verification_token="token_xxx")
   def on_action(card_event):
       open_message_id = card_event.event.context.open_message_id
       listener.bot.update_interactive_card(
           open_message_id,
           {
               "type": "template",
               "data": {
                   "template_id": "AAqC5c999",
                   "template_variable": {"status": "完成"}
               }
           }
       )
       return {"toast": {"type": "success", "content": "已更新"}}


推荐组合方式
------------

如果你只想做“收到消息就按类型回复”，优先用：

- ``listen(message_type="text" | "image" | "file" | "audio" | "post" | "interactive")``
- 因为你通常还要拿 ``message_id`` 做 ``reply_message`` 和 ``add_reaction``

如果你更在意附件自动下载和自动回传，优先用：

- ``image_handler``
- ``file_handler``
- ``audio_handler``
- ``media_handler``

如果你要的是事件驱动自动化，优先用：

- ``member_changed_handler``
- ``reaction_handler``
- ``message_read_handler``
- ``bot_added_handler``
- ``chat_updated_handler``


注意事项
--------

1. ``listen(message_type=...)`` 注册多个 handler 时，某条消息会依次经过多个处理器；不匹配的 handler 会直接跳过。
2. 自动下载类 decorator 会创建临时文件，处理结束后自动清理；如果你把返回路径指向了别的文件，也会尝试清理那个返回文件。
3. 如果你要“引用回复”某条消息，请始终保留 ``message_id``，不要只拿 ``chat_id``。
4. ``interactive`` 消息监听，处理的是“收到一条卡片消息”；卡片按钮点击不是消息，要走 ``card_action_handler``。
5. 私聊 / 群聊名称过滤便于配置，但不如按 ID 稳定；同名群、同名用户场景下建议自行增加二次校验。
