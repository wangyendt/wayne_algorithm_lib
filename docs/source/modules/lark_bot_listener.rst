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


重点能力
--------

如果你想快速抓住这个模块最核心的扩展能力，最值得关注的是这几组：

- 消息类型补齐

  - ``audio_handler``
  - ``media_handler``
  - ``sticker_handler``
  - ``listen(message_type="interactive")`` 可直接监听“收到一张卡片消息”

- 事件类型补齐

  - ``message_read_handler``
  - ``reaction_handler``
  - ``bot_added_handler`` / ``bot_removed_handler``
  - ``bot_p2p_chat_entered_handler``
  - ``member_changed_handler``
  - ``chat_updated_handler`` / ``chat_disbanded_handler``

- 机器人被 @ 的专门入口

  - ``mention_handler`` 只处理“文本里真正 @ 到机器人”的消息

- 基于 ``MessageContext`` 的流式卡片回复

  - ``reply_streaming_card``
  - ``update_streaming_card``
  - ``recolor_streaming_card``
  - ``stream_reply_card``
  - ``astream_reply_card``

- 流式卡片状态色切换

  - 流式进行中可用 ``template="blue"``
  - 完成时可用 ``final_template="green"``
  - 失败时可手动 ``recolor_streaming_card(..., template="red")``

- 卡片交互 HTTP 回调

  - ``card_action_handler`` + ``get_card_action_handler`` 可直接接入 Web 框架
  - 适合做按钮点击、表单提交、审批确认、二次操作等

最常见的闭环组合一般是：

- 收到消息 -> 临时加 reaction -> 引用回复 -> 取消 reaction
- 收到文本 -> 走 LLM 流式输出 -> 在同一张卡片内持续更新 -> 完成时变绿色
- 收到文件 / 图片 / 语音 -> 自动下载 -> 处理 -> 返回新文件路径自动回发
- 机器人被拉进群 -> 自动发欢迎说明
- 群成员变化 / 已读 / reaction 事件 -> 同步外部业务系统


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

示例：用 ``MessageContext`` 同时处理线程、群私聊、@ 信息

.. code-block:: python

   @listener.listen()
   async def inspect_context(ctx: MessageContext):
       if ctx.thread_id:
           wayne_print(
               f"线程消息 root={ctx.root_id} parent={ctx.parent_id}",
               color="yellow",
               bold=True
           )

       if ctx.mentions:
           wayne_print(f"本条消息 mentions={ctx.mentions}", color="green")

       if ctx.is_group:
           listener.bot.reply_message(
               ctx.message_id,
               "text",
               {"text": "这是群聊消息"}
           )
       else:
           listener.bot.reply_message(
               ctx.message_id,
               "text",
               {"text": "这是私聊消息"}
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

示例：只监听群聊文本，并且在线程中继续回复

.. code-block:: python

   @listener.listen(message_type="text", group_only=True)
   async def reply_in_thread_when_possible(ctx: MessageContext):
       listener.bot.reply_message(
           ctx.message_id,
           "text",
           {"text": f"群消息收到：{ctx.content}"},
           reply_in_thread=bool(ctx.thread_id)
       )

示例：单个 handler 做完整分流

.. code-block:: python

   @listener.listen()
   async def smart_router(ctx: MessageContext):
       if ctx.message_type == "text" and ctx.content.startswith("/pin "):
           reply = listener.bot.reply_message(
               ctx.message_id,
               "text",
               {"text": ctx.content[5:]}
           )
           listener.bot.pin_message(reply["message_id"])
           return

       if ctx.message_type == "post":
           listener.bot.add_reaction(ctx.message_id, "THUMBSUP")
           return

       if ctx.message_type == "interactive":
           listener.bot.reply_message(
               ctx.message_id,
               "text",
               {"text": "收到一条卡片消息"}
           )


发送 Markdown 回复
~~~~~~~~~~~~~~~~~~

.. py:method:: send_message(chat_id: str, content: str)

   一个轻量发送入口。内部使用 post + markdown 内容发送到指定 chat。

   它适合快速调试，不适合复杂业务。更推荐在正式业务中直接使用 ``listener.bot`` 里的完整接口，比如：

   - ``reply_message``
   - ``send_markdown_to_chat``
   - ``send_interactive_to_chat``


流式卡片回复
~~~~~~~~~~~~

``LarkBotListener`` 现在也提供了围绕 ``MessageContext`` 的流式卡片便捷方法：

- ``reply_streaming_card(target, ...)``
- ``update_streaming_card(card_message_id, md_text, ...)``
- ``recolor_streaming_card(card_message_id, md_text, ...)``
- ``stream_reply_card(target, text_stream, ...)``
- ``astream_reply_card(target, text_stream, ...)``

其中：

- ``target`` 可以直接传 ``MessageContext``，也可以传原始 ``message_id``
- ``update_streaming_card`` 更新的是“回复出来的卡片消息 ID”，不是原始用户消息 ID
- ``final_template`` 可用于在流式结束后自动切换卡片头部颜色，比如从 ``blue`` 切到 ``green``
- 常用模板色包括 ``blue``、``green``、``orange``、``red``、``grey`` 等；完整常用列表可用 ``CardContentV2.list_header_templates()`` 查看
- 流式更新传入的是“当前完整文本”，不是本次新增的 delta
- 如果你需要 1 个字 1 个字刷新，请把 ``update_interval`` 调小，但要注意单消息频率限制

示例：在监听器里直接接 LLM 流式输出

.. code-block:: python

   @listener.listen(message_type="text")
   async def on_text(ctx: MessageContext):
       async def fake_stream():
           yield "你好，"
           yield "这是"
           yield "一段流式卡片回复。"

       await listener.astream_reply_card(
           ctx,
           fake_stream(),
           title="AI 回复中",
           template="blue",
           final_template="green",
           final_status_text="已完成"
       )

示例：一边流式输出，一边在完成后自动变色

.. code-block:: python

   @listener.listen(message_type="text")
   async def on_text_stream(ctx: MessageContext):
       async def llm_stream():
           yield "第一段分析\n"
           yield "第二段分析\n"
           yield "第三段结论\n"

       await listener.astream_reply_card(
           ctx,
           llm_stream(),
           title="问题分析中",
           template="wathet",
           status_text="生成中...",
           final_template="green",
           final_status_text="分析完成",
           update_interval=0.4
       )

示例：先创建卡片，再手动多次更新

.. code-block:: python

   @listener.listen(message_type="text")
   async def on_text_manual_stream(ctx: MessageContext):
       reply = listener.reply_streaming_card(
           ctx,
           title="任务执行中",
           template="blue",
           initial_md="开始处理请求..."
       )

       card_message_id = reply["message_id"]
       listener.update_streaming_card(
           card_message_id,
           "步骤 1：读取输入完成",
           title="任务执行中"
       )
       listener.update_streaming_card(
           card_message_id,
           "步骤 1：读取输入完成\n步骤 2：执行分析完成",
           title="任务执行中"
       )
       listener.recolor_streaming_card(
           card_message_id,
           "步骤 1：读取输入完成\n步骤 2：执行分析完成\n步骤 3：输出结果完成",
           title="任务已完成",
           template="green",
           status_text="完成"
       )

示例：任务失败时把已有卡片改成红色

.. code-block:: python

   @listener.listen(message_type="text")
   async def on_text(ctx: MessageContext):
       reply = listener.reply_streaming_card(
           ctx,
           title="任务执行中",
           initial_md="开始处理..."
       )

       card_message_id = reply["message_id"]

       try:
           listener.update_streaming_card(
               card_message_id,
               "步骤 1 完成\\n步骤 2 完成",
               title="任务执行中"
           )
           raise RuntimeError("第三步失败")
       except Exception as exc:
           listener.recolor_streaming_card(
               card_message_id,
               f"步骤 1 完成\\n步骤 2 完成\\n\\n错误信息：{exc}",
               title="任务执行失败",
               template="red",
               status_text="执行失败"
           )


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

示例：文本指令 + 引用回复 + 群私聊差异化处理

.. code-block:: python

   @listener.text_handler()
   async def handle_text(text: str,
                         chat_id: str,
                         is_group: bool,
                         group_name: str,
                         user_name: str,
                         message_id: str):
       if text == "/whoami":
           target = f"群聊 {group_name}" if is_group else "私聊"
           listener.bot.reply_message(
               message_id,
               "text",
               {"text": f"{user_name}，你当前在{target}"}
           )
       elif text.startswith("/say "):
           listener.bot.send_text_to_chat(chat_id, text[5:])


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

示例：图片消息自动下载后，引用回复说明，再把处理结果回发

.. code-block:: python

   @listener.image_handler()
   async def detect_and_reply(image_path: Path, message_id: str, user_name: str) -> Path:
       listener.bot.reply_message(
           message_id,
           "text",
           {"text": f"{user_name}，图片已收到，开始处理"}
       )
       image = cv2.imread(str(image_path))
       cv2.rectangle(image, (20, 20), (220, 220), (0, 255, 0), 3)
       with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
           out = Path(f.name)
       cv2.imwrite(str(out), image)
       return out


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

示例：文件自动落盘后解析大小并引用回复

.. code-block:: python

   @listener.file_handler()
   async def summarize_file(file_path: Path, message_id: str, user_name: str):
       size = file_path.stat().st_size
       listener.bot.reply_message(
           message_id,
           "text",
           {"text": f"{user_name}，文件已收到，大小 {size} 字节"}
       )


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

示例：语音消息 -> 临时 reaction -> 引用回复 -> 转成别名文件回传

.. code-block:: python

   import shutil
   import tempfile
   from pathlib import Path

   @listener.audio_handler()
   async def handle_audio(audio_path: Path, message_id: str) -> Path:
       reaction = listener.bot.add_reaction(message_id, "WITTY")
       try:
           listener.bot.reply_message(message_id, "text", {"text": "收到音频，已开始处理"})
           with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f:
               out = Path(f.name)
           shutil.copy(audio_path, out)
           return out
       finally:
           listener.bot.delete_reaction(message_id, reaction["reaction_id"])


media_handler
~~~~~~~~~~~~~

.. py:method:: media_handler(group_only: bool = False, user_only: bool = False)

   自动下载视频 / 媒体文件，默认临时扩展名为 ``.mp4``；若返回新路径，会按 ``mp4`` 上传回发。

   可声明参数与 ``audio_handler`` 类似，只是资源参数名为 ``media_path``。

示例：收到视频后只回元信息，不回文件

.. code-block:: python

   @listener.media_handler()
   async def handle_media(media_path, message_id):
       size = media_path.stat().st_size
       listener.bot.reply_message(
           message_id,
           "text",
           {"text": f"收到媒体文件，大小 {size} 字节"}
       )


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

示例：不同贴纸触发不同文案

.. code-block:: python

   @listener.sticker_handler(group_only=True)
   async def respond_sticker(sticker_content, message_id: str):
       listener.bot.reply_message(
           message_id,
           "text",
           {"text": f"收到贴纸：{sticker_content}"}
       )


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

示例：群里只对被 @ 的消息做流式卡片回复

.. code-block:: python

   @listener.mention_handler(group_only=True)
   async def mention_stream(text: str, message_id: str):
       def answer_stream():
           yield "已收到 @ 请求\n"
           yield f"原始文本：{text}\n"
           yield "处理中完成"

       listener.stream_reply_card(
           message_id,
           answer_stream(),
           title="被 @ 后的自动处理",
           template="blue",
           final_template="green",
           final_status_text="已回复"
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

示例：撤回事件写审计日志

.. code-block:: python

   from pywayne.tools import wayne_print

   @listener.recall_handler()
   async def on_recalled(message_id: str, chat_id: str, recall_time: str):
       wayne_print(
           f"消息被撤回 chat={chat_id} message={message_id} time={recall_time}",
           color="red",
           bold=True
       )


message_read_handler
~~~~~~~~~~~~~~~~~~~~

.. py:method:: message_read_handler()

   处理消息已读事件。

   可声明参数：

   - ``reader``
   - ``message_id_list``
   - ``raw_event``

示例：把已读事件同步到外部状态表

.. code-block:: python

   READ_STATE = {}

   @listener.message_read_handler()
   async def on_read(message_id_list, reader):
       reader_open_id = reader["reader_id"]["open_id"]
       for message_id in message_id_list:
           READ_STATE[message_id] = reader_open_id


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

示例：区分 reaction 创建和删除

.. code-block:: python

   @listener.reaction_handler()
   async def on_reaction(action: str, message_id: str, emoji_type: str, operator_type: str):
       if action == "created":
           listener.bot.send_text_to_chat(
               "oc_xxx",
               f"{message_id} 新增 reaction: {emoji_type} by {operator_type}"
           )
       else:
           listener.bot.send_text_to_chat(
               "oc_xxx",
               f"{message_id} 移除了 reaction: {emoji_type}"
           )


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

示例：入群时自动发欢迎卡片，移除时打日志

.. code-block:: python

   from pywayne.tools import wayne_print

   @listener.bot_added_handler()
   async def on_bot_added(chat_id: str, name: str):
       listener.bot.send_markdown_to_chat(
           chat_id,
           md_text=f"# {name}\n\n机器人已加入当前群，可直接开始使用。",
           title="机器人入群"
       )

   @listener.bot_removed_handler()
   async def on_bot_removed(chat_id: str, operator_id: str):
       wayne_print(f"机器人被移出群 {chat_id}, operator={operator_id}", color="yellow", bold=True)


bot_p2p_chat_entered_handler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: bot_p2p_chat_entered_handler()

   处理机器人首次进入某个用户私聊的事件。

   适合做“首次欢迎语”场景。

示例：首次进入私聊就发 onboarding

.. code-block:: python

   @listener.bot_p2p_chat_entered_handler()
   async def on_enter_p2p(chat_id: str, operator_id: str):
       listener.bot.send_markdown_to_chat(
           chat_id,
           md_text=(
               "# 欢迎使用机器人\n\n"
               "- 直接发文本可触发问答\n"
               "- 发图片 / 文件可触发对应处理链路\n"
               "- 支持流式卡片输出"
           ),
           title=f"欢迎 {operator_id}"
       )


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

示例：成员加入 / 退出时同步群人数缓存

.. code-block:: python

   GROUP_MEMBER_COUNT = {}

   @listener.member_changed_handler()
   async def on_member_change(action: str, chat_id: str, users):
       delta = len(users or [])
       GROUP_MEMBER_COUNT.setdefault(chat_id, 0)
       if action == "added":
           GROUP_MEMBER_COUNT[chat_id] += delta
       elif action in {"deleted", "withdrawn"}:
           GROUP_MEMBER_COUNT[chat_id] = max(0, GROUP_MEMBER_COUNT[chat_id] - delta)


chat_updated_handler / chat_disbanded_handler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: chat_updated_handler()
.. py:method:: chat_disbanded_handler()

   用于监听群资料变更与群解散。

示例：群名变化后同步本地配置，群解散后清理状态

.. code-block:: python

   CHAT_META = {}

   @listener.chat_updated_handler()
   async def on_chat_updated(chat_id: str, before_change, after_change):
       CHAT_META[chat_id] = after_change

   @listener.chat_disbanded_handler()
   async def on_chat_disbanded(chat_id: str):
       CHAT_META.pop(chat_id, None)


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

示例：按钮点击后先返回 toast，再把原卡片改成绿色完成态

.. code-block:: python

   @listener.card_action_handler(verification_token="token_xxx")
   def on_finish(card_event):
       open_message_id = card_event.event.context.open_message_id
       listener.bot.update_streaming_card(
           open_message_id,
           "任务已确认完成",
           title="任务状态",
           template="green",
           done=True,
           status_text="已完成"
       )
       return {"toast": {"type": "success", "content": "已确认完成"}}


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


场景 2.1：收到文本后直接流式卡片回复
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @listener.listen(message_type="text")
   async def on_text_stream(ctx: MessageContext):
       async def llm_stream():
           yield "第一段输出\n"
           yield "第二段输出\n"
           yield "第三段输出\n"

       await listener.astream_reply_card(
           ctx,
           llm_stream(),
           title="分析中",
           status_text="生成中...",
           final_status_text="已完成"
       )


场景 2.2：逐字流式卡片回复，完成后自动变绿色
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio

   @listener.listen(message_type="text")
   async def on_text_char_stream(ctx: MessageContext):
       async def char_stream():
           sentence = "这是一个逐字流式卡片输出示例。"
           for ch in sentence:
               yield ch
               await asyncio.sleep(0.3)

       await listener.astream_reply_card(
           ctx,
           char_stream(),
           title="逐字输出中",
           template="blue",
           status_text="生成中...",
           final_template="green",
           final_status_text="已完成",
           update_interval=0.25
       )


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


场景 7.1：机器人首次进入私聊时主动发 onboarding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @listener.bot_p2p_chat_entered_handler()
   async def on_private_chat_ready(chat_id: str):
       listener.bot.send_text_to_chat(chat_id, "你好，我已经进入这个私聊，可以直接发消息给我。")


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


场景 10.1：收到文本先加 reaction，后续依据 reaction 事件做监控
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pywayne.tools import wayne_print

   @listener.listen(message_type="text")
   async def on_text(ctx: MessageContext):
       listener.bot.add_reaction(ctx.message_id, "OK")

   @listener.reaction_handler()
   async def monitor_reaction(action: str, message_id: str, emoji_type: str):
       wayne_print(f"{message_id} reaction {action}: {emoji_type}", color="cyan")


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


场景 16：同一个监听器里同时处理文本、图片、文件、音频、post、card
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio

   async def ack_with_temp_reaction(ctx: MessageContext, text: str):
       reaction = listener.bot.add_reaction(ctx.message_id, "WITTY")
       try:
           await asyncio.sleep(1)
           listener.bot.reply_message(ctx.message_id, "text", {"text": text})
       finally:
           listener.bot.delete_reaction(ctx.message_id, reaction["reaction_id"])

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


场景 17：按群名和用户名双重过滤，再进入统一处理链路
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ALLOWED_GROUP_NAMES = {"测试3", "项目群"}
   ALLOWED_PRIVATE_USER_NAMES = {"Wayne", "Alice"}

   def allow(group_name: str, user_name: str, is_group: bool) -> bool:
       if is_group:
           return group_name in ALLOWED_GROUP_NAMES
       return user_name in ALLOWED_PRIVATE_USER_NAMES

   @listener.listen()
   async def guarded_router(ctx: MessageContext):
       group_name, user_name = listener.bot.get_chat_and_user_name(ctx.chat_id, ctx.user_id)
       if not allow(group_name, user_name, ctx.is_group):
           return

       listener.bot.reply_message(
           ctx.message_id,
           "text",
           {"text": f"允许访问：{user_name}"}
       )


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
