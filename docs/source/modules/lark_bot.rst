飞书机器人 (lark_bot)
=====================

``pywayne.lark_bot`` 是应用机器人侧的主能力封装，覆盖主动发消息、引用回复、消息编辑、撤回、reaction、置顶、群管理、资源上传下载、历史消息查询等常见飞书 IM 能力。

如果把整个模块按对象分层，可以理解为：

- ``TextContent``: 文本格式工具，解决 @、加粗、链接等轻量文本拼接
- ``PostContent``: rich_text 富文本构造器，适合做结构化文本、代码块、Markdown 表格降级
- ``CardContentV2``: schema 2.0 卡片构造器，适合大段 Markdown 和轻交互展示
- ``LarkBot``: 统一 API 封装，负责真正和飞书 OpenAPI 通信


能力概览
--------

``LarkBot`` 当前重点覆盖这些能力：

- 主动发送消息

  - 文本
  - 图片
  - 音频
  - 媒体
  - 文件
  - rich_text 富文本
  - card 卡片
  - 分享群聊
  - 分享用户
  - 系统消息
  - Markdown 自动路由发送

- 消息后处理

  - 引用回复
  - 线程内回复
  - 转发消息
  - 撤回消息
  - 获取单条消息
  - 获取历史消息列表
  - 编辑文本 / rich_text / card 消息
  - 原位更新 card 卡片
  - 查询已读用户
  - 加急消息
  - 加 / 删 / 查 reaction
  - 置顶 / 取消置顶 / 查询置顶

- 群管理

  - 创建群聊
  - 删除群聊
  - 修改群名称 / 描述 / 头像 / 群主
  - 拉人入群
  - 移出成员
  - 设置 / 取消管理员
  - 转让群主
  - 获取群公告
  - patch 群公告

- 资源与基础信息

  - 上传 / 下载图片
  - 上传 / 下载文件
  - 下载消息资源
  - 批量下载消息内资源
  - 查询用户信息
  - 查询群列表
  - 通过群名找群 ID
  - 取群成员
  - 通过成员名找 open_id
  - 获取群名和用户名


重点能力
--------

如果你想快速抓住这个模块最核心的扩展能力，优先关注这几组：

- 消息生命周期补齐

  - ``reply_message``
  - ``forward_message``
  - ``recall_message``
  - ``get_message``
  - ``get_message_list``
  - ``edit_text_message``
  - ``edit_post_message``
  - ``edit_card_message``

- 消息状态操作

  - ``get_message_read_users``
  - ``urgent_message``
  - ``add_reaction`` / ``delete_reaction`` / ``list_reactions``
  - ``pin_message`` / ``unpin_message`` / ``list_pinned_messages``

- 流式卡片回复

  - ``reply_streaming_card``
  - ``update_streaming_card``
  - ``recolor_streaming_card``
  - ``stream_reply_card``
  - ``astream_reply_card``

- 群管理增强

  - ``create_chat`` / ``update_chat`` / ``delete_chat``
  - ``add_members_to_chat`` / ``remove_members_from_chat``
  - ``set_chat_admin`` / ``transfer_chat_owner``
  - ``get_chat_announcement`` / ``set_chat_announcement``

- 批量发消息

  - ``batch_send_message`` 支持按用户 / 部门触达

如果你在做机器人业务闭环，最常见的组合一般是：

- 收到用户消息 -> ``reply_message`` 引用回复
- 长任务开始 -> ``reply_streaming_card`` -> 多次 ``update_streaming_card`` -> 完成后 ``recolor_streaming_card(..., template="green")``
- 告警消息 -> ``forward_message`` 给值班人 -> ``urgent_message`` 加急
- 重要结论 -> ``reply_message`` -> ``pin_message``
- 任务处理中 -> ``add_reaction`` -> 完成 / 失败后 ``delete_reaction``
- 项目群初始化 -> ``create_chat`` -> ``add_members_to_chat`` -> ``set_chat_admin`` -> ``send_card_to_chat``


快速开始
--------

.. code-block:: python

   from pywayne.lark_bot import LarkBot

   bot = LarkBot(app_id="cli_xxx", app_secret="sec_xxx")

   bot.send_text_to_user("ou_xxx", "你好")
   bot.send_text_to_chat("oc_xxx", "群消息测试")


TextContent
-----------

``TextContent`` 适合快速拼出飞书文本消息里常见的富文本标记。

常用方法：

- ``make_at_all_pattern()``
- ``make_at_someone_pattern(someone_open_id, username, id_type)``
- ``make_bold_pattern(content)``
- ``make_italian_pattern(content)``
- ``make_underline_pattern(content)``
- ``make_delete_line_pattern(content)``
- ``make_url_pattern(url, text)``

示例：构造一条带 @ 和链接的文本

.. code-block:: python

   from pywayne.lark_bot import LarkBot, TextContent

   bot = LarkBot(app_id="cli_xxx", app_secret="sec_xxx")

   msg = (
       TextContent.make_at_someone_pattern("ou_xxx", "Wayne", "open_id")
       + " "
       + TextContent.make_bold_pattern("发布完成")
       + "，请查看 "
       + TextContent.make_url_pattern("https://example.com", "详情页")
   )
   bot.send_text_to_chat("oc_xxx", msg)


PostContent
-----------

``PostContent`` 适合构造复杂 rich_text 富文本，尤其适合：

- 多段结构化说明
- 一行混排文字 / 链接 / @ / 图片
- 代码块
- Markdown 分块发送
- Markdown 表格降级为等宽代码块

常用方法：

- ``make_text_content``
- ``make_link_content``
- ``make_at_content``
- ``make_image_content``
- ``make_media_content``
- ``make_emoji_content``
- ``make_hr_content``
- ``make_code_block_content``
- ``make_markdown_content``
- ``add_markdown``
- ``add_content_in_line``
- ``add_contents_in_line``
- ``add_content_in_new_line``
- ``add_contents_in_new_line``
- ``list_emoji_types``

示例：混排 rich_text 消息

.. code-block:: python

   from pywayne.lark_bot import LarkBot, PostContent

   bot = LarkBot(app_id="cli_xxx", app_secret="sec_xxx")

   post = PostContent(title="发布通知")
   post.add_contents_in_new_line([
       post.make_text_content("构建完成", styles=["bold"]),
       post.make_emoji_content("OK"),
   ])
   post.add_contents_in_new_line([
       post.make_link_content("查看 Jenkins", "https://jenkins.example.com"),
       post.make_at_content("ou_xxx"),
   ])
   post.add_content_in_new_line(
       post.make_code_block_content("bash", "deploy.sh --env prod")
   )

   bot.send_rich_text_to_chat("oc_xxx", post.get_content())

示例：Markdown 表格安全降级

.. code-block:: python

   md = """
   ## 回归结果

   | 模块 | 结果 | 负责人 |
   | --- | --- | --- |
   | 登录 | pass | Alice |
   | 支付 | pass | Bob |
   | 推荐 | running | Carol |
   """

   post = PostContent(title="测试日报")
   post.add_markdown(md, table_as="code_block", max_chunk_bytes=8000)
   bot.send_rich_text_to_chat("oc_xxx", post.get_content())


CardContentV2
-------------

``CardContentV2`` 是面向 schema 2.0 card 卡片的轻量构造器，适合大段 Markdown 公告、日报、状态展示。

常用方法：

- ``add_markdown``
- ``add_hr``
- ``add_image``
- ``get_card``
- ``list_header_templates``

示例：日报卡片

.. code-block:: python

   from pywayne.lark_bot import CardContentV2

   card = CardContentV2(title="日报", template="blue")
   card.add_markdown("# 今日进展\n\n- 接口联调完成\n- 修复 3 个问题\n- 代码已合并")
   card.add_hr()
   card.add_image("img_xxx")

   bot.send_card_to_chat("oc_xxx", card.get_card())

示例：查看常用卡片头部模板色

.. code-block:: python

   from pywayne.tools import wayne_print

   wayne_print(CardContentV2.list_header_templates(), color="cyan")


LarkBot 类
----------

.. py:class:: LarkBot(app_id: str, app_secret: str)

   主应用机器人封装。


发送消息
--------

文本
~~~~

- ``send_text_to_user(user_open_id, text)``
- ``send_text_to_chat(chat_id, text)``

.. code-block:: python

   bot.send_text_to_user("ou_xxx", "私聊你好")
   bot.send_text_to_chat("oc_xxx", "群里你好")


图片
~~~~

- ``upload_image(image_path)``
- ``send_image_to_user(user_open_id, image_key)``
- ``send_image_to_chat(chat_id, image_key)``
- ``download_image(image_key, image_save_path)``

.. code-block:: python

   image_key = bot.upload_image("/tmp/report.png")
   bot.send_image_to_chat("oc_xxx", image_key)


音频 / 媒体 / 文件
~~~~~~~~~~~~~~~~~~

- ``upload_file(file_path, file_type="stream")``
- ``send_audio_to_user`` / ``send_audio_to_chat``
- ``send_media_to_user`` / ``send_media_to_chat``
- ``send_file_to_user`` / ``send_file_to_chat``
- ``download_file(file_key, file_save_path)``

.. code-block:: python

   opus_key = bot.upload_file("/tmp/voice.opus", file_type="opus")
   bot.send_audio_to_chat("oc_xxx", opus_key)

   mp4_key = bot.upload_file("/tmp/demo.mp4", file_type="mp4")
   bot.send_media_to_chat("oc_xxx", mp4_key)

   pdf_key = bot.upload_file("/tmp/spec.pdf", file_type="pdf")
   bot.send_file_to_chat("oc_xxx", pdf_key)


rich_text 富文本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``send_rich_text_to_user``
- ``send_rich_text_to_chat``

.. code-block:: python

   post = PostContent(title="值班提醒")
   post.add_content_in_new_line(post.make_text_content("今晚 20:00 发布", styles=["bold"]))
   bot.send_rich_text_to_chat("oc_xxx", post.get_content())


card 卡片
~~~~~~~~~~~~~~~~

- ``send_card_to_user``
- ``send_card_to_chat``

.. code-block:: python

   bot.send_card_to_chat(
       "oc_xxx",
       {
           "header": {"title": {"content": "状态卡片", "tag": "plain_text"}},
           "elements": [{"tag": "markdown", "content": "**服务正常**"}]
       }
   )


分享消息
~~~~~~~~

- ``share_chat_to_user`` / ``share_chat_to_chat``
- ``share_user_to_user`` / ``share_user_to_chat``


系统消息
~~~~~~~~

- ``send_system_message_to_user``

适合做系统通知类场景，不走普通文本样式。


推荐入口：send_markdown_message_to_chat
----------------------------------------

.. py:method:: send_markdown_message_to_chat(chat_id: str, md_text: str, *, title: str = "", prefer: str = "card_v2", table_fallback: str = "code_block", max_message_bytes: Optional[int] = None)

这是推荐的高层发送入口，适合绝大多数“我要发 Markdown”场景。

特性：

- 自动按字节分包
- 支持 ``card_v2`` 与 ``post`` 两条发送路由
- ``post`` 路由下支持表格降级

示例 1：默认发 schema 2.0 卡片

.. code-block:: python

   bot.send_markdown_message_to_chat(
       "oc_xxx",
       md_text="# 发布完成\n\n- API: pass\n- Worker: pass",
       title="发布结果"
   )

示例 2：强制走 rich_text 路由，并处理 Markdown 表格

.. code-block:: python

   bot.send_markdown_message_to_chat(
       "oc_xxx",
       md_text="""
       # 回归看板

       | 模块 | 状态 |
       | --- | --- |
       | 登录 | pass |
       | 支付 | pass |
       """,
       title="回归结果",
       prefer="post",
       table_fallback="code_block"
   )

示例 3：超长日报自动分片

.. code-block:: python

   bot.send_markdown_message_to_chat(
       "oc_xxx",
       md_text=very_long_markdown,
       title="长文日报",
       prefer="card_v2",
       max_message_bytes=12000
   )


消息后处理
----------

引用回复
~~~~~~~~

.. py:method:: reply_message(message_id: str, msg_type: str, content, *, reply_in_thread: bool = False, uuid: str = "")

这是监听场景里最常用的方法。

示例：引用回复文本

.. code-block:: python

   bot.reply_message(
       message_id="om_xxx",
       msg_type="text",
       content={"text": "收到文本"}
   )

示例：在线程中回复

.. code-block:: python

   bot.reply_message(
       message_id="om_xxx",
       msg_type="text",
       content={"text": "这条回复会进入 thread"},
       reply_in_thread=True
   )

示例：引用回复卡片

.. code-block:: python

   card = CardContentV2(title="处理结果")
   card.add_markdown("已收到你的请求，正在执行。")
   bot.reply_message("om_xxx", "interactive", card.get_card())

示例：按消息类型动态引用回复

.. code-block:: python

   def reply_by_type(message_id: str, kind: str):
       mapping = {
           "text": "收到文本",
           "image": "收到图片",
           "file": "收到文件",
           "audio": "收到音频",
           "post": "收到 rich_text",
           "interactive": "收到 card",
       }
       bot.reply_message(
           message_id,
           "text",
           {"text": mapping.get(kind, f"收到 {kind}")}
       )


转发消息
~~~~~~~~

.. py:method:: forward_message(message_id: str, receive_id: str, *, receive_id_type: str = "chat_id", uuid: str = "")

示例：把群里的告警转发给值班人

.. code-block:: python

   bot.forward_message(
       message_id="om_alert_xxx",
       receive_id="ou_duty_xxx",
       receive_id_type="open_id"
   )

示例：先引用回复，再把原始消息转发给二线支持

.. code-block:: python

   bot.reply_message("om_xxx", "text", {"text": "已转交二线支持"})
   bot.forward_message("om_xxx", "ou_level2_xxx", receive_id_type="open_id")


撤回消息
~~~~~~~~

.. py:method:: recall_message(message_id: str)

适合“发错消息立即撤回”或“临时状态消息在成功后撤回”。

示例：短暂提示后立即撤回

.. code-block:: python

   sent = bot.send_text_to_chat("oc_xxx", "这是一条临时提示")
   bot.recall_message(sent["message_id"])


查询消息
~~~~~~~~

- ``get_message(message_id)``
- ``get_message_list(chat_id, start_time, end_time, sort_type="", page_size=50, page_token="")``

示例：拉取某时间范围内的历史消息

.. code-block:: python

   history = bot.get_message_list(
       chat_id="oc_xxx",
       start_time="1735603200000",
       end_time="1735689600000",
       sort_type="ByCreateTimeAsc"
   )

示例：先查单条消息，再决定是否转发

.. code-block:: python

   detail = bot.get_message("om_xxx")
   content = detail["body"]["content"]
   if "紧急" in content:
       bot.forward_message("om_xxx", "ou_duty_xxx", receive_id_type="open_id")


编辑消息
~~~~~~~~

- ``edit_text_message(message_id, text)``
- ``edit_post_message(message_id, post_content)``
- ``edit_card_message(message_id, card)``

注意：

- ``edit_text_message`` / ``edit_post_message`` 底层对应飞书 ``PUT /im/v1/messages/:message_id``，只适用于 ``text`` / ``post`` 消息，也就是文本 / rich_text 消息
- ``edit_card_message`` 底层对应飞书卡片更新接口，适用于交互式卡片；做“同一条消息持续刷新”的效果时，通常也是走卡片更新
- 飞书官方限制一条消息最多编辑 ``20`` 次；且只能编辑自己发送、未撤回、未超出可编辑时间的消息
- 文本 / 富文本和卡片更新接口是分开的，不能混用；例如不能拿卡片 JSON 去调 ``edit_text_message`` / ``edit_post_message``
- 如果你只是想按消息类型做原位编辑，优先使用这 3 个 ``edit_*`` 接口，不再需要区分旧的底层方法名

示例：把“处理中”卡片更新成“已完成”

.. code-block:: python

   bot.edit_card_message(
       message_id="om_xxx",
       card={
           "type": "template",
           "data": {
               "template_id": "AAqC5c999",
               "template_variable": {"status": "已完成"}
           }
       }
   )

示例：先发文本，再更新文本内容

.. code-block:: python

   sent = bot.send_text_to_chat("oc_xxx", "初版结论")
   bot.edit_text_message(sent["message_id"], "初版结论\n补充说明：影响范围仅限灰度环境")

示例：先发富文本，再更新富文本内容

.. code-block:: python

   post = PostContent(title="阶段播报")
   post.add_markdown("第一版结果")
   sent = bot.send_rich_text_to_chat("oc_xxx", post.get_content())

   post2 = PostContent(title="阶段播报")
   post2.add_markdown("第一版结果\n\n补充：已完成二次校验")
   bot.edit_post_message(sent["message_id"], post2.get_content())

示例：直接编辑已发送的卡片

.. code-block:: python

   bot.edit_card_message(
       "om_xxx",
       {
           "type": "template",
           "data": {
               "template_id": "AAqC5c999",
               "template_variable": {"status": "处理中", "detail": "第 2 步 / 共 5 步"}
           }
       }
   )


已读与加急
~~~~~~~~~~

- ``get_message_read_users(message_id, ...)``
- ``urgent_message(message_id, urgent_type, user_open_ids, ...)``

``urgent_type`` 支持：

- ``app``
- ``phone``
- ``sms``

示例：先发消息，再给值班人加急

.. code-block:: python

   sent = bot.send_text_to_chat("oc_xxx", "生产告警，请处理")
   bot.urgent_message(
       sent["message_id"],
       urgent_type="app",
       user_open_ids=["ou_duty_xxx"]
   )

示例：查询哪些人已读，再决定是否继续催办

.. code-block:: python

   readers = bot.get_message_read_users("om_xxx")
   if not readers.get("items"):
       bot.urgent_message("om_xxx", "sms", ["ou_duty_xxx"])


reaction
~~~~~~~~

- ``add_reaction(message_id, emoji_type)``
- ``delete_reaction(message_id, reaction_id)``
- ``list_reactions(message_id, ...)``

``emoji_type`` 使用飞书 reaction code，不是 Unicode 字符。

常用值包括：

- ``THUMBSUP``
- ``OK``
- ``HEART``
- ``HAHA``
- ``WITTY``

完整列表可通过 ``PostContent.list_emoji_types()`` 打开飞书官方表情页。

示例：临时 reaction

.. code-block:: python

   reaction = bot.add_reaction("om_xxx", "THUMBSUP")
   bot.delete_reaction("om_xxx", reaction["reaction_id"])

示例：统计某条消息有哪些 reaction

.. code-block:: python

   from pywayne.tools import wayne_print

   data = bot.list_reactions("om_xxx")
   wayne_print(data, color="cyan")

示例：把 reaction 当作任务状态灯

.. code-block:: python

   reaction = bot.add_reaction("om_xxx", "WITTY")
   try:
       bot.reply_message("om_xxx", "text", {"text": "处理中"})
   except Exception:
       bot.add_reaction("om_xxx", "HAHA")
       raise
   finally:
       bot.delete_reaction("om_xxx", reaction["reaction_id"])


置顶
~~~~

- ``pin_message(message_id)``
- ``unpin_message(message_id)``
- ``list_pinned_messages(chat_id, ...)``

示例：先回复，再置顶

.. code-block:: python

   reply = bot.reply_message("om_xxx", "text", {"text": "这是最终结论"})
   bot.pin_message(reply["message_id"])

示例：查询置顶消息并输出摘要

.. code-block:: python

   from pywayne.tools import wayne_print

   pinned = bot.list_pinned_messages("oc_xxx")
   for item in pinned.get("items", []):
       wayne_print(item["message_id"], color="cyan")


流式卡片回复
~~~~~~~~~~~~

这一组方法用于“先回复一张卡片，再不断原位刷新同一张卡片内容”，适合接 LLM 流式输出。

- ``build_streaming_card(md_text, ...)``
- ``reply_streaming_card(message_id, ...)``
- ``update_streaming_card(message_id, md_text, ...)``
- ``recolor_streaming_card(message_id, md_text, ...)``
- ``stream_reply_card(source_message_id, text_stream, ...)``
- ``astream_reply_card(source_message_id, text_stream, ...)``

设计约束：

- 走的是“先发 card 卡片，再反复更新该消息”的路径
- 更新时传入的是“当前完整文本”，不是 delta
- 默认 ``update_interval=0.25``，是为了尽量避开单消息高频更新限制
- 卡片默认带 ``config.update_multi=true``
- 颜色不是任意 RGB，而是飞书卡片头部的预设模板色
- ``final_template`` 用于流式结束后的最终颜色
- ``recolor_streaming_card`` 适合手动切换为成功 / 失败 / 警告状态色

常用头部模板色：

- ``blue``
- ``wathet``
- ``turquoise``
- ``green``
- ``yellow``
- ``orange``
- ``red``
- ``carmine``
- ``violet``
- ``purple``
- ``indigo``
- ``grey``

你也可以直接调用 ``CardContentV2.list_header_templates()`` 获取当前内置常用列表。

示例 1：手工启动 + 手工更新

.. code-block:: python

   reply = bot.reply_streaming_card(
       "om_xxx",
       title="AI 回复中",
       initial_md="正在思考..."
   )

   card_message_id = reply["message_id"]
   bot.update_streaming_card(card_message_id, "第一段输出", title="AI 回复中")
   bot.update_streaming_card(card_message_id, "完整输出内容", title="AI 回复完成", done=True)

示例 2：完成后自动从蓝色切到绿色

.. code-block:: python

   def fake_stream():
       yield "第一段输出\\n"
       yield "第二段输出\\n"
       yield "第三段输出\\n"

   result = bot.stream_reply_card(
       "om_xxx",
       fake_stream(),
       title="AI 回复中",
       template="blue",
       final_template="green",
       status_text="生成中...",
       final_status_text="已完成"
   )

示例 3：失败时改成红色

.. code-block:: python

   reply = bot.reply_streaming_card(
       "om_xxx",
       title="任务执行中",
       template="blue",
       initial_md="开始执行..."
   )

   card_message_id = reply["message_id"]

   try:
       bot.update_streaming_card(card_message_id, "步骤 1 成功\\n步骤 2 成功", title="任务执行中")
       raise RuntimeError("第三步失败")
   except Exception as exc:
       bot.recolor_streaming_card(
           card_message_id,
           f"步骤 1 成功\\n步骤 2 成功\\n\\n错误信息：{exc}",
           title="任务执行失败",
           template="red",
           status_text="执行失败",
           done=True
       )

示例 4：逐字更新，完成后自动变绿色

.. code-block:: python

   import time

   def char_stream():
       sentence = "这是一个逐字流式卡片回复示例。"
       for ch in sentence:
           yield ch
           time.sleep(0.3)

   bot.stream_reply_card(
       "om_xxx",
       char_stream(),
       title="逐字输出中",
       template="blue",
       final_template="green",
       status_text="生成中...",
       final_status_text="已完成",
       update_interval=0.25
   )

示例 5：处理中为蓝色，待人工确认切橙色，最终完成切绿色

.. code-block:: python

   reply = bot.reply_streaming_card(
       "om_xxx",
       title="审批任务",
       template="blue",
       initial_md="系统已开始自动分析"
   )

   card_message_id = reply["message_id"]
   bot.update_streaming_card(
       card_message_id,
       "自动分析完成，等待人工确认",
       title="审批任务",
       template="orange",
       done=False,
       status_text="等待人工确认"
   )
   bot.recolor_streaming_card(
       card_message_id,
       "自动分析完成\n人工确认通过",
       title="审批任务",
       template="green",
       status_text="已完成"
   )

示例 6：直接消费同步生成器

.. code-block:: python

   def fake_stream():
       yield "你好，"
       yield "这是"
       yield "一段流式输出。"

   result = bot.stream_reply_card(
       "om_xxx",
       fake_stream(),
       title="AI 回复中"
   )

示例 7：异步生成器版本

.. code-block:: python

   async def fake_astream():
       yield "第一段"
       yield "第二段"

   result = await bot.astream_reply_card(
       "om_xxx",
       fake_astream(),
       title="AI 回复中",
       template="wathet",
       final_template="green",
       final_status_text="回答完成"
   )


群管理
------

创建 / 删除 / 更新群
~~~~~~~~~~~~~~~~~~~~

- ``create_chat``
- ``delete_chat``
- ``update_chat``
- ``transfer_chat_owner``

示例：创建一个项目群并设置群主

.. code-block:: python

   chat = bot.create_chat(
       name="项目 Alpha",
       user_open_ids=["ou_a", "ou_b", "ou_c"],
       description="Alpha 项目协作群",
       owner_open_id="ou_a"
   )

示例：修改群资料

.. code-block:: python

   bot.update_chat(
       chat_id="oc_xxx",
       name="项目 Alpha - 灰度群",
       description="用于灰度发布值守"
   )

示例：转让群主后再更新群描述

.. code-block:: python

   bot.transfer_chat_owner("oc_xxx", "ou_new_owner")
   bot.update_chat(
       chat_id="oc_xxx",
       description="群主已更新，请按新流程协作"
   )


成员管理
~~~~~~~~

- ``add_members_to_chat``
- ``remove_members_from_chat``
- ``set_chat_admin``

示例：加人并设置管理员

.. code-block:: python

   bot.add_members_to_chat("oc_xxx", ["ou_dev1", "ou_dev2"])
   bot.set_chat_admin("oc_xxx", ["ou_dev1"], is_admin=True)

示例：移除管理员权限

.. code-block:: python

   bot.set_chat_admin("oc_xxx", ["ou_dev1"], is_admin=False)

示例：拉人、踢人、再补发说明

.. code-block:: python

   bot.add_members_to_chat("oc_xxx", ["ou_new1", "ou_new2"])
   bot.remove_members_from_chat("oc_xxx", ["ou_old1"])
   bot.send_text_to_chat("oc_xxx", "成员已调整，请关注新的值班安排")


群公告
~~~~~~

- ``get_chat_announcement``
- ``set_chat_announcement``

``set_chat_announcement`` 走的是飞书 patch 语义，参数 ``requests`` 需要直接传飞书公告 API 的 patch 操作列表。

示例：读取公告

.. code-block:: python

   announcement = bot.get_chat_announcement("oc_xxx")

示例：用 patch 语义更新公告

.. code-block:: python

   bot.set_chat_announcement(
       "oc_xxx",
       requests=[
           {
               "op": "replace",
               "path": "/content",
               "value": "今晚 23:00 维护，请提前保存工作内容。"
           }
       ]
   )


资源与下载
----------

下载单条消息中的资源
~~~~~~~~~~~~~~~~~~~~

- ``download_message_resource(message_id, resource_type, save_path, file_key=None)``

``resource_type`` 常见取值：

- ``image``
- ``file``
- ``audio``
- ``media``
- ``video``

示例：下载图片消息中的原图

.. code-block:: python

   bot.download_message_resource(
       message_id="om_xxx",
       resource_type="image",
       save_path="/tmp/msg.png",
       file_key="img_xxx"
   )


批量下载消息中的全部资源
~~~~~~~~~~~~~~~~~~~~~~~~

- ``download_message_resources(message_id, message_content, save_dir)``

适合“你已经拿到消息正文 JSON，想把里面所有 file_key/image_key 对应的资源一次性落盘”。

示例：下载一条消息中的全部资源后统一归档

.. code-block:: python

   import json

   detail = bot.get_message("om_xxx")
   content = json.loads(detail["body"]["content"])
   bot.download_message_resources("om_xxx", content, "/tmp/msg_assets")


用户与群信息
------------

- ``get_user_info(emails, mobiles)``
- ``get_group_list()``
- ``find_chat_ids_by_name(group_name)``
- ``get_chat_members(chat_id)``
- ``find_member_open_ids_by_name(chat_id, member_name)``
- ``get_chat_and_user_name(chat_id, user_id)``

示例：按群名查 ID，再按用户名查成员 open_id

.. code-block:: python

   chat_ids = bot.find_chat_ids_by_name("项目 Alpha")
   if chat_ids:
       open_ids = bot.find_member_open_ids_by_name(chat_ids[0], "Wayne")

示例：同时查群名和发件人姓名

.. code-block:: python

   group_name, user_name = bot.get_chat_and_user_name("oc_xxx", "ou_xxx")


批量发送
--------

.. py:method:: batch_send_message(msg_type: str, *, content=None, card=None, user_open_ids=None, department_ids=None, user_ids=None, union_ids=None)

这不是发到群，而是批量发给用户或部门。

特点：

- 底层走 ``/message/v4/batch_send/``
- 支持文本和卡片
- 支持 open_id / user_id / union_id / department_id 目标
- 批量消息不是普通群消息，不能像普通消息那样引用回复

示例：批量发文本通知

.. code-block:: python

   bot.batch_send_message(
       "text",
       content="今晚 23:00 系统维护",
       user_open_ids=["ou_a", "ou_b", "ou_c"]
   )

示例：批量发卡片

.. code-block:: python

   bot.batch_send_message(
       "interactive",
       card={
           "header": {"title": {"content": "运维通知", "tag": "plain_text"}},
           "elements": [{"tag": "markdown", "content": "请及时确认"}]
       },
       department_ids=["od_xxx"]
   )

示例：批量发通知后轮询进度

.. code-block:: python

   from pywayne.tools import wayne_print

   result = bot.batch_send_message(
       "text",
       content="请在今晚 18:00 前完成确认",
       user_open_ids=["ou_a", "ou_b"]
   )
   task_id = result.get("task_id")
   if task_id:
       progress = bot.get_batch_message_progress(task_id)
       wayne_print(progress, color="cyan")


组合场景示例
------------

场景 1：发日报卡片，用户点击后再原位更新
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   card = CardContentV2(title="日报提交")
   card.add_markdown("请点击按钮确认今天已提交日报。")
   bot.send_card_to_chat("oc_xxx", card.get_card())

   # 按钮点击后的更新逻辑放到 LarkBotListener.card_action_handler 中处理


场景 2：收到用户消息后引用回复，并把回复置顶
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   reply = bot.reply_message("om_xxx", "text", {"text": "这是最终处理结论"})
   bot.pin_message(reply["message_id"])


场景 3：先发“处理中”，完成后更新成最终状态
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   msg = bot.send_card_to_chat(
       "oc_xxx",
       {
           "header": {"title": {"content": "任务状态", "tag": "plain_text"}},
           "elements": [{"tag": "markdown", "content": "处理中..."}]
       }
   )

   bot.edit_card_message(
       msg["message_id"],
       {
           "header": {"title": {"content": "任务状态", "tag": "plain_text"}},
           "elements": [{"tag": "markdown", "content": "已完成"}]
       }
   )


场景 4：收到告警后，转发给值班人并加急
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   bot.forward_message("om_alert_xxx", "ou_duty_xxx", receive_id_type="open_id")
   bot.urgent_message("om_alert_xxx", "app", ["ou_duty_xxx"])


场景 5：使用 reaction 作为处理中状态
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   reaction = bot.add_reaction("om_xxx", "WITTY")
   try:
       bot.reply_message("om_xxx", "text", {"text": "处理中"})
   finally:
       bot.delete_reaction("om_xxx", reaction["reaction_id"])


场景 5.1：把 LLM 流式输出刷到同一张回复卡片上
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def llm_stream():
       yield "今天的分析如下：\n\n"
       yield "1. 指标整体稳定\n"
       yield "2. 风险点主要在支付链路\n"
       yield "3. 建议先观察 30 分钟\n"

   bot.stream_reply_card(
       "om_xxx",
       llm_stream(),
       title="分析结果生成中",
       final_status_text="已完成"
   )


场景 5.2：逐字流式输出，完成态改绿色，失败态改红色
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time

   reply = bot.reply_streaming_card(
       "om_xxx",
       title="逐字输出测试",
       template="blue",
       initial_md=""
   )

   card_message_id = reply["message_id"]
   text = "这是一段逐字输出的测试文本。"
   current = ""

   try:
       for ch in text:
           current += ch
           bot.update_streaming_card(
               card_message_id,
               current,
               title="逐字输出测试",
               template="blue",
               status_text="生成中..."
           )
           time.sleep(0.3)

       bot.recolor_streaming_card(
           card_message_id,
           current,
           title="逐字输出测试",
           template="green",
           status_text="已完成"
       )
   except Exception as exc:
       bot.recolor_streaming_card(
           card_message_id,
           current + f"\n\n错误信息：{exc}",
           title="逐字输出测试",
           template="red",
           status_text="失败"
       )


场景 6：创建专项群，拉人，设管理员，发欢迎卡片
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   group = bot.create_chat(
       name="专项推进群",
       user_open_ids=["ou_a", "ou_b", "ou_c"],
       description="专项推进"
   )
   chat_id = group["chat_id"]
   bot.set_chat_admin(chat_id, ["ou_a"], is_admin=True)

   card = CardContentV2(title="欢迎加入")
   card.add_markdown("请查看群公告并完成本周任务认领。")
   bot.send_card_to_chat(chat_id, card.get_card())


场景 7：长 Markdown 公告自动切片发送
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   bot.send_markdown_message_to_chat(
       "oc_xxx",
       md_text=huge_release_note,
       title="发布说明",
       prefer="card_v2",
       max_message_bytes=10000
   )


场景 8：按名字找目标群和目标人，再定向发送
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   chat_ids = bot.find_chat_ids_by_name("值班群")
   if chat_ids:
       bot.send_text_to_chat(chat_ids[0], "今晚注意观察监控")

   open_ids = bot.find_member_open_ids_by_name(chat_ids[0], "Wayne")
   if open_ids:
       bot.send_text_to_user(open_ids[0], "请确认值班")


场景 8.1：先按群名查群，再发送长 Markdown 卡片
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   chat_ids = bot.find_chat_ids_by_name("测试3")
   if chat_ids:
       bot.send_markdown_message_to_chat(
           chat_ids[0],
           md_text="# 自动通知\n\n- 功能已发布\n- 请在群内验证",
           title="系统通知"
       )


场景 9：下载消息附件后再二次转发
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   bot.download_message_resource("om_xxx", "file", "/tmp/input.pdf", "file_xxx")
   new_key = bot.upload_file("/tmp/input.pdf", file_type="pdf")
   bot.send_file_to_chat("oc_other_xxx", new_key)


场景 10：读取群公告并同步成卡片消息
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   data = bot.get_chat_announcement("oc_xxx")
   card = CardContentV2(title="当前群公告")
   card.add_markdown(str(data))
   bot.send_card_to_chat("oc_xxx", card.get_card())


场景 11：查消息详情 -> 转发 -> 加急 -> 置顶结论
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   detail = bot.get_message("om_alert_xxx")
   if "严重" in detail["body"]["content"]:
       bot.forward_message("om_alert_xxx", "ou_duty_xxx", receive_id_type="open_id")
       bot.urgent_message("om_alert_xxx", "app", ["ou_duty_xxx"])
       reply = bot.reply_message("om_alert_xxx", "text", {"text": "已转交值班并加急"})
       bot.pin_message(reply["message_id"])


注意事项
--------

1. ``reply_message`` 的 ``content`` 会被自动 JSON 序列化；文本消息通常传 ``{"text": "..."}``。
2. ``interactive`` 这个飞书类型值对应的是 card 卡片；发送与更新时，传入的是卡片 JSON，不需要你手动再做 ``json.dumps``。
3. ``reaction`` 的 ``emoji_type`` 不是表情符号本身，而是飞书定义的名称。
4. ``batch_send_message`` 面向用户 / 部门，不面向群；它和群消息的生命周期不同。
5. ``set_chat_announcement`` 目前直接暴露飞书 patch 风格参数，适合需要精确控制公告 patch 的场景。
6. 流式卡片更新传入的是“当前完整文本”，不是本次新增片段；如果你只传 delta，卡片内容会丢前文。
7. 卡片颜色使用的是飞书头部模板色，不是任意 RGB；完成自动变绿需要显式传 ``final_template="green"`` 或手动 ``recolor_streaming_card``。
