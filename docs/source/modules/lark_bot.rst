飞书机器人 (lark_bot)
=====================

``pywayne.lark_bot`` 是应用机器人侧的主能力封装，覆盖主动发消息、引用回复、消息编辑、撤回、reaction、置顶、群管理、资源上传下载、历史消息查询等常见飞书 IM 能力。

如果把整个模块按对象分层，可以理解为：

- ``TextContent``: 文本格式工具，解决 @、加粗、链接等轻量文本拼接
- ``PostContent``: post 富文本构造器，适合做结构化文本、代码块、Markdown 表格降级
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
  - post 富文本
  - interactive 卡片
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
  - 全量更新消息
  - patch 消息
  - 原位更新 interactive 卡片
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

``PostContent`` 适合构造复杂 post 富文本，尤其适合：

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

示例：混排 post 消息

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

   bot.send_post_to_chat("oc_xxx", post.get_content())

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
   bot.send_post_to_chat("oc_xxx", post.get_content())


CardContentV2
-------------

``CardContentV2`` 是面向 schema 2.0 interactive 卡片的轻量构造器，适合大段 Markdown 公告、日报、状态展示。

常用方法：

- ``add_markdown``
- ``add_hr``
- ``add_image``
- ``get_card``

示例：日报卡片

.. code-block:: python

   from pywayne.lark_bot import CardContentV2

   card = CardContentV2(title="日报", template="blue")
   card.add_markdown("# 今日进展\n\n- 接口联调完成\n- 修复 3 个问题\n- 代码已合并")
   card.add_hr()
   card.add_image("img_xxx")

   bot.send_interactive_to_chat("oc_xxx", card.get_card())


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


post 富文本
~~~~~~~~~~~

- ``send_post_to_user``
- ``send_post_to_chat``

.. code-block:: python

   post = PostContent(title="值班提醒")
   post.add_content_in_new_line(post.make_text_content("今晚 20:00 发布", styles=["bold"]))
   bot.send_post_to_chat("oc_xxx", post.get_content())


interactive 卡片
~~~~~~~~~~~~~~~~

- ``send_interactive_to_user``
- ``send_interactive_to_chat``

.. code-block:: python

   bot.send_interactive_to_chat(
       "oc_xxx",
       {
           "header": {"title": {"content": "状态卡片", "tag": "plain_text"}},
           "elements": [{"tag": "markdown", "content": "**服务正常**"}]
       }
   )


分享消息
~~~~~~~~

- ``send_shared_chat_to_user`` / ``send_shared_chat_to_chat``
- ``send_shared_user_to_user`` / ``send_shared_user_to_chat``


系统消息
~~~~~~~~

- ``send_system_msg_to_user``

适合做系统通知类场景，不走普通文本样式。


推荐入口：send_markdown_to_chat
--------------------------------

.. py:method:: send_markdown_to_chat(chat_id: str, md_text: str, *, title: str = "", prefer: str = "card_v2", table_fallback: str = "code_block", max_message_bytes: Optional[int] = None)

这是推荐的高层发送入口，适合绝大多数“我要发 Markdown”场景。

特性：

- 自动按字节分包
- 支持 ``card_v2`` 与 ``post`` 两条路由
- ``post`` 路由下支持表格降级

示例 1：默认发 schema 2.0 卡片

.. code-block:: python

   bot.send_markdown_to_chat(
       "oc_xxx",
       md_text="# 发布完成\n\n- API: pass\n- Worker: pass",
       title="发布结果"
   )

示例 2：强制走 post，并处理 Markdown 表格

.. code-block:: python

   bot.send_markdown_to_chat(
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

   bot.send_markdown_to_chat(
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


撤回消息
~~~~~~~~

.. py:method:: recall_message(message_id: str)

适合“发错消息立即撤回”或“临时状态消息在成功后撤回”。


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


更新消息
~~~~~~~~

- ``update_message(message_id, msg_type, content)``
- ``patch_message(message_id, content)``
- ``update_interactive_card(message_id, card)``

示例：把“处理中”卡片更新成“已完成”

.. code-block:: python

   bot.update_interactive_card(
       message_id="om_xxx",
       card={
           "type": "template",
           "data": {
               "template_id": "AAqC5c999",
               "template_variable": {"status": "已完成"}
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


置顶
~~~~

- ``pin_message(message_id)``
- ``unpin_message(message_id)``
- ``list_pinned_messages(chat_id, ...)``

示例：先回复，再置顶

.. code-block:: python

   reply = bot.reply_message("om_xxx", "text", {"text": "这是最终结论"})
   bot.pin_message(reply["message_id"])


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


群公告
~~~~~~

- ``get_chat_announcement``
- ``set_chat_announcement``

``set_chat_announcement`` 走的是飞书 patch 语义，参数 ``requests`` 需要直接传飞书公告 API 的 patch 操作列表。

示例：读取公告

.. code-block:: python

   announcement = bot.get_chat_announcement("oc_xxx")


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


用户与群信息
------------

- ``get_user_info(emails, mobiles)``
- ``get_group_list()``
- ``get_group_chat_id_by_name(group_name)``
- ``get_members_in_group_by_group_chat_id(chat_id)``
- ``get_member_open_id_by_name(chat_id, member_name)``
- ``get_chat_and_user_name(chat_id, user_id)``

示例：按群名查 ID，再按用户名查成员 open_id

.. code-block:: python

   chat_ids = bot.get_group_chat_id_by_name("项目 Alpha")
   if chat_ids:
       open_ids = bot.get_member_open_id_by_name(chat_ids[0], "Wayne")


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


组合场景示例
------------

场景 1：发日报卡片，用户点击后再原位更新
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   card = CardContentV2(title="日报提交")
   card.add_markdown("请点击按钮确认今天已提交日报。")
   bot.send_interactive_to_chat("oc_xxx", card.get_card())

   # 按钮点击后的更新逻辑放到 LarkBotListener.card_action_handler 中处理


场景 2：收到用户消息后引用回复，并把回复置顶
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   reply = bot.reply_message("om_xxx", "text", {"text": "这是最终处理结论"})
   bot.pin_message(reply["message_id"])


场景 3：先发“处理中”，完成后更新成最终状态
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   msg = bot.send_interactive_to_chat(
       "oc_xxx",
       {
           "header": {"title": {"content": "任务状态", "tag": "plain_text"}},
           "elements": [{"tag": "markdown", "content": "处理中..."}]
       }
   )

   bot.update_interactive_card(
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
   bot.send_interactive_to_chat(chat_id, card.get_card())


场景 7：长 Markdown 公告自动切片发送
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   bot.send_markdown_to_chat(
       "oc_xxx",
       md_text=huge_release_note,
       title="发布说明",
       prefer="card_v2",
       max_message_bytes=10000
   )


场景 8：按名字找目标群和目标人，再定向发送
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   chat_ids = bot.get_group_chat_id_by_name("值班群")
   if chat_ids:
       bot.send_text_to_chat(chat_ids[0], "今晚注意观察监控")

   open_ids = bot.get_member_open_id_by_name(chat_ids[0], "Wayne")
   if open_ids:
       bot.send_text_to_user(open_ids[0], "请确认值班")


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
   bot.send_interactive_to_chat("oc_xxx", card.get_card())


注意事项
--------

1. ``reply_message`` 的 ``content`` 会被自动 JSON 序列化；文本消息通常传 ``{"text": "..."}``。
2. ``interactive`` 类型发送与更新时，传入的是卡片 JSON，不需要你手动再做 ``json.dumps``。
3. ``reaction`` 的 ``emoji_type`` 不是表情符号本身，而是飞书定义的名称。
4. ``batch_send_message`` 面向用户 / 部门，不面向群；它和群消息的生命周期不同。
5. ``set_chat_announcement`` 目前直接暴露飞书 patch 风格参数，适合需要精确控制公告 patch 的场景。
