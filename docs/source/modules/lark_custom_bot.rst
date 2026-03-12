飞书自定义机器人 (lark_custom_bot)
==================================

``pywayne.lark_custom_bot`` 面向 webhook 机器人场景，适合：

- 已经有飞书群里的自定义机器人 webhook
- 只需要“主动发消息到某个群”
- 不需要监听消息
- 不需要引用回复、撤回、reaction、已读、群管理

它和 ``LarkBot`` 的边界非常明确：

- ``LarkCustomBot``: webhook 推送器，轻量，配置简单
- ``LarkBot``: 应用机器人，能力完整，适合双向交互

如果你需要“收到消息后自动回复”、“给某条消息加表情”、“更新卡片”、“监听按钮点击”，请直接使用 ``LarkBot`` + ``LarkBotListener``。


能力概览
--------

``LarkCustomBot`` 当前支持：

- 文本消息
- post 富文本
- 分享群聊
- 图片消息
- interactive 卡片消息
- 本地图片上传
- OpenCV 图像直接上传
- 签名校验

它当前不负责：

- 监听消息
- 回复某条已有消息
- reaction
- 撤回 / 编辑消息
- 查询历史消息
- 群管理


LarkCustomBot 类
----------------

.. py:class:: LarkCustomBot(webhook: str, secret: str = '', bot_app_id: str = '', bot_secret: str = '')

   创建一个飞书自定义机器人实例。

   **参数**

   - ``webhook``: 群自定义机器人的 webhook 地址
   - ``secret``: 如果群机器人开启签名校验，则填写 secret
   - ``bot_app_id``: 图片上传用的应用 app id，可选
   - ``bot_secret``: 图片上传用的应用 secret，可选

   其中 ``bot_app_id`` / ``bot_secret`` 主要用于：

   - ``upload_image``
   - ``upload_image_from_cv2``

   因为图片上传不走 webhook，而是走开放平台鉴权接口。


主要方法
--------

send_text
~~~~~~~~~

.. py:method:: send_text(text: str, mention_all: bool = False)

发送纯文本消息。

示例：最简单的文本通知

.. code-block:: python

   from pywayne.lark_custom_bot import LarkCustomBot

   bot = LarkCustomBot(
       webhook="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
   )
   bot.send_text("Hello, Feishu")

示例：@ 所有人

.. code-block:: python

   bot.send_text("请注意查看今日值班安排", mention_all=True)


send_post
~~~~~~~~~

.. py:method:: send_post(content: List[List[Dict]], title: Optional[str] = None)

发送 post 富文本消息。

``content`` 是二维结构：

- 外层列表表示“多行”
- 内层列表表示“同一行内的多个元素”


send_share_chat
~~~~~~~~~~~~~~~

.. py:method:: send_share_chat(share_chat_id: str)

发送分享群聊卡片。


send_image
~~~~~~~~~~

.. py:method:: send_image(image_key: str)

发送图片消息。图片一般配合 ``upload_image`` 或 ``upload_image_from_cv2`` 使用。


send_interactive
~~~~~~~~~~~~~~~~

.. py:method:: send_interactive(card: Dict)

发送 interactive 卡片消息。


upload_image
~~~~~~~~~~~~

.. py:method:: upload_image(file_path: str) -> str

上传本地图片并返回 ``image_key``。


upload_image_from_cv2
~~~~~~~~~~~~~~~~~~~~~

.. py:method:: upload_image_from_cv2(cv2_image: np.ndarray) -> str

直接上传 OpenCV 图像矩阵。

适合“算法生成结果图，然后直接推送到群里”的场景。


辅助函数
--------

模块级辅助函数用于构造 ``send_post`` 的 ``content`` 元素。

- ``create_text_content(text, unescape=False)``
- ``create_link_content(href, text)``
- ``create_at_content(user_id, user_name)``
- ``create_image_content(image_key, width=None, height=None)``

示例：手工构造一条 post

.. code-block:: python

   from pywayne.lark_custom_bot import (
       LarkCustomBot,
       create_text_content,
       create_link_content,
       create_at_content,
       create_image_content,
   )

   bot = LarkCustomBot(
       webhook="https://open.feishu.cn/open-apis/bot/v2/hook/xxx",
       bot_app_id="cli_xxx",
       bot_secret="sec_xxx"
   )

   img_key = bot.upload_image("/tmp/report.png")
   content = [
       [
           create_text_content("巡检完成"),
           create_at_content("all", "所有人"),
       ],
       [
           create_link_content("https://example.com", "查看详情"),
       ],
       [
           create_image_content(img_key, width=300, height=200),
       ],
   ]
   bot.send_post(content, title="巡检报告")


典型场景示例
------------

场景 1：定时任务往群里推送一条纯文本通知
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   bot = LarkCustomBot(webhook="https://open.feishu.cn/open-apis/bot/v2/hook/xxx")
   bot.send_text("每日 09:00 自动巡检已开始")


场景 2：构造 richer 的 post 公告
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   content = [
       [create_text_content("发布完成", unescape=False)],
       [create_link_content("https://example.com/release", "查看发布记录")],
       [create_text_content("请研发和测试同学及时确认。")],
   ]
   bot.send_post(content, title="版本播报")


场景 3：上传本地图片后发到群里
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   image_key = bot.upload_image("/tmp/chart.png")
   bot.send_image(image_key)


场景 4：OpenCV 结果图直接推送
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cv2
   import numpy as np

   image = np.zeros((400, 600, 3), dtype=np.uint8)
   cv2.putText(image, "PASS", (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)

   image_key = bot.upload_image_from_cv2(image)
   bot.send_image(image_key)


场景 5：文本 + 图片 + 链接组合为一条 post
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   image_key = bot.upload_image("/tmp/dashboard.png")
   content = [
       [create_text_content("监控快照如下：")],
       [create_image_content(image_key)],
       [create_link_content("https://grafana.example.com", "打开大盘")],
   ]
   bot.send_post(content, title="监控日报")


场景 6：交互卡片做一个轻量审批提醒
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   card = {
       "config": {"wide_screen_mode": True},
       "header": {
           "title": {"tag": "plain_text", "content": "审批提醒"}
       },
       "elements": [
           {"tag": "markdown", "content": "**工单 #1234** 等待审批"},
           {
               "tag": "action",
               "actions": [
                   {
                       "tag": "button",
                       "text": {"tag": "plain_text", "content": "打开审批页"},
                       "type": "primary",
                       "url": "https://example.com/ticket/1234"
                   }
               ]
           }
       ]
   }
   bot.send_interactive(card)


场景 7：带签名校验的安全发送
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   bot = LarkCustomBot(
       webhook="https://open.feishu.cn/open-apis/bot/v2/hook/xxx",
       secret="your_sign_secret"
   )
   bot.send_text("这条消息会自动带 timestamp 和 sign")


场景 8：运维值班报告，一次发文本、图片、卡片三连
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   bot.send_text("今晚值班巡检开始")

   image_key = bot.upload_image("/tmp/nightly.png")
   bot.send_image(image_key)

   bot.send_interactive({
       "header": {"title": {"tag": "plain_text", "content": "值班报告"}},
       "elements": [
           {"tag": "markdown", "content": "- 巡检: 已完成\n- 异常: 0\n- 待处理: 1"}
       ]
   })


场景 9：把群链接分享给另一个群
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   bot.send_share_chat("oc_xxx")


场景 10：日常构造器复用，封一个发布函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def send_release_report(bot: LarkCustomBot, title: str, report_url: str, image_path: str):
       image_key = bot.upload_image(image_path)
       content = [
           [create_text_content("发布完成")],
           [create_link_content(report_url, "查看详情")],
           [create_image_content(image_key)],
       ]
       bot.send_post(content, title=title)


和 LarkBot 的取舍
-----------------

优先用 ``LarkCustomBot`` 的场景：

- 只要往固定群发消息
- 不想配置应用事件订阅
- 不需要监听和消息生命周期管理
- 用 webhook 足够

优先用 ``LarkBot`` 的场景：

- 需要引用回复
- 需要撤回或编辑消息
- 需要 reaction / 置顶 / 已读
- 需要监听群聊和私聊
- 需要按钮点击回调
- 需要群管理


注意事项
--------

1. ``upload_image`` / ``upload_image_from_cv2`` 需要 ``bot_app_id`` 和 ``bot_secret``，否则只能发纯 webhook 文本 / post / card，不能上传图片。
2. ``send_post`` 的 ``content`` 必须是飞书要求的二维结构，不是普通字符串。
3. ``send_interactive`` 只能发卡片消息，不负责接收卡片按钮点击；按钮回调要走应用机器人监听体系。
4. 如果你的业务开始出现“收到消息后处理再回复”的诉求，说明应该切到 ``LarkBot`` + ``LarkBotListener``，而不是继续堆 webhook 逻辑。
