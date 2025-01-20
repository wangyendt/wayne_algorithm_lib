# author: wangye(Wayne)
# license: Apache Licence
# file: lark_bot_listener.py
# time: 2025-01-20-21:17:23
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import functools
import json
import time
import asyncio
from typing import Optional, Callable, Dict, Set, List
from dataclasses import dataclass
import lark_oapi as lark
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1
from pywayne.llm.chat_bot import LLMChat
from pywayne.lark_bot import LarkBot, PostContent


@dataclass
class MessageContext:
    """消息上下文"""
    chat_id: str
    user_id: str
    message_type: str
    content: str
    is_group: bool
    chat_type: str


class LarkBotListener:
    def __init__(self, app_id: str, app_secret: str, message_expiry_time: int = 60):
        """
        初始化飞书监听器
        :param app_id: 飞书应用ID
        :param app_secret: 飞书应用密钥
        :param message_expiry_time: 消息去重过期时间（秒）
        """
        self.app_id = app_id
        self.app_secret = app_secret
        self.message_expiry_time = message_expiry_time

        # 消息去重（每个处理函数独立去重）
        self.processed_messages: Dict[str, Set[str]] = {}  # handler_name -> set(message_ids)
        self.message_timestamps: Dict[str, Dict[str, float]] = {}  # handler_name -> {message_id: timestamp}

        # 用户/群组的chat_bot缓存
        self.chat_bots: Dict[str, LLMChat] = {}

        # 初始化飞书客户端
        self.client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()

        # 初始化飞书机器人
        self.bot = LarkBot(app_id=app_id, app_secret=app_secret)

        # 注册的处理函数
        self.handlers: List[Callable] = []

    def _clean_expired_messages(self, handler_name: str):
        """清理过期的消息ID"""
        if handler_name not in self.message_timestamps:
            return

        current_time = time.time()
        expired = [msg_id for msg_id, timestamp in self.message_timestamps[handler_name].items()
                   if current_time - timestamp > self.message_expiry_time]
        for msg_id in expired:
            self.processed_messages[handler_name].remove(msg_id)
            del self.message_timestamps[handler_name][msg_id]

    def _get_or_create_chat_bot(self, chat_id: str, llm_config: dict) -> LLMChat:
        """获取或创建新的chat_bot"""
        if chat_id not in self.chat_bots:
            print(f"为会话 {chat_id} 创建新的chat_bot")
            self.chat_bots[chat_id] = LLMChat(**llm_config)
        return self.chat_bots[chat_id]

    def send_message(self, chat_id: str, content: str):
        """发送消息到飞书（使用Markdown格式）"""
        try:
            print(f"正在发送消息到 {chat_id}: {content}")
            # 创建富文本消息
            post = PostContent()
            # 添加Markdown内容
            md_content = post.make_markdown_content(content)
            post.add_content_in_new_line(md_content)
            # 发送富文本消息
            self.bot.send_post_to_chat(chat_id, post.get_content())
            print(f"消息发送成功: {content}")
        except Exception as e:
            print(f"发送消息时发生错误: {e}")

    def listen(self, message_type: Optional[str] = None,
               group_only: bool = False,
               user_only: bool = False,
               llm_config: Optional[dict] = None):
        """
        飞书消息监听装饰器
        :param message_type: 消息类型，例如"text"，None表示所有类型
        :param group_only: 是否只监听群组消息
        :param user_only: 是否只监听用户消息
        :param llm_config: LLM配置，如果提供则自动创建chat_bot
        """

        def decorator(func: Callable):
            handler_name = func.__name__
            # 初始化处理函数的消息记录
            self.processed_messages[handler_name] = set()
            self.message_timestamps[handler_name] = {}

            @functools.wraps(func)
            async def message_handler(data: P2ImMessageReceiveV1) -> None:
                # 消息去重（针对当前处理函数）
                message_id = data.event.message.message_id
                if message_id in self.processed_messages[handler_name]:
                    print(f"消息 {message_id} 已被 {handler_name} 处理，跳过")
                    return

                # 解析消息
                msg_type = data.event.message.message_type
                if message_type and msg_type != message_type:
                    print(f"消息类型不匹配，期望 {message_type}，实际 {msg_type}")
                    return

                chat_id = data.event.message.chat_id
                user_id = data.event.sender.sender_id.user_id
                chat_type = data.event.message.chat_type
                is_group = chat_type == "group"

                print(f"消息类型: {chat_type}, 是否群组: {is_group}, 处理函数: {handler_name}")

                if (group_only and not is_group) or (user_only and is_group):
                    print(f"跳过消息处理: group_only={group_only}, user_only={user_only}")
                    return

                # 解析消息内容
                content = ""
                if msg_type == "text":
                    content = json.loads(data.event.message.content)["text"]

                # 创建消息上下文
                ctx = MessageContext(
                    chat_id=chat_id,
                    user_id=user_id,
                    message_type=msg_type,
                    content=content,
                    is_group=is_group,
                    chat_type=chat_type
                )

                # 如果提供了llm配置，获取对应的chat_bot
                if llm_config:
                    print(f"使用LLM配置处理消息: {content}")
                    chat_bot = self._get_or_create_chat_bot(chat_id, llm_config)
                    await func(ctx, chat_bot)
                else:
                    await func(ctx)

                # 标记消息为已处理（仅针对当前处理函数）
                self.processed_messages[handler_name].add(message_id)
                self.message_timestamps[handler_name][message_id] = time.time()
                self._clean_expired_messages(handler_name)

            # 注册处理函数
            print(f"注册处理函数: {handler_name}, group_only={group_only}, user_only={user_only}")
            self.handlers.append(message_handler)
            return message_handler

        return decorator

    def run(self):
        """启动监听服务"""

        # 创建事件处理器
        def handle_message_receive(data: P2ImMessageReceiveV1) -> None:
            # 获取当前事件循环
            loop = asyncio.get_event_loop()

            # 执行所有处理函数
            async def run_handlers():
                print(f"\n开始处理新消息...")
                for handler in self.handlers:
                    try:
                        await handler(data)
                    except Exception as e:
                        print(f"处理消息时发生错误: {e}")
                print("消息处理完成\n")

            # 在当前事件循环中运行
            try:
                loop.create_task(run_handlers())
            except Exception as e:
                print(f"运行事件循环时发生错误: {e}")

        event_handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(handle_message_receive)
            .build()
        )

        # 创建WebSocket客户端
        ws_client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.DEBUG
        )

        print("启动飞书消息监听服务...")
        # 启动服务
        ws_client.start()


# 使用示例
if __name__ == "__main__":
    # 创建监听器实例
    listener = LarkBotListener(
        app_id="xxx",
        app_secret="xxx"
    )


    # 示例1：处理所有文本消息
    @listener.listen(message_type="text")
    async def handle_text(ctx: MessageContext):
        print(f"收到消息: {ctx.content} (来自{'群组' if ctx.is_group else '私聊'})")


    # 示例2：AI对话机器人（仅群组）
    @listener.listen(
        message_type="text",
        # group_only=True,
        llm_config={
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-xxx"
        }
    )
    async def ai_chat(ctx: MessageContext, chat_bot: LLMChat):
        try:
            print(f"开始处理AI回复...")
            # 使用线程池执行同步的chat调用
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: ''.join(chat_bot.chat(ctx.content, stream=True)))
            print(f"AI回复: {response}")
            # 发送AI回复到飞书
            listener.send_message(ctx.chat_id, response)
        except Exception as e:
            print(f"AI处理消息时发生错误: {e}")
            listener.send_message(ctx.chat_id, f"抱歉，处理消息时发生错误: {e}")


    # 启动服务
    listener.run()
