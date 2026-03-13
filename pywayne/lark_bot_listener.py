# author: wangye(Wayne)
# license: Apache Licence
# file: lark_bot_listener.py
# time: 2025-01-20-21:17:23
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import inspect
import functools
import json
import time
import asyncio
import cv2
import tempfile
from pathlib import Path
from typing import Optional, Callable, Dict, Set, List, Any, Iterable, AsyncIterable, Union
from dataclasses import dataclass
import lark_oapi as lark
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1
from lark_oapi.card import CardActionHandler
from pywayne.lark_bot import LarkBot, PostContent
from pywayne.tools import wayne_print
from pywayne.llm.chat_bot import ChatManager
from pywayne.cv.apriltag_detector import ApriltagCornerDetector


def _print_info(text: str) -> None:
    wayne_print(text, color="cyan")


def _print_success(text: str) -> None:
    wayne_print(text, color="green", bold=True)


def _print_warn(text: str) -> None:
    wayne_print(text, color="yellow", bold=True)


def _print_error(text: str) -> None:
    wayne_print(text, color="red", bold=True)


@dataclass
class MessageContext:
    """消息上下文"""
    chat_id: str
    user_id: str
    message_type: str
    content: str
    is_group: bool
    chat_type: str
    message_id: str
    thread_id: str = ""
    root_id: str = ""
    parent_id: str = ""
    mentions: Optional[List[Any]] = None
    raw_event: Any = None


class LarkBotListener:
    _SUPPORTED_EVENT_REGISTRATIONS = {
        "register_p2_im_chat_access_event_bot_p2p_chat_entered_v1",
        "register_p2_im_chat_disbanded_v1",
        "register_p2_im_chat_member_bot_added_v1",
        "register_p2_im_chat_member_bot_deleted_v1",
        "register_p2_im_chat_member_user_added_v1",
        "register_p2_im_chat_member_user_deleted_v1",
        "register_p2_im_chat_member_user_withdrawn_v1",
        "register_p2_im_chat_updated_v1",
        "register_p2_im_message_message_read_v1",
        "register_p2_im_message_reaction_created_v1",
        "register_p2_im_message_reaction_deleted_v1",
        "register_p2_im_message_recalled_v1",
    }

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
        
        # 创建临时文件夹
        self.temp_dir = Path(tempfile.gettempdir()) / "lark_bot_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        _print_info(f"创建临时文件夹: {self.temp_dir}")

        # 消息去重（每个处理函数独立去重）
        self.processed_messages: Dict[str, Set[str]] = {}  # handler_name -> set(message_ids)
        self.message_timestamps: Dict[str, Dict[str, float]] = {}  # handler_name -> {message_id: timestamp}

        # 初始化飞书客户端
        self.client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()

        # 初始化飞书机器人
        self.bot = LarkBot(app_id=app_id, app_secret=app_secret)

        # 注册的处理函数
        self.handlers: List[Callable] = []
        self.event_handlers: Dict[str, List[Callable]] = {
            name: [] for name in self._SUPPORTED_EVENT_REGISTRATIONS
        }
        self._card_action_handler: Optional[CardActionHandler] = None

    @staticmethod
    def _select_kwargs(param_names: List[str], available_values: Dict[str, Any]) -> Dict[str, Any]:
        """Pick only the kwargs a handler explicitly declared."""
        return {name: available_values[name] for name in param_names if name in available_values}

    @staticmethod
    def _parse_content_json(content: str) -> Dict[str, Any]:
        """Best-effort parse for Feishu message content JSON."""
        if not content:
            return {}
        try:
            return json.loads(content)
        except Exception:
            return {}

    @staticmethod
    def _resolve_source_message_id(target: Union[str, MessageContext]) -> str:
        """Accept either a raw message_id or a MessageContext."""
        if isinstance(target, MessageContext):
            return target.message_id
        return target

    async def _invoke_handler(self, func: Callable, **kwargs) -> Any:
        """Support both async and sync handler functions."""
        result = func(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _register_event_handler(self, register_method_name: str, handler: Callable):
        """Register non-message-receive event handlers by dispatcher method name."""
        self.event_handlers.setdefault(register_method_name, []).append(handler)

    def _schedule_handlers(self, handlers: List[Callable], data: Any, label: str):
        """Run a batch of async handlers on the current event loop."""
        if not handlers:
            return

        async def run_handlers():
            _print_info(f"\n开始处理事件: {label}")
            for handler in handlers:
                try:
                    await handler(data)
                except Exception as e:
                    _print_error(f"处理事件 {label} 时发生错误: {e}")
            _print_info(f"事件处理完成: {label}\n")

        try:
            loop = asyncio.get_event_loop()
            loop.create_task(run_handlers())
        except RuntimeError:
            asyncio.run(run_handlers())

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

    def send_message(self, chat_id: str, content: str):
        """发送消息到飞书（使用Markdown格式）"""
        try:
            _print_info(f"正在发送消息到 {chat_id}: {content}")
            # 创建富文本消息
            post = PostContent()
            # 添加Markdown内容
            md_content = post.make_markdown_content(content)
            post.add_content_in_new_line(md_content)
            # 发送富文本消息
            self.bot.send_post_to_chat(chat_id, post.get_content())
            _print_success(f"消息发送成功: {content}")
        except Exception as e:
            _print_error(f"发送消息时发生错误: {e}")

    def reply_streaming_card(self,
                             target: Union[str, MessageContext],
                             *,
                             title: str = "Streaming Reply",
                             template: str = "blue",
                             initial_md: str = "",
                             reply_in_thread: bool = False,
                             uuid: str = "",
                             status_text: str = "Generating...",
                             max_chunk_bytes: int = 18_000) -> Dict:
        """
        Reply to a source message with an initial streaming card.

        Args:
            target: Source message_id or MessageContext.
        """
        return self.bot.reply_streaming_card(
            self._resolve_source_message_id(target),
            title=title,
            template=template,
            initial_md=initial_md,
            reply_in_thread=reply_in_thread,
            uuid=uuid,
            status_text=status_text,
            max_chunk_bytes=max_chunk_bytes,
        )

    def update_streaming_card(self,
                              card_message_id: str,
                              md_text: str,
                              *,
                              title: str = "Streaming Reply",
                              template: str = "blue",
                              done: bool = False,
                              status_text: str = "",
                              max_chunk_bytes: int = 18_000) -> Dict:
        """
        Update an existing streaming reply card.

        Note:
            `card_message_id` must be the message ID returned by
            `reply_streaming_card(...)`, not the original user message ID.
        """
        return self.bot.update_streaming_card(
            card_message_id,
            md_text,
            title=title,
            template=template,
            done=done,
            status_text=status_text,
            max_chunk_bytes=max_chunk_bytes,
        )

    def recolor_streaming_card(self,
                               card_message_id: str,
                               md_text: str,
                               *,
                               title: str = "Streaming Reply",
                               template: str = "green",
                               status_text: str = "Done",
                               done: bool = True,
                               max_chunk_bytes: int = 18_000) -> Dict:
        """
        Switch an existing streaming card to another Feishu header template.

        Typical usage:
            - green after success
            - red after failure
            - orange while waiting for manual review
        """
        return self.bot.recolor_streaming_card(
            card_message_id,
            md_text,
            title=title,
            template=template,
            status_text=status_text,
            done=done,
            max_chunk_bytes=max_chunk_bytes,
        )

    def stream_reply_card(self,
                          target: Union[str, MessageContext],
                          text_stream: Iterable[Any],
                          *,
                          title: str = "Streaming Reply",
                          template: str = "blue",
                          initial_md: str = "",
                          reply_in_thread: bool = False,
                          uuid: str = "",
                          update_interval: float = 0.25,
                          status_text: str = "Generating...",
                          final_status_text: str = "",
                          final_template: Optional[str] = "green",
                          max_chunk_bytes: int = 18_000) -> Dict[str, Any]:
        """
        Listener-level wrapper around `LarkBot.stream_reply_card`.

        Args:
            target: Source message_id or MessageContext.
            text_stream: Sync iterable that yields text chunks.
        """
        return self.bot.stream_reply_card(
            self._resolve_source_message_id(target),
            text_stream,
            title=title,
            template=template,
            initial_md=initial_md,
            reply_in_thread=reply_in_thread,
            uuid=uuid,
            update_interval=update_interval,
            status_text=status_text,
            final_status_text=final_status_text,
            final_template=final_template,
            max_chunk_bytes=max_chunk_bytes,
        )

    async def astream_reply_card(self,
                                 target: Union[str, MessageContext],
                                 text_stream: AsyncIterable[Any],
                                 *,
                                 title: str = "Streaming Reply",
                                 template: str = "blue",
                                 initial_md: str = "",
                                 reply_in_thread: bool = False,
                                 uuid: str = "",
                                 update_interval: float = 0.25,
                                 status_text: str = "Generating...",
                                 final_status_text: str = "",
                                 final_template: Optional[str] = "green",
                                 max_chunk_bytes: int = 18_000) -> Dict[str, Any]:
        """
        Async listener-level wrapper around `LarkBot.astream_reply_card`.

        Args:
            target: Source message_id or MessageContext.
            text_stream: Async iterable that yields text chunks.
        """
        return await self.bot.astream_reply_card(
            self._resolve_source_message_id(target),
            text_stream,
            title=title,
            template=template,
            initial_md=initial_md,
            reply_in_thread=reply_in_thread,
            uuid=uuid,
            update_interval=update_interval,
            status_text=status_text,
            final_status_text=final_status_text,
            final_template=final_template,
            max_chunk_bytes=max_chunk_bytes,
        )

    def listen(self, message_type: Optional[str] = None,
               group_only: bool = False,
               user_only: bool = False):
        """
        飞书消息监听装饰器
        :param message_type: 消息类型，例如"text"、"image"、"file"、"post"，None表示所有类型
        :param group_only: 是否只监听群组消息
        :param user_only: 是否只监听用户消息
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
                    _print_warn(f"消息 {message_id} 已被 {handler_name} 处理，跳过")
                    return

                # 解析消息
                msg_type = data.event.message.message_type
                if message_type and msg_type != message_type:
                    _print_warn(f"消息类型不匹配，期望 {message_type}，实际 {msg_type}")
                    return

                chat_id = data.event.message.chat_id
                user_id = data.event.sender.sender_id.open_id
                chat_type = data.event.message.chat_type
                is_group = chat_type == "group"

                _print_info(f"消息类型: {chat_type}, 是否群组: {is_group}, 处理函数: {handler_name}")

                if (group_only and not is_group) or (user_only and is_group):
                    _print_warn(f"跳过消息处理: group_only={group_only}, user_only={user_only}")
                    return

                # 解析消息内容
                content = ""
                if msg_type == "text":
                    content = json.loads(data.event.message.content)["text"]
                elif data.event.message.content is not None:
                    content = data.event.message.content

                # 创建消息上下文
                ctx = MessageContext(
                    chat_id=chat_id,
                    user_id=user_id,
                    message_type=msg_type,
                    content=content,
                    is_group=is_group,
                    chat_type=chat_type,
                    message_id=message_id,
                    thread_id=getattr(data.event.message, "thread_id", "") or "",
                    root_id=getattr(data.event.message, "root_id", "") or "",
                    parent_id=getattr(data.event.message, "parent_id", "") or "",
                    mentions=getattr(data.event.message, "mentions", None),
                    raw_event=data
                )

                # 调用用户定义的处理函数
                await func(ctx)

                # 标记消息为已处理（仅针对当前处理函数）
                self.processed_messages[handler_name].add(message_id)
                self.message_timestamps[handler_name][message_id] = time.time()
                self._clean_expired_messages(handler_name)

            # 注册处理函数
            _print_info(f"注册处理函数: {handler_name}, group_only={group_only}, user_only={user_only}")
            self.handlers.append(message_handler)
            return message_handler

        return decorator

    def run(self):
        """启动监听服务"""
        # 创建事件处理器
        def handle_message_receive(data: P2ImMessageReceiveV1) -> None:
            self._schedule_handlers(self.handlers, data, "message.receive")

        dispatcher_builder = lark.EventDispatcherHandler.builder("", "")
        if self.handlers:
            dispatcher_builder = dispatcher_builder.register_p2_im_message_receive_v1(handle_message_receive)

        for register_method_name, handlers in self.event_handlers.items():
            register_method = getattr(dispatcher_builder, register_method_name)

            def make_callback(event_handlers: List[Callable], label: str):
                def callback(data: Any) -> None:
                    self._schedule_handlers(event_handlers, data, label)
                return callback

            dispatcher_builder = register_method(make_callback(handlers, register_method_name))

        event_handler = dispatcher_builder.build()

        # 创建WebSocket客户端
        ws_client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.DEBUG
        )

        _print_success("启动飞书消息监听服务...")
        # 启动服务
        ws_client.start()

    def text_handler(self, group_only: bool = False, user_only: bool = False):
        """
        文本消息处理装饰器，直接处理文本内容
        
        Args:
            group_only: 是否只处理群组消息
            user_only: 是否只处理私聊消息
            
        装饰的函数可以接受以下参数（除text外都是可选的）：
            text (str): 文本内容（必需）
            chat_id (str, optional): 会话ID
            is_group (bool, optional): 是否群组消息
            group_name (str, optional): 群组名称（私聊时为空字符串）
            user_name (str, optional): 发送消息的用户姓名
        """
        def decorator(func):
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            @self.listen(message_type="text", group_only=group_only, user_only=user_only)
            async def wrapper(ctx: MessageContext):
                try:
                    # 获取群组名称和用户姓名
                    group_name, user_name = self.bot.get_chat_and_user_name(ctx.chat_id, ctx.user_id)

                    # 根据函数参数构建调用参数
                    kwargs = {}
                    if 'text' in param_names:
                        kwargs['text'] = ctx.content
                    if 'chat_id' in param_names:
                        kwargs['chat_id'] = ctx.chat_id
                    if 'is_group' in param_names:
                        kwargs['is_group'] = ctx.is_group
                    if 'group_name' in param_names:
                        kwargs['group_name'] = group_name
                    if 'user_name' in param_names:
                        kwargs['user_name'] = user_name

                    # 调用用户处理函数
                    await func(**kwargs)
                except Exception as e:
                    _print_error(f"处理文本消息时发生错误: {e}")
                    import traceback
                    _print_error(f"错误详情:\n{traceback.format_exc()}")
            return wrapper
        return decorator

    def image_handler(self, group_only: bool = False, user_only: bool = False):
        """
        图片消息处理装饰器，自动处理图片下载和清理
        
        Args:
            group_only: 是否只处理群组消息
            user_only: 是否只处理私聊消息
            
        装饰的函数可以接受以下参数（除image_path外都是可选的）：
            image_path (Path): 临时图片文件路径（必需）
            chat_id (str, optional): 会话ID
            is_group (bool, optional): 是否群组消息
            group_name (str, optional): 群组名称（私聊时为空字符串）
            user_name (str, optional): 发送消息的用户姓名
            
        函数可以返回一个新的图片路径，该图片会被发送回去
        如果返回None，则不发送任何图片
        """
        def decorator(func):
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            @self.listen(message_type="image", group_only=group_only, user_only=user_only)
            async def wrapper(ctx: MessageContext):
                result_path = None
                try:
                    image_key = json.loads(ctx.content).get("image_key")
                    if not image_key:
                        _print_warn("未找到图片key")
                        return

                    # 创建临时文件
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        temp_image_path = Path(temp_file.name)
                    
                    try:
                        # 下载图片
                        if not self.bot.download_message_resource(ctx.message_id, "image", str(temp_image_path), image_key):
                            _print_warn("图片下载失败")
                            return

                        # 获取群组名称和用户姓名
                        group_name, user_name = self.bot.get_chat_and_user_name(ctx.chat_id, ctx.user_id)

                        # 根据函数参数构建调用参数
                        kwargs = {}
                        if 'image_path' in param_names:
                            kwargs['image_path'] = temp_image_path
                        if 'chat_id' in param_names:
                            kwargs['chat_id'] = ctx.chat_id
                        if 'is_group' in param_names:
                            kwargs['is_group'] = ctx.is_group
                        if 'group_name' in param_names:
                            kwargs['group_name'] = group_name
                        if 'user_name' in param_names:
                            kwargs['user_name'] = user_name

                        # 调用用户处理函数
                        result_path = await func(**kwargs)
                        
                        # 如果返回了新图片路径，发送回去
                        if result_path is not None:
                            new_image_key = self.bot.upload_image(str(result_path))
                            if new_image_key:
                                self.bot.send_image_to_chat(ctx.chat_id, new_image_key)
                    finally:
                        # 清理临时文件
                        temp_image_path.unlink(missing_ok=True)
                        if result_path is not None and result_path != temp_image_path:
                            result_path.unlink(missing_ok=True)
                
                except Exception as e:
                    _print_error(f"处理图片消息时发生错误: {str(e)}")
                    import traceback
                    _print_error(f"错误详情:\n{traceback.format_exc()}")
            return wrapper
        return decorator

    def file_handler(self, group_only: bool = False, user_only: bool = False):
        """
        文件消息处理装饰器，自动处理文件下载和清理
        
        Args:
            group_only: 是否只处理群组消息
            user_only: 是否只处理私聊消息
            
        装饰的函数可以接受以下参数（除file_path外都是可选的）：
            file_path (Path): 临时文件路径（必需）
            chat_id (str, optional): 会话ID
            is_group (bool, optional): 是否群组消息
            group_name (str, optional): 群组名称（私聊时为空字符串）
            user_name (str, optional): 发送消息的用户姓名
            
        函数可以返回一个新的文件路径，该文件会被发送回去
        如果返回None，则不发送任何文件
        """
        def decorator(func):
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            @self.listen(message_type="file", group_only=group_only, user_only=user_only)
            async def wrapper(ctx: MessageContext):
                result_path = None
                try:
                    file_key = json.loads(ctx.content).get("file_key")
                    if not file_key:
                        _print_warn("未找到文件key")
                        return

                    # 创建临时文件
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file_path = Path(temp_file.name)
                    
                    try:
                        # 下载文件
                        if not self.bot.download_message_resource(ctx.message_id, "file", str(temp_file_path), file_key):
                            _print_warn("文件下载失败")
                            return

                        # 获取群组名称和用户姓名
                        group_name, user_name = self.bot.get_chat_and_user_name(ctx.chat_id, ctx.user_id)

                        # 根据函数参数构建调用参数
                        kwargs = {}
                        if 'file_path' in param_names:
                            kwargs['file_path'] = temp_file_path
                        if 'chat_id' in param_names:
                            kwargs['chat_id'] = ctx.chat_id
                        if 'is_group' in param_names:
                            kwargs['is_group'] = ctx.is_group
                        if 'group_name' in param_names:
                            kwargs['group_name'] = group_name
                        if 'user_name' in param_names:
                            kwargs['user_name'] = user_name

                        # 调用用户处理函数
                        result_path = await func(**kwargs)
                        
                        # 如果返回了新文件路径，发送回去
                        if result_path is not None:
                            new_file_key = self.bot.upload_file(str(result_path))
                            if new_file_key:
                                self.bot.send_file_to_chat(ctx.chat_id, new_file_key)
                    finally:
                        # 清理临时文件
                        temp_file_path.unlink(missing_ok=True)
                        if result_path is not None and result_path != temp_file_path:
                            result_path.unlink(missing_ok=True)
                
                except Exception as e:
                    _print_error(f"处理文件消息时发生错误: {str(e)}")
                    import traceback
                    _print_error(f"错误详情:\n{traceback.format_exc()}")
            return wrapper
        return decorator

    def audio_handler(self, group_only: bool = False, user_only: bool = False):
        """
        音频消息处理装饰器，自动下载音频并支持处理后回传。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            @self.listen(message_type="audio", group_only=group_only, user_only=user_only)
            async def wrapper(ctx: MessageContext):
                result_path = None
                try:
                    content = self._parse_content_json(ctx.content)
                    file_key = content.get("file_key")
                    if not file_key:
                        _print_warn("未找到音频 file_key")
                        return

                    with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as temp_file:
                        temp_audio_path = Path(temp_file.name)

                    try:
                        if not self.bot.download_message_resource(ctx.message_id, "audio", str(temp_audio_path), file_key):
                            _print_warn("音频下载失败")
                            return

                        group_name, user_name = self.bot.get_chat_and_user_name(ctx.chat_id, ctx.user_id)
                        kwargs = self._select_kwargs(param_names, {
                            "audio_path": temp_audio_path,
                            "chat_id": ctx.chat_id,
                            "is_group": ctx.is_group,
                            "group_name": group_name,
                            "user_name": user_name,
                            "message_id": ctx.message_id,
                            "thread_id": ctx.thread_id,
                        })
                        result_path = await self._invoke_handler(func, **kwargs)

                        if result_path is not None:
                            new_file_key = self.bot.upload_file(str(result_path), file_type="opus")
                            if new_file_key:
                                self.bot.send_audio_to_chat(ctx.chat_id, new_file_key)
                    finally:
                        temp_audio_path.unlink(missing_ok=True)
                        if result_path is not None and result_path != temp_audio_path:
                            result_path.unlink(missing_ok=True)
                except Exception as e:
                    _print_error(f"处理音频消息时发生错误: {str(e)}")
                    import traceback
                    _print_error(f"错误详情:\n{traceback.format_exc()}")
            return wrapper
        return decorator

    def media_handler(self, group_only: bool = False, user_only: bool = False):
        """
        视频/媒体消息处理装饰器，自动下载媒体并支持处理后回传。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            @self.listen(message_type="media", group_only=group_only, user_only=user_only)
            async def wrapper(ctx: MessageContext):
                result_path = None
                try:
                    content = self._parse_content_json(ctx.content)
                    file_key = content.get("file_key")
                    if not file_key:
                        _print_warn("未找到媒体 file_key")
                        return

                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                        temp_media_path = Path(temp_file.name)

                    try:
                        if not self.bot.download_message_resource(ctx.message_id, "media", str(temp_media_path), file_key):
                            _print_warn("媒体下载失败")
                            return

                        group_name, user_name = self.bot.get_chat_and_user_name(ctx.chat_id, ctx.user_id)
                        kwargs = self._select_kwargs(param_names, {
                            "media_path": temp_media_path,
                            "chat_id": ctx.chat_id,
                            "is_group": ctx.is_group,
                            "group_name": group_name,
                            "user_name": user_name,
                            "message_id": ctx.message_id,
                            "thread_id": ctx.thread_id,
                        })
                        result_path = await self._invoke_handler(func, **kwargs)

                        if result_path is not None:
                            new_file_key = self.bot.upload_file(str(result_path), file_type="mp4")
                            if new_file_key:
                                self.bot.send_media_to_chat(ctx.chat_id, new_file_key)
                    finally:
                        temp_media_path.unlink(missing_ok=True)
                        if result_path is not None and result_path != temp_media_path:
                            result_path.unlink(missing_ok=True)
                except Exception as e:
                    _print_error(f"处理媒体消息时发生错误: {str(e)}")
                    import traceback
                    _print_error(f"错误详情:\n{traceback.format_exc()}")
            return wrapper
        return decorator

    def sticker_handler(self, group_only: bool = False, user_only: bool = False):
        """
        贴纸消息处理装饰器，传入原始或解析后的贴纸内容。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            @self.listen(message_type="sticker", group_only=group_only, user_only=user_only)
            async def wrapper(ctx: MessageContext):
                try:
                    group_name, user_name = self.bot.get_chat_and_user_name(ctx.chat_id, ctx.user_id)
                    parsed_content = self._parse_content_json(ctx.content)
                    kwargs = self._select_kwargs(param_names, {
                        "sticker_content": parsed_content or ctx.content,
                        "raw_content": ctx.content,
                        "chat_id": ctx.chat_id,
                        "is_group": ctx.is_group,
                        "group_name": group_name,
                        "user_name": user_name,
                        "message_id": ctx.message_id,
                        "thread_id": ctx.thread_id,
                    })
                    await self._invoke_handler(func, **kwargs)
                except Exception as e:
                    _print_error(f"处理贴纸消息时发生错误: {str(e)}")
                    import traceback
                    _print_error(f"错误详情:\n{traceback.format_exc()}")
            return wrapper
        return decorator

    def recall_handler(self):
        """
        消息撤回事件处理装饰器。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            async def wrapper(data):
                event = data.event
                kwargs = self._select_kwargs(param_names, {
                    "message_id": event.message_id,
                    "chat_id": event.chat_id,
                    "recall_time": event.recall_time,
                    "recall_type": event.recall_type,
                    "raw_event": data,
                })
                await self._invoke_handler(func, **kwargs)

            self._register_event_handler("register_p2_im_message_recalled_v1", wrapper)
            return wrapper
        return decorator

    def message_read_handler(self):
        """
        消息已读事件处理装饰器。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            async def wrapper(data):
                event = data.event
                kwargs = self._select_kwargs(param_names, {
                    "reader": event.reader,
                    "message_id_list": event.message_id_list,
                    "raw_event": data,
                })
                await self._invoke_handler(func, **kwargs)

            self._register_event_handler("register_p2_im_message_message_read_v1", wrapper)
            return wrapper
        return decorator

    def bot_added_handler(self):
        """
        机器人被拉入群的事件处理装饰器。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            async def wrapper(data):
                event = data.event
                kwargs = self._select_kwargs(param_names, {
                    "chat_id": event.chat_id,
                    "operator_id": event.operator_id,
                    "external": event.external,
                    "operator_tenant_key": event.operator_tenant_key,
                    "name": event.name,
                    "raw_event": data,
                })
                await self._invoke_handler(func, **kwargs)

            self._register_event_handler("register_p2_im_chat_member_bot_added_v1", wrapper)
            return wrapper
        return decorator

    def bot_removed_handler(self):
        """
        机器人被移出群的事件处理装饰器。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            async def wrapper(data):
                event = data.event
                kwargs = self._select_kwargs(param_names, {
                    "chat_id": event.chat_id,
                    "operator_id": event.operator_id,
                    "external": event.external,
                    "operator_tenant_key": event.operator_tenant_key,
                    "name": event.name,
                    "raw_event": data,
                })
                await self._invoke_handler(func, **kwargs)

            self._register_event_handler("register_p2_im_chat_member_bot_deleted_v1", wrapper)
            return wrapper
        return decorator

    def bot_p2p_chat_entered_handler(self):
        """
        机器人进入某个用户私聊会话的事件处理装饰器。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            async def wrapper(data):
                event = data.event
                kwargs = self._select_kwargs(param_names, {
                    "chat_id": event.chat_id,
                    "operator_id": event.operator_id,
                    "last_message_id": event.last_message_id,
                    "last_message_create_time": event.last_message_create_time,
                    "raw_event": data,
                })
                await self._invoke_handler(func, **kwargs)

            self._register_event_handler("register_p2_im_chat_access_event_bot_p2p_chat_entered_v1", wrapper)
            return wrapper
        return decorator

    def member_changed_handler(self):
        """
        群成员加入/移除/退出事件处理装饰器。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            def make_wrapper(action: str):
                async def wrapper(data):
                    event = data.event
                    kwargs = self._select_kwargs(param_names, {
                        "action": action,
                        "chat_id": event.chat_id,
                        "operator_id": event.operator_id,
                        "users": event.users,
                        "external": event.external,
                        "operator_tenant_key": event.operator_tenant_key,
                        "name": event.name,
                        "raw_event": data,
                    })
                    await self._invoke_handler(func, **kwargs)
                return wrapper

            self._register_event_handler("register_p2_im_chat_member_user_added_v1", make_wrapper("added"))
            self._register_event_handler("register_p2_im_chat_member_user_deleted_v1", make_wrapper("deleted"))
            self._register_event_handler("register_p2_im_chat_member_user_withdrawn_v1", make_wrapper("withdrawn"))
            return func
        return decorator

    def chat_updated_handler(self):
        """
        群信息变更事件处理装饰器。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            async def wrapper(data):
                event = data.event
                kwargs = self._select_kwargs(param_names, {
                    "chat_id": event.chat_id,
                    "operator_id": event.operator_id,
                    "external": event.external,
                    "operator_tenant_key": event.operator_tenant_key,
                    "before_change": event.before_change,
                    "after_change": event.after_change,
                    "moderator_list": event.moderator_list,
                    "raw_event": data,
                })
                await self._invoke_handler(func, **kwargs)

            self._register_event_handler("register_p2_im_chat_updated_v1", wrapper)
            return wrapper
        return decorator

    def chat_disbanded_handler(self):
        """
        群解散事件处理装饰器。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            async def wrapper(data):
                event = data.event
                kwargs = self._select_kwargs(param_names, {
                    "chat_id": event.chat_id,
                    "operator_id": event.operator_id,
                    "external": event.external,
                    "operator_tenant_key": event.operator_tenant_key,
                    "name": event.name,
                    "raw_event": data,
                })
                await self._invoke_handler(func, **kwargs)

            self._register_event_handler("register_p2_im_chat_disbanded_v1", wrapper)
            return wrapper
        return decorator

    def reaction_handler(self):
        """
        消息表情回应事件处理装饰器。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            def make_wrapper(action: str):
                async def wrapper(data):
                    event = data.event
                    kwargs = self._select_kwargs(param_names, {
                        "action": action,
                        "message_id": event.message_id,
                        "emoji_type": getattr(event.reaction_type, "emoji_type", ""),
                        "operator_type": event.operator_type,
                        "user_id": event.user_id,
                        "app_id": event.app_id,
                        "action_time": event.action_time,
                        "raw_event": data,
                    })
                    await self._invoke_handler(func, **kwargs)
                return wrapper

            self._register_event_handler("register_p2_im_message_reaction_created_v1", make_wrapper("created"))
            self._register_event_handler("register_p2_im_message_reaction_deleted_v1", make_wrapper("deleted"))
            return func
        return decorator

    def mention_handler(self, group_only: bool = False, user_only: bool = False):
        """
        机器人被 @ 时的文本消息处理装饰器。
        """
        def decorator(func):
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())

            @self.listen(message_type="text", group_only=group_only, user_only=user_only)
            async def wrapper(ctx: MessageContext):
                if not ctx.mentions:
                    return
                try:
                    group_name, user_name = self.bot.get_chat_and_user_name(ctx.chat_id, ctx.user_id)
                    kwargs = self._select_kwargs(param_names, {
                        "text": ctx.content,
                        "mentions": ctx.mentions,
                        "chat_id": ctx.chat_id,
                        "is_group": ctx.is_group,
                        "group_name": group_name,
                        "user_name": user_name,
                        "message_id": ctx.message_id,
                        "thread_id": ctx.thread_id,
                        "root_id": ctx.root_id,
                        "parent_id": ctx.parent_id,
                        "raw_event": ctx.raw_event,
                    })
                    await self._invoke_handler(func, **kwargs)
                except Exception as e:
                    _print_error(f"处理 @ 消息时发生错误: {str(e)}")
                    import traceback
                    _print_error(f"错误详情:\n{traceback.format_exc()}")
            return wrapper
        return decorator

    def card_action_handler(self, verification_token: str, encrypt_key: str = ""):
        """
        交互卡片回调处理装饰器。装饰后可通过 get_card_action_handler() 取回 HTTP handler。
        """
        def decorator(func):
            self._card_action_handler = CardActionHandler.builder(encrypt_key, verification_token).register(func).build()
            return func
        return decorator

    def get_card_action_handler(self) -> CardActionHandler:
        """
        获取交互卡片 HTTP 回调处理器，用于挂到 Web 框架路由中。
        """
        if self._card_action_handler is None:
            raise RuntimeError("card_action_handler has not been registered yet")
        return self._card_action_handler


# 使用示例
if __name__ == "__main__":
    # 创建监听器实例
    listener = LarkBotListener(
        app_id="xxx",
        app_secret="xxx"
    )

    # 创建聊天管理器
    chat_manager = ChatManager(
        base_url="https://api.deepseek.com/v1",
        api_key="xxx",
        timeout=3600  # 1小时超时
    )

    detector = ApriltagCornerDetector()

    # ====== 使用高级接口的示例 ======
    
    # 示例1：AI文本处理（高级接口）
    @listener.text_handler()
    async def handle_text(text: str, chat_id: str, is_group: bool, group_name: str, user_name: str):
        try:
            _print_info("开始处理AI回复...")
            # 获取对应的聊天机器人实例
            chat_bot = chat_manager.get_chat(chat_id)
            # 使用线程池执行同步的chat调用
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: ''.join(chat_bot.chat(text, stream=True)))
            _print_success(f"AI回复: {response}")
            # 发送AI回复到飞书
            listener.send_message(chat_id, f'Hi, {group_name} - {user_name}\n{response}')
        except Exception as e:
            _print_error(f"AI处理消息时发生错误: {e}")
            listener.send_message(chat_id, f"抱歉，处理消息时发生错误: {e}")

    # 示例2：简单的图片处理（高级接口）
    @listener.image_handler()
    async def handle_image(image_path: Path, chat_id: str, is_group: bool, group_name: str, user_name: str) -> Optional[Path]:
        _print_info(f"处理图片: {image_path} from {group_name} - {user_name}")
        # 检测并绘制AprilTag
        detected_image = detector.detect_and_draw(image_path)
        # 保存处理后的图片到新的临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            result_path = Path(temp_file.name)
        cv2.imwrite(str(result_path), detected_image)
        return result_path  # 返回处理后的图片路径，会自动发送并清理

    # 示例3：简单的文件处理（高级接口）
    @listener.file_handler()
    async def handle_file(file_path: Path, chat_id: str, is_group: bool, group_name: str, user_name: str) -> Optional[Path]:
        _print_info(f"收到文件: {file_path} from {group_name} - {user_name}")
        return file_path  # 直接返回原文件路径，会自动发送并清理

    # ====== 使用原始接口的示例 ======

    # 示例4：AI文本处理（原始接口）
    @listener.listen(message_type="text")
    async def handle_text_raw(ctx: MessageContext):
        try:
            _print_info("开始处理AI回复...")
            # 获取对应的聊天机器人实例
            chat_bot = chat_manager.get_chat(ctx.chat_id)
            # 使用线程池执行同步的chat调用
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: ''.join(chat_bot.chat(ctx.content, stream=True)))
            _print_success(f"AI回复: {response}")
            # 发送AI回复到飞书
            listener.send_message(ctx.chat_id, response)
        except Exception as e:
            _print_error(f"AI处理消息时发生错误: {e}")
            listener.send_message(ctx.chat_id, f"抱歉，处理消息时发生错误: {e}")

    # 示例5：图片处理（原始接口）
    @listener.listen(message_type="image")
    async def handle_image_raw(ctx: MessageContext):
        try:
            image_key = json.loads(ctx.content).get("image_key")
            if not image_key:
                _print_warn("未找到图片key")
                return

            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_image_path = Path(temp_file.name)
            
            try:
                # 下载图片
                if not listener.bot.download_message_resource(ctx.message_id, "image", str(temp_image_path), image_key):
                    _print_warn("图片下载失败")
                    return

                # 检测并绘制AprilTag
                detected_image = detector.detect_and_draw(temp_image_path)
                cv2.imwrite(str(temp_image_path), detected_image)
                
                # 重新上传并发送回去
                new_image_key = listener.bot.upload_image(str(temp_image_path))
                if new_image_key:
                    listener.bot.send_image_to_chat(ctx.chat_id, new_image_key)
                    _print_success("图片已发送回去")
                else:
                    _print_warn("图片上传失败")
            finally:
                # 清理临时文件
                temp_image_path.unlink(missing_ok=True)
                
        except Exception as e:
            _print_error(f"处理图片消息时发生错误: {str(e)}")
            import traceback
            _print_error(f"错误详情:\n{traceback.format_exc()}")

    # 示例6：文件处理（原始接口）
    @listener.listen(message_type="file")
    async def handle_file_raw(ctx: MessageContext):
        try:
            file_key = json.loads(ctx.content).get("file_key")
            if not file_key:
                _print_warn("未找到文件key")
                return

            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = Path(temp_file.name)
            
            try:
                # 下载文件
                if not listener.bot.download_message_resource(ctx.message_id, "file", str(temp_file_path), file_key):
                    _print_warn("文件下载失败")
                    return

                # 重新上传并发送回去
                new_file_key = listener.bot.upload_file(str(temp_file_path))
                if new_file_key:
                    listener.bot.send_file_to_chat(ctx.chat_id, new_file_key)
                    _print_success("文件已发送回去")
                else:
                    _print_warn("文件上传失败")
            finally:
                # 清理临时文件
                temp_file_path.unlink(missing_ok=True)
                
        except Exception as e:
            _print_error(f"处理文件消息时发生错误: {str(e)}")
            import traceback
            _print_error(f"错误详情:\n{traceback.format_exc()}")

    # 示例7：富文本消息处理（原始接口）
    @listener.listen(message_type="post")
    async def handle_post_raw(ctx: MessageContext):
        post_content = json.loads(ctx.content)
        _print_info(f"收到富文本消息: {post_content} (来自{'群组' if ctx.is_group else '私聊'})")
        
        # 创建回复的富文本消息
        post = PostContent(title="回复富文本消息")
        text_content = post.make_text_content("收到您的富文本消息，这是回复", styles=["bold"])
        post.add_content_in_new_line(text_content)
        
        # 发送富文本消息
        listener.bot.send_post_to_chat(ctx.chat_id, post.get_content())
        _print_success("富文本消息已发送回去")

    # 启动服务
    listener.run()
