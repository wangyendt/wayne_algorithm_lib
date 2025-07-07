# author: wangye(Wayne)
# license: Apache Licence
# file: cross_comm.py
# time: 2025-01-20-22:00:00
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.

import asyncio
import json
import os
import time
import uuid
import functools
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.client import WebSocketClientProtocol
from dotenv import load_dotenv

from pywayne.tools import read_yaml_config, write_yaml_config, wayne_print
from pywayne.aliyun_oss import OssManager

# 加载环境变量
load_dotenv()

# 导出的公共接口
__all__ = ['CrossCommService', 'Message', 'CommMsgType']

class CommMsgType(Enum):
    """消息类型枚举"""
    TEXT = "text"
    JSON = "json"
    DICT = "dict"
    BYTES = "bytes"
    IMAGE = "image"
    FILE = "file"
    FOLDER = "folder"
    HEARTBEAT = "heartbeat"
    LOGIN = "login"
    LOGOUT = "logout"
    LIST_CLIENTS = "list_clients"
    LIST_CLIENTS_RESPONSE = "list_clients_response"
    LOGIN_RESPONSE = "login_response"

@dataclass
class Message:
    """消息结构体"""
    msg_id: str
    from_client_id: str
    to_client_id: str  # "all" 表示发送给所有在线客户端
    msg_type: CommMsgType  # 使用枚举类型
    content: Any
    timestamp: float
    oss_key: Optional[str] = None  # 用于文件传输

    def to_dict(self) -> Dict:
        data = asdict(self)
        # 将枚举转换为字符串值用于JSON序列化
        data['msg_type'] = self.msg_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        # 将字符串转换回枚举
        if isinstance(data['msg_type'], str):
            data['msg_type'] = CommMsgType(data['msg_type'])
        return cls(**data)

class CrossCommService:
    """跨语言、多设备通信服务"""
    
    def __init__(self, role: str, ip: str = '0.0.0.0', port: int = 9898, 
                 client_id: Optional[str] = None, heartbeat_interval: int = 30,
                 heartbeat_timeout: int = 60):
        """
        初始化通信服务
        
        Args:
            role: 'server' 或 'client'
            ip: IP地址
            port: 端口号
            client_id: 客户端唯一ID（如果为None则自动生成）
            heartbeat_interval: 心跳间隔（秒）
            heartbeat_timeout: 心跳超时时间（秒）
        """
        self.role = role
        self.ip = ip
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        
        # 生成客户端唯一ID
        if client_id is None:
            # 使用uuid.getnode()获取MAC地址，转换为16进制字符串
            mac_node = uuid.getnode()
            mac_address = f"{mac_node:012x}"  # 转换为12位16进制字符串
            unique_id = str(uuid.uuid4())[:8]
            self.client_id = f"{mac_address}_{unique_id}"
        else:
            self.client_id = client_id
            
        # 初始化OSS管理器
        self.oss_manager = OssManager(
            endpoint=os.getenv('OSS_ENDPOINT'),
            bucket_name=os.getenv('OSS_BUCKET_NAME'),
            api_key=os.getenv('OSS_ACCESS_KEY_ID'),
            api_secret=os.getenv('OSS_ACCESS_KEY_SECRET'),
            verbose=False
        )
        
        # 服务器状态
        if self.role == 'server':
            self.clients: Dict[str, Dict] = {}  # client_id -> {websocket, last_heartbeat, status}
            self.clients_config_file = 'cross_comm_clients.yaml'
            self.server = None
            self._load_clients_config()
        
        # 客户端状态
        if self.role == 'client':
            self.websocket: Optional[WebSocketClientProtocol] = None
            self.is_connected = False
            self.heartbeat_task = None
            
        # 消息监听器（存储处理器及其配置）
        self.message_handlers: List[Dict[str, Any]] = []
        
        # 客户端列表缓存
        self._last_client_list = None
        
        wayne_print(f"CrossCommService initialized: role={role}, client_id={self.client_id}", 'green')
    
    def download_file_manually(self, oss_key: str, save_directory: str) -> bool:
        """
        手动下载文件
        
        Args:
            oss_key: OSS中的文件key
            save_directory: 保存目录（OSS管理器会在此目录下重建路径结构）
            
        Returns:
            是否下载成功
        """
        if oss_key.endswith('/'):
            # 这是一个文件夹
            return self._download_folder_from_oss(oss_key, save_directory)
        else:
            # 这是一个文件
            return self._download_file_from_oss(oss_key, save_directory)
    
    def _load_clients_config(self):
        """加载客户端配置"""
        if os.path.exists(self.clients_config_file):
            config = read_yaml_config(self.clients_config_file)
            # 将所有客户端状态设置为offline（服务器重启后）
            for client_id in config:
                config[client_id]['status'] = 'offline'
            write_yaml_config(self.clients_config_file, config)
    
    def _save_clients_config(self):
        """保存客户端配置"""
        config = {}
        for client_id, info in self.clients.items():
            config[client_id] = {
                'status': info.get('status', 'offline'),
                'last_seen': info.get('last_heartbeat', time.time())
            }
        write_yaml_config(self.clients_config_file, config, use_lock=True)
    
    def _generate_msg_id(self) -> str:
        """生成消息ID"""
        return f"{self.client_id}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
    
    def _upload_file_for_message(self, content: Any, msg_type: CommMsgType) -> Tuple[Any, Optional[str]]:
        """
        根据消息类型处理文件上传
        
        Args:
            content: 消息内容
            msg_type: 消息类型
            
        Returns:
            Tuple[处理后的内容, OSS_KEY（如果是文件类型）]
        """
        oss_key = None
        
        if msg_type in [CommMsgType.FILE, CommMsgType.IMAGE, CommMsgType.FOLDER]:
            if isinstance(content, (str, Path)):
                content_str = str(content)
                
                # 检查文件或文件夹是否存在
                if os.path.exists(content_str):
                    if msg_type == CommMsgType.FOLDER and os.path.isdir(content_str):
                        # 上传文件夹到OSS
                        oss_key = self._upload_folder_to_oss(content_str)
                        if not oss_key:
                            wayne_print("上传文件夹到OSS失败", 'red')
                            return content, None
                    elif msg_type in [CommMsgType.FILE, CommMsgType.IMAGE] and os.path.isfile(content_str):
                        # 上传文件到OSS
                        oss_key = self._upload_file_to_oss(content_str)
                        if not oss_key:
                            wayne_print("上传文件到OSS失败", 'red')
                            return content, None
                    else:
                        wayne_print(f"指定的消息类型 {msg_type.value} 与内容类型不匹配", 'red')
                        return content, None
                else:
                    wayne_print(f"文件或文件夹不存在: {content_str}", 'red')
                    return content, None
                
                # 保存原始路径作为内容
                return content_str, oss_key
            else:
                wayne_print(f"消息类型 {msg_type.value} 需要文件路径字符串", 'red')
                return content, None
        
        return content, oss_key
    
    def _upload_file_to_oss(self, file_path: str) -> Optional[str]:
        """上传文件到OSS"""
        try:
            # 生成唯一的OSS key
            timestamp = int(time.time() * 1000)
            random_id = uuid.uuid4().hex[:8]
            file_ext = Path(file_path).suffix
            oss_key = f"cross_comm/{self.client_id}/{timestamp}_{random_id}{file_ext}"
            
            if self.oss_manager.upload_file(oss_key, file_path):
                return oss_key
            return None
        except Exception as e:
            wayne_print(f"上传文件到OSS失败: {e}", 'red')
            return None
    
    def _upload_folder_to_oss(self, folder_path: str) -> Optional[str]:
        """上传文件夹到OSS"""
        try:
            # 生成唯一的OSS前缀
            timestamp = int(time.time() * 1000)
            random_id = uuid.uuid4().hex[:8]
            oss_prefix = f"cross_comm/{self.client_id}/{timestamp}_{random_id}_folder/"
            
            if self.oss_manager.upload_directory(folder_path, oss_prefix):
                return oss_prefix
            return None
        except Exception as e:
            wayne_print(f"上传文件夹到OSS失败: {e}", 'red')
            return None
    
    def _download_file_from_oss(self, oss_key: str, save_path: str) -> bool:
        """从OSS下载文件"""
        try:
            return self.oss_manager.download_file(oss_key, save_path)
        except Exception as e:
            wayne_print(f"从OSS下载文件失败: {e}", 'red')
            return False
    
    def _download_folder_from_oss(self, oss_prefix: str, save_path: str) -> bool:
        """从OSS下载文件夹"""
        try:
            return self.oss_manager.download_directory(oss_prefix, save_path)
        except Exception as e:
            wayne_print(f"从OSS下载文件夹失败: {e}", 'red')
            return False
    
    async def _handle_message(self, message: Message, download_directory: Optional[str] = None):
        """
        处理接收到的消息
        
        Args:
            message: 消息对象
            download_directory: 指定的下载目录（用于文件类型消息）
        """
        # 处理字节数组消息的解码
        if message.msg_type == CommMsgType.BYTES and isinstance(message.content, str):
            try:
                import base64
                message.content = base64.b64decode(message.content.encode('utf-8'))
            except Exception as e:
                wayne_print(f"解码字节消息失败: {e}", 'red')
        
        # 调用所有注册的消息处理器
        for handler_info in self.message_handlers:
            handler = handler_info['handler']
            handler_msg_type = handler_info.get('msg_type')
            handler_from_client_id = handler_info.get('from_client_id')
            
            # 过滤消息来源（不处理自己发送的消息）
            if message.from_client_id == self.client_id:
                continue
            
            # 过滤消息类型
            if handler_msg_type and message.msg_type != handler_msg_type:
                continue
            
            # 过滤发送方
            if handler_from_client_id and message.from_client_id != handler_from_client_id:
                continue
            
            try:
                await handler(message)
            except Exception as e:
                wayne_print(f"消息处理器执行错误: {e}", 'red')
    
    # ========== 服务器端方法 ==========
    
    async def _server_handle_client(self, websocket, *args, **kwargs):
        """服务器处理客户端连接"""
        # 兼容不同版本的websockets库，有些版本会传递path参数，有些不会
        client_id = None
        try:
            async for message_data in websocket:
                try:
                    data = json.loads(message_data)
                    message = Message.from_dict(data)
                    
                    if message.msg_type == CommMsgType.LOGIN:
                        client_id = message.from_client_id
                        current_time = time.time()
                        self.clients[client_id] = {
                            'websocket': websocket,
                            'last_heartbeat': current_time,
                            'status': 'online',
                            'login_time': current_time
                        }
                        self._save_clients_config()
                        wayne_print(f"客户端 {client_id} 已连接", 'green')
                        
                        # 发送登录成功响应
                        response = Message(
                            msg_id=self._generate_msg_id(),
                            from_client_id='server',
                            to_client_id=client_id,
                            msg_type=CommMsgType.LOGIN_RESPONSE,
                            content={'status': 'success'},
                            timestamp=time.time()
                        )
                        await websocket.send(json.dumps(response.to_dict()))
                    
                    elif message.msg_type == CommMsgType.LOGOUT:
                        if client_id and client_id in self.clients:
                            self.clients[client_id]['status'] = 'offline'
                            self._save_clients_config()
                            wayne_print(f"客户端 {client_id} 已登出", 'yellow')
                    
                    elif message.msg_type == CommMsgType.HEARTBEAT:
                        if client_id and client_id in self.clients:
                            self.clients[client_id]['last_heartbeat'] = time.time()
                    
                    elif message.msg_type == CommMsgType.LIST_CLIENTS:
                        # 处理客户端列表请求
                        await self._handle_list_clients_request(message, websocket)
                    
                    else:
                        # 转发消息给目标客户端
                        await self._forward_message(message)
                        
                except json.JSONDecodeError:
                    wayne_print("收到无效的JSON消息", 'red')
                except Exception as e:
                    wayne_print(f"处理消息时出错: {e}", 'red')
                    
        except websockets.exceptions.ConnectionClosed:
            wayne_print(f"客户端 {client_id} 连接已断开", 'yellow')
        finally:
            if client_id and client_id in self.clients:
                self.clients[client_id]['status'] = 'offline'
                self._save_clients_config()
    
    async def _forward_message(self, message: Message):
        """转发消息给目标客户端"""
        if message.to_client_id == 'all':
            # 发送给所有在线客户端
            for client_id, client_info in self.clients.items():
                if (client_info['status'] == 'online' and 
                    client_id != message.from_client_id and
                    'websocket' in client_info):
                    try:
                        await client_info['websocket'].send(json.dumps(message.to_dict()))
                    except Exception as e:
                        wayne_print(f"向客户端 {client_id} 发送消息失败: {e}", 'red')
        else:
            # 发送给指定客户端
            if (message.to_client_id in self.clients and
                self.clients[message.to_client_id]['status'] == 'online'):
                try:
                    await self.clients[message.to_client_id]['websocket'].send(json.dumps(message.to_dict()))
                except Exception as e:
                    wayne_print(f"向客户端 {message.to_client_id} 发送消息失败: {e}", 'red')

    async def _handle_list_clients_request(self, message: Message, websocket):
        """处理客户端列表请求"""
        try:
            # 解析请求内容，获取是否只显示在线用户
            only_show_online = False
            if message.content:
                try:
                    content = json.loads(message.content) if isinstance(message.content, str) else message.content
                    only_show_online = content.get('only_show_online', False)
                except (json.JSONDecodeError, AttributeError):
                    # 如果解析失败，使用默认值
                    pass
            
            # 获取客户端列表
            current_time = time.time()
            client_list = []
            
            for client_id, client_info in self.clients.items():
                is_online = (current_time - client_info['last_heartbeat']) < self.heartbeat_timeout
                
                if only_show_online and not is_online:
                    continue
                
                client_data = {
                    'client_id': client_id,
                    'status': 'online' if is_online else 'offline',
                    'last_heartbeat': client_info['last_heartbeat'],
                    'login_time': client_info.get('login_time', None)
                }
                client_list.append(client_data)
            
            # 构造响应消息
            response = Message(
                msg_id=str(uuid.uuid4()),
                from_client_id='server',
                to_client_id=message.from_client_id,
                msg_type=CommMsgType.LIST_CLIENTS_RESPONSE,
                content=json.dumps({
                    'clients': client_list,
                    'total_count': len(client_list),
                    'only_show_online': only_show_online
                }),
                timestamp=time.time()
            )
            
            # 发送响应
            await websocket.send(json.dumps(response.to_dict()))
            wayne_print(f"发送客户端列表给 {message.from_client_id}: {len(client_list)} 个客户端", 'blue')
            
        except Exception as e:
            wayne_print(f"处理客户端列表请求失败: {str(e)}", 'red')
    
    async def _check_heartbeat(self):
        """检查客户端心跳"""
        while True:
            current_time = time.time()
            offline_clients = []
            
            for client_id, client_info in self.clients.items():
                if (client_info['status'] == 'online' and
                    current_time - client_info['last_heartbeat'] > self.heartbeat_timeout):
                    offline_clients.append(client_id)
            
            for client_id in offline_clients:
                self.clients[client_id]['status'] = 'offline'
                wayne_print(f"客户端 {client_id} 心跳超时，标记为离线", 'yellow')
            
            if offline_clients:
                self._save_clients_config()
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def start_server(self):
        """启动服务器"""
        if self.role != 'server':
            raise ValueError("只有服务器角色才能启动服务器")
        
        wayne_print(f"启动服务器，监听 {self.ip}:{self.port}", 'green')
        
        # 启动心跳检查任务
        asyncio.create_task(self._check_heartbeat())
        
        # 启动WebSocket服务器
        self.server = await websockets.serve(
            self._server_handle_client,
            self.ip,
            self.port
        )
        
        wayne_print("服务器已启动，等待客户端连接...", 'green')
        await self.server.wait_closed()
    
    def get_online_clients(self) -> List[str]:
        """获取在线客户端列表"""
        if self.role != 'server':
            raise ValueError("只有服务器角色才能获取在线客户端")
        
        return [client_id for client_id, info in self.clients.items() 
                if info['status'] == 'online']
    
    # ========== 客户端方法 ==========
    
    async def _client_handle_message(self):
        """客户端处理接收到的消息"""
        try:
            async for message_data in self.websocket:
                try:
                    data = json.loads(message_data)
                    message = Message.from_dict(data)
                    
                    # 处理客户端列表响应
                    if message.msg_type == CommMsgType.LIST_CLIENTS_RESPONSE:
                        self._handle_list_clients_response(message)
                        continue
                    
                    # 处理文件下载
                    if message.msg_type in [CommMsgType.FILE, CommMsgType.FOLDER, CommMsgType.IMAGE] and message.oss_key:
                        await self._handle_file_message(message)
                    else:
                        await self._handle_message(message)
                        
                except json.JSONDecodeError:
                    wayne_print("收到无效的JSON消息", 'red')
                except Exception as e:
                    wayne_print(f"处理消息时出错: {e}", 'red')
        except websockets.exceptions.ConnectionClosed:
            wayne_print("与服务器的连接已断开", 'yellow')
            self.is_connected = False
    
    async def _handle_file_message(self, message: Message):
        """处理文件消息"""
        try:
            # 查找是否有匹配的处理器设置了下载目录
            download_directory = None
            for handler_info in self.message_handlers:
                handler_msg_type = handler_info.get('msg_type')
                handler_from_client_id = handler_info.get('from_client_id')
                
                # 过滤消息来源（不处理自己发送的消息）
                if message.from_client_id == self.client_id:
                    continue
                
                # 检查是否匹配消息类型和发送方
                if handler_msg_type and message.msg_type != handler_msg_type:
                    continue
                
                if handler_from_client_id and message.from_client_id != handler_from_client_id:
                    continue
                
                # 找到匹配的处理器，获取其下载目录
                download_directory = handler_info.get('download_directory')
                if download_directory:
                    break
            
            # 如果没有找到设置下载目录的处理器，直接传递消息
            if not download_directory:
                wayne_print(f"收到文件消息但没有处理器设置下载目录，不下载: {message.oss_key}", 'yellow')
                await self._handle_message(message)
                return
            
            # 创建下载目录
            download_dir = Path(download_directory)
            download_dir.mkdir(parents=True, exist_ok=True)
            
            if message.msg_type == CommMsgType.FILE or message.msg_type == CommMsgType.IMAGE:
                # 下载文件 - 直接使用download_directory作为目标目录
                # OSS管理器会在目录下重建完整的oss_key路径结构
                if self._download_file_from_oss(message.oss_key, str(download_dir)):
                    # 计算下载后的实际文件路径（OSS管理器会重建路径结构）
                    actual_file_path = download_dir / message.oss_key
                    
                    # 如果文件已存在但有冲突，处理重命名
                    if not actual_file_path.exists():
                        # 如果预期路径不存在，说明可能有其他路径结构，尝试找到实际文件
                        file_name = Path(message.oss_key).name
                        possible_files = list(download_dir.rglob(file_name))
                        if possible_files:
                            actual_file_path = possible_files[0]
                    
                    # 更新消息内容为本地文件路径
                    message.content = str(actual_file_path)
                    await self._handle_message(message, download_directory)
                    wayne_print(f"下载文件成功: {message.oss_key} -> {actual_file_path}", 'green')
                else:
                    wayne_print(f"下载文件失败: {message.oss_key}", 'red')
                    # 即使下载失败，也传递原始消息，让用户知道有文件消息
                    await self._handle_message(message, download_directory)
            
            elif message.msg_type == CommMsgType.FOLDER:
                # 下载文件夹 - 直接使用download_directory作为目标目录
                # 为避免冲突，在目录名后添加时间戳
                timestamp = int(time.time())
                save_path = download_dir / f"folder_{timestamp}"
                
                if self._download_folder_from_oss(message.oss_key, str(save_path)):
                    # 更新消息内容为本地文件夹路径
                    message.content = str(save_path)
                    await self._handle_message(message, download_directory)
                    wayne_print(f"下载文件夹成功: {message.oss_key} -> {save_path}", 'green')
                else:
                    wayne_print(f"下载文件夹失败: {message.oss_key}", 'red')
                    # 即使下载失败，也传递原始消息，让用户知道有文件夹消息
                    await self._handle_message(message, download_directory)
                    
        except Exception as e:
            wayne_print(f"处理文件消息时出错: {e}", 'red')
            # 发生异常时也传递原始消息
            await self._handle_message(message)
    
    def _handle_list_clients_response(self, message: Message):
        """处理客户端列表响应"""
        try:
            content = json.loads(message.content) if isinstance(message.content, str) else message.content
            self._last_client_list = content
            wayne_print(f"收到客户端列表，共 {content['total_count']} 个客户端", 'blue')
        except Exception as e:
            wayne_print(f"处理客户端列表响应失败: {e}", 'red')
    
    async def _send_heartbeat(self):
        """发送心跳"""
        while self.is_connected:
            try:
                heartbeat_msg = Message(
                    msg_id=self._generate_msg_id(),
                    from_client_id=self.client_id,
                    to_client_id='server',
                    msg_type=CommMsgType.HEARTBEAT,
                    content={},
                    timestamp=time.time()
                )
                await self.websocket.send(json.dumps(heartbeat_msg.to_dict()))
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                wayne_print(f"发送心跳失败: {e}", 'red')
                break
    
    async def login(self) -> bool:
        """客户端登录"""
        if self.role != 'client':
            raise ValueError("只有客户端角色才能登录")
        
        try:
            # 连接到服务器
            uri = f"ws://{self.ip}:{self.port}"
            self.websocket = await websockets.connect(uri)
            self.is_connected = True
            
            # 发送登录消息
            login_msg = Message(
                msg_id=self._generate_msg_id(),
                from_client_id=self.client_id,
                to_client_id='server',
                msg_type=CommMsgType.LOGIN,
                content={},
                timestamp=time.time()
            )
            await self.websocket.send(json.dumps(login_msg.to_dict()))
            
            # 启动消息接收任务
            asyncio.create_task(self._client_handle_message())
            
            # 启动心跳任务
            self.heartbeat_task = asyncio.create_task(self._send_heartbeat())
            
            wayne_print(f"客户端 {self.client_id} 已连接到服务器", 'green')
            return True
            
        except Exception as e:
            wayne_print(f"连接服务器失败: {e}", 'red')
            return False
    
    async def logout(self):
        """客户端登出"""
        if self.role != 'client':
            raise ValueError("只有客户端角色才能登出")
        
        if self.is_connected and self.websocket:
            try:
                # 发送登出消息
                logout_msg = Message(
                    msg_id=self._generate_msg_id(),
                    from_client_id=self.client_id,
                    to_client_id='server',
                    msg_type=CommMsgType.LOGOUT,
                    content={},
                    timestamp=time.time()
                )
                await self.websocket.send(json.dumps(logout_msg.to_dict()))
                
                # 取消心跳任务
                if self.heartbeat_task:
                    self.heartbeat_task.cancel()
                
                # 关闭连接
                await self.websocket.close()
                self.is_connected = False
                
                wayne_print(f"客户端 {self.client_id} 已登出", 'yellow')
                
            except Exception as e:
                wayne_print(f"登出时出错: {e}", 'red')
    
    async def send_message(self, content: Any, msg_type: CommMsgType, 
                          to_client_id: str = 'all') -> bool:
        """
        发送消息
        
        Args:
            content: 消息内容
            msg_type: 消息类型（必须指定）
            to_client_id: 目标客户端ID，'all'表示发送给所有在线客户端
        """
        if self.role != 'client':
            raise ValueError("只有客户端角色才能发送消息")
        
        if not self.is_connected or not self.websocket:
            wayne_print("客户端未连接到服务器", 'red')
            return False
        
        try:
            # 根据消息类型处理内容和文件上传
            processed_content, oss_key = self._upload_file_for_message(content, msg_type)
            
            # 对于字节类型，需要转换为base64字符串以便JSON序列化
            if msg_type == CommMsgType.BYTES and isinstance(processed_content, (bytes, bytearray)):
                import base64
                processed_content = base64.b64encode(processed_content).decode('utf-8')
            elif msg_type == CommMsgType.TEXT and not isinstance(processed_content, str):
                # 对于文本类型，确保内容是字符串
                processed_content = str(processed_content)
            elif msg_type == CommMsgType.JSON and isinstance(processed_content, str):
                # 验证JSON格式
                try:
                    json.loads(processed_content)
                except (json.JSONDecodeError, ValueError):
                    wayne_print("指定为JSON类型但内容不是有效的JSON格式", 'red')
                    return False
            elif msg_type == CommMsgType.JSON and not isinstance(processed_content, str):
                # 将非字符串内容转换为JSON字符串
                try:
                    processed_content = json.dumps(processed_content)
                except Exception as e:
                    wayne_print(f"无法将内容转换为JSON: {e}", 'red')
                    return False
            
            # 创建消息
            message = Message(
                msg_id=self._generate_msg_id(),
                from_client_id=self.client_id,
                to_client_id=to_client_id,
                msg_type=msg_type,
                content=processed_content,
                timestamp=time.time(),
                oss_key=oss_key
            )
            
            # 发送消息
            await self.websocket.send(json.dumps(message.to_dict()))
            wayne_print(f"消息已发送: {msg_type.value} -> {to_client_id}", 'green')
            return True
            
        except Exception as e:
            wayne_print(f"发送消息失败: {e}", 'red')
            return False
    
    async def list_clients(self, only_show_online: bool = True, timeout: float = 5.0) -> Optional[dict]:
        """
        获取服务器上的客户端列表
        
        Args:
            only_show_online: 是否只显示在线客户端
            timeout: 等待响应的超时时间（秒）
            
        Returns:
            包含客户端列表的字典，格式为：
            {
                'clients': [
                    {
                        'client_id': str,
                        'status': 'online'/'offline',
                        'last_heartbeat': float,
                        'login_time': float
                    },
                    ...
                ],
                'total_count': int,
                'only_show_online': bool
            }
        """
        if self.role != 'client':
            raise ValueError("只有客户端角色才能请求客户端列表")
        
        if not self.is_connected or not self.websocket:
            wayne_print("客户端未连接到服务器", 'red')
            return None
        
        try:
            # 清空之前的响应
            self._last_client_list = None
            
            # 发送客户端列表请求
            request_msg = Message(
                msg_id=self._generate_msg_id(),
                from_client_id=self.client_id,
                to_client_id='server',
                msg_type=CommMsgType.LIST_CLIENTS,
                content=json.dumps({'only_show_online': only_show_online}),
                timestamp=time.time()
            )
            
            await self.websocket.send(json.dumps(request_msg.to_dict()))
            wayne_print(f"已发送客户端列表请求 (only_online={only_show_online})", 'blue')
            
            # 等待响应
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self._last_client_list is not None:
                    return self._last_client_list
                await asyncio.sleep(0.1)
            
            wayne_print(f"请求客户端列表超时 ({timeout}秒)", 'yellow')
            return None
            
        except Exception as e:
            wayne_print(f"请求客户端列表失败: {e}", 'red')
            return None
    
    # ========== 装饰器方法 ==========
    
    def message_listener(self, msg_type: Optional[CommMsgType] = None, 
                        from_client_id: Optional[str] = None,
                        download_directory: Optional[str] = None):
        """
        消息监听装饰器
        
        Args:
            msg_type: 监听的消息类型，None表示监听所有类型
            from_client_id: 监听特定发送方的消息，None表示监听所有发送方
            download_directory: 文件下载目录，仅对文件类型消息有效（FILE, IMAGE, FOLDER）
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(message: Message):
                # 调用用户处理函数
                try:
                    await func(message)
                except Exception as e:
                    wayne_print(f"消息处理函数执行错误: {e}", 'red')
            
            # 注册消息处理器配置
            handler_info = {
                'handler': wrapper,
                'msg_type': msg_type,
                'from_client_id': from_client_id,
                'download_directory': download_directory
            }
            self.message_handlers.append(handler_info)
            
            # 如果设置了下载目录，打印提示信息
            if download_directory and msg_type in [CommMsgType.FILE, CommMsgType.IMAGE, CommMsgType.FOLDER]:
                wayne_print(f"已为 {msg_type.value} 类型消息设置下载目录: {download_directory}", 'cyan')
            
            return wrapper
        
        return decorator
    
    def stop(self):
        """停止服务"""
        if self.role == 'server' and self.server:
            self.server.close()
        elif self.role == 'client':
            self.is_connected = False


if __name__ == '__main__':
    import sys
    
    async def server_example():
        """服务器示例"""
        # 创建服务器
        server = CrossCommService(
            role='server',
            ip='0.0.0.0',  # 监听所有网络接口
            port=9898,
            heartbeat_interval=30,
            heartbeat_timeout=60
        )
        
        wayne_print(f"服务器ID: {server.client_id}", 'cyan')
        wayne_print("启动服务器...", 'blue')
        
        # 启动服务器（这会阻塞直到服务器停止）
        await server.start_server()

    async def client_example():
        """客户端示例"""
        # 创建客户端
        client = CrossCommService(
            role='client',
            ip='localhost',  # 连接到本地服务器
            port=9898,
            client_id='example_client',  # 可选：指定客户端ID
            heartbeat_interval=30,
            heartbeat_timeout=60
        )
        
        wayne_print(f"客户端ID: {client.client_id}", 'cyan')
        
        # 注册消息监听器
        @client.message_listener(msg_type=CommMsgType.TEXT)
        async def handle_text_message(message: Message):
            wayne_print(f"收到文本消息: {message.content}", 'green')
            wayne_print(f"来自: {message.from_client_id}", 'white')
            wayne_print(f"时间: {message.timestamp}", 'white')
        
        # 为文件类型消息指定下载目录
        @client.message_listener(msg_type=CommMsgType.FILE, download_directory="./downloads/files")
        async def handle_file_message(message: Message):
            # message.content 包含下载后的本地文件路径
            wayne_print(f"收到文件并已下载: {message.content}", 'yellow')
            wayne_print(f"来自: {message.from_client_id}", 'white')
            
            # 可以直接使用本地文件
            try:
                with open(message.content, 'r', encoding='utf-8') as f:
                    file_content = f.read()[:100]  # 只读取前100个字符
                    wayne_print(f"文件内容预览: {file_content}...", 'white')
            except Exception as e:
                wayne_print(f"读取文件内容失败: {e}", 'red')
        
        # 为图片类型消息指定下载目录
        @client.message_listener(msg_type=CommMsgType.IMAGE, download_directory="./downloads/images")
        async def handle_image_message(message: Message):
            wayne_print(f"收到图片并已下载: {message.content}", 'cyan')
            wayne_print(f"来自: {message.from_client_id}", 'white')
            
            # 可以进一步处理图片文件
            file_size = Path(message.content).stat().st_size
            wayne_print(f"图片大小: {file_size} 字节", 'white')
        
        # 为文件夹类型消息指定下载目录
        @client.message_listener(msg_type=CommMsgType.FOLDER, download_directory="./downloads/folders")
        async def handle_folder_message(message: Message):
            wayne_print(f"收到文件夹并已下载: {message.content}", 'magenta')
            wayne_print(f"来自: {message.from_client_id}", 'white')
            
            # 列出文件夹内容
            try:
                folder_path = Path(message.content)
                files = list(folder_path.iterdir())
                wayne_print(f"文件夹包含 {len(files)} 个项目", 'white')
            except Exception as e:
                wayne_print(f"读取文件夹内容失败: {e}", 'red')
        
        # 不设置下载目录的处理器 - 不会自动下载文件
        @client.message_listener(msg_type=CommMsgType.FILE, from_client_id="special_client")
        async def handle_special_file_message(message: Message):
            # 不会自动下载，只显示消息信息
            wayne_print(f"收到特殊客户端的文件消息: {message.content}", 'yellow')
            wayne_print(f"OSS Key: {message.oss_key}", 'white')
            
            # 可以选择手动下载
            # success = client.download_file_manually(message.oss_key, "./manual_downloads/")
            # if success:
            #     wayne_print("文件下载成功", 'green')
        
        @client.message_listener()  # 监听所有消息类型
        async def handle_all_messages(message: Message):
            wayne_print(f"[通用处理器] {message.msg_type.value}: {message.content}", 'white')
        
        try:
            # 连接到服务器
            success = await client.login()
            if not success:
                wayne_print("连接服务器失败", 'red')
                return
            
            wayne_print("连接成功！", 'green')
            
            # 发送各种类型的消息
            
            # 1. 发送文本消息给所有在线客户端
            await client.send_message("Hello everyone!", CommMsgType.TEXT)
            
            # 2. 发送JSON消息
            await client.send_message('{"type": "notification", "data": "test"}', CommMsgType.JSON)
            
            # 3. 发送字典消息
            await client.send_message({"action": "update", "id": 123}, CommMsgType.DICT)
            
            # 4. 发送字节数据
            await client.send_message(b"Binary data", CommMsgType.BYTES)
            
            # 5. 发送文件（会自动上传到OSS）
            # await client.send_message("/path/to/your/file.txt", CommMsgType.FILE)
            
            # 6. 发送图片（会自动上传到OSS）
            # await client.send_message("/path/to/your/image.png", CommMsgType.IMAGE)
            
            # 7. 发送文件夹（会自动上传到OSS）
            # await client.send_message("/path/to/your/folder", CommMsgType.FOLDER)
            
            # 8. 发送给指定客户端
            # await client.send_message("Private message", CommMsgType.TEXT, to_client_id='target_client_id')
            
            # 9. 获取客户端列表
            wayne_print("\\n=== 获取客户端列表示例 ===", 'cyan')
            
            # 获取所有客户端（包括离线）
            all_clients = await client.list_clients(only_show_online=False)
            if all_clients:
                wayne_print(f"所有客户端数量: {all_clients['total_count']}", 'white')
                for client_info in all_clients['clients']:
                    status_color = 'green' if client_info['status'] == 'online' else 'yellow'
                    wayne_print(f"  - {client_info['client_id']}: {client_info['status']}", status_color)
            
            # 获取在线客户端
            online_clients = await client.list_clients(only_show_online=True)
            if online_clients:
                wayne_print(f"在线客户端数量: {online_clients['total_count']}", 'green')
                for client_info in online_clients['clients']:
                    wayne_print(f"  - {client_info['client_id']}: {client_info['status']}", 'green')
            
            # 10. 文件下载控制示例
            wayne_print("\\n=== 文件下载控制示例 ===", 'cyan')
            wayne_print("文件下载目录现在在消息监听器装饰器中指定：", 'white')
            wayne_print("- 指定下载目录的处理器会自动下载文件", 'white')
            wayne_print("- 不指定下载目录的处理器不会自动下载，节省流量", 'white')
            wayne_print("- 可以为不同类型的文件设置不同的下载目录", 'white')
            
            # 手动下载文件示例（适用于没有设置下载目录的处理器）
            # success = client.download_file_manually("oss_key", "./manual_downloads/")
            # if success:
            #     wayne_print("文件下载成功", 'green')
            # else:
            #     wayne_print("文件下载失败", 'red')
            
            # 保持连接
            while client.is_connected:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            wayne_print("用户中断", 'yellow')
        finally:
            # 登出
            if client.is_connected:
                await client.logout()

    def run_server():
        """运行服务器"""
        wayne_print("=== 跨语言通信服务器示例 ===", 'cyan', True)
        asyncio.run(server_example())

    def run_client():
        """运行客户端"""
        wayne_print("=== 跨语言通信客户端示例 ===", 'cyan', True)
        asyncio.run(client_example())

    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        run_server()
    elif len(sys.argv) > 1 and sys.argv[1] == 'client':
        run_client()
    else:
        wayne_print("使用方法:", 'white')
        wayne_print("  python -m pywayne.cross_comm server   # 运行服务器", 'white')
        wayne_print("  python -m pywayne.cross_comm client   # 运行客户端", 'white') 