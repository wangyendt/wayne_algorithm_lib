跨语言通信 (cross_comm)
========================

本模块提供了一个基于WebSocket的跨语言、多设备通信解决方案。通过CrossCommService类，用户可以轻松构建支持多种消息类型的实时通信应用，包括文本、JSON、字典、字节数组、图片、文件和文件夹传输。

核心特性
--------

- **多角色支持**：支持服务器和客户端角色
- **实时通信**：基于WebSocket协议的双向实时通信
- **多种消息类型**：支持文本、JSON、字典、字节数组、图片、文件、文件夹
- **客户端管理**：唯一ID生成、在线状态管理、心跳检测
- **文件传输**：集成阿里云OSS进行大文件传输
- **消息监听**：装饰器方式注册消息处理器
- **状态持久化**：YAML文件记录客户端状态

CrossCommService 类详细说明
---------------------------

.. py:class:: CrossCommService(role: str, ip: str = 'localhost', port: int = 9898, client_id: Optional[str] = None, heartbeat_interval: float = 30.0, heartbeat_timeout: float = 60.0, oss_manager: Optional[OssManager] = None, clients_file: str = 'cross_comm_clients.yaml')

   跨语言通信服务的主要类，支持服务器和客户端两种角色。

   **初始化参数**:
   
   - **role**: 服务角色，'server' 或 'client'
   - **ip**: IP地址，服务器监听地址或客户端连接地址
   - **port**: 端口号，默认9898
   - **client_id**: 客户端唯一ID，如不指定则自动生成
   - **heartbeat_interval**: 心跳间隔（秒），默认30秒
   - **heartbeat_timeout**: 心跳超时（秒），默认60秒
   - **oss_manager**: OSS管理器实例，用于文件传输
   - **clients_file**: 客户端状态文件路径，默认'cross_comm_clients.yaml'

   **主要方法**:

   .. py:method:: start_server() -> None
      
      启动服务器，开始监听客户端连接。仅在role为'server'时可用。
      
      该方法会阻塞执行，直到服务器停止。

   .. py:method:: connect() -> bool
      
      客户端连接到服务器。仅在role为'client'时可用。
      
      **返回**:
      
      - 布尔值，表示连接是否成功

   .. py:method:: disconnect() -> None
      
      断开连接并清理资源。

   .. py:method:: send_message(content: Any, to_client_id: str = 'all', msg_type: Optional[str] = None) -> bool
      
      发送消息到指定客户端或广播到所有客户端。
      
      **参数**:
      
      - **content**: 消息内容，支持多种类型
      - **to_client_id**: 接收方客户端ID，'all'表示广播
      - **msg_type**: 消息类型，如不指定则自动检测
      
      **返回**:
      
      - 布尔值，表示发送是否成功

   .. py:method:: message_listener(msg_type: Optional[str] = None, from_client_id: Optional[str] = None)
      
      装饰器方法，用于注册消息监听器。
      
      **参数**:
      
      - **msg_type**: 要监听的消息类型，None表示监听所有类型
      - **from_client_id**: 要监听的发送方ID，None表示监听所有发送方
      
      **示例**:
      
      .. code-block:: python
      
         @service.message_listener(msg_type='text')
         async def handle_text(message: Message):
             print(f"收到文本消息: {message.content}")

   .. py:method:: get_online_clients() -> List[str]
      
      获取当前在线的客户端ID列表（仅服务器角色可用）。
      
      **返回**:
      
      - 在线客户端ID列表

   .. py:method:: list_clients(only_show_online: bool = False, timeout: float = 5.0) -> Optional[dict]
      
      客户端请求服务器上的客户端列表（仅客户端角色可用）。
      
      **参数**:
      
      - **only_show_online**: 是否只显示在线客户端，默认False
      - **timeout**: 等待响应的超时时间（秒），默认5.0
      
      **返回**:
      
      - 包含客户端列表的字典，格式为：
        
        .. code-block:: python
        
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

   .. py:method:: is_client_online(client_id: str) -> bool
      
      检查指定客户端是否在线。
      
      **参数**:
      
      - **client_id**: 客户端ID
      
      **返回**:
      
      - 布尔值，表示客户端是否在线

Message 类详细说明
------------------

.. py:class:: Message

   消息对象，包含完整的消息信息。

   **属性**:
   
   - **msg_id**: 消息唯一ID
   - **from_client_id**: 发送方客户端ID
   - **to_client_id**: 接收方客户端ID
   - **msg_type**: 消息类型
   - **content**: 消息内容
   - **timestamp**: 消息时间戳
   - **oss_key**: OSS键值（用于文件传输）

   **方法**:

   .. py:method:: to_dict() -> Dict[str, Any]
      
      将消息对象转换为字典格式。

   .. py:method:: from_dict(data: Dict[str, Any]) -> Message
      
      从字典创建消息对象（类方法）。

   .. py:method:: to_json() -> str
      
      将消息对象转换为JSON字符串。

   .. py:method:: from_json(json_str: str) -> Message
      
      从JSON字符串创建消息对象（类方法）。

支持的消息类型
--------------

- **text**: 纯文本消息
- **json**: JSON字符串消息
- **dict**: Python字典消息
- **bytes**: 字节数组消息
- **image**: 图片文件消息
- **file**: 普通文件消息
- **folder**: 文件夹消息

环境配置
--------

在使用前需要配置阿里云OSS环境变量（用于文件传输）：

.. code-block:: bash

   # .env 文件
   OSS_ENDPOINT=xxx
   OSS_BUCKET_NAME=xxx
   OSS_ACCESS_KEY_ID=xxx
   OSS_ACCESS_KEY_SECRET=xxx

使用示例
--------

以下示例展示了如何使用跨语言通信模块：

**服务器端示例**:

.. code-block:: python

   import asyncio
   from pywayne.cross_comm import CrossCommService, Message
   
   async def run_server():
       # 创建服务器
       server = CrossCommService(
           role='server',
           ip='0.0.0.0',
           port=9898,
           heartbeat_interval=30,
           heartbeat_timeout=60
       )
       
       # 注册消息监听器
       @server.message_listener(msg_type='text')
       async def handle_text(message: Message):
           print(f"收到来自 {message.from_client_id} 的文本: {message.content}")
           # 回复消息
           await server.send_message(
               content=f"服务器收到: {message.content}",
               to_client_id=message.from_client_id
           )
       
       @server.message_listener(msg_type='image')
       async def handle_image(message: Message):
           print(f"收到来自 {message.from_client_id} 的图片: {message.oss_key}")
       
       # 启动服务器
       await server.start_server()
   
   # 运行服务器
   asyncio.run(run_server())

**客户端示例**:

.. code-block:: python

   import asyncio
   from pywayne.cross_comm import CrossCommService, Message
   
   async def run_client():
       # 创建客户端
       client = CrossCommService(
           role='client',
           ip='localhost',
           port=9898,
           client_id='my_client'
       )
       
       # 注册消息监听器
       @client.message_listener()
       async def handle_all_messages(message: Message):
           print(f"收到消息: {message.content}")
       
       # 连接到服务器
       if await client.connect():
           print("连接成功!")
           
           # 发送文本消息
           await client.send_message("Hello Server!")
           
           # 发送字典消息
           await client.send_message({
               "type": "data",
               "value": 123,
               "status": "ok"
           })
           
           # 发送图片文件
           await client.send_message("/path/to/image.jpg")
           
           # 广播消息
           await client.send_message("Hello Everyone!", to_client_id='all')
           
           # 获取客户端列表
           all_clients = await client.list_clients(only_show_online=False)
           if all_clients:
               print(f"总客户端数量: {all_clients['total_count']}")
               for client_info in all_clients['clients']:
                   print(f"  - {client_info['client_id']}: {client_info['status']}")
           
           # 只获取在线客户端
           online_clients = await client.list_clients(only_show_online=True)
           if online_clients:
               print(f"在线客户端数量: {online_clients['total_count']}")
           
           # 保持连接
           await asyncio.sleep(60)
           
       # 断开连接
       await client.disconnect()
   
   # 运行客户端
   asyncio.run(run_client())

**多客户端通信示例**:

.. code-block:: python

   # 客户端A
   clientA = CrossCommService(role='client', client_id='clientA')
   await clientA.connect()
   
   # 客户端B
   clientB = CrossCommService(role='client', client_id='clientB')
   await clientB.connect()
   
   # A向B发送消息
   await clientA.send_message("Hello B!", to_client_id='clientB')
   
   # B向A发送文件
   await clientB.send_message("/path/to/file.pdf", to_client_id='clientA')

**文件传输示例**:

.. code-block:: python

   # 发送图片
   await client.send_message("/path/to/photo.jpg")
   
   # 发送文档
   await client.send_message("/path/to/document.pdf")
   
   # 发送文件夹（会压缩后传输）
   await client.send_message("/path/to/folder/")

命令行使用
----------

模块支持直接从命令行运行：

.. code-block:: bash

   # 运行服务器示例
   python -m pywayne.cross_comm server
   
   # 运行客户端示例
   python -m pywayne.cross_comm client
   
   # 查看帮助
   python -m pywayne.cross_comm --help

心跳机制
--------

系统内置心跳机制确保连接稳定性：

- 客户端定期向服务器发送心跳包
- 服务器检测客户端心跳超时自动标记为离线
- 支持自定义心跳间隔和超时时间
- 自动重连机制（客户端）

状态管理
--------

客户端状态通过YAML文件持久化存储：

.. code-block:: yaml

   clients:
     client_001:
       status: online
       last_heartbeat: 2024-01-01T12:00:00
       ip: 192.168.1.100
       connected_at: 2024-01-01T11:00:00
     client_002:
       status: offline
       last_heartbeat: 2024-01-01T11:55:00
       ip: 192.168.1.101
       connected_at: 2024-01-01T11:00:00

错误处理
--------

模块提供完善的错误处理机制：

- 连接异常自动重试
- 消息发送失败通知
- 文件上传/下载错误处理
- 详细的日志记录

注意事项
--------

1. **文件传输**：大文件通过OSS传输，需要正确配置OSS环境变量
2. **网络安全**：生产环境建议使用TLS/SSL加密
3. **性能考虑**：大量并发连接时考虑调整心跳参数
4. **资源清理**：程序退出时确保调用disconnect()方法
5. **唯一ID**：客户端ID基于MAC地址和UUID生成，确保全局唯一

扩展建议
--------

未来可考虑的扩展功能：

- 消息加密和数字签名
- 消息持久化和离线消息
- 群组和频道功能
- 消息路由和负载均衡
- 更多消息类型支持（音频、视频等）
- Web界面监控和管理
- 消息统计和分析功能 