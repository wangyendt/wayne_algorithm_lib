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
- **消息监听**：装饰器方式注册消息处理器，支持类型和来源过滤
- **状态持久化**：YAML文件记录客户端状态
- **文件下载控制**：支持指定下载目录，避免不必要的流量消耗

CommMsgType 枚举类详细说明
--------------------------

.. py:class:: CommMsgType(Enum)

   消息类型枚举，定义了所有支持的消息类型。

   **枚举值**:
   
   - **TEXT**: "text" - 文本消息
   - **JSON**: "json" - JSON字符串消息
   - **DICT**: "dict" - Python字典消息
   - **BYTES**: "bytes" - 字节数组消息
   - **IMAGE**: "image" - 图片文件消息
   - **FILE**: "file" - 普通文件消息
   - **FOLDER**: "folder" - 文件夹消息
   - **HEARTBEAT**: "heartbeat" - 心跳消息（内部使用）
   - **LOGIN**: "login" - 登录消息（内部使用）
   - **LOGOUT**: "logout" - 登出消息（内部使用）
   - **LIST_CLIENTS**: "list_clients" - 客户端列表请求（内部使用）
   - **LIST_CLIENTS_RESPONSE**: "list_clients_response" - 客户端列表响应（内部使用）
   - **LOGIN_RESPONSE**: "login_response" - 登录响应（内部使用）

CrossCommService 类详细说明
---------------------------

.. py:class:: CrossCommService(role: str, ip: str = '0.0.0.0', port: int = 9898, client_id: Optional[str] = None, heartbeat_interval: int = 30, heartbeat_timeout: int = 60)

   跨语言通信服务的主要类，支持服务器和客户端两种角色。

   **初始化参数**:
   
   - **role**: 服务角色，'server' 或 'client'
   - **ip**: IP地址，服务器监听地址或客户端连接地址
   - **port**: 端口号，默认9898
   - **client_id**: 客户端唯一ID，如不指定则自动生成（基于MAC地址和UUID）
   - **heartbeat_interval**: 心跳间隔（秒），默认30秒
   - **heartbeat_timeout**: 心跳超时（秒），默认60秒

   **主要方法**:

   .. py:method:: start_server() -> None
      
      启动服务器，开始监听客户端连接。仅在role为'server'时可用。
      
      该方法会阻塞执行，直到服务器停止。

   .. py:method:: login() -> bool
      
      客户端登录到服务器。仅在role为'client'时可用。
      
      **返回**:
      
      - 布尔值，表示登录是否成功

   .. py:method:: logout() -> None
      
      客户端登出并断开连接。仅在role为'client'时可用。

   .. py:method:: send_message(content: Any, msg_type: CommMsgType, to_client_id: str = 'all') -> bool
      
      发送消息到指定客户端或广播到所有客户端。
      
      **参数**:
      
      - **content**: 消息内容，支持多种类型
      - **msg_type**: 消息类型，必须指定CommMsgType枚举值
      - **to_client_id**: 接收方客户端ID，'all'表示广播
      
      **返回**:
      
      - 布尔值，表示发送是否成功
      
      **注意**:
      
      - 文件类型消息（FILE、IMAGE、FOLDER）会自动上传到OSS
      - 字节类型消息会自动进行base64编码
      - JSON类型消息会验证JSON格式的有效性

   .. py:method:: download_file_manually(oss_key: str, save_path: str) -> bool
      
      手动下载文件或文件夹。
      
      **参数**:
      
      - **oss_key**: OSS中的文件键值
      - **save_path**: 本地保存路径
      
      **返回**:
      
      - 布尔值，表示下载是否成功

   .. py:method:: message_listener(msg_type: Optional[CommMsgType] = None, from_client_id: Optional[str] = None, download_directory: Optional[str] = None)
      
      装饰器方法，用于注册消息监听器。支持按消息类型和发送方过滤，可指定文件下载目录。
      
      **参数**:
      
      - **msg_type**: 要监听的消息类型，None表示监听所有类型
      - **from_client_id**: 要监听的发送方ID，None表示监听所有发送方
      - **download_directory**: 文件下载目录，仅对文件类型消息有效（FILE、IMAGE、FOLDER）
      
      **特性**:
      
      - 自动过滤自己发送的消息
      - 指定下载目录的处理器会自动下载文件
      - 未指定下载目录的处理器不会自动下载，节省流量
      - 可以为不同类型的文件设置不同的下载目录
      
      **示例**:
      
      .. code-block:: python
      
         @service.message_listener(msg_type=CommMsgType.TEXT)
         async def handle_text(message: Message):
             print(f"收到文本消息: {message.content}")
         
         @service.message_listener(msg_type=CommMsgType.FILE, download_directory="./downloads")
         async def handle_file(message: Message):
             # message.content 包含下载后的本地文件路径
             print(f"文件已下载到: {message.content}")

   .. py:method:: get_online_clients() -> List[str]
      
      获取当前在线的客户端ID列表（仅服务器角色可用）。
      
      **返回**:
      
      - 在线客户端ID列表

   .. py:method:: list_clients(only_show_online: bool = True, timeout: float = 5.0) -> Optional[dict]
      
      客户端请求服务器上的客户端列表（仅客户端角色可用）。
      
      **参数**:
      
      - **only_show_online**: 是否只显示在线客户端，默认True
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

Message 类详细说明
------------------

.. py:class:: Message

   消息对象，包含完整的消息信息。

   **属性**:
   
   - **msg_id**: 消息唯一ID
   - **from_client_id**: 发送方客户端ID
   - **to_client_id**: 接收方客户端ID
   - **msg_type**: 消息类型（CommMsgType枚举）
   - **content**: 消息内容
   - **timestamp**: 消息时间戳
   - **oss_key**: OSS键值（用于文件传输）

   **方法**:

   .. py:method:: to_dict() -> Dict[str, Any]
      
      将消息对象转换为字典格式，枚举类型会转换为字符串值。

   .. py:method:: from_dict(data: Dict[str, Any]) -> Message
      
      从字典创建消息对象，字符串会转换回枚举类型（类方法）。

支持的消息类型
--------------

用户可直接使用的消息类型：

- **CommMsgType.TEXT**: 纯文本消息
- **CommMsgType.JSON**: JSON字符串消息
- **CommMsgType.DICT**: Python字典消息
- **CommMsgType.BYTES**: 字节数组消息
- **CommMsgType.IMAGE**: 图片文件消息
- **CommMsgType.FILE**: 普通文件消息
- **CommMsgType.FOLDER**: 文件夹消息

系统内部使用的消息类型（用户无需关注）：

- **CommMsgType.HEARTBEAT**: 心跳消息
- **CommMsgType.LOGIN**: 登录消息
- **CommMsgType.LOGOUT**: 登出消息
- **CommMsgType.LIST_CLIENTS**: 客户端列表请求
- **CommMsgType.LIST_CLIENTS_RESPONSE**: 客户端列表响应
- **CommMsgType.LOGIN_RESPONSE**: 登录响应

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
   from pywayne.cross_comm import CrossCommService, Message, CommMsgType
   
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
       @server.message_listener(msg_type=CommMsgType.TEXT)
       async def handle_text(message: Message):
           print(f"收到来自 {message.from_client_id} 的文本: {message.content}")
       
       @server.message_listener(msg_type=CommMsgType.IMAGE, download_directory="./server_images")
       async def handle_image(message: Message):
           # message.content 包含下载后的本地文件路径
           print(f"收到来自 {message.from_client_id} 的图片: {message.content}")
       
       # 启动服务器
       await server.start_server()
   
   # 运行服务器
   asyncio.run(run_server())

**客户端示例**:

.. code-block:: python

   import asyncio
   from pywayne.cross_comm import CrossCommService, Message, CommMsgType
   
   async def run_client():
       # 创建客户端
       client = CrossCommService(
           role='client',
           ip='localhost',
           port=9898,
           client_id='my_client'
       )
       
       # 注册消息监听器
       @client.message_listener(msg_type=CommMsgType.TEXT)
       async def handle_text(message: Message):
           print(f"收到文本消息: {message.content}")
       
       @client.message_listener(msg_type=CommMsgType.FILE, download_directory="./downloads/files")
       async def handle_file(message: Message):
           # message.content 包含下载后的本地文件路径
           print(f"文件已下载到: {message.content}")
           
           # 可以直接使用本地文件
           try:
               with open(message.content, 'r', encoding='utf-8') as f:
                   content = f.read()[:100]  # 只读取前100个字符
                   print(f"文件内容预览: {content}...")
           except Exception as e:
               print(f"读取文件失败: {e}")
       
       # 不设置下载目录的处理器 - 不会自动下载文件
       @client.message_listener(msg_type=CommMsgType.FILE, from_client_id="special_client")
       async def handle_special_file(message: Message):
           print(f"收到特殊客户端的文件消息: {message.content}")
           print(f"OSS Key: {message.oss_key}")
           
           # 可以选择手动下载
           success = client.download_file_manually(message.oss_key, "special_file.txt")
           if success:
               print("文件下载成功")
       
       # 登录到服务器
       if await client.login():
           print("登录成功!")
           
           # 发送各种类型的消息
           
           # 1. 发送文本消息
           await client.send_message("Hello Server!", CommMsgType.TEXT)
           
           # 2. 发送JSON消息
           await client.send_message('{"type": "notification", "data": "test"}', CommMsgType.JSON)
           
           # 3. 发送字典消息
           await client.send_message({
               "type": "data",
               "value": 123,
               "status": "ok"
           }, CommMsgType.DICT)
           
           # 4. 发送字节数据
           await client.send_message(b"Binary data", CommMsgType.BYTES)
           
           # 5. 发送图片文件（会自动上传到OSS）
           # await client.send_message("/path/to/image.jpg", CommMsgType.IMAGE)
           
           # 6. 发送文件（会自动上传到OSS）
           # await client.send_message("/path/to/document.pdf", CommMsgType.FILE)
           
           # 7. 发送文件夹（会自动上传到OSS）
           # await client.send_message("/path/to/folder", CommMsgType.FOLDER)
           
           # 8. 发送给指定客户端
           # await client.send_message("Private message", CommMsgType.TEXT, to_client_id='target_client_id')
           
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
           
       # 登出
       await client.logout()
   
   # 运行客户端
   asyncio.run(run_client())

**多客户端通信示例**:

.. code-block:: python

   # 客户端A
   clientA = CrossCommService(role='client', client_id='clientA')
   await clientA.login()
   
   # 客户端B
   clientB = CrossCommService(role='client', client_id='clientB')
   await clientB.login()
   
   # A向B发送消息
   await clientA.send_message("Hello B!", CommMsgType.TEXT, to_client_id='clientB')
   
   # B向A发送文件
   await clientB.send_message("/path/to/file.pdf", CommMsgType.FILE, to_client_id='clientA')

**文件下载控制示例**:

.. code-block:: python

   # 为不同类型的文件设置不同的下载目录
   @client.message_listener(msg_type=CommMsgType.IMAGE, download_directory="./downloads/images")
   async def handle_image(message: Message):
       print(f"图片已下载到: {message.content}")
   
   @client.message_listener(msg_type=CommMsgType.FILE, download_directory="./downloads/documents") 
   async def handle_document(message: Message):
       print(f"文档已下载到: {message.content}")
   
   # 不设置下载目录 - 节省流量，只获取消息信息
   @client.message_listener(msg_type=CommMsgType.FILE, from_client_id="low_priority_client")
   async def handle_low_priority_file(message: Message):
       print(f"收到低优先级文件消息，不自动下载: {message.oss_key}")
       
       # 根据需要手动下载
       if should_download(message):
           success = client.download_file_manually(message.oss_key, "manual_download.txt")
           if success:
               print("手动下载成功")

命令行使用
----------

模块支持直接从命令行运行：

.. code-block:: bash

   # 运行服务器示例
   python -m pywayne.cross_comm server
   
   # 运行客户端示例
   python -m pywayne.cross_comm client

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

   client_001:
     status: online
     last_seen: 1640995200.123
   client_002:
     status: offline
     last_seen: 1640995100.456

文件传输和下载控制
------------------

**文件上传**:

- 文件类型消息（FILE、IMAGE、FOLDER）会自动上传到阿里云OSS
- 生成唯一的OSS键值避免冲突
- 支持大文件和文件夹传输

**文件下载控制**:

- 通过message_listener装饰器的download_directory参数控制下载行为
- 指定下载目录：自动下载文件到指定目录
- 不指定下载目录：不自动下载，节省流量和存储空间
- 支持手动下载：使用download_file_manually方法
- 可为不同消息类型设置不同下载目录
- 支持按发送方过滤下载策略

错误处理
--------

模块提供完善的错误处理机制：

- 连接异常自动重试
- 消息发送失败通知
- 文件上传/下载错误处理
- 详细的日志记录
- 消息过滤避免处理自己发送的消息

注意事项
--------

1. **文件传输**：大文件通过OSS传输，需要正确配置OSS环境变量
2. **消息类型**：必须使用CommMsgType枚举，不再支持字符串类型
3. **文件下载**：合理设置下载目录，避免不必要的流量消耗
4. **网络安全**：生产环境建议使用TLS/SSL加密
5. **性能考虑**：大量并发连接时考虑调整心跳参数
6. **资源清理**：程序退出时确保调用logout()方法
7. **唯一ID**：客户端ID基于MAC地址和UUID生成，确保全局唯一

API变更说明
-----------

**v2.0 主要变更**:

- **Breaking Change**: 引入CommMsgType枚举，替代字符串类型
- **Breaking Change**: send_message方法参数顺序调整，msg_type变为必需参数
- **新增**: download_file_manually方法支持手动文件下载
- **增强**: message_listener装饰器增加download_directory参数
- **增强**: 消息过滤机制，自动过滤自己发送的消息
- **优化**: 文件下载控制，避免不必要的流量消耗

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
- 分布式部署支持 