加密解密 (crypto)
===================

本模块提供了安全的字符串加密和解密功能，支持多种加密策略和自定义密码保护。该模块采用了先进的代码混淆技术，确保加密算法的实现细节得到有效保护。

特性
------

- **双重加密策略**: 优先使用 Fernet 对称加密（基于 AES 128），当 cryptography 库不可用时自动回退到 XOR 加密
- **自定义密码支持**: 支持用户自定义密码，也可使用内置的默认密钥
- **Unicode 支持**: 完美处理中文等多字节字符
- **代码混淆保护**: 核心算法通过 base64 编码和动态函数生成技术进行混淆
- **错误处理**: 完整的异常处理机制和用户友好的错误信息
- **零依赖回退**: 即使没有额外依赖也能正常工作

安装
------

使用基本功能（XOR 加密）::

   pip install pywayne

使用高级功能（Fernet 加密）::

   pip install pywayne[crypto]

快速开始
----------

基本用法
~~~~~~~~~

.. code-block:: python

   from pywayne.crypto import encrypt, decrypt
   
   # 使用默认密钥加密
   encrypted = encrypt("Hello World")
   decrypted = decrypt(encrypted)
   print(decrypted)  # 输出: Hello World

自定义密码加密
~~~~~~~~~~~~~~

.. code-block:: python

   from pywayne.crypto import encrypt, decrypt
   
   # 使用自定义密码
   password = "my_secret_password"
   encrypted = encrypt("机密信息", password)
   decrypted = decrypt(encrypted, password)
   print(decrypted)  # 输出: 机密信息

处理不同数据类型
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pywayne.crypto import encrypt, decrypt
   
   # 字符串加密
   text_encrypted = encrypt("文本信息", "password123")
   
   # 字节数据加密
   byte_data = b"binary data"
   byte_encrypted = encrypt(byte_data, "password123")
   
   # 解密
   text_decrypted = decrypt(text_encrypted, "password123")
   byte_decrypted = decrypt(byte_encrypted, "password123")

API 参考
----------

encrypt 函数
~~~~~~~~~~~~~

.. py:function:: encrypt(text: Union[str, bytes], password: str = None) -> str

   加密字符串或字节数据。

   **参数**:
   
   - **text** (*str* or *bytes*): 要加密的文本数据
   - **password** (*str*, 可选): 自定义密码。如果不提供，则使用默认密钥

   **返回值**:
   
   - **str**: 加密后的 base64 编码字符串

   **异常**:
   
   - **ValueError**: 当输入参数类型无效时抛出

   **示例**::

      # 基本用法
      encrypted = encrypt("Hello World")
      
      # 使用密码
      encrypted = encrypt("Secret message", "my_password")
      
      # 处理字节数据
      encrypted = encrypt(b"binary data", "password")

decrypt 函数
~~~~~~~~~~~~~

.. py:function:: decrypt(encrypted_text: str, password: str = None) -> str

   解密字符串。

   **参数**:
   
   - **encrypted_text** (*str*): 要解密的 base64 编码字符串
   - **password** (*str*, 可选): 解密密码。必须与加密时使用的密码相同

   **返回值**:
   
   - **str**: 解密后的原始字符串

   **异常**:
   
   - **ValueError**: 当解密失败时抛出（数据损坏或密码错误）

   **示例**::

      # 基本用法
      original = decrypt(encrypted_text)
      
      # 使用密码解密
      original = decrypt(encrypted_text, "my_password")

错误处理
----------

常见错误类型
~~~~~~~~~~~~

.. code-block:: python

   from pywayne.crypto import encrypt, decrypt
   
   try:
       # 错误的数据类型
       encrypt(123)
   except ValueError as e:
       print("错误: 输入类型必须是字符串或字节")
   
   try:
       # 错误的密码
       decrypt(encrypted_text, "wrong_password")
   except ValueError as e:
       print("错误: 解密失败，密码不正确")
   
   try:
       # 损坏的数据
       decrypt("invalid_base64_data")
   except ValueError as e:
       print("错误: 数据格式无效")

最佳实践
----------

密码安全
~~~~~~~~

.. code-block:: python

   import os
   from pywayne.crypto import encrypt, decrypt
   
   # 使用环境变量存储密码
   password = os.getenv('ENCRYPTION_PASSWORD', 'default_password')
   
   # 加密敏感数据
   sensitive_data = "用户的隐私信息"
   encrypted = encrypt(sensitive_data, password)
   
   # 安全解密
   try:
       decrypted = decrypt(encrypted, password)
       print(f"解密成功: {decrypted}")
   except ValueError:
       print("解密失败: 密码错误或数据损坏")

批量处理
~~~~~~~~

.. code-block:: python

   from pywayne.crypto import encrypt, decrypt
   
   def encrypt_batch(data_list, password):
       """批量加密数据"""
       return [encrypt(item, password) for item in data_list]
   
   def decrypt_batch(encrypted_list, password):
       """批量解密数据"""
       results = []
       for item in encrypted_list:
           try:
               results.append(decrypt(item, password))
           except ValueError:
               results.append(None)  # 解密失败的项目
       return results
   
   # 使用示例
   data = ["数据1", "数据2", "数据3"]
   password = "batch_password"
   
   encrypted = encrypt_batch(data, password)
   decrypted = decrypt_batch(encrypted, password)

技术细节
----------

加密算法
~~~~~~~~

本模块采用分层加密策略：

1. **Fernet 加密** (推荐)
   
   - 基于 AES 128 对称加密
   - 包含消息完整性验证
   - 使用 SHA256 进行密钥派生
   - 需要 ``cryptography`` 库支持

2. **XOR 加密** (回退方案)
   
   - 简单的异或加密算法
   - 无需额外依赖
   - 适用于基本的数据混淆

代码混淆
~~~~~~~~

为了保护加密算法的实现细节，本模块采用了多种混淆技术：

- **Base64 编码隐藏**: 核心算法代码通过 base64 编码存储
- **动态函数生成**: 运行时动态创建函数，避免静态分析
- **命名空间清理**: 执行后立即删除敏感变量
- **分散式架构**: 将核心逻辑分散到多个独立的编码块

注意事项
----------

安全提醒
~~~~~~~~

- 本模块主要用于数据混淆和基本加密保护
- 对于高安全性要求的应用，建议使用专业的加密库
- 密码应当妥善保管，避免硬编码在源码中
- 定期更新 ``cryptography`` 库以获得最新的安全特性

性能考虑
~~~~~~~~

- Fernet 加密的性能优于 XOR 加密的安全性
- 对于大量数据的加密，建议分块处理
- 密钥派生过程会消耗一定的计算资源

兼容性
~~~~~~

- 支持 Python 3.6+
- 跨平台兼容（Windows、macOS、Linux）
- 与其他 pywayne 模块完全兼容

示例应用
----------

配置文件加密
~~~~~~~~~~~~

.. code-block:: python

   import json
   from pywayne.crypto import encrypt, decrypt
   
   def save_config(config_dict, password, filename):
       """保存加密的配置文件"""
       config_json = json.dumps(config_dict, ensure_ascii=False)
       encrypted_config = encrypt(config_json, password)
       
       with open(filename, 'w') as f:
           f.write(encrypted_config)
   
   def load_config(password, filename):
       """加载并解密配置文件"""
       with open(filename, 'r') as f:
           encrypted_config = f.read()
       
       try:
           config_json = decrypt(encrypted_config, password)
           return json.loads(config_json)
       except ValueError:
           raise ValueError("配置文件解密失败，请检查密码")
   
   # 使用示例
   config = {
       "database": {
           "host": "localhost",
           "username": "admin",
           "password": "secret123"
       }
   }
   
   save_config(config, "master_password", "config.enc")
   loaded_config = load_config("master_password", "config.enc")

日志加密
~~~~~~~~

.. code-block:: python

   import logging
   from pywayne.crypto import encrypt
   
   class EncryptedFileHandler(logging.FileHandler):
       """加密的日志文件处理器"""
       
       def __init__(self, filename, password, mode='a', encoding=None):
           self.password = password
           super().__init__(filename, mode, encoding)
       
       def emit(self, record):
           """加密日志记录后写入文件"""
           try:
               msg = self.format(record)
               encrypted_msg = encrypt(msg, self.password)
               
               # 写入加密的日志
               with open(self.baseFilename, 'a', encoding='utf-8') as f:
                   f.write(encrypted_msg + '\n')
           except Exception:
               self.handleError(record)
   
   # 使用示例
   logger = logging.getLogger('encrypted_logger')
   handler = EncryptedFileHandler('app.log.enc', 'log_password')
   formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
   handler.setFormatter(formatter)
   logger.addHandler(handler)
   logger.setLevel(logging.INFO)
   
   logger.info("这条日志将被加密存储")
