文本转语音 (TTS)
===================

本模块提供了文本转语音（Text-to-Speech, TTS）功能，支持将文本内容转换为 opus 或 MP3 格式的音频文件。模块集成了系统原生 TTS 功能（macOS 的 say 命令）以及 Google Text-to-Speech (gTTS) 服务，并使用 ffmpeg 进行音频格式转换。

主要功能
---------

模块提供了两个主要函数，分别用于生成 opus 和 MP3 格式的音频文件：

text_to_speech_output_opus
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: text_to_speech_output_opus(text: str, opus_filename: str, use_say: bool = True)

   将文本转换为 opus 格式的音频文件。

   **参数**:

   - text (str): 需要转换的文本内容。
   - opus_filename (str): 输出的 opus 文件名。
   - use_say (bool): 在 macOS 系统上是否使用系统的 say 命令，默认为 True。

   **功能说明**:

   - 在 macOS 上优先使用系统的 say 命令进行转换。
   - 在其他系统或指定不使用 say 命令时，使用 gTTS 服务。
   - 使用 ffmpeg 将音频转换为 opus 格式，采样率为 16kHz，单声道。

   **示例**::

      >>> from pywayne.tts import text_to_speech_output_opus
      >>> text_to_speech_output_opus("你好，世界", "hello.opus")


text_to_speech_output_mp3
~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: text_to_speech_output_mp3(text: str, mp3_filename: str, use_say: bool = True)

   将文本转换为 MP3 格式的音频文件。

   **参数**:

   - text (str): 需要转换的文本内容。
   - mp3_filename (str): 输出的 MP3 文件名。
   - use_say (bool): 在 macOS 系统上是否使用系统的 say 命令，默认为 True。

   **功能说明**:

   - 在 macOS 上优先使用系统的 say 命令进行转换。
   - 在其他系统或指定不使用 say 命令时，使用 gTTS 服务。
   - 使用 ffmpeg 将音频转换为 MP3 格式。

   **示例**::

      >>> from pywayne.tts import text_to_speech_output_mp3
      >>> text_to_speech_output_mp3("你好，世界", "hello.mp3")


依赖要求
---------

使用该模块需要满足以下依赖条件：

1. ffmpeg：

   - macOS 用户可通过 Homebrew 安装：``brew install ffmpeg``
   - Windows 用户需要从 https://ffmpeg.org 下载并配置环境变量
   - Linux 用户可使用包管理器安装：``sudo apt install ffmpeg``

2. Python 依赖：

   - gtts：用于调用 Google Text-to-Speech 服务
   - 其他基础 Python 库：tempfile, subprocess, platform, shutil 等

使用建议
---------

1. 音频格式选择：

   - opus 格式适合语音通话等场景，文件小且音质好
   - MP3 格式兼容性更好，适合多媒体播放场景

2. 性能考虑：

   - macOS 上使用系统 say 命令速度较快
   - 使用 gTTS 服务时需要网络连接
   - 建议对频繁使用的音频进行缓存

3. 错误处理：

   - 使用前请确保 ffmpeg 已正确安装
   - 注意处理网络问题导致的 gTTS 服务调用失败
   - 确保输出路径具有写入权限

示例代码
---------

以下是一个完整的使用示例：

.. code-block:: python

   from pywayne.tts import text_to_speech_output_opus, text_to_speech_output_mp3
   
   # 生成 opus 格式音频
   text_to_speech_output_opus(
       text="欢迎使用文本转语音服务",
       opus_filename="welcome.opus"
   )
   
   # 生成 MP3 格式音频
   text_to_speech_output_mp3(
       text="这是一条测试消息",
       mp3_filename="test.mp3",
       use_say=False  # 强制使用 gTTS 服务
   )

注意事项
---------

1. 系统要求：

   - 确保系统已安装 ffmpeg 并可在命令行中访问
   - macOS 用户如需使用 say 命令，请确保系统音频服务正常

2. 网络要求：

   - 使用 gTTS 服务时需要稳定的网络连接
   - 建议添加网络超时处理机制

3. 文件处理：

   - 注意检查输出文件路径的权限
   - 建议在处理完成后及时清理临时文件

模块扩展建议
-------------

未来可以考虑在以下方面扩展模块功能：

1. 支持更多 TTS 服务：

   - 集成其他开源或商业 TTS 引擎
   - 添加更多语音合成选项（如声音、语速等）

2. 增强音频处理能力：

   - 支持更多音频格式
   - 提供音频后处理功能（如音量调节、降噪等）

3. 优化性能：

   - 实现音频缓存机制
   - 支持批量转换和并行处理

常见问题 (FAQ)
--------------

1. ffmpeg 相关问题：

   Q: 提示找不到 ffmpeg 命令怎么办？
   
   A: 请根据您的操作系统安装 ffmpeg：
      - macOS：使用 ``brew install ffmpeg``
      - Linux：使用 ``sudo apt install ffmpeg`` 或对应的包管理器命令
      - Windows：从官网下载并添加到系统 PATH

   Q: ffmpeg 转换时报错怎么处理？
   
   A: 检查输出目录是否有写入权限，确保输入文件存在且格式正确。

2. 语音服务问题：

   Q: macOS 的 say 命令无法使用？
   
   A: 检查系统音频服务是否正常，或尝试将 use_say 参数设置为 False 以使用 gTTS。

   Q: gTTS 服务无法访问？
   
   A: 检查网络连接，必要时配置代理。如果问题持续，可以尝试使用本地 TTS 引擎。

3. 音频质量问题：

   Q: 生成的音频质量不理想？
   A: 可以尝试调整 ffmpeg 的转换参数，或选择更适合的音频格式。opus 格式通常在相同文件大小下有更好的音质。

4. 其他常见问题：

   Q: 如何处理长文本转换？
   A: 建议将长文本分段处理，避免单次转换时间过长。

   Q: 临时文件占用空间过大？
   A: 程序会自动清理临时文件，如果发现残留，可以手动清理系统临时目录。 