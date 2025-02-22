# pywayne: 强大的 Python 工具库 ✨

pywayne 是一款集成多种实用功能的工具库，旨在为 Python 与 C++ 开发提供全面支持。无论你在处理信号、数据结构、数学运算、图形可视化，还是在与飞书机器人、阿里云 OSS 或文本转语音(TTS)等领域工作，pywayne 都能提供一站式解决方案。

## 目录
- [快速开始](#快速-开始)
- [核心功能](#核心功能)
- [模块详情](#模块详情)
- [安装说明](#安装说明)
- [文档](#文档)
- [联系方式](#联系方式)
- [贡献说明](#贡献说明)

## 快速 开始 🚀
1. 安装 pywayne：
   ```bash
   pip install -U pywayne
   ```
2. 快速示例：
   ```python
   from pywayne.tools import list_all_files
   files = list_all_files(".")
   print(files)
   ```

## 核心功能 ✨
- **工具函数**：提供文件处理、日志记录、计时器、单例模式等常用工具。
- **信号处理（dsp）**：内置 Butterworth 滤波器、局部极值检测、DTW 等多种数字信号处理算法。
- **图形用户界面（gui）**：支持热键注册、窗口操作、鼠标键盘自动化操作，助你轻松实现 GUI 交互。
- **数学工具（maths）**：实现因数分解、快速乘法（Karatsuba）及其他数学实用工具。
- **数据结构**：包括条件树和并查集实现，优化数据存储和查询。
- **绘图工具（plot）**：支持频谱图、定制 Colormap 绘图和其他数据可视化功能。
- **姿态与校准（ahrs, calibration）**：实现 SE3 转换、四元数处理，以及时空数据校准。
- **飞书机器人及监听**：整合 lark_custom_bot、lark_bot 和 lark_bot_listener 模块，实现文本、图片、文件等多种消息交互。
- **文本转语音（tts）**：生成 opus 或 MP3 格式语音文件，为项目增添语音播报功能。
- **云存储支持（aliyun_oss）**：与阿里云 OSS 集成，提供文件上传、下载和批量操作。
- **辅助模块（helper）**：包含配置管理与常用辅助函数，助力项目开发。

## 模块详情 📚

| 模块名称             | 主要功能描述                                                                   | Emoji  |
| -------------------- | ------------------------------------------------------------------------------ | ------ |
| **tools**            | 常用工具函数：文件操作、日志记录、计时器、单例模式等                            | 🛠️     |
| **dsp**              | 信号处理工具：滤波器、局部极值检测、动态时间规整 (DTW) 等                        | 🔊     |
| **gui**              | 图形用户界面自动化：热键绑定、窗口操作、鼠标键盘控制                               | 🖥️     |
| **maths**            | 数学工具：因数分解、卡拉楚巴乘法、快速计算等                                      | ➕➖    |
| **data_structure**   | 数据结构实现：条件树、并查集等                                                   | 🌲     |
| **plot**             | 绘图工具：频谱图、定制 Colormap 绘制、数据可视化                                  | 📊     |
| **ahrs**             | 姿态估计：SE3 与 pose 转换、姿态可视化                                           | 🧭     |
| **calibration**      | 校准工具：时空数据校准与处理                                                    | 🔧     |
| **lark_custom_bot**  | 飞书自定义机器人：支持多种消息类型（文本、图片、文件）                             | 🤖     |
| **lark_bot**         | 飞书机器人交互：文本、图片、文件消息发送                                         | 💬     |
| **lark_bot_listener**| 飞书消息监听：实时监听文本、图片、文件消息并处理                                   | 👂     |
| **tts**              | 文本转语音工具：生成 Opus 或 MP3 格式音频                                       | 🔈     |
| **aliyun_oss**       | 阿里云 OSS 文件管理：文件上传、下载、删除及目录操作                                | ☁️     |
| **helper**           | 辅助模块：配置管理、常用辅助函数                                                 | 🧰     |

## 安装说明 🔧

安装方法非常简单：

```bash
pip install -U pywayne
```

若需了解更多安装细节及依赖，请参阅 [requirements.txt](./requirements.txt) 和 [setup.py](./setup.py)。

## 文档 📖

详细文档请访问：

[pywayne 文档](https://wayne-algorithm-lib.readthedocs.io/)

## 联系方式 📬

如果在使用过程中遇到问题，欢迎通过以下方式联系：

- **邮箱**：
  - wang121ye@hotmail.com
  - y-w22@mails.tsinghua.edu.cn
- **个人网站**：
  - [http://wangye.xin](http://wangye.xin)
  - [http://cvllm.com](http://cvllm.com)
- **LeetCode**：[http://leetcode.com/wangyehope](http://leetcode.com/wangyehope)
- **GitHub**：[http://github.com/wangyendt](http://github.com/wangyendt)

## 贡献说明 🤝

欢迎使用 pywayne，并提出 Pull Request 和 issue！
无论你是对现有功能的改进建议，还是希望增加新的模块，我们都非常欢迎你的贡献。

---

我们希望 pywayne 能在你的项目中发挥重要作用，带来高效与便利！

Happy coding! 😄
