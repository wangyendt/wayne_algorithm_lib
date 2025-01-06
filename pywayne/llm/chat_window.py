from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QTextEdit,
                             QLineEdit, QPushButton)
from PyQt5.QtCore import Qt
from openai import OpenAI
import sys
from typing import Optional, List, Dict, Any


class ChatWindow(QWidget):
    def __init__(
            self,
            base_url: str,
            api_key: str,
            model: str = "deepseek-chat",
            temperature: float = 0.7,
            max_tokens: int = 2048,
            top_p: float = 1.0,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            system_prompt: str = "你是一个严谨的助手",
            window_title: str = "AI Chat",
            window_width: int = 600,
            window_height: int = 800,
            window_x: int = 300,
            window_y: int = 300
    ):
        # 确保QApplication实例存在
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        super().__init__()

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        # 保存配置
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        # 初始化消息历史
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        # 窗口配置
        self.window_title = window_title
        self.window_width = window_width
        self.window_height = window_height
        self.window_x = window_x
        self.window_y = window_y

        # 初始化UI
        self._init_ui()

    def _init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle(self.window_title)
        self.setGeometry(
            self.window_x,
            self.window_y,
            self.window_width,
            self.window_height
        )

        # 主布局
        layout = QVBoxLayout()

        # 聊天历史显示区域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        # 输入区域
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("在此输入消息...")
        self.input_field.returnPressed.connect(self._send_message)
        layout.addWidget(self.input_field)

        # 发送按钮
        send_btn = QPushButton('发送')
        send_btn.clicked.connect(self._send_message)
        layout.addWidget(send_btn)

        self.setLayout(layout)

    def _send_message(self):
        """处理消息发送"""
        user_input = self.input_field.text()
        if not user_input:
            return

        self.messages.append({"role": "user", "content": user_input})
        self.chat_display.append(f"你: {user_input}\n")
        self.input_field.clear()

        # 创建流式响应
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )

        self.chat_display.append("AI: ")
        assistant_response = []

        # 处理流式响应
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                self.chat_display.insertPlainText(content)
                assistant_response.append(content)
                QApplication.processEvents()

        self.chat_display.append("\n")
        self.messages.append(
            {"role": "assistant", "content": "".join(assistant_response)}
        )

    def run(self):
        """运行聊天机器人"""
        self.show()
        # 如果是新创建的QApplication实例，则需要启动事件循环
        if self.app.instance() is self.app:  # 检查是否是我们创建的实例
            self.app.exec_()

    @classmethod
    def launch(cls, base_url: str, api_key: str, system_messages: List[Dict[str, str]] = None, **kwargs):
        """
        快速启动一个聊天窗口
        
        Args:
            base_url: API基础URL
            api_key: API密钥
            system_messages: 系统消息列表，每个消息是一个包含role和content的字典
                           例如：[{"role": "system", "content": "你是一个Python专家"}]
            **kwargs: 其他传递给ChatWindow的参数
        """
        app = QApplication.instance() or QApplication(sys.argv)
        chat = cls(base_url=base_url, api_key=api_key, **kwargs)
        if system_messages:
            chat.set_system_messages(system_messages)
        chat.show()
        app.exec_()

    def set_system_messages(self, messages: List[Dict[str, str]]):
        """
        设置系统消息
        
        Args:
            messages: 系统消息列表，每个消息是一个包含role和content的字典
        """
        self.messages = messages.copy()
        # 清空显示
        self.chat_display.clear()

    def add_system_message(self, content: str):
        """
        添加单条系统消息
        
        Args:
            content: 系统消息内容
        """
        self.messages.append({"role": "system", "content": content})


if __name__ == '__main__':
    # 基础用法
    ChatWindow.launch(
        base_url="https://api.deepseek.com/v1",
        api_key="your-api-key"
    )

    # 高级用法 - 设置系统消息
    ChatWindow.launch(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-4556e299ea3b4401b87adcbda3cebb19",
        system_messages=[
            {"role": "system", "content": "你是一个Python专家"},
            {"role": "system", "content": "你的回答要简洁且包含代码示例"}
        ],
        window_title="Python助手",
        temperature=0.8
    )
