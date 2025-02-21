from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QHBoxLayout)
from PyQt5.QtCore import Qt
from openai import OpenAI
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class ChatConfig:
    base_url: str
    api_key: str
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    system_prompt: str = "你是一个严谨的助手"
    window_title: str = "AI Chat"
    window_width: int = 600
    window_height: int = 800
    window_x: int = 300
    window_y: int = 300


class ChatWindow(QWidget):
    def __init__(self, config: ChatConfig):
        # 确保QApplication实例存在
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        super().__init__()

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

        # 保存配置
        self.config = config

        # 初始化消息历史
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": config.system_prompt}
        ]

        # 添加停止标志
        self.is_generating = False

        # 初始化UI
        self._init_ui()

    def _init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle(self.config.window_title)
        self.setGeometry(
            self.config.window_x,
            self.config.window_y,
            self.config.window_width,
            self.config.window_height
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
        self.input_field.returnPressed.connect(self._handle_button_click)
        layout.addWidget(self.input_field)

        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 发送/停止按钮
        self.action_btn = QPushButton('发送')
        self.action_btn.clicked.connect(self._handle_button_click)
        button_layout.addWidget(self.action_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _handle_button_click(self):
        """处理按钮点击事件"""
        if self.is_generating:
            self._stop_generation()
        else:
            self._send_message()

    def _stop_generation(self):
        """停止生成回答"""
        self.is_generating = False
        self.action_btn.setText('发送')
        self.input_field.setEnabled(True)

    def _send_message(self):
        """处理消息发送"""
        user_input = self.input_field.text()
        if not user_input:
            return

        # 设置生成状态
        self.is_generating = True
        self.action_btn.setText('停止')
        self.input_field.setEnabled(False)

        self.messages.append({"role": "user", "content": user_input})
        self.chat_display.append(f"你: {user_input}\n")
        self.input_field.clear()

        # 创建流式响应
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=self.messages,
            stream=True,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty
        )

        self.chat_display.append("AI: ")
        assistant_response = []

        # 处理流式响应
        for chunk in response:
            if not self.is_generating:
                break
            
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                cursor = self.chat_display.textCursor()
                cursor.movePosition(cursor.End)
                self.chat_display.setTextCursor(cursor)
                self.chat_display.insertPlainText(content)
                assistant_response.append(content)
                QApplication.processEvents()

        self.chat_display.append("\n")
        self.messages.append(
            {"role": "assistant", "content": "".join(assistant_response)}
        )

        # 恢复界面状态
        self.is_generating = False
        self.action_btn.setText('发送')
        self.input_field.setEnabled(True)

    def run(self):
        """运行聊天机器人"""
        self.show()
        # 如果是新创建的QApplication实例，则需要启动事件循环
        if self.app.instance() is self.app:  # 检查是否是我们创建的实例
            self.app.exec_()

    @classmethod
    def launch(cls, base_url: str, api_key: str, model: str = "deepseek-chat", system_messages: List[Dict[str, str]] = None, **kwargs):
        """
        快速启动一个聊天窗口

        Args:
            base_url: API基础URL
            api_key: API密钥
            model: 要使用的模型名称，默认为"deepseek-chat"
            system_messages: 系统消息列表，每个消息是一个包含role和content的字典
                           例如：[{"role": "system", "content": "你是一个Python专家"}]
            **kwargs: 其他传递给ChatWindow的参数
        """
        config = ChatConfig(base_url=base_url, api_key=api_key, model=model, **kwargs)
        app = QApplication.instance() or QApplication(sys.argv)
        chat = cls(config=config)
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
        api_key="xxx",
        model="deepseek-chat"  # 可以指定模型
    )

    # 高级用法 - 设置系统消息
    config = ChatConfig(
        base_url="https://api.deepseek.com/v1",
        api_key="xxx",
        model="deepseek-coder",  # 使用不同的模型
        temperature=0.8,
        window_title="Python助手"
    )
    chat = ChatWindow(config)
    chat.set_system_messages([
        {"role": "system", "content": "你是一个Python专家"},
        {"role": "system", "content": "你的回答要简洁且包含代码示例"}
    ])
    chat.run()
