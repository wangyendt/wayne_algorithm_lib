# author: wangye(Wayne)
# license: Apache Licence
# file: chat_ollama_gradio.py
# time: 2025-02-11-02:46:22
# contact: wang121ye@hotmail.com
# site:  wangyendt@github.com
# software: PyCharm
# code is far away from bugs.


import gradio as gr
import subprocess
import time
from pywayne.llm.chat_bot import ChatManager, LLMConfig


class OllamaChatGradio:
    def __init__(self, base_url="http://localhost:11434/v1", server_name="0.0.0.0", 
                 server_port=7870, root_path="", api_key="ollama"):
        self.base_url = base_url
        self.server_name = server_name
        self.server_port = server_port
        self.root_path = root_path
        self.api_key = api_key
        
        # 初始化类变量
        self.chat_manager = None
        self.current_chat_id = None
        self.chat_history_dict = {}  # 用于存储所有会话的历史记录
        
        # 初始化聊天管理器
        self.init_chat_manager()

    def get_ollama_models(self):
        """获取所有可用的Ollama模型"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            # 解析输出并提取模型名称
            lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
            models = []
            # 排除包含 'embed' 的模型名称
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    if 'embed' not in model_name.lower():
                        models.append(model_name)
            return models if models else ["qwen2.5:0.5b"]  # 如果没有找到模型，返回默认模型
        except Exception as e:
            print(f"获取模型列表失败: {e}")
            return ["qwen2.5:0.5b"]  # 出错时返回默认模型

    def init_chat_manager(self):
        """初始化聊天管理器"""
        if self.chat_manager is None:
            self.chat_manager = ChatManager(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=float('inf')  # 不设置超时
            )

    def create_new_chat(self):
        """创建新的聊天会话"""
        self.current_chat_id = f"chat_{int(time.time())}"
        self.chat_history_dict[self.current_chat_id] = []
        # 返回所有会话ID列表
        chat_ids = list(self.chat_history_dict.keys())
        # 更新Radio组件的选项和当前值
        return gr.update(value=self.current_chat_id), [], gr.update(choices=chat_ids, value=self.current_chat_id)

    def switch_chat(self, selected_chat_id):
        """切换到选定的聊天会话"""
        if selected_chat_id:
            self.current_chat_id = selected_chat_id
            return self.chat_history_dict.get(self.current_chat_id, [])
        return []

    def format_history(self, history):
        """格式化历史记录用于显示"""
        formatted = []
        for user_msg, bot_msg in history:
            formatted.append([user_msg, bot_msg])
        return formatted

    def chat(self, message, history, model_name):
        """处理聊天消息"""
        if not message:
            return "", history, gr.update()

        if not self.current_chat_id:
            self.current_chat_id = f"chat_{int(time.time())}"
            self.chat_history_dict[self.current_chat_id] = []

        # 获取或创建聊天实例
        config = LLMConfig(
            base_url=self.base_url,
            api_key=self.api_key,
            model=model_name
        )
        chat_instance = self.chat_manager.get_chat(self.current_chat_id, config=config)

        # 添加用户消息到历史记录
        history.append([message, ""])
        self.chat_history_dict[self.current_chat_id] = history

        # 流式生成回复
        partial_message = ""
        chat_ids = list(self.chat_history_dict.keys())

        for token in chat_instance.chat(message):
            partial_message += token
            history[-1][1] = partial_message
            yield "", history, gr.update(choices=chat_ids, value=self.current_chat_id)

        self.chat_history_dict[self.current_chat_id] = history
        return "", history, gr.update(choices=chat_ids, value=self.current_chat_id)

    def create_demo(self):
        """创建Gradio演示界面"""
        # 获取可用模型列表
        available_models = self.get_ollama_models()

        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(height=600)
                    with gr.Row():
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="输入您的消息...",
                            container=False
                        )
                        submit_btn = gr.Button("发送")

                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        value=available_models[0] if available_models else "qwen2.5:0.5b",
                        label="选择模型"
                    )
                    chat_id_text = gr.Textbox(
                        label="当前会话ID",
                        interactive=False
                    )
                    new_chat_btn = gr.Button("新建会话")
                    chat_history_list = gr.Radio(
                        label="历史会话",
                        choices=[],
                        interactive=True,
                        value=None
                    )

            # 事件处理
            msg_submit_click = submit_btn.click(
                self.chat,
                inputs=[msg, chatbot, model_dropdown],
                outputs=[msg, chatbot, chat_history_list]
            )

            msg_submit_enter = msg.submit(
                self.chat,
                inputs=[msg, chatbot, model_dropdown],
                outputs=[msg, chatbot, chat_history_list]
            )

            new_chat_click = new_chat_btn.click(
                self.create_new_chat,
                outputs=[chat_id_text, chatbot, chat_history_list]
            )

            chat_history_select = chat_history_list.change(
                self.switch_chat,
                inputs=[chat_history_list],
                outputs=[chatbot]
            )

        return demo

    def launch(self):
        """启动Gradio应用"""
        demo = self.create_demo()
        demo.queue()
        demo.launch(
            server_name=self.server_name,
            server_port=self.server_port,
            share=False,
            root_path=self.root_path
        )


if __name__ == "__main__":
    app = OllamaChatGradio()
    app.launch()
