from typing import Optional, List, Dict, Any, Generator, Union
from openai import OpenAI
import time


class LLMChat:
    """
    一个简单的LLM对话类，支持单次对话和历史对话，以及流式输出。
    
    注意：这是同步版本的实现。如果需要异步版本，建议：
    1. 使用 aiohttp 替代普通的 http 请求
    2. 将方法改为 async def
    3. 使用 async for 处理流式响应
    4. 考虑使用 asyncio.Queue 处理并发请求
    """

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
            system_prompt: str = "你是一个严谨的助手"
    ):
        """
        初始化LLM对话实例

        Args:
            base_url: API基础URL
            api_key: API密钥
            model: 使用的模型名称
            temperature: 温度参数，控制输出的随机性
            max_tokens: 最大输出token数
            top_p: 控制输出多样性
            frequency_penalty: 频率惩罚参数
            presence_penalty: 存在惩罚参数
            system_prompt: 系统提示语
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        # 初始化消息历史
        self._history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    def ask(
            self,
            prompt: str,
            stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        单次对话，不考虑历史记录

        Args:
            prompt: 用户输入的提示语
            stream: 是否使用流式输出

        Returns:
            如果stream=False，返回完整的回复字符串
            如果stream=True，返回一个生成器，逐字输出回复
        """
        messages = [
            {"role": "system", "content": self._history[0]["content"]},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )

        if not stream:
            return response.choices[0].message.content

        def response_generator():
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        return response_generator()

    def chat(
            self,
            prompt: str,
            stream: bool = True
    ) -> Union[str, Generator[str, None, None]]:
        """
        带历史记录的对话

        Args:
            prompt: 用户输入的提示语
            stream: 是否使用流式输出，默认为True

        Returns:
            如果stream=False，返回完整的回复字符串
            如果stream=True，返回一个生成器，逐字输出回复
        """
        # 添加用户输入到历史记录
        self._history.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._history,
            stream=stream,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )

        if not stream:
            reply = response.choices[0].message.content
            # 添加助手回复到历史记录
            self._history.append({"role": "assistant", "content": reply})
            return reply

        def response_generator():
            assistant_response = []
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    assistant_response.append(content)
                    yield content
            # 添加完整的助手回复到历史记录
            self._history.append(
                {"role": "assistant", "content": "".join(assistant_response)}
            )

        return response_generator()

    def update_system_prompt(self, system_prompt: str) -> None:
        """
        更新系统提示语

        Args:
            system_prompt: 新的系统提示语
        """
        # 更新第一条系统消息
        self._history[0]["content"] = system_prompt
        # 如果历史记录中有其他系统消息，也一并更新
        for msg in self._history:
            if msg["role"] == "system":
                msg["content"] = system_prompt

    def clear_history(self) -> None:
        """
        清除对话历史，只保留系统提示语
        """
        system_prompt = self._history[0]["content"]
        self._history = [{"role": "system", "content": system_prompt}]

    @property
    def history(self) -> List[Dict[str, str]]:
        """
        获取当前对话历史

        Returns:
            List[Dict[str, str]]: 对话历史列表
        """
        return self._history.copy()


class LLMConfig:
    """LLM配置类，用于管理LLM的所有可配置参数"""
    def __init__(
            self,
            base_url: str,
            api_key: str,
            model: str = "deepseek-chat",
            temperature: float = 0.7,
            max_tokens: int = 8192,
            top_p: float = 1.0,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            system_prompt: str = "你是一个严谨的助手"
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.system_prompt = system_prompt

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典形式"""
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "system_prompt": self.system_prompt
        }


class ChatManager:
    """聊天管理器，用于管理多个聊天实例"""
    def __init__(
            self,
            base_url: str,
            api_key: str,
            timeout: float = float('inf'),
            model: str = "deepseek-chat",
            temperature: float = 0.7,
            max_tokens: int = 8192,
            top_p: float = 1.0,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            system_prompt: str = "你是一个严谨的助手"
    ):
        """
        初始化聊天管理器
        
        Args:
            base_url: API基础URL
            api_key: API密钥
            timeout: 会话超时时间（秒），默认为无穷大
            model: 使用的模型名称
            temperature: 温度参数，控制输出的随机性
            max_tokens: 最大输出token数
            top_p: 控制输出多样性
            frequency_penalty: 频率惩罚参数
            presence_penalty: 存在惩罚参数
            system_prompt: 系统提示语
        """
        self.default_config = LLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            system_prompt=system_prompt
        )
        self.timeout = timeout
        self.chats: Dict[str, LLMChat] = {}
        self.last_used: Dict[str, float] = {}  # 记录最后使用时间

    def get_chat(
            self,
            chat_id: str,
            stream: bool = True,
            config: Optional[LLMConfig] = None
    ) -> LLMChat:
        """
        获取或创建聊天实例
        
        Args:
            chat_id: 聊天ID（可以是群组ID或用户ID）
            stream: 是否使用流式输出
            config: LLM配置，如果为None则使用默认配置
            
        Returns:
            LLMChat: 聊天实例
        """
        current_time = time.time()
        # 清理超时的会话
        self._cleanup_expired(current_time)
        
        # 获取或创建会话
        if chat_id not in self.chats:
            print(f"为会话 {chat_id} 创建新的chat_bot")
            cfg = config or self.default_config
            self.chats[chat_id] = LLMChat(**cfg.to_dict())
            
        # 更新最后使用时间
        self.last_used[chat_id] = current_time
        return self.chats[chat_id]

    def remove_chat(self, chat_id: str) -> None:
        """
        移除聊天实例
        
        Args:
            chat_id: 聊天ID
        """
        if chat_id in self.chats:
            del self.chats[chat_id]
            if chat_id in self.last_used:
                del self.last_used[chat_id]

    def _cleanup_expired(self, current_time: float) -> None:
        """
        清理超时的会话
        
        Args:
            current_time: 当前时间戳
        """
        if self.timeout == float('inf'):
            return
            
        expired = [
            chat_id for chat_id, last_used in self.last_used.items()
            if current_time - last_used > self.timeout
        ]
        for chat_id in expired:
            print(f"清理超时会话: {chat_id}")
            self.remove_chat(chat_id)


if __name__ == '__main__':
    # API配置
    API_BASE = "http://xxx:11434/v1"
    API_KEY = "ollama"
    MODEL = "qwen2.5:7b" # test

    def test_llm_chat():
        """测试LLMChat的基本功能"""
        print("\n=== 测试LLMChat基本功能 ===")
        
        # 创建对话实例
        llm = LLMChat(
            base_url=API_BASE,
            api_key=API_KEY,
            model=MODEL,
            temperature=0.7
        )

        try:
            # 测试单次对话（非流式）
            print("测试单次对话（非流式）:")
            response = llm.ask("写一个Python函数，计算斐波那契数列的第n项。", stream=False)
            print(response)
            print("\n" + "="*50 + "\n")

            # 测试单次对话（流式）
            print("测试单次对话（流式）:")
            for token in llm.ask("解释一下什么是递归？", stream=True):
                print(token, end='', flush=True)
            print("\n" + "="*50 + "\n")

            # 测试带历史的对话（流式）
            print("测试带历史的对话（流式）:")
            # 第一轮
            print("User: Python中如何创建一个类？")
            print("Assistant: ", end='', flush=True)
            for token in llm.chat("Python中如何创建一个类？"):
                print(token, end='', flush=True)
            print("\n")
            
            # 第二轮
            print("User: 如何在类中定义构造函数？")
            print("Assistant: ", end='', flush=True)
            for token in llm.chat("如何在类中定义构造函数？"):
                print(token, end='', flush=True)
            print("\n")

            # 显示当前历史
            print("当前对话历史:")
            for msg in llm.history:
                if msg["role"] != "system":
                    print(f"{msg['role'].capitalize()}: {msg['content']}")
            print("\n" + "="*50 + "\n")

            # 测试更改系统提示语
            print("测试更改系统提示语:")
            llm.update_system_prompt("你现在是一个Python专家，回答要简洁且包含代码示例")
            print("User: 如何使用装饰器？")
            print("Assistant: ", end='', flush=True)
            for token in llm.chat("如何使用装饰器？"):
                print(token, end='', flush=True)
            print("\n" + "="*50 + "\n")

            # 测试清除历史
            print("测试清除历史:")
            llm.clear_history()
            print("User: 之前我们讨论了什么？")
            print("Assistant: ", end='', flush=True)
            for token in llm.chat("之前我们讨论了什么？"):
                print(token, end='', flush=True)
            print("\n")

        except Exception as e:
            print(f"错误: {str(e)}")

    def test_chat_manager():
        """测试ChatManager的功能"""
        print("\n=== 测试ChatManager功能 ===")
        
        try:
            # 创建ChatManager实例，设置较长的超时时间
            manager = ChatManager(
                base_url=API_BASE,
                api_key=API_KEY,
                model=MODEL,
                timeout=10  # 增加超时时间，确保测试3不会提前超时
            )
            
            # 测试1：创建和获取聊天实例
            print("\n1. 测试创建和获取聊天实例:")
            chat1 = manager.get_chat("user1")
            response = chat1.ask("你好，请做个自我介绍", stream=False)
            print(f"user1的回复: {response}")
            
            # 测试2：使用自定义配置创建聊天实例
            print("\n2. 测试使用自定义配置:")
            custom_config = LLMConfig(
                base_url=API_BASE,
                api_key=API_KEY,
                model=MODEL,
                temperature=0.9,
                system_prompt="你是一个Python专家"
            )
            chat2 = manager.get_chat("user2", config=custom_config)
            response = chat2.ask("解释一下Python的装饰器", stream=False)
            print(f"user2的回复: {response}")
            
            # 测试3：重复获取同一个聊天实例
            print("\n3. 测试重复获取同一个聊天实例:")
            chat1_again = manager.get_chat("user1")
            print(f"chat1 和 chat1_again 是同一个实例: {chat1 is chat1_again}")
            
            # 测试4：手动移除聊天实例
            print("\n4. 测试手动移除聊天实例:")
            manager.remove_chat("user2")
            chat2_new = manager.get_chat("user2")  # 应该创建新的实例
            print("已移除并重新创建user2的聊天实例")
            
            # 创建一个新的manager用于测试超时
            print("\n5. 测试超时清理机制:")
            timeout_manager = ChatManager(
                base_url=API_BASE,
                api_key=API_KEY,
                model=MODEL,
                timeout=5  # 使用较短的超时时间
            )
            
            # 创建一个聊天实例并等待超时
            timeout_chat = timeout_manager.get_chat("timeout_user")
            print("等待6秒以触发超时...")
            time.sleep(6)
            timeout_chat_new = timeout_manager.get_chat("timeout_user")
            print(f"超时后，timeout_chat和timeout_chat_new是同一个实例: {timeout_chat is timeout_chat_new}")
            
            # 测试6：流式输出
            print("\n6. 测试流式输出:")
            chat = manager.get_chat("user3", stream=True)
            print("流式回复: ", end='', flush=True)
            for token in chat.chat("写一个简单的Python函数"):
                print(token, end='', flush=True)
            print("\n")

        except Exception as e:
            print(f"ChatManager测试过程中出现错误: {str(e)}")

    # 运行所有测试
    test_llm_chat()
    test_chat_manager()