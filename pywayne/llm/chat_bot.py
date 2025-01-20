from typing import Optional, List, Dict, Any, Generator, Union
from openai import OpenAI


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


if __name__ == '__main__':
    # 创建对话实例
    llm = LLMChat(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-4556e299ea3b4401b87adcbda3cebb19",  # 替换为你的API密钥
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