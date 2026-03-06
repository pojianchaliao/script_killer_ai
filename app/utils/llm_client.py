"""
LLM Client - 智谱 AI SDK 封装
包含流式输出、tool_calls 解析和自动降级兜底逻辑
"""
import os
import json
from typing import List, Dict, Any, Optional, Generator
from zhipuai import ZhipuAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class ZhipuClient:
    """
    智谱 AI 客户端类
    
    核心特性:
    1. 支持普通对话 (invoke 方法)
    2. 支持流式调用并自动降级兜底 (stream_with_tool_fallback 方法)
    3. 默认使用 glm-4-flash (低成本) 和 glm-4 (高智力)
    """
    
    def __init__(self, model: str = "glm-4-flash"):
        """
        初始化智谱 AI 客户端
        
        Args:
            model: 模型名称，默认使用 glm-4-flash (低成本)
                   可选：glm-4 (高智力), glm-4-flash, glm-4-air 等
        """
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY 环境变量未设置")
        
        self.client = ZhipuAI(api_key=api_key)
        self.model = model
        self.high_intelligence_model = "glm-4"
        self.low_cost_model = "glm-4-flash"
    
    def invoke(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        普通对话接口 (非流式)
        
        Args:
            messages: 消息列表，格式为 [{"role": "user/system/assistant", "content": "..."}]
            temperature: 温度参数，控制随机性 (0-1)
            max_tokens: 最大生成 token 数
            tools: 工具定义列表 (可选)
            tool_choice: 工具选择策略 ("auto", "none", "required" 或具体工具名)
        
        Returns:
            完整的响应字典，包含 message、tool_calls 等信息
        """
        try:
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            if tools:
                request_params["tools"] = tools
            
            if tool_choice:
                request_params["tool_choice"] = tool_choice
            
            response = self.client.chat.completions.create(**request_params)
            
            return {
                "success": True,
                "data": response.choices[0].message,
                "usage": response.usage,
                "model": self.model
            }
        
        except Exception as e:
            error_msg = f"智谱 AI 调用失败：{str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "data": None
            }
    
    def _parse_stream_tools(self, response_stream) -> tuple[Optional[List[Dict]], str]:
        """
        尝试从流式响应中解析 tool_calls
        
        Args:
            response_stream: 流式响应迭代器
        
        Returns:
            (tool_calls, full_content): 
                - tool_calls: 解析出的工具调用列表，如果失败则为 None
                - full_content: 完整的文本内容
        """
        tool_call_fragments = {}  # 用于累积流式片段 {index: {"id": ..., "function": {...}}}
        full_content = ""
        current_tool_index = None
        
        try:
            for chunk in response_stream:
                if not chunk.choices or not chunk.choices[0].delta:
                    continue
                
                delta = chunk.choices[0].delta
                
                # 累积文本内容
                if delta.content:
                    full_content += delta.content
                
                # 处理 tool_calls 流式片段
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        index = tool_call.index
                        
                        # 初始化该片段的工具调用
                        if index not in tool_call_fragments:
                            tool_call_fragments[index] = {
                                "id": None,
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }
                        
                        fragment = tool_call_fragments[index]
                        
                        # 累积工具 ID
                        if tool_call.id:
                            fragment["id"] = tool_call.id
                        
                        # 累积函数名
                        if tool_call.function and tool_call.function.name:
                            fragment["function"]["name"] += tool_call.function.name
                        
                        # 累积函数参数
                        if tool_call.function and tool_call.function.arguments:
                            fragment["function"]["arguments"] += tool_call.function.arguments
            
            # 验证解析结果是否完整
            tool_calls = list(tool_call_fragments.values())
            
            # 检查每个 tool_call 的完整性
            for tool_call in tool_calls:
                # 必需字段检查
                if not tool_call.get("id"):
                    print("⚠️  流式解析失败：tool_call 缺少 id 字段")
                    return None, full_content
                
                if not tool_call.get("function", {}).get("name"):
                    print("⚠️  流式解析失败：tool_call 缺少 function.name 字段")
                    return None, full_content
                
                # 尝试解析参数字符串为 JSON
                try:
                    args_str = tool_call["function"]["arguments"]
                    if args_str:  # 如果有参数
                        tool_call["function"]["arguments"] = json.loads(args_str)
                except json.JSONDecodeError as e:
                    print(f"⚠️  流式解析失败：tool_call 参数 JSON 解析错误 - {str(e)}")
                    return None, full_content
            
            if tool_calls:
                print(f"✅ 流式成功解析 {len(tool_calls)} 个 tool_calls")
                return tool_calls, full_content
            else:
                return None, full_content
        
        except Exception as e:
            print(f"⚠️  流式工具调用解析异常：{str(e)}")
            return None, full_content
    
    def stream_with_tool_fallback(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        核心亮点方法：流式优先，失败自动降级为非流式
        
        工作流程:
        1. 首先尝试流式调用，解析 tool_calls
        2. 如果检测到 JSON 解析失败或流式返回不完整，自动切换为非流式模式重试
        3. 确保 tool_call_id 和参数完整
        
        Args:
            messages: 消息列表
            tools: 工具定义列表 (可选)
            tool_choice: 工具选择策略 (可选)
            temperature: 温度参数
            max_tokens: 最大生成 token 数
        
        Returns:
            包含响应类型 (stream/fallback) 和数据的字典
        """
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if tools:
            request_params["tools"] = tools
        
        if tool_choice:
            request_params["tool_choice"] = tool_choice
        
        # ========== 第一步：尝试流式调用 ==========
        try:
            print(f"🔄 尝试流式调用 (模型：{self.model})...")
            
            response_stream = self.client.chat.completions.create(
                **request_params,
                stream=True
            )
            
            # 尝试从流式中解析 tool_calls
            tool_calls, content = self._parse_stream_tools(response_stream)
            
            # 如果成功解析出 tool_calls，返回流式结果
            if tool_calls:
                print("✅ 流式调用成功，返回 tool_calls")
                return {
                    "type": "stream",
                    "success": True,
                    "data": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls
                    },
                    "model": self.model
                }
            
            # 如果没有 tools 要求，直接返回文本内容
            if not tools:
                print("✅ 流式调用成功 (无工具)，返回文本内容")
                return {
                    "type": "stream",
                    "success": True,
                    "data": {
                        "role": "assistant",
                        "content": content
                    },
                    "model": self.model
                }
            
            # 如果需要 tools 但流式解析失败，触发降级
            print("⚠️  流式工具调用不完整，触发非流式兜底...")
        
        except Exception as e:
            print(f"❌ 流式调用失败：{str(e)}")
            print("🔄 切换到非流式模式重试...")
        
        # ========== 第二步：降级为非流式调用 ==========
        try:
            print(f"🔄 非流式兜底调用 (模型：{self.model})...")
            
            non_stream_resp = self.client.chat.completions.create(
                **request_params,
                stream=False
            )
            
            message = non_stream_resp.choices[0].message
            
            # 验证非流式响应的 tool_calls 完整性
            if tools and message.tool_calls:
                for tool_call in message.tool_calls:
                    try:
                        # 验证参数是否为有效 JSON
                        if tool_call.function.arguments:
                            json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        error_msg = f"非流式响应 tool_call 参数 JSON 无效：{str(e)}"
                        print(f"❌ {error_msg}")
                        return {
                            "type": "fallback",
                            "success": False,
                            "error": error_msg,
                            "data": None
                        }
                
                print(f"✅ 非流式兜底成功，返回 {len(message.tool_calls)} 个 tool_calls")
            
            return {
                "type": "fallback",
                "success": True,
                "data": message,
                "usage": non_stream_resp.usage,
                "model": self.model
            }
        
        except Exception as e:
            error_msg = f"非流式兜底调用失败：{str(e)}"
            print(f"❌ {error_msg}")
            return {
                "type": "fallback",
                "success": False,
                "error": error_msg,
                "data": None
            }
    
    def switch_model(self, model: str):
        """
        切换模型
        
        Args:
            model: 新的模型名称
        """
        self.model = model
        print(f"✅ 模型已切换为：{model}")
    
    def use_high_intelligence(self):
        """切换到高智力模型 (glm-4)"""
        self.switch_model(self.high_intelligence_model)
    
    def use_low_cost(self):
        """切换到低成本模型 (glm-4-flash)"""
        self.switch_model(self.low_cost_model)


# ========== 便捷函数 ==========

def call_llm(
    prompt: str,
    system_message: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    use_stream: bool = False,
    model: str = "glm-4-flash"
) -> str:
    """
    调用 LLM 的便捷函数
    
    Args:
        prompt: 用户输入
        system_message: 系统消息 (可选)
        history: 历史对话 (可选)
        temperature: 温度参数
        max_tokens: 最大 token 数
        use_stream: 是否使用流式
        model: 模型名称
    
    Returns:
        LLM 生成的文本响应
    """
    client = ZhipuClient(model=model)
    
    # 构建消息列表
    messages = []
    
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": prompt})
    
    try:
        if use_stream:
            # 流式调用 (不使用工具)
            result = client.stream_with_tool_fallback(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if result["success"]:
                return result["data"]["content"]
            else:
                return f"[错误] {result.get('error', '未知错误')}"
        
        else:
            # 非流式调用
            result = client.invoke(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if result["success"]:
                # 兼容两种情况：字典或 CompletionMessage 对象
                data = result["data"]
                if isinstance(data, dict):
                    return data.get("content", "")
                else:
                    # CompletionMessage 对象，使用 .content 属性
                    return getattr(data, 'content', '')
            else:
                return f"[错误] {result.get('error', '未知错误')}"
    
    except Exception as e:
        error_msg = f"LLM 调用失败：{str(e)}"
        print(f"❌ {error_msg}")
        return f"[错误] {error_msg}"


# ========== 全局单例 ==========

# 默认使用低成本模型
zhipu_client = ZhipuClient(model="glm-4-flash")

# 如果需要高智力模型，可以使用:
# zhipu_client_high = ZhipuClient(model="glm-4")


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例 1: 普通对话
    print("=" * 60)
    print("示例 1: 普通对话")
    print("=" * 60)
    
    client = ZhipuClient(model="glm-4-flash")
    
    messages = [
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ]
    
    result = client.invoke(messages)
    if result["success"]:
        print(f"回复：{result['data'].content}")
    else:
        print(f"错误：{result['error']}")
    
    # 示例 2: 流式对话 (无工具)
    print("\n" + "=" * 60)
    print("示例 2: 流式对话 (无工具)")
    print("=" * 60)
    
    result = client.stream_with_tool_fallback(messages)
    if result["success"]:
        print(f"回复类型：{result['type']}")
        print(f"回复：{result['data']['content']}")
    
    # 示例 3: 流式对话 + 工具调用 (带兜底)
    print("\n" + "=" * 60)
    print("示例 3: 流式对话 + 工具调用 (带兜底)")
    print("=" * 60)
    
    # 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    messages_with_tool = [
        {"role": "user", "content": "北京今天天气怎么样？"}
    ]
    
    result = client.stream_with_tool_fallback(
        messages=messages_with_tool,
        tools=tools,
        tool_choice="auto"
    )
    
    if result["success"]:
        print(f"响应类型：{result['type']}")
        if result["data"]['tool_calls']:
            print(f"检测到 {len(result['data']['tool_calls'])} 个工具调用:")
            for tool_call in result["data"]['tool_calls']:
                print(f"  - 工具名：{tool_call['function']['name']}")
                print(f"    参数：{tool_call['function']['arguments']}")
    else:
        print(f"错误：{result.get('error', '未知错误')}")
    
    # 示例 4: 模型切换
    print("\n" + "=" * 60)
    print("示例 4: 模型切换")
    print("=" * 60)
    
    client.use_high_intelligence()  # 切换到 glm-4
    result = client.invoke([{"role": "user", "content": "1+1 等于几？"}])
    if result["success"]:
        print(f"glm-4 回复：{result['data'].content}")

    client.use_low_cost()  # 切换回 glm-4-flash
