"""
LLM Client - 智谱 API 封装
包含流式输出和错误处理逻辑
"""
from typing import AsyncGenerator, Dict, Any, Optional, List
import httpx
from ..config import settings


class ZhipuClient:
    """智谱 AI API 客户端"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None
    ):
        self.api_key = api_key or settings.ZHIPU_API_KEY
        self.model = model or settings.ZHIPU_MODEL
        self.base_url = base_url or settings.ZHIPU_BASE_URL or "https://open.bigmodel.cn/api/paas/v4"
        self.timeout = 60.0
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        """聊天补全接口（非流式）"""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                headers=self._get_headers(),
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:
        """流式聊天接口"""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                url,
                headers=self._get_headers(),
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        
                        import json
                        try:
                            parsed = json.loads(data)
                            delta = parsed["choices"][0]["delta"]
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError):
                            continue


async def call_llm(
    prompt: str,
    system_message: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    use_stream: bool = False
) -> str:
    """调用 LLM 的便捷函数"""
    client = ZhipuClient()
    
    messages = []
    
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": prompt})
    
    try:
        if use_stream:
            full_response = ""
            async for chunk in client.chat_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                full_response += chunk
            return full_response
        else:
            response = await client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response["choices"][0]["message"]["content"]
    
    except httpx.HTTPStatusError as e:
        error_msg = f"API 请求失败：{e.response.status_code}"
        print(error_msg)
        return f"[错误] {error_msg}"
    
    except httpx.RequestError as e:
        error_msg = f"网络连接失败：{str(e)}"
        print(error_msg)
        return f"[错误] {error_msg}"
    
    except Exception as e:
        error_msg = f"未知错误：{str(e)}"
        print(error_msg)
        return f"[错误] {error_msg}"


zhipu_client = ZhipuClient()
