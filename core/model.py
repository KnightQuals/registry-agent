"""
core/model.py — 模型抽象层（Execution 层的模型接入部分）

设计要点（对应用户需求）：
- 默认接实验室服务器本地 LLM（vLLM 部署的 Qwen3，速度快、可定制）。
- 同时开放覆盖接口：用户可通过环境变量或参数自定义 base_url / api_key / model，
  无缝切换到 Claude / GPT / 中转站等任意 OpenAI 兼容服务。
- 统一走 OpenAI Chat Completions 协议：vLLM 原生兼容它，且支持原生 tool_calls。
  （vLLM 起服务时加 --enable-auto-tool-choice --tool-call-parser hermes 即可返回标准 tool_calls）

本模块只负责"如何调模型"，不含业务逻辑。ReAct 循环在 core/loop.py。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

try:
    from openai import AsyncOpenAI
except ImportError:  # 允许在未装依赖时被 import（结构冒烟测试用）
    AsyncOpenAI = None  # type: ignore


# ---- 实验室本地默认（可被环境变量覆盖）----
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8502/v1")
DEFAULT_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")  # 本地 vLLM 通常不校验 key
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen")


@dataclass
class ModelConfig:
    """模型配置。三个字段都可被用户覆盖；不填则用实验室本地默认。"""
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """从环境变量构造。未设置的项回落到本地默认。"""
        return cls(
            base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.getenv("LLM_API_KEY", DEFAULT_API_KEY),
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )


class ModelClient:
    """
    OpenAI 兼容的异步模型客户端。

    用法：
        client = ModelClient(ModelConfig.from_env())
        resp = await client.chat(messages, tools=tools_schema)
        # resp 是原生 ChatCompletion，含 .choices[0].message.tool_calls
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig.from_env()
        if AsyncOpenAI is None:
            self._client = None
        else:
            self._client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )

    def describe(self) -> str:
        return f"model={self.config.model} @ {self.config.base_url}"

    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
    ) -> Any:
        """
        发起一次 chat completion。若传入 tools，则启用原生 tool_calls。
        返回原生 ChatCompletion 对象（调用方从 .choices[0].message 取 content / tool_calls）。
        """
        if self._client is None:
            raise RuntimeError("openai SDK 未安装，请先 pip install openai")

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        return await self._client.chat.completions.create(**kwargs)
