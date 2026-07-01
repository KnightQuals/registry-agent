"""
guardrails/policy.py — 护栏层（Governance）

对应本轮决策：接入公有 MCP 的安全底线。
- MCP 来的外部工具默认标记为"需留意"。
- 会产生真实副作用的操作（下单花钱、删除、发送等）默认需二次确认，防止 Agent 误触发。

check(name, arguments) -> (allowed: bool, reason: str)
  - allowed=True  ：放行。
  - allowed=False ：拦截，reason 说明原因（会作为 observation 回注给模型）。

确认机制通过 confirm_hook 注入：
  - 默认（无 hook）：命中危险模式即拦截，让模型转而询问用户，安全优先。
  - 交互场景：可注入一个 confirm_hook(name, args)->bool，由前端弹窗让用户确认。
"""

from __future__ import annotations

import re
from typing import Callable, Optional

# 危险操作关键词（工具名匹配）。命中则默认需要确认。
DANGEROUS_PATTERNS = [
    r"order", r"pay", r"purchase", r"checkout", r"buy",   # 下单/支付
    r"delete", r"remove", r"drop", r"destroy",            # 删除
    r"send", r"post", r"publish", r"transfer",            # 外发/转账
    r"update", r"write", r"create",                       # 写操作（较宽，可按需收紧）
]


class Guardrail:
    def __init__(self, confirm_hook: Optional[Callable[[str, dict], bool]] = None,
                 extra_patterns: Optional[list[str]] = None,
                 enabled: bool = True):
        self.confirm_hook = confirm_hook
        self.enabled = enabled
        pats = list(DANGEROUS_PATTERNS) + list(extra_patterns or [])
        self._regex = re.compile("|".join(pats), re.IGNORECASE)

    def is_dangerous(self, name: str) -> bool:
        return bool(self._regex.search(name))

    def check(self, name: str, arguments: dict) -> tuple[bool, str]:
        if not self.enabled:
            return True, ""
        if not self.is_dangerous(name):
            return True, ""
        # 命中危险模式
        if self.confirm_hook is not None:
            ok = self.confirm_hook(name, arguments)
            return (True, "") if ok else (False, "用户未确认该操作")
        # 无确认通道：安全优先，拦截并提示模型改为向用户询问
        return False, (
            f"工具 {name} 属于可能产生真实副作用的操作（如下单/支付/删除/外发），"
            f"需用户显式确认后才能执行。请先向用户说明将要执行的操作并征得同意。"
        )
