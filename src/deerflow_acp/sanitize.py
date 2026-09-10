"""秘密脱敏。

桥永远不知道 DeerFlow、LangGraph、provider SDK 或某个工具会把什么塞进异常
消息里——provider 常见做法就是把整个请求头（含 ``Authorization``）或连接串
回显进报错。因此凡是**要离开桥进程**的文本（JSON-RPC 响应体、stderr 日志、
客户端可见的事件内容），都必须先过这一层。

设计取舍：宁可多抹一点，也不让秘密漏出去；但不能把 host:port、错误类型、
重试次数这类真正有诊断价值的信息一起毁掉——那样错误就不可诊断了，
等于用一个问题换另一个问题。
"""

from __future__ import annotations

import re
from collections.abc import Callable

MASK = "[REDACTED]"

_Rule = tuple[re.Pattern[str], Callable[[re.Match[str]], str]]


def _mask_all(_: re.Match[str]) -> str:
    """整段命中都是秘密，全部抹掉。"""
    return MASK


def _keep_prefix(match: re.Match[str]) -> str:
    """保留键名 / 协议前缀，只抹值——让人仍能看出是哪一类凭据出了问题。"""
    return f"{match.group(1)}{MASK}"


def _keep_around(match: re.Match[str]) -> str:
    """保留前后定界符（如 URL userinfo 的 ``user:`` 与 ``@``），只抹中间的值。"""
    return f"{match.group(1)}{MASK}{match.group(3)}"


# 每条规则自带替换函数：新增规则不需要维护任何下标，
# 从结构上排除「加了一条正则，替换分支错位」这类静默失效。
_RULES: tuple[_Rule, ...] = (
    # 1) 有明确前缀的 provider key
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"), _mask_all),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{16,}"), _mask_all),
    (re.compile(r"\b(?:gsk|xoxb|xoxp|xoxa|xapp)-[A-Za-z0-9-]{16,}"), _mask_all),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), _mask_all),
    # 2) JWT：三段 base64url
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), _mask_all),
    # 3) Authorization 头：保留认证方案名，抹掉凭据本身
    (re.compile(r"(?i)\b((?:bearer|basic)\s+)[A-Za-z0-9+/=._~-]{12,}"), _keep_prefix),
    # 4) key=value / key: value 形态，键名含 key/token/secret/password/credential 等
    (
        re.compile(
            r"(?i)\b([A-Za-z0-9_.-]*(?:api[_-]?key|access[_-]?key|secret[_-]?key|auth[_-]?token"
            r"|api[_-]?token|access[_-]?token|refresh[_-]?token|token|secret|password|passwd|pwd"
            r"|credential|private[_-]?key)\s*[:=]\s*)"
            r"[\"']?[^\s\"',;)\]}]{6,}[\"']?"
        ),
        _keep_prefix,
    ),
    # 5) URL 里的 userinfo：scheme://user:password@host —— 保留 host 供诊断
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^\s:/@]+:)([^\s@/]+)(@)"), _keep_around),
    # 6) 裸的长十六进制串（≥32 位）——典型的 token / 签名
    (re.compile(r"\b[0-9a-fA-F]{32,}\b"), _mask_all),
    # 7) PEM 私钥块
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        _mask_all,
    ),
)


def redact_text(value: object) -> str:
    """抹掉文本中的疑似秘密，返回仍可诊断的脱敏结果。

    非字符串输入会被强制转成字符串（``None`` → 空串），因为调用点常常拿到的
    是上游给的任意对象，不能因为类型意外就跳过脱敏。
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)

    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)

    return text


def describe_exception(exc: BaseException | None) -> str:
    """把异常压缩成**只有类型名**的分类标签。

    这是对外错误的默认形态：类型名足以区分「导入失败 / 连接超时 / 权限拒绝」，
    而异常消息、args、traceback 全都可能夹带凭据或本地路径，一律不外传。
    """
    if exc is None:
        return "UnknownError"
    return type(exc).__name__
