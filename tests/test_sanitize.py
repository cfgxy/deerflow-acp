"""脱敏层测试。

这些用例里的「秘密」全部是本地构造的假串，不是任何真实凭据。
"""

from __future__ import annotations

import pytest

from deerflow_acp.sanitize import describe_exception, redact_text

# 构造的假秘密：形态逼真但完全无效
FAKE_OPENAI_KEY = "sk-proj-Ab3xQ9zK7mNpR2vT5wY8cE1dF4gH6jL0oP"
FAKE_ANTHROPIC_KEY = "sk-ant-api03-ZzYyXxWwVvUuTtSsRrQqPpOoNnMmLlKkJj"
FAKE_HEX_TOKEN = "9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c5b4a3928"
FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"


class TestRedactText:
    @pytest.mark.parametrize(
        "secret",
        [
            FAKE_OPENAI_KEY,
            FAKE_ANTHROPIC_KEY,
            FAKE_JWT,
            FAKE_HEX_TOKEN,
        ],
    )
    def test_known_secret_shapes_are_masked(self, secret: str) -> None:
        text = f"调用失败：凭据 {secret} 被拒绝"
        result = redact_text(text)
        assert secret not in result
        assert "[REDACTED]" in result

    @pytest.mark.parametrize(
        "raw",
        [
            "api_key=Ab3xQ9zK7mNpR2vT5wY8cE1d",
            "API-KEY: Ab3xQ9zK7mNpR2vT5wY8cE1d",
            'token="Ab3xQ9zK7mNpR2vT5wY8cE1d"',
            "password: hunter2hunter2hunter2",
            "secret = Ab3xQ9zK7mNpR2vT5wY8cE1d",
            "authorization: Bearer Ab3xQ9zK7mNpR2vT5wY8cE1d",
        ],
    )
    def test_key_value_secrets_are_masked(self, raw: str) -> None:
        result = redact_text(raw)
        assert "Ab3xQ9zK7mNpR2vT5wY8cE1d" not in result
        assert "hunter2" not in result
        assert "[REDACTED]" in result

    def test_url_userinfo_is_masked(self) -> None:
        result = redact_text("connect postgres://dfuser:Sup3rS3cretPw@127.0.0.1:5432/deerflow")
        assert "Sup3rS3cretPw" not in result
        assert "127.0.0.1:5432" in result, "主机名是有用的诊断信息，不应被一并抹掉"

    def test_env_var_assignment_is_masked(self) -> None:
        result = redact_text("OPENAI_API_KEY=Ab3xQ9zK7mNpR2vT5wY8cE1d not set correctly")
        assert "Ab3xQ9zK7mNpR2vT5wY8cE1d" not in result

    def test_plain_diagnostics_survive(self) -> None:
        """脱敏不能把普通诊断信息一起毁掉，否则错误就不可诊断了。"""
        text = "连接 127.0.0.1:5432 超时（3 次重试后放弃）"
        assert redact_text(text) == text

    def test_short_identifiers_survive(self) -> None:
        text = "thread df-abc123 不存在"
        assert redact_text(text) == text

    def test_non_string_input_is_coerced(self) -> None:
        assert redact_text(None) == ""
        assert redact_text(12345) == "12345"

    def test_multiple_secrets_all_masked(self) -> None:
        text = f"first={FAKE_OPENAI_KEY} second={FAKE_HEX_TOKEN}"
        result = redact_text(text)
        assert FAKE_OPENAI_KEY not in result
        assert FAKE_HEX_TOKEN not in result


class TestDescribeException:
    def test_returns_type_name_only(self) -> None:
        exc = RuntimeError(f"连接 provider 失败：key={FAKE_OPENAI_KEY}")
        assert describe_exception(exc) == "RuntimeError"

    def test_custom_exception_type(self) -> None:
        class WeirdError(Exception):
            pass

        assert describe_exception(WeirdError("x")) == "WeirdError"

    def test_none_is_handled(self) -> None:
        assert describe_exception(None) == "UnknownError"
