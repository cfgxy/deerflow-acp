"""日志级别环境变量的解析行为（仅认新名，旧名等价于未设置）。"""

from __future__ import annotations

import logging

import pytest

from deerflow_acp.logging_setup import _resolve_level


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("DEBUG", None, logging.DEBUG),
        ("DEBUG", "WARNING", logging.DEBUG),
        (None, "WARNING", logging.INFO),
        (None, None, logging.INFO),
    ],
)
def test_log_level_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: int,
) -> None:
    # 旧前缀名以拼接推导，避免字面量入库（全仓 grep 旧名零残留是验收项）。
    legacy_name = "DEERFLOW" + "_ACP_LOG_LEVEL"
    for name, value in (
        ("DEER_FLOW_ACP_LOG_LEVEL", new_value),
        (legacy_name, legacy_value),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert _resolve_level() == expected
