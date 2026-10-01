"""日志级别环境变量（新名优先、旧名兼容回退）的解析行为。"""

from __future__ import annotations

import logging

import pytest

from deerflow_acp.logging_setup import _resolve_level


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("DEBUG", None, logging.DEBUG),
        (None, "WARNING", logging.WARNING),
        ("DEBUG", "WARNING", logging.DEBUG),
        (None, None, logging.INFO),
    ],
)
def test_log_level_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: int,
) -> None:
    for name, value in (
        ("DEER_FLOW_ACP_LOG_LEVEL", new_value),
        ("DEERFLOW_ACP_LOG_LEVEL", legacy_value),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert _resolve_level() == expected
