"""桥接器环境变量配置。"""

from __future__ import annotations

import pytest

from deerflow_acp.config import (
    DEFAULT_CANCEL_GRACE_SECONDS,
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    BridgeConfig,
)


@pytest.mark.parametrize(
    ("new_path", "legacy_path", "expected"),
    [
        ("/config/new.yaml", None, "/config/new.yaml"),
        (None, "/config/legacy.yaml", "/config/legacy.yaml"),
        ("/config/new.yaml", "/config/legacy.yaml", "/config/new.yaml"),
        (None, None, None),
    ],
)
def test_config_path_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_path: str | None,
    legacy_path: str | None,
    expected: str | None,
) -> None:
    for name, value in (
        ("DEER_FLOW_CONFIG_PATH", new_path),
        ("DEERFLOW_ACP_CONFIG_PATH", legacy_path),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert BridgeConfig.from_env().deerflow_config_path == expected


def _set_env_pair(
    monkeypatch: pytest.MonkeyPatch,
    new_name: str,
    legacy_name: str,
    new_value: str | None,
    legacy_value: str | None,
) -> None:
    for name, value in ((new_name, new_value), (legacy_name, legacy_value)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("gpt-new", None, "gpt-new"),
        (None, "gpt-legacy", "gpt-legacy"),
        ("gpt-new", "gpt-legacy", "gpt-new"),
        (None, None, None),
    ],
)
def test_model_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: str | None,
) -> None:
    _set_env_pair(
        monkeypatch,
        "DEER_FLOW_ACP_MODEL",
        "DEERFLOW_ACP_MODEL",
        new_value,
        legacy_value,
    )
    assert BridgeConfig.from_env().model_name == expected


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("false", None, False),
        (None, "false", False),
        ("false", "true", False),
        (None, None, True),
    ],
)
def test_thinking_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: bool,
) -> None:
    _set_env_pair(
        monkeypatch,
        "DEER_FLOW_ACP_THINKING",
        "DEERFLOW_ACP_THINKING",
        new_value,
        legacy_value,
    )
    assert BridgeConfig.from_env().thinking_enabled is expected


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("0.5", None, 0.5),
        (None, "0.5", 0.5),
        ("0.5", "9", 0.5),
        (None, None, DEFAULT_CANCEL_GRACE_SECONDS),
    ],
)
def test_cancel_grace_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: float,
) -> None:
    _set_env_pair(
        monkeypatch,
        "DEER_FLOW_ACP_CANCEL_GRACE_SECONDS",
        "DEERFLOW_ACP_CANCEL_GRACE_SECONDS",
        new_value,
        legacy_value,
    )
    assert BridgeConfig.from_env().cancel_grace_seconds == expected


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("0.5", None, 0.5),
        (None, "0.5", 0.5),
        ("0.5", "9", 0.5),
        (None, None, DEFAULT_SHUTDOWN_GRACE_SECONDS),
    ],
)
def test_shutdown_grace_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: float,
) -> None:
    _set_env_pair(
        monkeypatch,
        "DEER_FLOW_ACP_SHUTDOWN_GRACE_SECONDS",
        "DEERFLOW_ACP_SHUTDOWN_GRACE_SECONDS",
        new_value,
        legacy_value,
    )
    assert BridgeConfig.from_env().shutdown_grace_seconds == expected


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("true", None, True),
        (None, "true", True),
        ("true", "false", True),
        (None, None, False),
    ],
)
def test_emit_usage_update_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: bool,
) -> None:
    _set_env_pair(
        monkeypatch,
        "DEER_FLOW_ACP_EMIT_USAGE_UPDATE",
        "DEERFLOW_ACP_EMIT_USAGE_UPDATE",
        new_value,
        legacy_value,
    )
    # emit 依赖窗口大小同时设置才生效，统一钉一个窗口值，使断言只反映 flag 本身。
    monkeypatch.setenv("DEER_FLOW_ACP_CONTEXT_WINDOW_TOKENS", "4096")
    assert BridgeConfig.from_env().emit_usage_update is expected


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("4096", None, 4096),
        (None, "2048", 2048),
        ("4096", "2048", 4096),
        (None, None, None),
    ],
)
def test_context_window_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: int | None,
) -> None:
    _set_env_pair(
        monkeypatch,
        "DEER_FLOW_ACP_CONTEXT_WINDOW_TOKENS",
        "DEERFLOW_ACP_CONTEXT_WINDOW_TOKENS",
        new_value,
        legacy_value,
    )
    assert BridgeConfig.from_env().context_window_tokens == expected
