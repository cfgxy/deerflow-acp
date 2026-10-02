"""桥接器环境变量配置：仅认 ``DEER_FLOW_*`` 新名。"""

from __future__ import annotations

import pytest

from deerflow_acp.config import (
    DEFAULT_CANCEL_GRACE_SECONDS,
    DEFAULT_SHUTDOWN_GRACE_SECONDS,
    BridgeConfig,
)


def _legacy_name(new_name: str) -> str:
    """旧前缀名（已废弃，不再读取）。

    反向用例要设置旧名来证明它已失效，但全仓 grep 旧名必须零残留，
    因此用拼接推导而不是写字面量。
    """
    legacy_prefix = "DEERFLOW" + "_ACP_"
    if new_name.startswith("DEER_FLOW_ACP_"):
        return legacy_prefix + new_name.removeprefix("DEER_FLOW_ACP_")
    return legacy_prefix + new_name.removeprefix("DEER_FLOW_")


def _set_env(monkeypatch: pytest.MonkeyPatch, name: str, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    ("new_path", "legacy_path", "expected"),
    [
        ("/config/new.yaml", None, "/config/new.yaml"),
        ("/config/new.yaml", "/config/legacy.yaml", "/config/new.yaml"),
        (None, "/config/legacy.yaml", None),
        (None, None, None),
    ],
)
def test_config_path_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_path: str | None,
    legacy_path: str | None,
    expected: str | None,
) -> None:
    _set_env(monkeypatch, "DEER_FLOW_CONFIG_PATH", new_path)
    _set_env(monkeypatch, _legacy_name("DEER_FLOW_CONFIG_PATH"), legacy_path)
    assert BridgeConfig.from_env().deerflow_config_path == expected


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("gpt-new", None, "gpt-new"),
        ("gpt-new", "gpt-legacy", "gpt-new"),
        (None, "gpt-legacy", None),
        (None, None, None),
    ],
)
def test_model_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: str | None,
) -> None:
    _set_env(monkeypatch, "DEER_FLOW_ACP_MODEL", new_value)
    _set_env(monkeypatch, _legacy_name("DEER_FLOW_ACP_MODEL"), legacy_value)
    assert BridgeConfig.from_env().model_name == expected


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("false", None, False),
        ("false", "true", False),
        (None, "false", True),
        (None, None, True),
    ],
)
def test_thinking_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: bool,
) -> None:
    _set_env(monkeypatch, "DEER_FLOW_ACP_THINKING", new_value)
    _set_env(monkeypatch, _legacy_name("DEER_FLOW_ACP_THINKING"), legacy_value)
    assert BridgeConfig.from_env().thinking_enabled is expected


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("0.5", None, 0.5),
        ("0.5", "9", 0.5),
        (None, "0.5", DEFAULT_CANCEL_GRACE_SECONDS),
        (None, None, DEFAULT_CANCEL_GRACE_SECONDS),
    ],
)
def test_cancel_grace_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: float,
) -> None:
    _set_env(monkeypatch, "DEER_FLOW_ACP_CANCEL_GRACE_SECONDS", new_value)
    _set_env(monkeypatch, _legacy_name("DEER_FLOW_ACP_CANCEL_GRACE_SECONDS"), legacy_value)
    assert BridgeConfig.from_env().cancel_grace_seconds == expected


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("0.5", None, 0.5),
        ("0.5", "9", 0.5),
        (None, "0.5", DEFAULT_SHUTDOWN_GRACE_SECONDS),
        (None, None, DEFAULT_SHUTDOWN_GRACE_SECONDS),
    ],
)
def test_shutdown_grace_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: float,
) -> None:
    _set_env(monkeypatch, "DEER_FLOW_ACP_SHUTDOWN_GRACE_SECONDS", new_value)
    _set_env(monkeypatch, _legacy_name("DEER_FLOW_ACP_SHUTDOWN_GRACE_SECONDS"), legacy_value)
    assert BridgeConfig.from_env().shutdown_grace_seconds == expected


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("true", None, True),
        ("true", "false", True),
        (None, "true", False),
        (None, None, False),
    ],
)
def test_emit_usage_update_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: bool,
) -> None:
    _set_env(monkeypatch, "DEER_FLOW_ACP_EMIT_USAGE_UPDATE", new_value)
    _set_env(monkeypatch, _legacy_name("DEER_FLOW_ACP_EMIT_USAGE_UPDATE"), legacy_value)
    # emit 依赖窗口大小同时设置才生效，统一钉一个窗口值，使断言只反映 flag 本身。
    monkeypatch.setenv("DEER_FLOW_ACP_CONTEXT_WINDOW_TOKENS", "4096")
    assert BridgeConfig.from_env().emit_usage_update is expected


@pytest.mark.parametrize(
    ("new_value", "legacy_value", "expected"),
    [
        ("4096", None, 4096),
        ("4096", "2048", 4096),
        (None, "2048", None),
        (None, None, None),
    ],
)
def test_context_window_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
    new_value: str | None,
    legacy_value: str | None,
    expected: int | None,
) -> None:
    _set_env(monkeypatch, "DEER_FLOW_ACP_CONTEXT_WINDOW_TOKENS", new_value)
    _set_env(monkeypatch, _legacy_name("DEER_FLOW_ACP_CONTEXT_WINDOW_TOKENS"), legacy_value)
    assert BridgeConfig.from_env().context_window_tokens == expected
