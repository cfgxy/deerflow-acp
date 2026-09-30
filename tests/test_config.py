"""桥接器环境变量配置。"""

from __future__ import annotations

import pytest

from deerflow_acp.config import BridgeConfig


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
