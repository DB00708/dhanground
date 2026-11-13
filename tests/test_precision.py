import pytest

from adapters.common import precision


def test_config_default_is_bf16(monkeypatch):
    monkeypatch.delenv("DHANGROUND_PRECISION", raising=False)
    assert precision({"precision": "bf16"}) == "bf16"


def test_env_overrides_config_for_t4(monkeypatch):
    monkeypatch.setenv("DHANGROUND_PRECISION", "fp16")
    assert precision({"precision": "bf16"}) == "fp16"


def test_rejects_unknown_precision(monkeypatch):
    monkeypatch.setenv("DHANGROUND_PRECISION", "int4")
    with pytest.raises(SystemExit):
        precision({"precision": "bf16"})
