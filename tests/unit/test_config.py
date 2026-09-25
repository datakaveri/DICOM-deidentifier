import importlib
import os


def _reload_config():
    import config
    return importlib.reload(config)


def test_defaults_resolve_under_app_dir():
    for var in ("SKALD_DATA_DIR", "SKALD_CONFIG_DIR", "SKALD_OUTPUT_DIR"):
        os.environ.pop(var, None)
    config = _reload_config()

    assert config.DATA_DIR == os.path.join(config.BASE_DIR, "data")
    assert config.CONFIG_DIR == os.path.join(config.BASE_DIR, "config")
    assert config.OUTPUT_DIR == os.path.join(config.BASE_DIR, "output")
    assert config.KEYSTORE_DIR == os.path.join(config.OUTPUT_DIR, "keystore")


def test_env_vars_override_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("SKALD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SKALD_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("SKALD_OUTPUT_DIR", str(tmp_path / "output"))
    config = _reload_config()

    assert config.DATA_DIR == str(tmp_path / "data")
    assert config.CONFIG_DIR == str(tmp_path / "config")
    assert config.OUTPUT_DIR == str(tmp_path / "output")
    assert config.KEYSTORE_DIR == str(tmp_path / "output" / "keystore")
