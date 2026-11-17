"""main.py routing: where each artefact goes, and how a bad config fails."""

import json

import pytest

import main


@pytest.fixture(autouse=True)
def reset_globals(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(main, "_SCRATCH_DIR", None)
    monkeypatch.setattr(main, "_OUT_NAMES", {})
    monkeypatch.setattr(main, "_JOB_CONFIG", None)


def test_same_basename_in_two_folders_gets_two_output_dirs():
    names = main._allocate_out_names(["/in/a/img.dcm", "/in/b/img.dcm", "/in/c.dcm"])
    assert names == {"/in/a/img.dcm": "img", "/in/b/img.dcm": "img_2", "/in/c.dcm": "c"}


def test_without_job_config_everything_stays_beside_the_output(tmp_path):
    p = main._paths("/in/img.dcm")
    out = str(tmp_path / "out" / "img")
    assert p["out_dir"] == p["work_dir"] == out
    assert p["before"].startswith(out)


def test_with_job_config_identified_files_never_reach_the_output(tmp_path, monkeypatch):
    scratch = str(tmp_path / "scratch")
    monkeypatch.setattr(main, "_SCRATCH_DIR", scratch)
    p = main._paths("/in/img.dcm")
    out = str(tmp_path / "out")
    for key in ("before", "data", "phi_tags"):
        assert p[key].startswith(scratch) and not p[key].startswith(out), key
    for key in ("final", "audit", "tag_audit"):
        assert p[key].startswith(out), key


def test_invalid_job_config_exits_1_before_any_work(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "operations": ["dicom_deidentify", "generalize"],
                "dicom_deidentify": {"tag_actions": {}},
            }
        )
    )
    import config

    monkeypatch.setattr(config, "JOB_CONFIG_FILE", str(cfg))
    assert main.main() == 1
    assert "invalid job config" in capsys.readouterr().err
