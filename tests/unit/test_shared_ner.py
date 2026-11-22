"""
The shared Anamika NER engine (added 2026-11-23): the socket client, the
free_text_anonymization tag action, and the extra PHI signal on burned-in
text. The failures that matter here are silent ones -- a value nobody could
check written out as if it were clean, or a run without the feature behaving
differently from before.
"""

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydicom.dataset import Dataset

import shared_ner
from classify import classify_phi
from de_identification.actions import RunSecrets
from de_identification.deidentify import _redact_spans, deidentify_dataset_from_config
from de_identification.job_config import JobConfigError, parse_job_config
from de_identification.keystore import KeyStore


def make_config(tag_actions, shared=None):
    block = {"default_action": "keep", "tag_actions": tag_actions}
    if shared is not None:
        block["shared_ner"] = shared
    return parse_job_config(
        {
            "data_type": "chest_xray",
            "format": "dicom",
            "operations": ["dicom_deidentify", "free_text_anonymization"],
            "dicom_deidentify": block,
        }
    )


@pytest.fixture
def keystore(tmp_path):
    return KeyStore(str(tmp_path / "secured.json"))


@pytest.fixture(autouse=True)
def fresh_client(monkeypatch):
    monkeypatch.setattr(shared_ner, "_broken", False)


def _fake_engine(monkeypatch, finder):
    """shared_ner.spans_many answering from `finder(text) -> spans`."""
    seen = []

    def spans_many(texts, minimum_confidence=0.75, model_labels=("PERSON",)):
        seen.append((list(texts), minimum_confidence, tuple(model_labels)))
        return [finder(t) for t in texts]

    monkeypatch.setattr(shared_ner, "spans_many", spans_many)
    return seen


def _name_finder(text):
    i = text.find("RAMESH KUMAR")
    return [{"start": i, "end": i + 12, "label": "PERSON", "confidence": 0.99}] if i >= 0 else []


# ── job config ───────────────────────────────────────────────────────────────


def test_free_text_action_is_accepted_and_defaults_apply():
    cfg = make_config({"StudyDescription": {"action": "free_text_anonymization"}})
    assert cfg.tag_actions[(0x0008, 0x1030)].action == "free_text_anonymization"
    assert cfg.shared_ner.pixel_text is False
    assert cfg.shared_ner.minimum_confidence == 0.75
    assert cfg.shared_ner.model_labels == ("PERSON",)


def test_shared_ner_block_is_validated():
    cfg = make_config({}, {"pixel_text": True, "minimum_confidence": 0.6, "model_labels": ["PERSON", "LOCATION"]})
    assert cfg.shared_ner.pixel_text is True
    assert cfg.shared_ner.model_labels == ("PERSON", "LOCATION")
    for bad in ({"pixel_text": "yes"}, {"minimum_confidence": 2}, {"model_labels": ["DATE"]}, []):
        with pytest.raises(JobConfigError):
            make_config({}, bad)


# ── free_text_anonymization on tags ──────────────────────────────────────────


def test_names_inside_a_description_are_replaced_and_the_rest_kept(monkeypatch, keystore):
    seen = _fake_engine(monkeypatch, _name_finder)
    ds = Dataset()
    ds.StudyDescription = "XR CHEST PA - RAMESH KUMAR"
    ds.SeriesDescription = "CHEST PA"
    cfg = make_config(
        {
            "StudyDescription": {"action": "free_text_anonymization"},
            "SeriesDescription": {"action": "free_text_anonymization"},
        }
    )
    audit = deidentify_dataset_from_config(ds, keystore, RunSecrets(), cfg)
    assert ds.StudyDescription == "XR CHEST PA - [PERSON]"
    assert ds.SeriesDescription == "CHEST PA"
    assert len(seen) == 1  # one engine call for the whole file
    actions = {a["field"]: a["action"] for a in audit}
    assert actions == {"StudyDescription": "redacted-1-span(s)", "SeriesDescription": "no-personal-data-found"}
    assert "RAMESH" not in json.dumps(audit)  # the audit never carries values


def test_engine_unavailable_blanks_the_tag(monkeypatch, keystore):
    monkeypatch.setattr(shared_ner, "spans_many", lambda *a, **k: None)
    ds = Dataset()
    ds.StudyDescription = "XR CHEST PA - RAMESH KUMAR"
    ds.Modality = "CR"
    cfg = make_config({"StudyDescription": {"action": "free_text_anonymization"}})
    audit = deidentify_dataset_from_config(ds, keystore, RunSecrets(), cfg)
    assert ds.StudyDescription == ""
    assert "StudyDescription" in ds  # blanked, tag retained
    assert ds.Modality == "CR"
    assert audit[0]["action"] == "blanked-ner-unavailable"


def test_no_socket_means_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(shared_ner, "SOCKET_PATH", str(tmp_path / "absent.sock"))
    assert shared_ner.available() is False
    assert shared_ner.spans_many(["RAMESH KUMAR"]) is None


def test_non_text_vr_is_blanked_not_passed_through(monkeypatch, keystore):
    _fake_engine(monkeypatch, _name_finder)
    ds = Dataset()
    ds.StudyDate = "20240115"
    cfg = make_config({"StudyDate": {"action": "free_text_anonymization"}})
    audit = deidentify_dataset_from_config(ds, keystore, RunSecrets(), cfg)
    assert ds.StudyDate == ""
    assert audit[0]["action"] == "blanked-not-free-text-vr-DA"


def test_nested_sequence_values_are_covered(monkeypatch, keystore):
    _fake_engine(monkeypatch, _name_finder)
    item = Dataset()
    item.StudyDescription = "REFERRED BY RAMESH KUMAR"
    ds = Dataset()
    ds.RequestAttributesSequence = [item]
    cfg = make_config({"StudyDescription": {"action": "free_text_anonymization"}})
    deidentify_dataset_from_config(ds, keystore, RunSecrets(), cfg)
    assert ds.RequestAttributesSequence[0].StudyDescription == "REFERRED BY [PERSON]"


def test_replacement_too_long_for_the_vr_keeps_the_length():
    # SH holds 16 characters: "[PERSON]" for "AB" would make 22.
    text = "XR 12345678 AB C"
    out = _redact_spans(text, [{"start": 12, "end": 14, "label": "PERSON"}], "SH")
    assert out == "XR 12345678 ** C"
    assert len(out) == len(text)


def test_overlapping_spans_become_one_replacement():
    text = "Dr Anita Sharma, CT"
    spans = [
        {"start": 3, "end": 8, "label": "PERSON"},
        {"start": 3, "end": 15, "label": "PERSON"},
    ]
    assert _redact_spans(text, spans, "LO") == "Dr [PERSON], CT"


# ── burned-in text ───────────────────────────────────────────────────────────


def _det(text, x=10):
    return {"text": text, "bbox": [x, 10, x + 50, 30]}


def test_shared_ner_hit_is_redacted_even_when_the_matrix_keeps_it():
    # With no models loaded, letters are redacted anyway ("fallback:
    # unclassified_text"), so use a text the matrix KEEPS: an ID-like number
    # with no letters ("fallback:numeric_non_phi"), e.g. a hospital number.
    merged = [_det("SAROJ DEVI"), _det("PA", 100), _det("45871/23", 200)]
    without = classify_phi(merged, (512, 512))
    with_ner = classify_phi(merged, (512, 512), shared_ner_phi={"45871/23"})
    assert {r["text"] for r in without} == {"SAROJ DEVI"}
    assert {r["text"] for r in with_ner} == {"SAROJ DEVI", "45871/23"}


def test_clinical_allowlist_wins_over_shared_ner():
    merged = [_det("PA")]
    assert classify_phi(merged, (512, 512), shared_ner_phi={"PA"}) == []


# ── the socket client against a real Unix socket ─────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    reply = None

    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps(_Handler.reply(req)).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _UnixServer(HTTPServer):
    address_family = socket.AF_UNIX

    def server_bind(self):
        self.socket.bind(self.server_address)

    def get_request(self):
        conn, _ = self.socket.accept()
        return conn, ("local", 0)


@pytest.fixture
def unix_engine(monkeypatch):
    # AF_UNIX paths are limited to ~108 bytes; pytest's tmp_path can exceed it.
    path = f"/tmp/ner-test-{os.getpid()}.sock"
    if os.path.exists(path):
        os.remove(path)
    srv = _UnixServer(path, _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(shared_ner, "SOCKET_PATH", path)
    yield _Handler
    srv.shutdown()
    os.remove(path)


def test_client_round_trip_filters_labels_and_bad_spans(unix_engine):
    unix_engine.reply = lambda req: {
        "results": [
            {
                "spans": [
                    {"start": 0, "end": 12, "label": "PERSON", "confidence": 0.99},
                    {"start": 15, "end": 23, "label": "ORGANIZATION", "confidence": 0.9},
                    {"start": 5, "end": 999, "label": "PERSON", "confidence": 0.9},
                ],
                "error": None,
            },
            {"spans": [], "error": "engine failed on this one"},
        ]
    }
    got = shared_ner.spans_many(["RAMESH KUMAR - CHEST PA", "x"], 0.8, ("PERSON",))
    assert got[0] == [{"start": 0, "end": 12, "label": "PERSON", "confidence": 0.99}]
    assert got[1] is None


def test_client_failure_disables_it_for_the_batch(unix_engine):
    unix_engine.reply = lambda req: {"results": []}  # wrong count
    assert shared_ner.spans_many(["a"]) is None
    assert shared_ner.available() is False
