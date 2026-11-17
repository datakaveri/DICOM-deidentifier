"""classify_phi runs each model only when its answer can change the decision.

The decisions themselves were checked against the previous run-everything
implementation with the real models (docs/performance.md); these tests pin
the call pattern and the decision matrix with fake models.
"""

from types import SimpleNamespace

import pytest

import classify


class Fakes:
    """Fake models that flag the texts they are given and count calls."""

    def __init__(self, deid=(), presidio=(), medical=(), gliner=()):
        self.flags = {
            "deid": set(deid),
            "presidio": set(presidio),
            "medical": set(medical),
            "gliner": set(gliner),
        }
        self.calls = {k: 0 for k in self.flags}
        # Stable handles: the decision cache keys on model identity, and a
        # bound method is a new object on every attribute access.
        self.analyzer = SimpleNamespace(analyze=self._presidio)
        self.gliner = SimpleNamespace(predict_entities=self._gliner)
        self.deid_model = self.deid
        self.medical_model = self.medical

    def deid(self, text):
        self.calls["deid"] += 1
        hit = text in self.flags["deid"]
        return [{"entity_group": "PATIENT", "score": 0.9}] if hit else []

    def medical(self, text):
        self.calls["medical"] += 1
        hit = text in self.flags["medical"]
        return [{"entity_group": "Disease_disorder"}] if hit else []

    def _presidio(self, text, language):
        self.calls["presidio"] += 1
        if text in self.flags["presidio"]:
            return [SimpleNamespace(entity_type="PERSON", score=0.9)]
        return []

    def _gliner(self, text, labels, threshold):
        self.calls["gliner"] += 1
        return [{"label": "disease"}] if text in self.flags["gliner"] else []

    def redacts(self, text):
        merged = [{"text": text, "bbox": [10, 10, 60, 30]}]
        out = classify.classify_phi(
            merged,
            (1000, 1000),
            self.analyzer,
            self.gliner,
            self.medical_model,
            self.deid_model,
        )
        return bool(out)


@pytest.fixture(autouse=True)
def fresh_cache():
    classify._DECISION_CACHE.clear()
    yield
    classify._DECISION_CACHE.clear()


def test_allowlisted_text_is_kept_without_any_model():
    f = Fakes(deid={"PA"}, presidio={"PA"})
    assert f.redacts("PA") is False
    assert sum(f.calls.values()) == 0


def test_demographic_label_redacts_without_any_model():
    f = Fakes(medical={"DR. SMITH"}, gliner={"DR. SMITH"})
    assert f.redacts("DR. SMITH") is True
    assert sum(f.calls.values()) == 0


def test_presidio_hit_redacts_and_skips_the_rest():
    f = Fakes(presidio={"XYZQ WORD"}, medical={"XYZQ WORD"}, gliner={"XYZQ WORD"})
    assert f.redacts("XYZQ WORD") is True
    assert f.calls == {"deid": 0, "presidio": 1, "medical": 0, "gliner": 0}


def test_stanford_hit_redacts_even_with_a_medical_signal():
    f = Fakes(deid={"XYZQ WORD"}, medical={"XYZQ WORD"}, gliner={"XYZQ WORD"})
    assert f.redacts("XYZQ WORD") is True
    assert f.calls["medical"] == 0 and f.calls["gliner"] == 0


@pytest.mark.parametrize(
    "medical, gliner, expect_redact, gliner_calls",
    [
        (True, False, False, 0),  # biomedical NER settles it; GLiNER skipped
        (False, True, False, 1),
        (False, False, True, 1),
    ],
)
def test_regex_only_phi_is_decided_by_the_medical_signal(
    medical, gliner, expect_redact, gliner_calls
):
    text = "9876543210"  # phone-number regex only: no label, no NER hit
    f = Fakes(medical={text} if medical else (), gliner={text} if gliner else ())
    assert f.redacts(text) is expect_redact
    assert f.calls["gliner"] == gliner_calls


@pytest.mark.parametrize("text", ["120", "0.5 - 1.5", "12345678"])
def test_numbers_without_phi_signal_are_kept_without_medical_models(text):
    f = Fakes()
    assert f.redacts(text) is False
    assert f.calls["medical"] == 0 and f.calls["gliner"] == 0


@pytest.mark.parametrize("gliner, expect_redact", [(True, False), (False, True)])
def test_unclassified_words_need_a_medical_signal_to_be_kept(gliner, expect_redact):
    f = Fakes(gliner={"XYZQ WORD"} if gliner else ())
    assert f.redacts("XYZQ WORD") is expect_redact


def test_decisions_are_cached_per_text_and_model_set():
    f = Fakes()
    f.redacts("XYZQ WORD")
    first = dict(f.calls)
    f.redacts("XYZQ WORD")
    assert f.calls == first

    # verify.py classifies with Presidio only: a separate cache entry.
    merged = [{"text": "XYZQ WORD", "bbox": [10, 10, 60, 30]}]
    assert classify.classify_phi(merged, (1000, 1000), f.analyzer)
    assert f.calls["presidio"] == first["presidio"] + 1
