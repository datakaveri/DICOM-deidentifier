"""
shared_ner.py -- PII spans from the shared Anamika NER engine, reached through
the enclave.

Added 2026-11-23 (Kailash: DICOM should use the same NER as the Anamika
k-anonymisation pipeline, "on extracted text + if there are any remarks tags
which need NER that too").

Where the engine is: on the SPIDEr enclave that runs this container, the
enclave manager serves a Unix socket (mounted here by the compose file at
SOCKET_PATH). Behind it, the enclave sends the text over attested RA-TLS to
the NER TEE, a separate confidential VM that holds the models, and returns
only character spans. This container has no network route to the NER TEE and
needs none; the text never leaves confidential memory.

Never a hard dependency:
  - No socket (any other deployment, an enclave on older code, a local run):
    `available()` is False and every caller keeps its existing behaviour.
  - A failed or slow call returns None ("unknown"), never [] ("nothing
    found"), so callers can tell "clean" from "could not check" and take
    their safe path. After a failure the socket is not asked again for this
    process, so a dead engine costs one timeout per batch, not per file.

Request/response: POST /spans {"texts", "minimum_confidence", "labels"} ->
{"results": [{"spans": [{"start", "end", "label", "source", "confidence"}],
"error"}]}. Offsets index each text exactly as sent.
"""

import http.client
import json
import os
import socket

from config import log

SOCKET_PATH = os.environ.get("NER_SPANS_SOCKET", "/run/anamika-ner/ner.sock")
# The first call may wait while the enclave attests a NER TEE that is still
# booting (it waits up to 180 s for one); later calls take about a second.
TIMEOUT_SECONDS = float(os.environ.get("NER_SPANS_TIMEOUT", "300"))
MAX_TEXTS_PER_CALL = 2000

# Model labels the engine can report. Its ORGANIZATION / LOCATION guesses are
# noisy on upper-case radiology text ("CHEST PA", "KV", "MAS" came back as
# ORGANIZATION at 0.77-1.0 in a 2026-11-23 probe), so by default only PERSON
# is asked for; a config can add the others (shared_ner.model_labels).
MODEL_LABELS = ("PERSON", "LOCATION", "ORGANIZATION")
DEFAULT_MODEL_LABELS = ("PERSON",)

_broken = False


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


def available():
    return not _broken and os.path.exists(SOCKET_PATH)


def _call(texts, minimum_confidence, model_labels):
    body = json.dumps(
        {
            "texts": texts,
            "minimum_confidence": minimum_confidence,
            "labels": list(model_labels),
        }
    )
    conn = _UnixHTTPConnection(SOCKET_PATH, TIMEOUT_SECONDS)
    try:
        conn.request(
            "POST", "/spans", body=body, headers={"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        raw = resp.read()
    finally:
        conn.close()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}")
    results = json.loads(raw)["results"]
    if len(results) != len(texts):
        raise RuntimeError(f"{len(results)} results for {len(texts)} texts")
    return results


def spans_many(texts, minimum_confidence=0.75, model_labels=DEFAULT_MODEL_LABELS):
    """Spans per text, or None when the engine could not be asked.

    Each span is {"start", "end", "label", "confidence"}. A text the engine
    reported an error for gets None in its slot, like an unreachable engine.
    """
    global _broken
    texts = list(texts)
    if not texts:
        return []
    if not available():
        return None
    labels = tuple(x for x in model_labels if x in MODEL_LABELS) or DEFAULT_MODEL_LABELS
    out = []
    try:
        for i in range(0, len(texts), MAX_TEXTS_PER_CALL):
            chunk = texts[i : i + MAX_TEXTS_PER_CALL]
            for text, res in zip(chunk, _call(chunk, minimum_confidence, labels)):
                if not isinstance(res, dict) or res.get("error"):
                    out.append(None)
                    continue
                spans = []
                for s in res.get("spans") or []:
                    try:
                        a, b = int(s["start"]), int(s["end"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    label = str(s.get("label") or "")
                    if not 0 <= a < b <= len(text):
                        continue
                    if label in MODEL_LABELS and label not in labels:
                        continue
                    spans.append(
                        {
                            "start": a,
                            "end": b,
                            "label": label,
                            "confidence": float(s.get("confidence") or 0),
                        }
                    )
                out.append(spans)
    except Exception as e:  # noqa: BLE001 -- never fail a file because NER did
        _broken = True
        # The error names the transport, never any text.
        log.warning(
            f"  [shared NER] unavailable ({type(e).__name__}: {e}); using this "
            f"container's own detection for the rest of the batch."
        )
        return None
    return out


# ── Injectable client (added for single_file.Engines) ────────────────────────
# A caller that already holds an NER engine (a Ray actor with its own client)
# can pass any object with `spans(texts) -> list[list[dict]]`, a span being
# {"start", "end", "label"[, "confidence"]}. The contract is the same as
# spans_many's: None for the whole call (or for one text's slot) means "could
# not be checked", and the caller then blanks the tag / keeps its own
# classifier. The socket client below stays the default, so production is
# unchanged.


class SocketNerClient:
    """The default client: the enclave's unix socket, via spans_many."""

    def __init__(self, minimum_confidence=0.75, model_labels=DEFAULT_MODEL_LABELS):
        self.minimum_confidence = minimum_confidence
        self.model_labels = tuple(model_labels)

    def spans(self, texts):
        return spans_many(texts, self.minimum_confidence, self.model_labels)


class _CheckedClient:
    """Wraps an injected client so it obeys the job's settings the way the
    socket path does: unknown labels, bad offsets and low-confidence spans are
    dropped, and a client that raises counts as unavailable (never fail a file
    because NER did)."""

    def __init__(self, client, settings):
        self._client = client
        self._labels = tuple(x for x in settings.model_labels if x in MODEL_LABELS) or DEFAULT_MODEL_LABELS
        self._min = settings.minimum_confidence

    def spans(self, texts):
        texts = list(texts)
        if not texts:
            return []
        try:
            raw = self._client.spans(texts)
        except Exception as e:  # noqa: BLE001
            log.warning(f"  [injected NER] unavailable ({type(e).__name__}); treating as unchecked.")
            return None
        if raw is None or len(raw) != len(texts):
            return None
        out = []
        for text, found in zip(texts, raw):
            if found is None:
                out.append(None)
                continue
            kept = []
            for s in found:
                try:
                    a, b = int(s["start"]), int(s["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                label = str(s.get("label") or "")
                if not 0 <= a < b <= len(text):
                    continue
                if label in MODEL_LABELS and label not in self._labels:
                    continue
                conf = s.get("confidence")
                if conf is not None and float(conf) < self._min:
                    continue
                kept.append({"start": a, "end": b, "label": label, "confidence": float(conf or 0)})
            out.append(kept)
        return out


def resolve_client(ner, settings):
    """The client to ask: `ner` (checked against `settings`) when injected,
    else the socket client with the job's settings."""
    if ner is None:
        return SocketNerClient(settings.minimum_confidence, settings.model_labels)
    return _CheckedClient(ner, settings)
