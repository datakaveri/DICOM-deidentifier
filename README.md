# SKALD-DICOM

De-identification pipeline for radiology DICOM files, built for the SPIDEr platform. Removes Protected Health Information (PHI) at two levels in a single pass:

1. **Burned-in pixel text** — OCR (EasyOCR) finds names, IDs, dates, phone numbers, and other PHI printed directly into the image, classifies it against a clinical allowlist and NLP (Presidio), and redacts it with structure-preserving inpainting (no quality loss to the underlying anatomy). PaddleOCR is supported as an optional second engine in `engines.py`/`ocr_detect.py` for local, non-containerized use, but is deliberately left out of the Docker image — running PaddlePaddle and PyTorch in the same process corrupts the heap once both load real model weights.
2. **DICOM tag values** — every header tag is passed through a per-tag technique, plus private/vendor tag stripping and UID regeneration. Which technique hits which tag comes from one of two policies, chosen by the job config on the `/app/config` mount (see [Job config](#job-config) below): the user's own per-tag choices, or the built-in fixed policy in [`app/de_identification/tag_mapping.py`](app/de_identification/tag_mapping.py).

The output is always a valid `.dcm` file — never a PNG/JPEG export — so downstream DICOM tooling keeps working.

## Project layout

```
app/
  main.py                  # batch driver: every *.dcm under DATA_DIR, manifest + per-file audits
  de_identification/run.py # container entrypoint (Dockerfile CMD) — delegates to main.py
  config.py                # paths (env-var overridable), PII patterns, clinical allowlist
  pipeline.py               # 7-stage pixel redaction pipeline
  text_region_detect.py     # shape-based candidate text regions, fed to OCR
  classify.py               # PHI vs clinical decision over the OCR detections
  de_identification/        # tag-level de-identification (hashing, tokenisation, FPE, keystore)
  config/                   # local dev mirror of the container's mounted config volume
data/                       # input DICOM files (local dev mirror of the container's mounted volume)
output/                     # results, audit logs, keystore (local dev mirror of the container's mounted volume)
tests/                      # pytest suite
Dockerfile
requirements.txt
```

## Running locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Drop one or more .dcm files into data/, then:
cd app
python -m de_identification.run
```

Results land under `output/<filename>/`: `data.json` (original tag snapshot), `phi_tags.json` (identified PHI tags), `before_deidentification.dcm` (pixel-redacted), `after_deidentification.dcm` (pixel-redacted + tag de-identified), `pipeline_audit.json`, and `tag_audit.json`. A run-level `output/manifest.json` summarizes every file processed. Tokenisation/encryption keys persist in `output/keystore/` across runs.

## Running with Docker

```bash
docker build -t skald-dicom .

docker run --rm \
  -v /path/to/input/dicoms:/app/data \
  -v /path/to/config:/app/config \
  -v /path/to/output:/app/output \
  skald-dicom
```

The container processes every `.dcm` file found under `/app/data` (recursively) and writes results to `/app/output`, in the same layout described above. OCR/NLP model weights are baked into the image at build time, so the container needs no outbound network access at runtime — this matters for air-gapped/TEE deployments.

Environment variables (already set in the image, override if needed):

| Variable | Default | Purpose |
|---|---|---|
| `SKALD_DATA_DIR` | `/app/data` | Input DICOM files (recursively scanned for `*.dcm`) |
| `SKALD_CONFIG_DIR` | `/app/config` | Where the middleware drops the decrypted job config (`config.json`) |
| `SKALD_JOB_CONFIG` | `$SKALD_CONFIG_DIR/config.json` | Explicit path to the job config, if it is not at the default name |
| `SKALD_TEMP_DIR` | `/tmp/skald-dicom` | Scratch for the still-identified intermediates. Must stay under `/tmp` — the app refuses to start otherwise |
| `SKALD_OUTPUT_DIR` | `/app/output` | Per-file results, audit logs, and the persistent keystore |
| `SKALD_SAVE_BBOX_PREVIEW` | unset (off) | Set to `1` to also write `bbox_regions.png` per file — the frame **before** redaction with detected PHI boxed. It still shows the burned-in PHI in the clear, so it's a detection-tuning aid only, never for a de-identified output volume. |

## Job config

SPIDEr dispatches a job config for `application: "skald_dicom"`, Fernet-encrypted as `config_ciphertext`. The middleware decrypts it and drops the JSON at `$SKALD_CONFIG_DIR/config.json`; nothing in this repo decrypts anything.

**No config, or a config with no `dicom_deidentify` block** — the pre-contract document shape — runs the built-in fixed policy from `tag_mapping.py`, unchanged.

**A config with a `dicom_deidentify` block** runs the user's own per-tag choices:

```jsonc
{
  "data_type": "chest_xray",
  "dataset_name": "chest_xray.dcm",
  "format": "dicom",
  "operations": ["dicom_deidentify", "suppress", "hashing_with_salt", "masking"],

  "dicom_deidentify": {
    "default_action": "keep",
    "tag_actions": {
      "PatientName":      { "action": "suppress" },
      "PatientID":        { "action": "hashing_with_salt" },
      "PatientBirthDate": { "action": "masking", "masking_char": "*",
                            "characters_to_mask": [5, 6, 7, 8] },
      "InstitutionName":  { "action": "suppress" }
    }
  }
}
```

`tag_actions` is keyed by tag **keyword** (what the browser parses and the user selected), resolved via `job_config.KEYWORD_TO_TAG` and then pydicom's dictionary. The six actions are the same k-anonymisation techniques, with the same names and meanings, as the tabular (CSV) pipeline — each dispatches to the SKALD primitive that already implements it:

| Action | Behaviour |
|---|---|
| `suppress` | Blanks the attribute — zero length, **tag retained**. Deleting it breaks readers expecting a Type 2 attribute. |
| `hashing_with_salt` | Salted one-way hash. The salt is **per-run and never leaves the enclave**: same input → same output within a run so records still join, different across runs so there is no cross-run linkage. VR-length preserving. |
| `encrypt` | `format_preserving: true` keeps the input's shape and VR — the only variant safe to write into a typed element. With `false` the value is widened. |
| `masking` | Replaces the characters at `characters_to_mask` (**1-based**) with `masking_char`. Also how a date is coarsened: DICOM dates are `YYYYMMDD`, so `[5,6,7,8]` turns `19780412` into `1978****`. |
| `tokenization` | Opaque generated value; no relation to the input, no stability across records. |
| `charcloak` | Character-level obfuscation. |

There is deliberately no `generalize`, `date_shift`, `uid_remap` or `free_text_anonymization` — an `operations` entry naming one is rejected at parse time, before the first file is opened.

`default_action` is always `keep`: **a tag absent from `tag_actions` is written through unchanged.** `StudyDescription`/`SeriesDescription` in particular pass through byte-identical unless named — with no free-text redaction, deleting them takes "CHEST PA" along with "Jane Doe". Two things are not "unchanged", both so this path is never weaker than the fixed policy it replaces: UIDs not named get the existing deterministic remap, and private tags are stripped as always.

### Fields that are fixed server-side

The config arrives from a browser, so everything below is read from `config.py` and **ignored on the wire** — a tampered document cannot move it. A disagreeing wire value is logged and discarded.

| Field | Enforced | Why |
|---|---|---|
| `temp_dir` | `/tmp/skald-dicom` | Anything under `/app/output` writes the original, still-identified DICOM onto the volume that leaves the enclave |
| `emit_pixel_text_report` | `false` | The recognised burned-in text is patient names and MRNs; a plaintext sidecar keeps the PHI readable even though the pixels were blacked out |
| `emit_uid_crosswalk` | `false` | Nothing remaps UIDs through a crosswalk today, but an enclave that grows the capability must not start publishing a re-identification key beside the output |
| `input_dir` / `output_dir` | `/app/data`, `/app/output` | Fixed mount points |
| `fail_on_unparsable` | `true` | A DICOM that cannot be parsed fails loudly, rather than passing through un-de-identified because nothing could be found to remove |
| `pixel` | `{ redact_burned_in_text: true, method: "black", verify: true }` | Turning redaction off leaves PHI in pixels no tag action can reach, and `blur` is partially reversible on text |

`verify` produces `pixel_verification_status` (`PASSED` / `FAILED` / `SKIPPED`). **A file that fails verification is still written**, with the failed status — the user needs to see that redaction was incomplete, not receive nothing and no explanation.

### Where the output goes

Under a job config, the three artefacts that still hold original values — `data.json`, `phi_tags.json` and `before_deidentification.dcm` — go to a scratch tree under `SKALD_TEMP_DIR` and are deleted when the batch ends. Only `after_deidentification.dcm`, the audits and `manifest.json` reach `/app/output`, and `pipeline_audit.json` has the recognised burned-in text stripped out of it. The keystore lives in the scratch tree too and dies with the run. Without a job config, everything stays where it always was.

`manifest.json` keys `tags_by_technique` off the same six action identifiers the config sent, so they line up exactly. Private-tag stripping and the UID remap are counted separately (`private_tag_policy`, `uid_policy`) rather than folded into the user's own `suppress` count.

### Burned-in text detection

Classification runs a signal stack: clinical allowlist/shorthand, regex PII patterns (Aadhaar, ABHA, phone, UHID/MRN, dates, age/sex), and Presidio NLP. `engines.py` additionally supports three optional transformer models (`StanfordAIMI/stanford-deidentifier-base`, `d4data/biomedical-ner-all`, `urchade/gliner_large-v2.1`) that sharpen the PHI-vs-clinical decision. They are **not** installed in the image — `transformers` and `gliner` are absent from `requirements.txt`, so those three initialise to `None` and the container runs on the allowlist + regex + Presidio path. Install them locally if you want the full stack; expect the image to roughly triple in size and RAM use if you bake them in.

## Testing

```bash
pytest
```

## Policy notes

- Tag values are never modified by the pixel pipeline — only private/vendor tags are stripped and UIDs regenerated, so a file can't be linked back to the original study before tag-level de-identification runs.
- The tag-level de-identification technique per field is explicit and auditable — either in the job config the user sent, or in `tag_mapping.py`. Nothing is inferred at runtime.
- The `hashing_with_salt` salt is minted per run, held only in memory, and written nowhere. That is what makes hashes join within a batch and stay unlinkable across batches; it is not an oversight that it does not persist.
- `output/keystore/secured.json` (tokenisation/encryption key material) is never committed to git — it's runtime state. Without a job config it persists via the `/app/output` mount so tokens and keys stay stable across runs; deleting it re-mints every key, so previously de-identified files stop correlating with new ones. **Under a job config it is written to the scratch tree under `/tmp` instead and discarded with the run**, so no key material reaches the volume that leaves the enclave. Cross-run correlation of encrypted/tokenised values is traded away for that; a deployment that needs it should add proper key escrow rather than moving the keystore back onto the output volume.
