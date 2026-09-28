# Parallel DICOM De-Identification Architecture & Memory Specification

---

## 1. High-Level System Overview & Dual-Track Workflow

The **DICOM De-Identification Pipeline** operates as two distinct, synchronized tracks that execute in a single unified pass per file:
1. **Track 1: Burned-in Pixel Text Redaction** — Deep learning vision and NLP stack (PaddleOCR, Stanford De-ID, Biomedical NER, GLiNER-BioMed, Microsoft Presidio, and Navier-Stokes Inpainting).
2. **Track 2: DICOM Tag Metadata De-Identification** — Cryptographic tag engine applying keyed double hashing, tokenization vaults, date masking, and private tag stripping per DICOM PS3.15 Annex E.

```mermaid
flowchart TD
    DCM["Raw DICOM File (.dcm)"] --> DUMP["Step 0: Snapshot Original Header Tags -> data.json"]
    
    subgraph TrackA ["Track 1: Burned-In Pixel Text Redaction (Heavy AI Pipeline)"]
        direction TB
        DUMP --> S1["Stage 1: Pixel Scaling & MONOCHROME1 Inversion"]
        S1 --> S2["Stage 2: Shape-Based Candidate Text Region Detection"]
        S2 --> S3["Stage 3: Region-Based PaddleOCR (Candidate Crops Only)"]
        S3 --> S4["Stage 4: Multi-Model Ensemble Matrix (5 Models + Allowlist)"]
        S4 --> S4_CROSS["Cross-Check: Match OCR text with original data.json tags"]
        S4_CROSS --> S5["Stage 5: Navier-Stokes Inpainting (Preserve Bone & Tissue)"]
        S5 --> S6["Stage 6: Two-Tier Verification Gate (<1ms Stroke Residue Check)"]
        S6 --> OUT_BEFORE["Checkpoint Saved: before_deidentification.dcm<br>(Pixels Redacted, Original Header Tags Intact)"]
    end

    subgraph TrackB ["Track 2: DICOM Tag De-Identification (Cryptographic Engine)"]
        direction TB
        OUT_BEFORE --> KEY["KeyStore Vault (secured.json)"]
        KEY --> S7_HASH["Keyed Double Hash: PatientID, StudyUIDs"]
        KEY --> S7_TOK["Reversible Vault Tokenization"]
        KEY --> S7_MASK["Date Masking: Keep Year, Mask Month/Day"]
        KEY --> S7_STRIP["Strip Private & Vendor Tags (Odd Groups)"]
        S7_STRIP --> OUT_AFTER["Final Output Saved: after_deidentification.dcm<br>(Pixels Redacted + Header Tags De-Identified)"]
    end

    OUT_AFTER --> PREV["Generate normalized after_preview.png"]
    OUT_AFTER --> AUDIT["Generate pipeline_audit.json (Timing, BBoxes, Verif Status)"]
```

---

## 2. Accurately Resolved Worker Allocation

The pipeline dynamically determines active parallel workers using a 3-tier resolution hierarchy in `app/main_parallel.py`:

```mermaid
flowchart LR
    CLI["1. CLI Argument: --workers N<br>(Highest Priority)"] --> RES["Resolved Worker Count"]
    ENV["2. Environment: SKALD_NUM_WORKERS<br>(Second Priority)"] --> RES
    AUTO["3. Auto-Tuner: _auto_tune()<br>(Default Fallback)"] --> RES
```

### Auto-Tuner Formula
```python
threads = 2                  # PyTorch / OpenMP / MKL thread cap
paddle_internal_threads = 2  # Internal threads spawned by Paddle runtime
effective_per_worker = 4     # Effective threads per worker
ram_per_worker_gb = 3.5      # ~3.2 GB model weights + 0.3 GB image buffers

max_by_cpu = max(1, int(TOTAL_CPUS * 1.25 / effective_per_worker))
max_by_ram = max(1, int((available_gb - 8) / ram_per_worker_gb))

workers = max(2, min(max_by_cpu, max_by_ram, 10))
```

### Worker & Thread Allocation Across Environments

| Environment | CPU Cores | Available RAM | Active Workers | Effective Threads | Total RAM Used | Operational Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Production Multi-Core VM** *(Commit 975daef)* | **16 Cores** | **125 GB** | **6 Workers** | 24 threads | **~21.0 GB** | High-throughput VM batching; 90–95% CPU saturation |
| **Current Local Machine** | **16 Cores** | **~2.1 GB free** | **2 Workers** | 8 threads | **~7.0 GB** | Safe floor clamped by auto-tuner to prevent OOM |
| **Custom User Override** | Any | Any | **N Workers** | $N \times 4$ | $N \times 3.5$ GB | Invoked with `--workers <N>` |

---

## 3. Parallel Processing VM Architecture

```mermaid
flowchart TB
    subgraph HostSystem ["Host VM: 16 CPU Cores | 128 GB RAM (forkserver context)"]
        direction TB

        subgraph MasterProc ["Main Process (PID: Master) ~500 MB RAM"]
            direction TB
            M1["1. Parse CLI Args & Scan input/ (*.dcm)"] --> M2["2. Auto-Tuning Calculation"]
            M2 --> M3["3. Model Cache Pre-Warming (Pre-downloads weights once)"]
            M3 --> M4["4. Launch multiprocessing.Pool (6 Persistent Workers)"]
            M4 --> M5["5. Dynamic Work Queue (pool.imap_unordered)"]
        end

        subgraph DiskCache ["Local Storage Cache (~3.6 GB Disk)"]
            D1["~/.paddleocr/ (PP-OCRv4 detection + recognition)"]
            D2["~/.cache/huggingface/ (Stanford De-ID, BioNER, GLiNER)"]
        end

        M3 -.->|"Downloads once prior to spawning"| DiskCache

        subgraph WorkerPool ["Worker Process Pool (6 Isolated Processes, ~21 GB RAM Total)"]
            direction LR

            subgraph W1 ["Worker 1 (PID 101)"]
                W1_M["5 Models: 3.2 GB<br>Buffers: 0.3 GB<br>Total: ~3.5 GB"]
                W1_T["2 Math Threads<br>+ 2 Paddle Threads"]
            end

            subgraph W2 ["Worker 2 (PID 102)"]
                W2_M["5 Models: 3.2 GB<br>Buffers: 0.3 GB<br>Total: ~3.5 GB"]
                W2_T["2 Math Threads<br>+ 2 Paddle Threads"]
            end

            subgraph W3 ["Worker 3 (PID 103)"]
                W3_M["5 Models: 3.2 GB<br>Buffers: 0.3 GB<br>Total: ~3.5 GB"]
                W3_T["2 Math Threads<br>+ 2 Paddle Threads"]
            end

            subgraph W4 ["Worker 4 (PID 104)"]
                W4_M["5 Models: 3.2 GB<br>Buffers: 0.3 GB<br>Total: ~3.5 GB"]
                W4_T["2 Math Threads<br>+ 2 Paddle Threads"]
            end

            subgraph W5 ["Worker 5 (PID 105)"]
                W5_M["5 Models: 3.2 GB<br>Buffers: 0.3 GB<br>Total: ~3.5 GB"]
                W5_T["2 Math Threads<br>+ 2 Paddle Threads"]
            end

            subgraph W6 ["Worker 6 (PID 106)"]
                W6_M["5 Models: 3.2 GB<br>Buffers: 0.3 GB<br>Total: ~3.5 GB"]
                W6_T["2 Math Threads<br>+ 2 Paddle Threads"]
            end
        end

        DiskCache ==>|"Zero-download fast disk load"| WorkerPool
        M5 ==>|"Streams DICOM file paths"| WorkerPool

        subgraph FileOutputs ["Storage Output (/app/output/<sample>/)"]
            OUT1["after_deidentification.dcm (Pixel & Header Anonymized)"]
            OUT2["before_deidentification.dcm (Pixel Redacted, Original Tags)"]
            OUT3["after_preview.png (Visual Inspection Preview)"]
            OUT4["bbox_regions.png (Detected PHI Bounding Boxes)"]
            OUT5["pipeline_audit.json (Timing, Status, Redacted Regions)"]
            OUT6["secured.json (Shared KeyStore Cryptographic Vault)"]
        end

        WorkerPool --> FileOutputs
    end
```

---

## 4. RAM (Memory) Distribution per Worker Process

```mermaid
pie title RAM Allocation per Worker Process (~3.5 GB Total)
    "GLiNER-BioMed (DeBERTa-v3-large)" : 1400
    "Biomedical NER (BioLinkBERT)" : 1300
    "Stanford De-ID (RoBERTa-base)" : 600
    "PaddleOCR (DBNet + SVTR)" : 400
    "Microsoft Presidio + spaCy" : 250
    "Python Runtime, OpenCV & 16-bit Buffers" : 350
```

### Exact Memory Footprint Breakdown

| Engine / Component | Architecture / Weights | Disk Cache Size | Resident RAM (per Worker) | Primary Role |
| :--- | :--- | :---: | :---: | :--- |
| **GLiNER-BioMed** | `urchade/gliner_large-v2.1` | ~1.7 GB | **~1,400 MB** | Zero-shot detection of 30+ radiology anatomy & technique labels |
| **Biomedical NER** | `d4data/biomedical-ner-all` | ~1.4 GB | **~1,300 MB** | Clinical entity recognition (diseases, symptoms, organs) |
| **Stanford De-ID** | `StanfordAIMI/stanford-deidentifier-base` | ~500 MB | **~600 MB** | Radiology-specific PHI identification (Patient, MRN, Hospital) |
| **PaddleOCR** | PP-OCRv4 (`det` + `rec`) | ~45 MB | **~400 MB** | Cropped text detection and character reading |
| **Microsoft Presidio** | Rule Analyzer + `en_core_web_sm` | ~15 MB | **~250 MB** | Structured PII patterns (DOB, dates, phone numbers, locations) |
| **Pixel Buffers & OpenCV** | 16-bit DICOM arrays (up to 3000×3000) | — | **~350 MB** | Image scaling, LUT mappings, inpainting scratch arrays |
| **Total per Worker** | — | **~3.6 GB** | **~3.5 GB to 3.8 GB** | **Isolated RAM per child process** |

---

## 5. Detailed 7-Stage Pipeline Lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant FS as DICOM File System
    participant WP as Worker Process
    participant TR as Shape Region Detector (Stage 2)
    participant OCR as PaddleOCR (Stage 3)
    participant CLF as Multi-Model Matrix (Stage 4)
    participant INP as Navier-Stokes Inpainter (Stage 5)
    participant VRF as Two-Tier Verifier (Stage 6)
    participant TAG as Tag De-ID Engine (Stage 7)

    FS->>WP: Load raw .dcm file
    WP->>WP: Stage 1: MONOCHROME1 flip & 1-99% contrast stretch
    WP->>TR: Stage 2: Connected-component & shape grouping
    TR-->>WP: Candidate text bounding boxes (No OCR run yet)
    WP->>OCR: Stage 3: Crop candidate regions and run PaddleOCR
    OCR-->>WP: Detected raw text strings + coordinates
    WP->>CLF: Stage 4: Run Stanford De-ID, BioNER, GLiNER, Presidio, Allowlists
    WP->>CLF: Cross-check detected words against original data.json tags
    CLF-->>WP: Final resolved PHI bounding boxes to redact
    WP->>INP: Stage 5: Isolate character strokes & apply Navier-Stokes inpainting
    INP-->>WP: Cleaned pixel array (Tissue & bone textures preserved)
    WP->>VRF: Stage 6: Fast Tier 1 stroke energy residual check (<1ms)
    VRF-->>WP: Verification PASSED (or Tier 2 OCR escalation if residue found)
    WP->>FS: Save checkpoint: before_deidentification.dcm (Tags original)
    WP->>TAG: Stage 7: Apply PS3.15 tag mapping (Hash, Tokenize, Mask, Suppress)
    TAG-->>WP: Tag-scrubbed dataset
    WP->>FS: Save final: after_deidentification.dcm + after_preview.png + pipeline_audit.json
```

---

## 6. Latency Profile & Stage Bottlenecks

| Stage | Name | Average Latency | % of Total Time | Bottleneck Type | Optimization Implemented |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **1** | Ingestion & Dynamic Scaling | ~0.08s | 2% | Disk I/O & NumPy percentiles | Vectorized contrast stretching |
| **2** | Text Region Detection | ~0.05s | 1% | Morphological filtering | Shape-only candidate filtering (no OCR) |
| **3** | Region-Based PaddleOCR | ~0.60s – 1.10s | **25%** | Deep learning inference | **Runs on cropped regions only** (avoids full image) |
| **4** | Multi-Model Classification Matrix | ~1.20s – 2.00s | **50%** | 3 Transformers + Presidio sequential pass | Clinical allowlist bypass; block expansion |
| **5** | Navier-Stokes Inpainting | ~0.25s – 0.45s | **12%** | PDE boundary value iterations | Character stroke isolation (only letters inpainted) |
| **6** | Verification Gate | ~0.01s – 0.05s | 2% | Re-verification | **Tier 1 stroke residual gate (<1ms)** |
| **7** | Tag De-Identification & Save | ~0.15s – 0.25s | 8% | Cryptographic hashing & file writes | Reused in-memory KeyStore |
| **All** | **Total per File (Single Worker)** | **~2.3s – 4.0s** | **100%** | — | **Throughput across 6 Workers: ~0.4s to 0.7s / file** |
