"""
debug_stage.py — temporary diagnostic: prints candidate regions, raw OCR
detections, merged detections, and final PHI classification for
data/input.dcm, one stage at a time. Delete once the burned-in-text
detection issue is resolved.

Run from inside app/:  python debug_stage.py
"""

import numpy as np
import pydicom

from text_region_detect import detect_text_regions
from ocr_detect import detect_text_in_regions
from classify import merge_detections, classify_phi, expand_phi_blocks
from engines import check_gpu_available, initialize_engines

ds = pydicom.dcmread("data/input.dcm", force=True)
pixels = ds.pixel_array.copy()
pix_min, pix_max = pixels.min(), pixels.max()
norm_8 = ((pixels - pix_min) / (pix_max - pix_min) * 255.0).astype(np.uint8)

use_gpu = check_gpu_available()
paddle_ocr, easy_ocr, analyzer = initialize_engines(use_gpu=use_gpu)
print(f"\npaddle_ocr={'OK' if paddle_ocr else 'None'}  easy_ocr={'OK' if easy_ocr else 'None'}  analyzer={'OK' if analyzer else 'None'}")

regions = detect_text_regions(norm_8)
print(f"\n[Stage 2] Candidate regions: {len(regions)}")
for r in regions:
    print(" ", r)

raw = detect_text_in_regions(norm_8, regions, paddle_ocr, easy_ocr)
print(f"\n[Stage 3] Raw OCR detections: {len(raw)}")
for r in raw:
    print(" ", r)

merged = merge_detections(raw)
print(f"\n[Stage 4] Merged detections: {len(merged)}")
for m in merged:
    print(" ", m)

phi = classify_phi(merged, norm_8.shape, analyzer)
phi = expand_phi_blocks(merged, phi, norm_8.shape)
print(f"\n[Stage 4] Classified as PHI (to redact, after block-expand): {len(phi)}")
for p in phi:
    print(" ", p)
