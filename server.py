"""
server.py — DICOM Tag De-identification Flask server.

Follows the SPIDEr service pattern (see SKALD_server.py in k-anonymisation).

Endpoints
---------
GET  /test_DICOM_deidentifier   health-check
POST /process_DICOM             de-identify PHI tags in an uploaded DICOM file
POST /inspect_DICOM             list PHI tags found in an uploaded DICOM file
"""

import base64
import configparser
import io
import json
import os

import pydicom
from flask import Flask, jsonify, request
from flask_cors import CORS

from deidentifier import deidentify, inspect

app = Flask(__name__)
CORS(app)

server_config = configparser.ConfigParser()
server_config.read("server_config.cfg")

HOST = server_config.get("DICOM_DEIDENTIFIER_SERVER", "ip",   fallback="0.0.0.0")
PORT = server_config.getint("DICOM_DEIDENTIFIER_SERVER", "port", fallback=5001)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_dicom(file_storage) -> pydicom.Dataset:
    raw = file_storage.read()
    return pydicom.dcmread(io.BytesIO(raw))


def _ds_to_b64(ds: pydicom.Dataset) -> str:
    buf = io.BytesIO()
    ds.save_as(buf)
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/test_DICOM_deidentifier", methods=["GET"])
def health():
    return jsonify({"message": "DICOM tag de-identification server running!"})


@app.route("/process_DICOM", methods=["POST"])
def process_dicom():
    """
    Accept a DICOM file, de-identify PHI tags, return audit + cleaned DICOM.

    Request: multipart/form-data
        file  — the DICOM (.dcm) file

    Response JSON:
        status               "success" | "failed"
        audit                list of {field, old_value, action}
        deidentified_dicom   base64-encoded de-identified .dcm bytes
    """
    try:
        if "file" not in request.files:
            return jsonify({"status": "failed", "error_message": "No 'file' field in request"}), 400

        f = request.files["file"]
        if not f.filename:
            return jsonify({"status": "failed", "error_message": "Empty filename"}), 400

        ds = _read_dicom(f)
        cleaned_ds, audit = deidentify(ds)

        return jsonify({
            "status": "success",
            "audit": audit,
            "fields_processed": len(audit),
            "deidentified_dicom": _ds_to_b64(cleaned_ds),
        })

    except Exception as e:
        return jsonify({"status": "failed", "error_message": str(e)}), 500


@app.route("/inspect_DICOM", methods=["POST"])
def inspect_dicom():
    """
    Accept a DICOM file, return list of PHI tags found in it.

    Request: multipart/form-data
        file  — the DICOM (.dcm) file

    Response JSON:
        status     "success" | "failed"
        phi_tags   list of {field, value}
        count      int
    """
    try:
        if "file" not in request.files:
            return jsonify({"status": "failed", "error_message": "No 'file' field in request"}), 400

        ds = _read_dicom(request.files["file"])
        tags = inspect(ds)

        return jsonify({"status": "success", "phi_tags": tags, "count": len(tags)})

    except Exception as e:
        return jsonify({"status": "failed", "error_message": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
