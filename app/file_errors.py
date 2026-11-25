"""
file_errors.py -- the stable error codes a single-file call can fail with.

Added so single_file.process_file can return FileResult(ok=False, error_code=...)
with a code a caller can switch on, instead of a message it would have to
parse. FileError subclasses RuntimeError, so every existing `except Exception`
(the CLI's per-file handler) and every message it prints is unchanged.
"""

NOT_DICOM = "NOT_DICOM"
DECODE_FAILED = "DECODE_FAILED"
UNSUPPORTED_TRANSFER_SYNTAX = "UNSUPPORTED_TRANSFER_SYNTAX"
UNSUPPORTED_COLOUR = "UNSUPPORTED_COLOUR"
NO_PIXEL_DATA = "NO_PIXEL_DATA"
PIXEL_STAGE_FAILED = "PIXEL_STAGE_FAILED"
TAG_STAGE_FAILED = "TAG_STAGE_FAILED"
INVALID_CONFIG = "INVALID_CONFIG"


class FileError(RuntimeError):
    """A per-file failure with a stable `code`."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
