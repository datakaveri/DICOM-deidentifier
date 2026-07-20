"""
masking.py — Python port of SKALD's Rust `masking` module
(k-anonymisation/SKALD/src/pipeline/preprocess/masking.rs).

Provides the exact same masking engine as the Rust source:
  - RegexPatternKind / RegexPatternConfig / MaskingConfigLite — the
    configuration shapes (a literal regex or a semantic "before"/"after"/
    "in_between" descriptor; per-pattern masking char / length / capture-
    group modes; per-column masking_char, characters_to_mask, apply_order,
    class_masking_mode).
  - parse_masking_config — parses a config entry (dict, mirroring the
    Rust source's JSON object) into a MaskingConfigLite.
  - derive_regex — builds a regex string from a semantic type descriptor.
  - apply_delimiter_length_mask — masks N characters before/after a
    delimiter (position-based, not full-value regex).
  - apply_regex_pattern — applies one RegexPatternConfig to one value.
  - apply_regex_group_mask — full/partial masking of specific capture
    groups within every regex match.
  - apply_masking_value — runs the "characters" / "regex" / "class" steps
    against one value, in the order given by `apply_order`.

Ported line-for-line from the Rust implementation (see masking.rs for the
original comments this file mirrors). The DICOM-specific configs that use
this engine (date masking, free-text PII scrubbing) live in
tag_mapping.py / operations.py, since SKALD itself has no notion of a
DICOM tag — only CSV columns.
"""

import re


# ── Regex pattern kind ────────────────────────────────────────────────────────

class RegexPatternKind:
    """Either a literal regex string, or a semantic descriptor converted to
    a regex at apply time (mirrors the Rust `RegexPatternKind` enum)."""

    LITERAL = "literal"
    DERIVED = "derived"

    def __init__(self, kind, regex=None, pattern_type=None, delimiter=None, start=None, end=None):
        self.kind = kind
        self.regex = regex
        self.pattern_type = pattern_type
        self.delimiter = delimiter
        self.start = start
        self.end = end

    @staticmethod
    def literal(regex: str) -> "RegexPatternKind":
        return RegexPatternKind(RegexPatternKind.LITERAL, regex=regex)

    @staticmethod
    def derived(pattern_type: str = "", delimiter=None, start=None, end=None) -> "RegexPatternKind":
        return RegexPatternKind(RegexPatternKind.DERIVED, pattern_type=pattern_type,
                                 delimiter=delimiter, start=start, end=end)


class RegexPatternConfig:
    """Per-pattern masking configuration within a column's regex_patterns list."""

    def __init__(self, kind: RegexPatternKind, masking_char: str = "*", length=None,
                 mask_groups=None, pattern_type=None, delimiter=None, start=None, end=None):
        self.kind = kind
        self.masking_char = masking_char
        self.length = length
        self.mask_groups = mask_groups or []  # list of (group_idx: int, mode: "full"|"partial")
        # Stored regardless of `kind` so delimiter-length masking fires even
        # when a literal `regex` is also present (matches the Rust source).
        self.pattern_type = pattern_type
        self.delimiter = delimiter
        self.start = start
        self.end = end


class MaskingConfigLite:
    """Full masking configuration for one column/tag."""

    def __init__(self, column: str, masking_char: str = "*", characters_to_mask=None,
                 regex_patterns=None, apply_order=None, class_masking_mode=None,
                 class_letter: str = "X", class_digit: str = "0"):
        self.column = column
        self.masking_char = masking_char
        self.characters_to_mask = characters_to_mask or []  # 1-based positions
        self.regex_patterns = regex_patterns or []
        self.apply_order = apply_order or ["characters", "regex", "class"]
        self.class_masking_mode = class_masking_mode  # "random_class" | "fixed_class" | None
        self.class_letter = class_letter
        self.class_digit = class_digit


# ── Config parsing ────────────────────────────────────────────────────────────

def parse_masking_config(entry: dict) -> MaskingConfigLite:
    """Parses a single masking config entry (dict) into a MaskingConfigLite."""
    if not isinstance(entry, dict):
        raise ValueError("Each masking entry must be an object")

    column = entry.get("column")
    if not column:
        raise ValueError("Masking config missing 'column'")

    masking_char_s = entry.get("masking_char") or entry.get("masking_character") or "*"
    if not masking_char_s:
        raise ValueError(f"masking_char cannot be empty ({column})")
    masking_char = masking_char_s[0]

    characters_to_mask = []
    for pos in entry.get("characters_to_mask", []) or []:
        if not isinstance(pos, int):
            raise ValueError(f"characters_to_mask must be positive integers ({column})")
        if pos > 0:
            characters_to_mask.append(pos)

    regex_patterns = []
    for pat in entry.get("regex_patterns", []) or []:
        if not isinstance(pat, dict):
            raise ValueError(f"Each regex_patterns entry must be an object ({column})")

        pat_masking_char_s = pat.get("masking_char")
        pat_masking_char = pat_masking_char_s[0] if pat_masking_char_s else masking_char

        length = pat.get("length")
        length = max(0, length) if isinstance(length, int) else None

        mask_groups = []
        for k, v in (pat.get("mask_groups") or {}).items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, str):
                mask_groups.append((idx, v))

        pattern_type_val = (pat.get("type") or "").lower() or None
        delimiter_val = pat.get("delimiter")
        start_val = pat.get("start")
        end_val = pat.get("end")

        if pat.get("regex"):
            kind = RegexPatternKind.literal(pat["regex"])
        else:
            kind = RegexPatternKind.derived(pattern_type_val or "", delimiter_val, start_val, end_val)

        regex_patterns.append(RegexPatternConfig(
            kind=kind, masking_char=pat_masking_char, length=length, mask_groups=mask_groups,
            pattern_type=pattern_type_val, delimiter=delimiter_val, start=start_val, end=end_val,
        ))

    apply_order = entry.get("apply_order") or ["characters", "regex", "class"]
    class_masking_mode = entry.get("class_masking_mode")
    class_letter_s = entry.get("class_mask_letter") or "X"
    class_letter = class_letter_s[0] if class_letter_s else "X"
    class_digit_s = entry.get("class_mask_digit") or "0"
    class_digit = class_digit_s[0] if class_digit_s else "0"

    return MaskingConfigLite(
        column=column, masking_char=masking_char, characters_to_mask=characters_to_mask,
        regex_patterns=regex_patterns, apply_order=apply_order,
        class_masking_mode=class_masking_mode, class_letter=class_letter, class_digit=class_digit,
    )


# ── Regex helpers ─────────────────────────────────────────────────────────────

def derive_regex(pattern_type: str, delimiter=None, start=None, end=None) -> str:
    """
    Builds a regex string from a semantic type descriptor — mirrors the
    Rust `derive_regex`. All patterns use capturing group 1 for the text
    to mask (avoids lookbehind, same reasoning as the Rust `regex` crate).
    """
    if pattern_type == "before":
        if delimiter is not None:
            return f"^(.+?){re.escape(delimiter)}"
    elif pattern_type == "after":
        if delimiter is not None:
            return f"{re.escape(delimiter)}(.+)$"
    elif pattern_type == "in_between":
        if start is not None and end is not None:
            if end == "$":
                return f"{re.escape(start)}(.+)$"
            return f"{re.escape(start)}(.+?){re.escape(end)}"
    return ""


def apply_delimiter_length_mask(text: str, pattern_type: str, delimiter: str, length: int, mask_char: str):
    """
    Masks `length` characters immediately before ("before") or after
    ("after") every occurrence of `delimiter` in `text`.
    Returns (masked_text, changed).
    """
    if not delimiter or length == 0:
        return text, False

    chars = list(text)
    changed = False
    search_start = 0

    while True:
        idx = text.find(delimiter, search_start)
        if idx == -1:
            break

        delim_len = len(delimiter)
        if pattern_type == "after":
            s = idx + delim_len
            e = min(s + length, len(chars))
        else:  # "before"
            e = idx
            s = max(0, e - length)

        for i in range(s, e):
            chars[i] = mask_char
            changed = True

        search_start = idx + delim_len
        if search_start >= len(text):
            break

    return "".join(chars), changed


def apply_regex_group_mask(value: str, compiled: "re.Pattern", mask_groups, default_mask: str) -> str:
    """
    Applies group-based masking to all matches of `compiled` in `value`.
    "full" -> replace every character with default_mask.
    "partial" -> keep the first character, replace the rest.
    """
    def _replace(m: "re.Match") -> str:
        rebuilt = m.group(0)
        for group_idx, mode in mask_groups:
            try:
                original = m.group(group_idx)
            except IndexError:
                original = None
            if original is None:
                continue
            if mode == "full":
                replacement = default_mask * len(original)
            elif mode == "partial":
                if len(original) <= 1:
                    replacement = original
                else:
                    replacement = original[0] + default_mask * (len(original) - 1)
            else:
                replacement = original
            rebuilt = rebuilt.replace(original, replacement, 1)
        return rebuilt

    return compiled.sub(_replace, value)


def apply_regex_pattern(value: str, pat: RegexPatternConfig, column: str) -> str:
    """
    Applies a single RegexPatternConfig to one string value:
      1. Delimiter-length masking (fires regardless of kind, using the
         stored semantic fields) if it produces a change.
      2. Build + compile the regex (literal or derived).
      3. Apply — explicit group masking, auto group-1 for derived
         patterns, or full-match masking.
    """
    mask_char = pat.masking_char

    if pat.length is not None:
        pattern_type = pat.pattern_type or ""
        delimiter = pat.delimiter or ""
        if delimiter and pattern_type in ("before", "after"):
            result, changed = apply_delimiter_length_mask(value, pattern_type, delimiter, pat.length, mask_char)
            if changed:
                return result

    is_derived = pat.kind.kind == RegexPatternKind.DERIVED
    if pat.kind.kind == RegexPatternKind.LITERAL:
        regex_str = pat.kind.regex or ""
    else:
        regex_str = derive_regex(pat.kind.pattern_type, pat.kind.delimiter, pat.kind.start, pat.kind.end)

    if not regex_str:
        return value

    try:
        compiled = re.compile(regex_str)
    except re.error:
        derived = derive_regex(pat.pattern_type or "", pat.delimiter, pat.start, pat.end)
        if not derived:
            return value
        try:
            compiled = re.compile(derived)
            is_derived = True
        except re.error:
            return value

    if pat.mask_groups:
        return apply_regex_group_mask(value, compiled, pat.mask_groups, mask_char)
    if is_derived:
        return apply_regex_group_mask(value, compiled, [(1, "full")], mask_char)

    # Full-match masking: replace each match span with mask_char * match length
    out = []
    last = 0
    for m in compiled.finditer(value):
        out.append(value[last:m.start()])
        out.append(mask_char * len(m.group(0)))
        last = m.end()
    out.append(value[last:])
    return "".join(out)


def apply_masking_value(value: str, cfg: MaskingConfigLite, randomize_fn) -> str:
    """
    Applies all masking steps configured in `cfg` to `value`, in the order
    given by `cfg.apply_order`. `randomize_fn` implements class-preserving
    randomization for the "random_class" class-masking mode (pass
    crypto.randomize_preserving_class).
    """
    masked = value

    for step in cfg.apply_order:
        if step == "characters" and cfg.characters_to_mask:
            chars = list(masked)
            for pos in cfg.characters_to_mask:
                idx = pos - 1
                if 0 <= idx < len(chars):
                    chars[idx] = cfg.masking_char
            masked = "".join(chars)

        elif step == "regex" and cfg.regex_patterns:
            for pat in cfg.regex_patterns:
                masked = apply_regex_pattern(masked, pat, cfg.column)

        elif step == "class":
            if cfg.class_masking_mode == "random_class":
                masked = randomize_fn(masked)
            elif cfg.class_masking_mode == "fixed_class":
                out = []
                for c in masked:
                    if c.isdigit():
                        out.append(cfg.class_digit)
                    elif c.isalpha():
                        out.append(cfg.class_letter)
                    else:
                        out.append(c)
                masked = "".join(out)

    return masked
