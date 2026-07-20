"""
tag_mapping.py — DICOM Standard Tags -> Anonymization Technique Mapping,
transcribed from "DICOM Tag -> Anonymization Technique Mapping (SKALD).pdf"
(PS3.15 Annex E Basic Confidentiality Profile plus common identifying
elements, mapped to SKALD preprocessing techniques).

Each entry is keyed by (group, element) as ints and describes:
  technique — the primary technique applied (see operations.py for the
              executable implementation of each):
                suppress    | element dropped entirely
                hash        | SHA-256 (or, for UI-VR tags, a deterministic
                              DICOM-valid pseudo-UID derived from SHA-256)
                tokenise    | sequential opaque token via a reversible vault
                encrypt     | pseudo-encryption, XOR keystream ("ENC$" hex)
                encrypt_fpe | format-preserving encryption (keeps length/charset)
                charcloak   | per-character random substitution, same class
                mask        | partial redaction (dates: blank the day)
                scrub       | "Remove PII and retain": regex-redact PII
                              spans in free text, keep the rest
                retain      | no transformation
  alt        — the PDF's parenthetical/slash-separated alternative
               technique, kept for documentation (not auto-applied).
  reason     — the PDF's stated rationale, kept for audit/documentation.

Tags not present in this table are left untouched by deidentify.py's
default policy EXCEPT private tags (odd group numbers), which are always
suppressed per the PDF's private-tag policy note.
"""

# technique constants
SUPPRESS    = "suppress"
HASH        = "hash"
TOKENISE    = "tokenise"
ENCRYPT     = "encrypt"
ENCRYPT_FPE = "encrypt_fpe"
CHARCLOAK   = "charcloak"
MASK        = "mask"
SCRUB       = "scrub"
RETAIN      = "retain"

TAG_MAPPING = {
    # 1. Patient identity & demographics
    (0x0010, 0x0010): {"name": "Patient's Name", "technique": SUPPRESS,
                        "reason": "Direct identifier; never required for analysis."},
    (0x0010, 0x0020): {"name": "Patient ID", "technique": HASH, "alt": TOKENISE,
                        "reason": "Primary linkage key (e.g. UHID). Hash for a stable, "
                                  "non-reversible pseudonym; tokenise if a controlled "
                                  "re-link to source is required."},
    (0x0010, 0x0021): {"name": "Issuer of Patient ID", "technique": SUPPRESS,
                        "reason": "Names the issuing authority/site; identifying and not needed."},
    (0x0010, 0x0024): {"name": "Issuer of Patient ID Qualifiers Sequence", "technique": SUPPRESS,
                        "reason": "Qualifies the issuer; identifying context."},
    (0x0010, 0x1000): {"name": "Other Patient IDs", "technique": HASH,
                        "reason": "Secondary identifiers (ABHA, MRN); pseudonymise to "
                                  "preserve any linkage value."},
    (0x0010, 0x1001): {"name": "Other Patient Names", "technique": SUPPRESS,
                        "reason": "Alias names; direct identifiers."},
    (0x0010, 0x1002): {"name": "Other Patient IDs Sequence", "technique": HASH,
                        "reason": "Structured form of secondary IDs; same treatment as "
                                  "Other Patient IDs."},
    (0x0010, 0x0030): {"name": "Patient's Birth Date", "technique": MASK,
                        "reason": "Quasi-identifier; retain year, blank month/day to "
                                  "preserve age cohort while removing exact DOB."},
    (0x0010, 0x0032): {"name": "Patient's Birth Time", "technique": SUPPRESS,
                        "reason": "No analytic value; only adds re-identification surface."},
    (0x0010, 0x0040): {"name": "Patient's Sex", "technique": RETAIN,
                        "reason": "Clinically required; low identifying power on its own."},
    (0x0010, 0x1010): {"name": "Patient's Age", "technique": RETAIN,
                        "reason": "Quasi-identifier; coarsen to age band to limit re-identification."},
    (0x0010, 0x1020): {"name": "Patient's Size", "technique": RETAIN,
                        "reason": "Clinical measurement; not identifying."},
    (0x0010, 0x1030): {"name": "Patient's Weight", "technique": RETAIN,
                        "reason": "Clinical measurement; not identifying."},
    (0x0010, 0x1040): {"name": "Patient's Address", "technique": SUPPRESS,
                        "reason": "Direct identifier."},
    (0x0010, 0x2154): {"name": "Patient's Telephone Numbers", "technique": SUPPRESS,
                        "reason": "Direct identifier."},
    (0x0010, 0x1060): {"name": "Patient's Mother's Birth Name", "technique": SUPPRESS,
                        "reason": "Family identifier."},
    (0x0010, 0x2160): {"name": "Ethnic Group", "technique": SUPPRESS,
                        "reason": "Sensitive attribute; rarely needed, high re-identification and harm risk."},
    (0x0010, 0x2180): {"name": "Occupation", "technique": SUPPRESS,
                        "reason": "Quasi-identifier free text."},
    (0x0010, 0x1080): {"name": "Military Rank", "technique": SUPPRESS,
                        "reason": "Quasi-identifier."},
    (0x0010, 0x1081): {"name": "Branch of Service", "technique": SUPPRESS,
                        "reason": "Quasi-identifier."},
    (0x0010, 0x2150): {"name": "Country of Residence", "technique": SUPPRESS,
                        "reason": "Geographic quasi-identifier."},
    (0x0010, 0x2152): {"name": "Region of Residence", "technique": SUPPRESS,
                        "reason": "Geographic quasi-identifier (finer than country)."},
    (0x0010, 0x21F0): {"name": "Patient's Religious Preference", "technique": SUPPRESS,
                        "reason": "Sensitive attribute; not needed."},
    (0x0010, 0x2000): {"name": "Medical Alerts", "technique": SUPPRESS,
                        "reason": "Free text may embed identifiers; low analytic value."},
    (0x0010, 0x2110): {"name": "Allergies", "technique": SCRUB,
                        "reason": "Free text; redact PII, prefer NLP free-text anonymisation upstream."},
    (0x0010, 0x21B0): {"name": "Additional Patient History", "technique": SCRUB,
                        "reason": "Free text likely to contain PII; anonymise as free text."},
    (0x0010, 0x21D0): {"name": "Last Menstrual Date", "technique": MASK,
                        "reason": "Date quasi-identifier; retain year/month, blank day if clinically needed."},
    (0x0010, 0x4000): {"name": "Patient Comments", "technique": SUPPRESS,
                        "reason": "Free text; high PII risk, no structured value."},
    (0x0010, 0x1090): {"name": "Medical Record Locator", "technique": SUPPRESS,
                        "reason": "Points to the source record; identifying."},

    # 2. Physicians, operators & institutions
    (0x0008, 0x0090): {"name": "Referring Physician's Name", "technique": SUPPRESS,
                        "reason": "Staff identifier; indirectly identifies patient via care team."},
    (0x0008, 0x0092): {"name": "Referring Physician's Address", "technique": SUPPRESS,
                        "reason": "Direct identifier."},
    (0x0008, 0x0094): {"name": "Referring Physician's Telephone Numbers", "technique": SUPPRESS,
                        "reason": "Direct identifier."},
    (0x0008, 0x0096): {"name": "Referring Physician Identification Sequence", "technique": SUPPRESS,
                        "reason": "Structured staff identifiers."},
    (0x0008, 0x1048): {"name": "Physician(s) of Record", "technique": SUPPRESS,
                        "reason": "Staff identifier."},
    (0x0008, 0x1050): {"name": "Performing Physician's Name", "technique": SUPPRESS,
                        "reason": "Staff identifier."},
    (0x0008, 0x1060): {"name": "Name of Physician(s) Reading Study", "technique": SUPPRESS,
                        "reason": "Radiologist name; staff identifier."},
    (0x0008, 0x1070): {"name": "Operators' Name", "technique": SUPPRESS,
                        "reason": "Technician name; staff identifier."},
    (0x0008, 0x1072): {"name": "Operator Identification Sequence", "technique": SUPPRESS,
                        "reason": "Structured operator identifiers."},
    (0x0032, 0x1032): {"name": "Requesting Physician", "technique": SUPPRESS,
                        "reason": "Staff identifier."},
    (0x0040, 0x0006): {"name": "Scheduled Performing Physician's Name", "technique": SUPPRESS,
                        "reason": "Staff identifier."},
    (0x0040, 0x1010): {"name": "Names of Intended Recipients of Results", "technique": SUPPRESS,
                        "reason": "Third-party identifiers."},
    (0x0008, 0x0080): {"name": "Institution Name", "technique": SUPPRESS,
                        "reason": "Facility identifier; enables geographic re-identification."},
    (0x0008, 0x0081): {"name": "Institution Address", "technique": SUPPRESS,
                        "reason": "Geographic identifier."},
    (0x0008, 0x0082): {"name": "Institution Code Sequence", "technique": SUPPRESS,
                        "reason": "Coded facility identifier."},
    (0x0008, 0x1010): {"name": "Station Name", "technique": HASH,
                        "reason": "Identifies acquisition workstation/site; hash if device "
                                  "grouping is analytically useful, else suppress."},
    (0x0008, 0x1040): {"name": "Institutional Department Name", "technique": RETAIN,
                        "reason": "Department/ward retained; useful, low identifying power."},

    # 3. Study / series / instance identifiers
    (0x0020, 0x000D): {"name": "Study Instance UID", "technique": HASH,
                        "reason": "Must be remapped consistently so all objects of a study "
                                  "stay linked; deterministic hash preserves referential "
                                  "integrity while breaking the tie to source UIDs."},
    (0x0020, 0x000E): {"name": "Series Instance UID", "technique": HASH,
                        "reason": "Same rationale; deterministic remap keeps series grouping intact."},
    (0x0008, 0x0018): {"name": "SOP Instance UID", "technique": HASH,
                        "reason": "Per-object UID; deterministic remap maintains cross-reference consistency."},
    (0x0020, 0x0052): {"name": "Frame of Reference UID", "technique": HASH,
                        "reason": "Spatial linkage across series; must stay internally consistent after remap."},
    (0x0008, 0x0058): {"name": "Failed SOP Instance UID List", "technique": HASH,
                        "reason": "References other instances; remap consistently."},
    (0x0008, 0x0016): {"name": "SOP Class UID", "technique": RETAIN,
                        "reason": "Identifies object type, not the patient; required to interpret the object."},
    (0x0020, 0x0010): {"name": "Study ID", "technique": HASH, "alt": TOKENISE,
                        "reason": "RIS-generated study key; hash for pseudonym, tokenise if "
                                  "reversal to source is needed."},
    (0x0008, 0x0050): {"name": "Accession Number", "technique": TOKENISE, "alt": ENCRYPT_FPE,
                        "reason": "Order key often reconciled with RIS later; tokenise for a "
                                  "reversible vault link, or FPE to keep the fixed-width format "
                                  "while remaining reversible."},
    (0x0040, 0x1001): {"name": "Requested Procedure ID", "technique": HASH,
                        "reason": "Order identifier; pseudonymise."},
    (0x0040, 0x0009): {"name": "Scheduled Procedure Step ID", "technique": HASH,
                        "reason": "Workflow identifier."},
    (0x0040, 0x0253): {"name": "Performed Procedure Step ID", "technique": HASH,
                        "reason": "Workflow identifier."},
    (0x0040, 0x2016): {"name": "Placer Order Number/Imaging Service Request", "technique": HASH, "alt": TOKENISE,
                        "reason": "Order-system key; pseudonymise, tokenise if reconciliation needed."},
    (0x0040, 0x2017): {"name": "Filler Order Number/Imaging Service Request", "technique": HASH, "alt": TOKENISE,
                        "reason": "Order-system key; same treatment."},
    (0x0088, 0x0140): {"name": "Storage Media File-set UID", "technique": HASH,
                        "reason": "UID referencing media; remap consistently."},

    # 4. Dates & times
    (0x0008, 0x0020): {"name": "Study Date", "technique": MASK,
                        "reason": "Retain year (and month if needed), blank day; temporal "
                                  "cohort is analytically useful."},
    (0x0008, 0x0021): {"name": "Series Date", "technique": MASK,
                        "reason": "Same rationale; keep consistent masking across a study."},
    (0x0008, 0x0022): {"name": "Acquisition Date", "technique": MASK, "reason": "Same rationale."},
    (0x0008, 0x0023): {"name": "Content Date", "technique": MASK, "reason": "Same rationale."},
    (0x0008, 0x0030): {"name": "Study Time", "technique": SUPPRESS,
                        "reason": "Time-of-day is a re-identification vector with little analytic value."},
    (0x0008, 0x0031): {"name": "Series Time", "technique": SUPPRESS, "reason": "Same rationale."},
    (0x0008, 0x0032): {"name": "Acquisition Time", "technique": SUPPRESS, "reason": "Same rationale."},
    (0x0008, 0x0033): {"name": "Content Time", "technique": SUPPRESS, "reason": "Same rationale."},
    (0x0038, 0x0020): {"name": "Admitting Date", "technique": MASK, "reason": "Date quasi-identifier; coarsen."},
    (0x0038, 0x0021): {"name": "Admitting Time", "technique": SUPPRESS,
                        "reason": "Time-of-day re-identification vector."},

    # 5. Visit / admission / order workflow (RIS)
    (0x0038, 0x0010): {"name": "Admission ID", "technique": HASH,
                        "reason": "Encounter identifier; pseudonymise (needed only if joining encounters)."},
    (0x0038, 0x0011): {"name": "Issuer of Admission ID", "technique": SUPPRESS,
                        "reason": "Names issuing facility; identifying."},
    (0x0038, 0x0060): {"name": "Service Episode ID", "technique": HASH,
                        "reason": "Episode linkage key; pseudonymise."},
    (0x0038, 0x0300): {"name": "Current Patient Location", "technique": SUPPRESS,
                        "reason": "Ward/room; geographic quasi-identifier."},
    (0x0038, 0x0400): {"name": "Patient's Institution Residence", "technique": SUPPRESS,
                        "reason": "Residential facility; identifying."},
    (0x0038, 0x0500): {"name": "Patient State", "technique": SUPPRESS,
                        "reason": "Free text status; possible PII."},
    (0x0038, 0x4000): {"name": "Visit Comments", "technique": CHARCLOAK, "alt": SUPPRESS,
                        "reason": "Free text; anonymise or drop."},
    (0x0032, 0x1060): {"name": "Requested Procedure Description", "technique": RETAIN,
                        "reason": "Clinical descriptor; retain unless free text embeds PII (then charcloak)."},
    (0x0040, 0x0254): {"name": "Performed Procedure Step Description", "technique": RETAIN,
                        "reason": "Clinical descriptor; retain."},
    (0x0040, 0x0275): {"name": "Request Attributes Sequence", "technique": SUPPRESS,
                        "reason": "Nested order/accession identifiers."},
    (0x0040, 0x0241): {"name": "Performed Station AE Title", "technique": SUPPRESS,
                        "reason": "Network node identifier tied to site."},
    (0x0040, 0x3001): {"name": "Confidentiality Constraint on Patient Data Description", "technique": SUPPRESS,
                        "reason": "Free text confidentiality note; may name the patient."},

    # 6. Equipment & device
    (0x0018, 0x1000): {"name": "Device Serial Number", "technique": SUPPRESS,
                        "reason": "Uniquely identifies the machine/site."},
    (0x0018, 0x1002): {"name": "Device UID", "technique": HASH,
                        "reason": "Device identifier; hash if device grouping is useful, else suppress."},
    (0x0018, 0x700A): {"name": "Detector ID", "technique": HASH, "alt": SUPPRESS,
                        "reason": "Component identifier tied to a specific unit."},
    (0x0008, 0x0070): {"name": "Manufacturer", "technique": RETAIN,
                        "reason": "Needed for ML workloads; not identifying."},
    (0x0008, 0x1090): {"name": "Manufacturer's Model Name", "technique": RETAIN,
                        "reason": "Useful for model quality; not identifying."},
    (0x0018, 0x1020): {"name": "Software Versions", "technique": RETAIN,
                        "reason": "Technical metadata; not identifying."},
    (0x0018, 0x1030): {"name": "Protocol Name", "technique": RETAIN,
                        "reason": "Acquisition protocol; clinically useful, not identifying."},

    # 7. Free-text, comments & report content
    (0x0008, 0x1030): {"name": "Study Description", "technique": RETAIN,
                        "reason": "Usually a clinical code; retain (review if free text embeds names/IDs)."},
    (0x0008, 0x103E): {"name": "Series Description", "technique": RETAIN, "reason": "Same rationale."},
    (0x0008, 0x2111): {"name": "Derivation Description", "technique": CHARCLOAK, "alt": SUPPRESS,
                        "reason": "Free text describing processing; may embed identifiers."},
    (0x0020, 0x4000): {"name": "Image Comments", "technique": SCRUB, "reason": "Free text; high PII risk."},
    (0x0040, 0xA160): {"name": "Text Value (SR content)", "technique": SUPPRESS,
                        "reason": "Structured-report narrative; run free-text anonymisation, drop if not needed."},
    (0x0040, 0xA123): {"name": "Person Name (SR content)", "technique": SUPPRESS,
                        "reason": "Names embedded in report content."},
    (0x4008, 0x010C): {"name": "Interpretation Author", "technique": SUPPRESS,
                        "reason": "Radiologist name (retired attribute, still seen)."},
    (0x4008, 0x0114): {"name": "Physician Approving Interpretation", "technique": SUPPRESS,
                        "reason": "Staff name."},
    (0x4008, 0x0119): {"name": "Distribution Name", "technique": SUPPRESS, "reason": "Recipient name."},

    # 8. Other identifying references
    (0x0020, 0x0200): {"name": "Synchronization Frame of Reference UID", "technique": HASH,
                        "reason": "Cross-object reference; remap consistently."},
    (0x0040, 0xA124): {"name": "UID (SR content)", "technique": HASH,
                        "reason": "Referenced UID in report content; keep consistent with remapped instance UIDs."},
    (0x0400, 0x0561): {"name": "Original Attributes Sequence", "technique": SUPPRESS,
                        "reason": "Records pre-anonymisation original values — must be removed "
                                  "or it leaks everything just anonymised."},
    (0x0012, 0x0010): {"name": "Clinical Trial Sponsor Name", "technique": SUPPRESS,
                        "reason": "Study/organisation identifier."},
    (0x0012, 0x0020): {"name": "Clinical Trial Protocol ID", "technique": HASH,
                        "reason": "Protocol linkage key; pseudonymise or drop."},
    (0x0012, 0x0040): {"name": "Clinical Trial Subject ID", "technique": HASH,
                        "reason": "Subject linkage key; pseudonymise."},
}

# UID-VR tags that must remain syntactically valid DICOM UIDs after hashing
# (dotted numeric string, <=64 chars) rather than becoming a raw hex digest.
UID_VALUED_TAGS = {
    (0x0020, 0x000D),  # Study Instance UID
    (0x0020, 0x000E),  # Series Instance UID
    (0x0008, 0x0018),  # SOP Instance UID
    (0x0020, 0x0052),  # Frame of Reference UID
    (0x0008, 0x0058),  # Failed SOP Instance UID List
    (0x0088, 0x0140),  # Storage Media File-set UID
    (0x0018, 0x1002),  # Device UID
    (0x0020, 0x0200),  # Synchronization Frame of Reference UID
    (0x0040, 0xA124),  # UID (SR content)
}
