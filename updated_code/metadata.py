"""
metadata.py — Stage 1: metadata sanitization.

sanitize_metadata() and METADATA_TAGS_TO_CLEAR below are taken as-is from
the existing de-identification pipeline (DICOM-deidentifier/dicom_anonymizer_pipeline.py),
not reimplemented — this just wires that already-written logic in here.
"""

from pydicom.uid import generate_uid

from config import log

# Metadata tags to clear (DICOM de-identification profile)
METADATA_TAGS_TO_CLEAR = [
    'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
    'PatientAge', 'PatientAddress', 'PatientTelephoneNumbers',
    'PatientMotherBirthName', 'PatientBirthName',
    'ReferringPhysicianName', 'ReferringPhysicianAddress',
    'InstitutionName', 'InstitutionAddress', 'InstitutionalDepartmentName',
    'StationName', 'AccessionNumber', 'StudyID',
    'StudyDate', 'SeriesDate', 'AcquisitionDate', 'ContentDate',
    'StudyTime', 'SeriesTime', 'AcquisitionTime', 'ContentTime',
    'PhysiciansOfRecord', 'PerformingPhysicianName',
    'NameOfPhysiciansReadingStudy', 'OperatorsName',
    'AdmittingDiagnosesDescription', 'PatientWeight',
    'RequestingPhysician', 'RequestedProcedureDescription',
    'ScheduledPerformingPhysicianName', 'RequestedProcedureID',
]


def sanitize_metadata(ds):
    """Clears all PHI DICOM metadata tags and re-generates UIDs."""
    log.info("  [Stage 1] Sanitizing DICOM metadata tags...")

    for attr in METADATA_TAGS_TO_CLEAR:
        if hasattr(ds, attr):
            if attr == 'PatientName':
                ds.PatientName = ""
            elif attr == 'PatientID':
                ds.PatientID = ""
            else:
                try:
                    setattr(ds, attr, "")
                except Exception:
                    pass

    # Re-generate UIDs so this file cannot be linked to the original study
    for uid_attr in ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']:
        if hasattr(ds, uid_attr):
            setattr(ds, uid_attr, generate_uid())

    # Remove all private / vendor tags (group numbers are odd)
    ds.remove_private_tags()

    log.info("  [Stage 1] Done. All PHI metadata cleared.")
    return ds
