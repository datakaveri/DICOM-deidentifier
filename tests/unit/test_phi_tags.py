import pydicom
from pydicom.dataset import Dataset
from phi_tags import _is_id_field, identify_phi_tags


def test_is_id_field():
    assert _is_id_field("PatientID") is True
    assert _is_id_field("StudyID") is True
    assert _is_id_field("SOPInstanceUID") is False
    assert _is_id_field("SeriesInstanceUID") is False
    assert _is_id_field("") is False


def test_identify_phi_tags():
    ds = Dataset()
    ds.PatientName = "Doe^John"
    ds.PatientID = "123456"
    ds.PatientAge = "045Y"
    ds.StudyDate = "20260101"
    ds.StudyTime = "120000"
    ds.AccessionNumber = "ACC9988"
    ds.Modality = "CR"

    phi = identify_phi_tags(ds)
    fields = [item["field"] for item in phi]

    assert "PatientName" in fields
    assert "PatientID" in fields
    assert "PatientAge" in fields
    assert "StudyDate" in fields
    assert "StudyTime" in fields
    assert "AccessionNumber" in fields
    assert "Modality" not in fields
