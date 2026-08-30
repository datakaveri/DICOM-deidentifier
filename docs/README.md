# DICOM De-Identification Pipeline Documentation

Welcome to the project documentation for the **DICOM Burned-In Text & Metadata De-Identification Pipeline**.

---

## Documentation Index

1. **[Ubuntu WSL2 Setup & Execution Guide](file:///c:/Users/Shashank/OneDrive/Desktop/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset%20(1)/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset/docs/UBUNTU_WSL_SETUP_GUIDE.md)**
   - Complete step-by-step instructions to install and configure Ubuntu Linux via WSL2 on Windows.
   - Native installation commands for PaddlePaddle, PaddleOCR, and AI NLP dependencies.
   - How to navigate Windows paths from Linux and execute the pipeline.

2. **[Pipeline Architecture & Implementation Changelog](file:///c:/Users/Shashank/OneDrive/Desktop/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset%20(1)/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset/docs/PIPELINE_ARCHITECTURE_AND_CHANGELOG.md)**
   - Detailed diagram and explanation of the 7-stage anonymization architecture.
   - Comprehensive file-by-file breakdown of changes made across the repository.
   - PaddleOCR prioritization & single-model verification design.
   - Unified character-stroke inpainting (dropping the border vs. anatomy split).
   - Full batch dataset validation results across all 6 test DICOM files.

---

## Quick Reference Commands

### Run on Local Environment:
```bash
python app/main.py
```

### Run inside Ubuntu (WSL):
```bash
cd "/mnt/c/Users/Shashank/OneDrive/Desktop/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset (1)/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset"
python3 app/main.py
```
