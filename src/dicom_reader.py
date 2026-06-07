"""
dicom_reader.py
---------------
Stage 1 of the anonymization pipeline.

Reads all .dcm files from a folder and groups them into a
    DicomStudy > DicomSeries > DicomInstance
hierarchy, sorted by InstanceNumber within each series.

Pixel data is NOT loaded during the scan pass (stop_before_pixels=True),
so this is fast even for large studies. Pixel data is loaded lazily
when you access instance.dataset.pixel_array later in the pipeline.

Usage
-----
    from dicom_reader import read_and_group, print_study_tree

    study = read_and_group("path/to/dicom_folder")
    print_study_tree(study)          # optional — human-readable summary
"""

import pydicom
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  Data classes — Study > Series > Instance
# ─────────────────────────────────────────────

@dataclass
class DicomInstance:
    """One .dcm file — one image/slice."""
    path: Path
    sop_instance_uid: str
    instance_number: Optional[int]
    # Dataset with metadata only (no pixel data) from the scan pass.
    # Call pydicom.dcmread(instance.path) to get pixel data when needed.
    dataset: pydicom.Dataset = field(repr=False)


@dataclass
class DicomSeries:
    """One acquisition run (e.g. one MRI sequence, one CT stack)."""
    series_uid: str
    modality: str
    series_number: Optional[int]
    series_description: str
    instances: list[DicomInstance] = field(default_factory=list)

    @property
    def first(self) -> DicomInstance:
        """Convenience: first instance (lowest InstanceNumber)."""
        return self.instances[0]


@dataclass
class DicomStudy:
    """
    One clinical imaging encounter — contains one or more series.
    All patient-level metadata is stored here (extracted from the
    first readable file in the folder).
    """
    study_uid: str
    study_date: str
    study_description: str
    patient_name: str
    patient_id: str
    patient_dob: str
    # keys are SeriesInstanceUID strings
    series: dict[str, DicomSeries] = field(default_factory=dict)

    @property
    def all_instances(self) -> list[DicomInstance]:
        """Flat list of every instance across all series."""
        return [inst for s in self.series.values() for inst in s.instances]

    @property
    def total_instances(self) -> int:
        return sum(len(s.instances) for s in self.series.values())


# ─────────────────────────────────────────────
#  Main function
# ─────────────────────────────────────────────

def read_and_group(dicom_dir: str) -> DicomStudy:
    """
    Scan *dicom_dir* for .dcm files and return a populated DicomStudy.

    Parameters
    ----------
    dicom_dir : str
        Path to a flat folder of .dcm files for one patient encounter.
        Nested sub-folders are not scanned — put all files at the top level.

    Returns
    -------
    DicomStudy
        Fully populated study with series grouped by SeriesInstanceUID
        and instances sorted by InstanceNumber within each series.

    Raises
    ------
    ValueError
        If no .dcm files are found, or none could be read successfully.
    """
    dicom_dir = Path(dicom_dir)
    dcm_files = sorted(dicom_dir.glob("*.dcm"))

    if not dcm_files:
        raise ValueError(f"No .dcm files found in: {dicom_dir}")

    study: Optional[DicomStudy] = None
    skipped = 0

    for fpath in dcm_files:
        try:
            # stop_before_pixels=True — fast metadata-only read
            ds = pydicom.dcmread(str(fpath), stop_before_pixels=True, force=True)
        except Exception as e:
            print(f"[WARN] Skipping {fpath.name}: {e}")
            skipped += 1
            continue

        # ── Study level: populate once from the first readable file ──
        study_uid = str(getattr(ds, "StudyInstanceUID", "UNKNOWN"))
        if study is None:
            study = DicomStudy(
                study_uid         = study_uid,
                study_date        = str(getattr(ds, "StudyDate", "")),
                study_description = str(getattr(ds, "StudyDescription", "")),
                patient_name      = str(getattr(ds, "PatientName", "")),
                patient_id        = str(getattr(ds, "PatientID", "")),
                patient_dob       = str(getattr(ds, "PatientBirthDate", "")),
            )

        # ── Series level: create entry on first encounter ──
        series_uid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN"))
        if series_uid not in study.series:
            study.series[series_uid] = DicomSeries(
                series_uid         = series_uid,
                modality           = str(getattr(ds, "Modality", "")),
                series_number      = getattr(ds, "SeriesNumber", None),
                series_description = str(getattr(ds, "SeriesDescription", "")),
            )

        # ── Instance level ──
        instance_number = getattr(ds, "InstanceNumber", None)
        try:
            instance_number = int(instance_number)
        except (TypeError, ValueError):
            instance_number = None

        study.series[series_uid].instances.append(
            DicomInstance(
                path             = fpath,
                sop_instance_uid = str(getattr(ds, "SOPInstanceUID", "")),
                instance_number  = instance_number,
                dataset          = ds,
            )
        )

    if study is None:
        raise ValueError(f"No valid DICOM files could be read in: {dicom_dir}")

    if skipped:
        print(f"[INFO] {skipped} file(s) skipped due to read errors.")

    # Sort instances within each series by InstanceNumber (None sorts last)
    for s in study.series.values():
        s.instances.sort(key=lambda i: (i.instance_number is None, i.instance_number))

    return study


# ─────────────────────────────────────────────
#  Debug helper
# ─────────────────────────────────────────────

def print_study_tree(study: DicomStudy) -> None:
    """Print a human-readable tree of the study structure."""
    print("Study")
    print(f"  PatientName   : {study.patient_name}")
    print(f"  PatientID     : {study.patient_id}")
    print(f"  PatientDOB    : {study.patient_dob}")
    print(f"  StudyDate     : {study.study_date}")
    print(f"  Description   : {study.study_description}")
    print(f"  StudyUID      : {study.study_uid}")
    print(f"  Series count  : {len(study.series)}")
    print(f"  Total files   : {study.total_instances}")
    print()
    for idx, (uid, series) in enumerate(study.series.items(), 1):
        print(f"  Series {idx}: {series.modality or '?'}")
        print(f"    SeriesUID   : {series.series_uid}")
        print(f"    Description : {series.series_description or '(none)'}")
        print(f"    Instances   : {len(series.instances)}")
        for inst in series.instances:
            num = f"{inst.instance_number:>4}" if inst.instance_number is not None else "   ?"
            print(f"      [{num}] {inst.path.name}  SOP: ...{inst.sop_instance_uid[-10:]}")
        print()


# ─────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    study = read_and_group(folder)
    print_study_tree(study)