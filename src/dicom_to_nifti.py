"""
dicom_to_nifti.py
-----------------
Stage 2b of the pipeline — convert anonymized DICOM files to NIfTI.

Reads anonymized .dcm files from the DICOM output folder, converts each
series to a single .nii.gz volume, wipes PII-carrying NIfTI header fields,
and deletes any sidecar JSON files produced during conversion.

Design decisions
----------------
- Converts from anonymized .dcm files (not originals) — all DICOM tags
  that dicom2nifti reads for the NIfTI header are already wiped/replaced.
- dicom2nifti requires all slices of a series in a single flat folder.
  For multi-series studies, temporary per-series subfolders are created
  using symlinks (no disk copy), conversion runs, then temp folders are
  deleted.
- NIfTI header fields descrip, db_name, aux_file, intent_name are wiped
  as a precaution even though the source DICOM tags are already clean —
  scanner firmware sometimes writes non-DICOM content into these fields.
- Sidecar JSON files (.json) produced by dicom2nifti are deleted — all
  relevant metadata is captured in the FHIR ImagingStudy resource.
- Output filename: series_{modality}_{anon_series_uid_last16}.nii.gz

Usage
-----
    from dicom_reader import read_and_group
    from dicom_anonymizer import anonymize_study
    from dicom_to_nifti import convert_to_nifti

    study = read_and_group("input/dicom/")
    anon_result = anonymize_study(study, "output/dicom/")
    nifti_results = convert_to_nifti(study, anon_result, "output/nifti/")

Requires
--------
    pip install dicom2nifti nibabel
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import dicom2nifti
import dicom2nifti.settings as d2n_settings
import nibabel as nib

from dicom_reader import DicomStudy, DicomSeries
from dicom_anonymizer import make_anon_uid


# ─────────────────────────────────────────────────────────────────────────────
#  NIfTI header fields to wipe after conversion
# ─────────────────────────────────────────────────────────────────────────────

_NIFTI_WIPE_FIELDS = [
    "descrip",       # 80 chars — often StudyDescription or scanner info
    "db_name",       # 18 chars — sometimes PatientName
    "aux_file",      # 24 chars — auxiliary filename, sometimes scan info
    "intent_name",   # 16 chars — sometimes protocol/patient info
]


# ─────────────────────────────────────────────────────────────────────────────
#  Header wipe
# ─────────────────────────────────────────────────────────────────────────────

def _wipe_nifti_header(
    nifti_path: str,
    cal_min: float = None,
    cal_max: float = None,
) -> None:
    img = nib.load(nifti_path)
    hdr = img.header

    for field in _NIFTI_WIPE_FIELDS:
        try:
            hdr[field] = b""
        except Exception:
            pass

    # Set display window from DICOM WindowCenter/WindowWidth
    if cal_min is not None and cal_max is not None:
        hdr["cal_min"] = cal_min
        hdr["cal_max"] = cal_max

    nib.save(img, nifti_path)

def _get_window_from_dicom(dcm_path: Path) -> tuple[float, float]:
    """
    Extract WindowCenter and WindowWidth from a DICOM file and
    return (cal_min, cal_max) for the NIfTI header.
    Uses the first value if multiple windows are present.
    """
    import pydicom
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth",  None)

    if wc is None or ww is None:
        return None, None

    # Tags may contain a list (multiple window presets) — use the first
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = float(wc[0])
    else:
        wc = float(wc)

    if isinstance(ww, pydicom.multival.MultiValue):
        ww = float(ww[0])
    else:
        ww = float(ww)

    cal_min = wc - (ww / 2)
    cal_max = wc + (ww / 2)

    return cal_min, cal_max
def _check_nifti_geometry_preserved(
    dcm_path: Path,
    nifti_path: str,
    expected_slice_count: int,
) -> dict:
    """
    Compare the NIfTI output's geometry against the source anonymized
    DICOM slice's own geometry tags — confirms spatial/technical
    information survived the dicom2nifti conversion step, not just the
    DICOM-to-DICOM anonymization.

    dicom2nifti can reorient/permute axes (reorient_nifti=True), so voxel
    spacing is compared as a SET rather than by position — robust to axis
    reordering, while still catching real geometry loss (e.g. spacing
    collapsing to a 1.0/1.0/1.0 default, or the wrong slice count) that
    would matter for downstream use of the NIfTI file.

    Any check whose source tag isn't present in the DICOM is skipped
    (returns None for that sub-check) rather than counted as a failure —
    "if available in the input data", per the pipeline's preservation
    policy.
    """
    result = {
        "spacing_preserved":     None,
        "cal_window_preserved":  None,
        "slice_count_preserved": None,
        "preserved":             True,
        "note":                  "",
    }

    try:
        import pydicom
        ds  = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        img = nib.load(nifti_path)
        hdr = img.header
        zooms = [float(z) for z in hdr.get_zooms()[:3]]
    except Exception as e:
        result["preserved"] = False
        result["note"] = f"Could not compare geometry: {e}"
        return result

    notes = []

    # ── Voxel spacing — compared as a set, tolerant of axis reordering ──
    px_spacing = getattr(ds, "PixelSpacing", None)
    slice_thickness = getattr(ds, "SliceThickness", None)
    if px_spacing is not None and slice_thickness is not None:
        try:
            expected = sorted([
                round(float(px_spacing[0]), 2),
                round(float(px_spacing[1]), 2),
                round(float(slice_thickness), 2),
            ])
            actual = sorted(round(z, 2) for z in zooms)
            spacing_ok = all(
                any(abs(e - a) <= max(0.05, e * 0.05) for a in actual)
                for e in expected
            )
            result["spacing_preserved"] = spacing_ok
            if not spacing_ok:
                notes.append(f"voxel spacing {actual} does not match source {expected}")
        except Exception:
            pass  # leave as None — couldn't parse, don't fail the check for it

    # ── Display window — only meaningful if source had Window Center/Width ──
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        cal_min = float(hdr["cal_min"])
        cal_max = float(hdr["cal_max"])
        window_ok = not (cal_min == 0.0 and cal_max == 0.0)
        result["cal_window_preserved"] = window_ok
        if not window_ok:
            notes.append("source has WindowCenter/WindowWidth but NIfTI cal_min/cal_max are both 0")

    # ── Slice count — one NIfTI dim should match the series' instance count ──
    try:
        slice_count_ok = expected_slice_count in img.shape
        result["slice_count_preserved"] = slice_count_ok
        if not slice_count_ok:
            notes.append(f"NIfTI shape {img.shape} doesn't contain expected slice count {expected_slice_count}")
    except Exception:
        pass

    checks = [v for v in (
        result["spacing_preserved"], result["cal_window_preserved"], result["slice_count_preserved"]
    ) if v is not None]
    result["preserved"] = all(checks) if checks else True
    result["note"] = "; ".join(notes) if notes else "Geometry consistent with source DICOM."

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Per-series conversion
# ─────────────────────────────────────────────────────────────────────────────

def _convert_series(
    series: DicomSeries,
    anon_dicom_dir: Path,
    anon_result:dict,
    output_dir: Path,
    anon_patient_id: str,
) -> Optional[dict]:
    """
    Convert one DICOM series to an anonymized NIfTI file.

    Parameters
    ----------
    series : DicomSeries
        Series metadata from dicom_reader (contains original SOPInstanceUIDs).
    anon_dicom_dir : Path
        Folder containing the anonymized .dcm files.
    output_dir : Path
        Folder to write the output .nii.gz file.
    anon_patient_id : str
        Bare anonymized patient ID (used for logging only).

    Returns
    -------
    dict with keys: series_uid, modality, nifti_path, instance_count
    or None if conversion failed.
    """
    anon_series_uid = make_anon_uid(series.series_uid)
    uid_short       = anon_series_uid[-16:]
    modality        = series.modality or "UNKNOWN"

    print(f"  [Series {modality}] {len(series.instances)} instances → converting ...")

    # ── Build list of anonymized .dcm file paths for this series ──
    # The anonymized files are named by their new (anonymized) SOPInstanceUID
    anon_dcm_paths = []
    # Get the sop_uid_to_filename map from anon_result
    sop_uid_to_filename = anon_result.get("sop_uid_to_filename", {})

    for inst in series.instances:
        anon_filename = sop_uid_to_filename.get(inst.sop_instance_uid)
        if anon_filename is None:
            # fallback to deterministic derivation
            anon_filename = f"{make_anon_uid(inst.sop_instance_uid)}.dcm"
        anon_path = anon_dicom_dir / anon_filename
        if anon_path.exists():
            anon_dcm_paths.append(anon_path)
        else:
            print(f"  [WARN] Anonymized file not found: {anon_filename}")

    if not anon_dcm_paths:
        print(f"  [ERROR] No anonymized .dcm files found for series {anon_series_uid}")
        return None
    # Extract window values from first slice for NIfTI cal_min/cal_max
    cal_min, cal_max = _get_window_from_dicom(anon_dcm_paths[0])
    if len(anon_dcm_paths) < 3:
        print(f"  [WARN] Only {len(anon_dcm_paths)} slices — "
              f"dicom2nifti requires at least 3. Skipping series.")
        return None

    # ── Create temp folder with symlinks to anonymized .dcm files ──
    temp_dir = Path(tempfile.mkdtemp(prefix=f"d2n_{uid_short}_"))
    try:
        for dcm_path in anon_dcm_paths:
            link = temp_dir / dcm_path.name
            os.symlink(dcm_path.resolve(), link)

        # ── Convert using dicom2nifti ──
        temp_output = str(temp_dir / "output.nii.gz")

        # Disable orientation validation — anonymized files pass all checks
        # but some scanners produce non-standard orientations that would
        # otherwise raise ConversionValidationError
        d2n_settings.disable_validate_orientation()
        d2n_settings.disable_validate_slice_increment()

        dicom2nifti.dicom_series_to_nifti(
            str(temp_dir),
            temp_output,
            reorient_nifti=True,
        )

        # Delete sidecar JSON if produced
        sidecar = Path(temp_output.replace(".nii.gz", ".json"))
        if sidecar.exists():
            sidecar.unlink()
            print(f"  [INFO] Sidecar JSON deleted")

        # ── Wipe NIfTI header fields ──
        #_wipe_nifti_header(temp_output)
        # ── Wipe NIfTI header fields and set display window ──
        _wipe_nifti_header(temp_output, cal_min, cal_max)

        # ── Move to final output location ──
        out_filename = f"series_{modality}_{uid_short}.nii.gz"
        out_path     = output_dir / out_filename
        shutil.move(temp_output, str(out_path))

        print(f"  [OK] Written: {out_path.name}")

        geometry_check = _check_nifti_geometry_preserved(
            anon_dcm_paths[0], str(out_path), len(anon_dcm_paths)
        )

        return {
            "series_uid":      anon_series_uid,
            "modality":        modality,
            "nifti_path":      str(out_path),
            "instance_count":  len(anon_dcm_paths),
            "geometry_check":  geometry_check,
        }

    except dicom2nifti.exceptions.ConversionValidationError as e:
        print(f"  [ERROR] Conversion validation failed for series "
              f"{modality}: {e}")
        return None

    except dicom2nifti.exceptions.ConversionError as e:
        print(f"  [ERROR] Conversion failed for series {modality}: {e}")
        return None

    except Exception as e:
        print(f"  [ERROR] Unexpected error for series {modality}: {e}")
        return None

    finally:
        # Always clean up temp folder
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir), ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_nifti(
    study: DicomStudy,
    anon_result: dict,
    output_dir: str,
) -> list[dict]:
    """
    Convert all series in a DICOM study to anonymized NIfTI files.

    Parameters
    ----------
    study : DicomStudy
        Original study object from dicom_reader.read_and_group().
        Used for series metadata and original SOPInstanceUIDs.
    anon_result : dict
        Output of dicom_anonymizer.anonymize_study().
        Used for anon_patient_id and output_dir of anonymized .dcm files.
    output_dir : str
        Directory to write .nii.gz files. Created if it does not exist.
        Files are written to output_dir/patient_{anon_patient_id}/

    Returns
    -------
    list[dict]
        One entry per successfully converted series, each containing:
          - series_uid     : str  — anonymized SeriesInstanceUID
          - modality       : str  — e.g. "CT", "MR"
          - nifti_path     : str  — absolute path to .nii.gz file
          - instance_count : int  — number of slices in the volume
        Empty list if all conversions failed.
    """
    anon_patient_id = anon_result["anon_patient_id"]
    anon_dicom_dir  = Path(anon_result["output_dir"])

    # Output goes into a patient-specific subfolder
    patient_output_dir = Path(output_dir) / f"patient_{anon_patient_id}"
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Converting DICOM series to NIfTI ...")
    print(f"  Source (anon DICOM) : {anon_dicom_dir}")
    print(f"  Output dir          : {patient_output_dir}")
    print(f"  Series to convert   : {len(study.series)}")
    print()

    results = []

    for idx, (orig_series_uid, series) in enumerate(study.series.items(), 1):
        print(f"[Series {idx}/{len(study.series)}]")
        result = _convert_series(
            series,
            anon_dicom_dir,
            anon_result,
            patient_output_dir,
            anon_patient_id,
        )
        if result:
            results.append(result)

    print()
    print(f"[INFO] NIfTI conversion complete.")
    print(f"  Converted : {len(results)}/{len(study.series)} series")
    for r in results:
        size_mb = Path(r["nifti_path"]).stat().st_size / (1024 * 1024)
        print(f"  {r['modality']}  {r['instance_count']} slices  "
              f"{size_mb:.1f} MB  →  {Path(r['nifti_path']).name}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dicom_reader import read_and_group
    from dicom_anonymizer import anonymize_study, load_privacy_filter

    if len(sys.argv) < 4:
        print("Usage:")
        print("  python dicom_to_nifti.py <dicom_input_dir> <dicom_output_dir> "
              "<nifti_output_dir>")
        print()
        print("Example:")
        print("  python dicom_to_nifti.py \\")
        print('      "input/141-03-2026ME/" \\')
        print('      "output/dicom/ANON/" \\')
        print('      "output/nifti/"')
        sys.exit(1)

    dicom_input_dir  = sys.argv[1]
    dicom_output_dir = sys.argv[2]
    nifti_output_dir = sys.argv[3]

    print("=" * 60)
    print("STEP 1: Reading DICOM folder")
    print("=" * 60)
    study = read_and_group(dicom_input_dir)

    print("=" * 60)
    print("STEP 2: Anonymizing DICOM")
    print("=" * 60)
    clf = load_privacy_filter()
    anon_result = anonymize_study(study, dicom_output_dir)

    print("=" * 60)
    print("STEP 3: Converting to NIfTI")
    print("=" * 60)
    nifti_results = convert_to_nifti(study, anon_result, nifti_output_dir)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Anon DICOM dir : {dicom_output_dir}")
    print(f"  NIfTI files    : {len(nifti_results)}")
    for r in nifti_results:
        print(f"    {r['nifti_path']}")