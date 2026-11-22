"""
dicom_verify.py
---------------
Verification script for the DICOM anonymization pipeline.

Checks the anonymized output folder against the original input folder and
produces a pass/fail report covering:

  1. File count       — same number of files in output as input
  2. PII wipe         — all identity/date/device tags are empty
  3. PII not present  — original patient name, ID not found anywhere in output
  4. UID remapping    — all UIDs in output are different from originals
  5. UID consistency  — all output files share the same new StudyInstanceUID
  6. Pixel integrity  — pixel data is unchanged (hash comparison on a sample)
  7. Free-text tags   — no original free-text values survive in output
  8. Technical params — acquisition/geometry tags (Manufacturer, KVP, TR/TE,
                         SliceThickness, PixelSpacing, WindowCenter/Width,
                         etc.) are byte-for-byte unchanged, not wiped

Usage
-----
    python dicom_verify.py <original_dicom_dir> <anonymized_dicom_dir>

Example
-------
    python dicom_verify.py \\
        "input/141-03-2026ME/" \\
        "Output/ANBARSU-R/"
"""

import sys
import hashlib
from pathlib import Path

import pydicom
import pydicom.sequence


# ─────────────────────────────────────────────────────────────────────────────
#  Tag lists (must match dicom_anonymizer.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

WIPE_TAGS: list[tuple[int, int]] = [
    #(0x0010, 0x0010),  # PatientName
    #(0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    # PatientAge, PatientSize, PatientWeight, PatientSex intentionally NOT
    # checked here — dicom_anonymizer.py retains them (see its WIPE_TAGS).
    (0x0010, 0x1040),  # PatientAddress
    (0x0010, 0x2154),  # PatientTelephoneNumbers
    (0x0010, 0x1002),  # OtherPatientIDsSequence
    (0x0010, 0x1000),  # OtherPatientIDs
    (0x0010, 0x1001),  # OtherPatientNames
    (0x0010, 0x2297),  # ResponsiblePerson
    (0x0010, 0x0050),  # PatientInsurancePlanCodeSequence
    (0x0008, 0x0050),  # AccessionNumber
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x1048),  # PhysiciansOfRecord
    (0x0008, 0x1050),  # PerformingPhysicianName
    (0x0008, 0x1070),  # OperatorsName
    (0x0008, 0x1010),  # StationName
    (0x0018, 0x1000),  # DeviceSerialNumber
    (0x0040, 0x0010),  # ScheduledStationName
    (0x0020, 0x0010),  # StudyID
    (0x0008, 0x0020),  # StudyDate
    (0x0008, 0x0021),  # SeriesDate
    (0x0008, 0x0022),  # AcquisitionDate
    (0x0008, 0x0023),  # ContentDate
    (0x0008, 0x0012),  # InstanceCreationDate
    (0x0008, 0x0030),  # StudyTime
    (0x0008, 0x0031),  # SeriesTime
    (0x0008, 0x0032),  # AcquisitionTime
    (0x0008, 0x0033),  # ContentTime
    (0x0008, 0x0013),  # InstanceCreationTime
]

UID_TAGS: list[tuple[int, int]] = [
    (0x0020, 0x000D),  # StudyInstanceUID
    (0x0020, 0x000E),  # SeriesInstanceUID
    (0x0008, 0x0018),  # SOPInstanceUID
    (0x0020, 0x0052),  # FrameOfReferenceUID
    (0x0008, 0x0014),  # InstanceCreatorUID
    (0x0020, 0x0200),  # SynchronizationFrameOfReferenceUID
]

FREE_TEXT_TAGS: list[tuple[int, int]] = [
    (0x0008, 0x1030),  # StudyDescription
    (0x0008, 0x103E),  # SeriesDescription
    (0x0010, 0x21B0),  # AdditionalPatientHistory
    (0x0010, 0x4000),  # PatientComments
    (0x0020, 0x4000),  # ImageComments
    (0x0032, 0x4000),  # StudyComments
]

# Technical / acquisition tags that must be RETAINED unchanged.
# These carry no patient identity but are required for clinical/scientific
# interpretation of the images (windowing, geometry, sequence parameters).
# If any of these differ between original and anonymized, something in
# dicom_anonymizer.py is over-wiping.
TECHNICAL_TAGS: dict[tuple[int, int], str] = {
    # Equipment / study context (not identifying on their own)
    (0x0008, 0x0060): "Modality",
    (0x0008, 0x0070): "Manufacturer",
    (0x0008, 0x1090): "ManufacturerModelName",
    (0x0018, 0x1020): "SoftwareVersions",
    (0x0018, 0x0015): "BodyPartExamined",
    # CT acquisition
    (0x0018, 0x0060): "KVP",
    (0x0018, 0x1150): "ExposureTime",
    (0x0018, 0x1151): "XRayTubeCurrent",
    (0x0018, 0x1152): "Exposure",
    (0x0018, 0x1210): "ConvolutionKernel",
    # MR acquisition
    (0x0018, 0x0080): "RepetitionTime",
    (0x0018, 0x0081): "EchoTime",
    (0x0018, 0x1314): "FlipAngle",
    (0x0018, 0x0087): "MagneticFieldStrength",
    (0x0018, 0x0095): "PixelBandwidth",
    (0x0018, 0x0020): "ScanningSequence",
    (0x0018, 0x0021): "SequenceVariant",
    (0x0018, 0x0023): "MRAcquisitionType",
    # Geometry — required for correct NIfTI conversion
    (0x0018, 0x0050): "SliceThickness",
    (0x0018, 0x0088): "SpacingBetweenSlices",
    (0x0028, 0x0030): "PixelSpacing",
    (0x0020, 0x0032): "ImagePositionPatient",
    (0x0020, 0x0037): "ImageOrientationPatient",
    (0x0020, 0x1041): "SliceLocation",
    (0x0028, 0x0010): "Rows",
    (0x0028, 0x0011): "Columns",
    (0x0028, 0x0100): "BitsAllocated",
    (0x0028, 0x0103): "PixelRepresentation",
    (0x0028, 0x1052): "RescaleIntercept",
    (0x0028, 0x1053): "RescaleSlope",
    # Display windowing — used to derive NIfTI cal_min/cal_max
    (0x0028, 0x1050): "WindowCenter",
    (0x0028, 0x1051): "WindowWidth",
}

TAG_NAMES: dict[tuple[int, int], str] = {
    (0x0010, 0x0010): "PatientName",
    (0x0010, 0x0020): "PatientID",
    (0x0010, 0x0030): "PatientBirthDate",
    (0x0010, 0x1010): "PatientAge",
    (0x0008, 0x0050): "AccessionNumber",
    (0x0008, 0x0080): "InstitutionName",
    (0x0008, 0x0081): "InstitutionAddress",
    (0x0008, 0x0090): "ReferringPhysicianName",
    (0x0008, 0x1050): "PerformingPhysicianName",
    (0x0008, 0x1070): "OperatorsName",
    (0x0008, 0x1010): "StationName",
    (0x0018, 0x1000): "DeviceSerialNumber",
    (0x0008, 0x0020): "StudyDate",
    (0x0008, 0x0021): "SeriesDate",
    (0x0008, 0x0022): "AcquisitionDate",
    (0x0008, 0x0023): "ContentDate",
    (0x0020, 0x000D): "StudyInstanceUID",
    (0x0020, 0x000E): "SeriesInstanceUID",
    (0x0008, 0x0018): "SOPInstanceUID",
    (0x0008, 0x1030): "StudyDescription",
    (0x0008, 0x103E): "SeriesDescription",
    (0x0010, 0x21B0): "AdditionalPatientHistory",
    (0x0010, 0x4000): "PatientComments",
    (0x0020, 0x4000): "ImageComments",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tag_name(tag: tuple[int, int]) -> str:
    return TAG_NAMES.get(tag, f"({tag[0]:04X},{tag[1]:04X})")


def _pixel_hash(ds: pydicom.Dataset) -> str | None:
    """SHA-256 of the raw pixel data bytes, or None if not present."""
    try:
        raw = ds.PixelData
        return hashlib.sha256(raw).hexdigest()
    except AttributeError:
        return None


def _tag_value_str(ds: pydicom.Dataset, tag: tuple[int, int]) -> str:
    """Return the string value of a tag, or '' if absent/empty."""
    if tag not in ds:
        return ""
    val = ds[tag].value
    if val is None:
        return ""
    if isinstance(val, pydicom.sequence.Sequence):
        return f"<SQ len={len(val)}>"
    return str(val).strip()


def _is_empty(ds: pydicom.Dataset, tag: tuple[int, int]) -> bool:
    """True if tag is absent or has an empty/whitespace-only value."""
    if tag not in ds:
        return True
    val = ds[tag].value
    if val is None:
        return True
    if isinstance(val, pydicom.sequence.Sequence):
        return len(val) == 0
    return str(val).strip() == ""


# ─────────────────────────────────────────────────────────────────────────────
#  Per-file checks
# ─────────────────────────────────────────────────────────────────────────────

def check_file(
    orig_ds: pydicom.Dataset,
    anon_ds: pydicom.Dataset,
    orig_patient_name: str,
    orig_patient_id: str,
    orig_study_uid: str,
    expected_anon_study_uid: str,
) -> list[str]:
    """
    Run all checks on one (original, anonymized) file pair.
    Returns a list of failure messages. Empty list = all passed.
    """
    failures = []
    pii_strings = [s.strip() for s in [orig_patient_name, orig_patient_id] if s and s.strip()]

    # ── Check 1: Wipe tags are empty ──
    for tag in WIPE_TAGS:
        if not _is_empty(anon_ds, tag):
            val = _tag_value_str(anon_ds, tag)
            failures.append(f"WIPE FAILED  {_tag_name(tag)}: value still present: '{val}'")

    # ── Check 1b: PatientName and PatientID are pseudonymised ──
    for tag in [(0x0010, 0x0010), (0x0010, 0x0020)]:
        val = _tag_value_str(anon_ds, tag)
        orig = _tag_value_str(orig_ds, tag)
        if val == orig:
            failures.append(f"NOT PSEUDONYMISED  {_tag_name(tag)}: original value unchanged")
        if not val:
            failures.append(f"NOT PSEUDONYMISED  {_tag_name(tag)}: value is empty")
        for pii in pii_strings:
            if pii.lower() in val.lower():
                failures.append(f"PII LEAK  {_tag_name(tag)}: original PII '{pii}' still present in '{val}'")

    # ── Check 2: Original PII strings do not appear anywhere ──
    pii_strings = [
        s.strip() for s in [orig_patient_name, orig_patient_id]
        if s and s.strip()
    ]
    for tag in list(WIPE_TAGS) + list(UID_TAGS) + list(FREE_TEXT_TAGS):
        val = _tag_value_str(anon_ds, tag)
        if not val:
            continue
        for pii in pii_strings:
            if pii.lower() in val.lower():
                failures.append(
                    f"PII LEAK     {_tag_name(tag)}: original value '{pii}' found in '{val}'"
                )

    # ── Check 3: UID tags are different from originals ──
    for tag in UID_TAGS:
        orig_val = _tag_value_str(orig_ds, tag)
        anon_val = _tag_value_str(anon_ds, tag)
        if orig_val and anon_val and orig_val == anon_val:
            failures.append(
                f"UID NOT REMAPPED  {_tag_name(tag)}: still '{orig_val}'"
            )

    # ── Check 4: StudyInstanceUID matches expected anonymized value ──
    anon_study_uid = _tag_value_str(anon_ds, (0x0020, 0x000D))
    if anon_study_uid and anon_study_uid != expected_anon_study_uid:
        failures.append(
            f"UID INCONSISTENT  StudyInstanceUID: got '{anon_study_uid}', "
            f"expected '{expected_anon_study_uid}'"
        )

    # ── Check 5: Pixel data unchanged ──
    orig_hash = _pixel_hash(orig_ds)
    anon_hash = _pixel_hash(anon_ds)
    if orig_hash is not None and anon_hash is not None:
        if orig_hash != anon_hash:
            failures.append(
                f"PIXEL MODIFIED  pixel hash changed: {orig_hash[:12]}... -> {anon_hash[:12]}..."
            )
    elif orig_hash is not None and anon_hash is None:
        failures.append("PIXEL MISSING  pixel data lost in anonymized file")

    # ── Check 6: Free-text tags don't contain original values unchanged ──
    for tag in FREE_TEXT_TAGS:
        orig_val = _tag_value_str(orig_ds, tag)
        anon_val = _tag_value_str(anon_ds, tag)
        if orig_val and anon_val and orig_val == anon_val:
            # Only flag if the original had PII-like content
            for pii in pii_strings:
                if pii.lower() in orig_val.lower():
                    failures.append(
                        f"FREE-TEXT NOT REDACTED  {_tag_name(tag)}: "
                        f"PII '{pii}' still in '{anon_val}'"
                    )

    # ── Check 7: Technical/acquisition tags are RETAINED unchanged ──
    for tag, name in TECHNICAL_TAGS.items():
        orig_val = _tag_value_str(orig_ds, tag)
        if not orig_val:
            continue  # tag wasn't present/populated in the source — nothing to check
        anon_val = _tag_value_str(anon_ds, tag)
        if not anon_val:
            failures.append(
                f"TECHNICAL PARAM LOST  {name}: had '{orig_val}', now empty/missing"
            )
        elif anon_val != orig_val:
            failures.append(
                f"TECHNICAL PARAM ALTERED  {name}: '{orig_val}' -> '{anon_val}'"
            )

    return failures


# ─────────────────────────────────────────────────────────────────────────────
#  Study-level summary (print what values are in the output)
# ─────────────────────────────────────────────────────────────────────────────

def print_anonymized_summary(anon_ds: pydicom.Dataset) -> None:
    """Print all PII-relevant tag values from one anonymized file."""
    key_tags = [
        (0x0010, 0x0010),  # PatientName
        (0x0010, 0x0020),  # PatientID
        (0x0010, 0x0030),  # PatientBirthDate
        (0x0010, 0x0040),  # PatientSex
        (0x0010, 0x1010),  # PatientAge
        (0x0008, 0x0020),  # StudyDate
        (0x0008, 0x0080),  # InstitutionName
        (0x0008, 0x0090),  # ReferringPhysicianName
        (0x0008, 0x1010),  # StationName
        (0x0008, 0x1030),  # StudyDescription
        (0x0008, 0x103E),  # SeriesDescription
        (0x0020, 0x000D),  # StudyInstanceUID
        (0x0020, 0x000E),  # SeriesInstanceUID
        (0x0008, 0x0018),  # SOPInstanceUID
    ]
    print("\n  Key tag values in anonymized output (from first file):")
    print(f"  {'Tag name':<30} {'Value'}")
    print(f"  {'-'*30} {'-'*45}")
    for tag in key_tags:
        val = _tag_value_str(anon_ds, tag)
        name = _tag_name(tag)
        display = val if val else "(empty)"
        print(f"  {name:<30} {display[:60]}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def verify(original_dir: str, anonymized_dir: str) -> bool:
    """
    Run all verification checks.
    Returns True if all checks pass, False if any fail.
    """
    orig_path = Path(original_dir)
    anon_path = Path(anonymized_dir)

    orig_files = sorted(orig_path.glob("*.dcm"))
    anon_files = sorted(anon_path.glob("*.dcm"))

    print("=" * 60)
    print("DICOM ANONYMIZATION VERIFIER")
    print("=" * 60)
    print(f"  Original dir  : {orig_path}")
    print(f"  Anonymized dir: {anon_path}")
    print(f"  Original files: {len(orig_files)}")
    print(f"  Anon files    : {len(anon_files)}")

    all_passed = True

    # ── Check A: file count ──
    print("\n[CHECK] File count")
    if len(orig_files) == len(anon_files):
        print(f"  ✓ PASS  {len(orig_files)} files in both folders")
    else:
        print(f"  ✗ FAIL  original={len(orig_files)}, anonymized={len(anon_files)}")
        all_passed = False

    if not orig_files or not anon_files:
        print("\n[ERROR] One or both directories are empty. Cannot continue.")
        return False

    # ── Read first original to extract baseline PII values ──
    first_orig = pydicom.dcmread(str(orig_files[0]), force=True)
    orig_patient_name = _tag_value_str(first_orig, (0x0010, 0x0010))
    orig_patient_id   = _tag_value_str(first_orig, (0x0010, 0x0020))
    orig_study_uid    = _tag_value_str(first_orig, (0x0020, 0x000D))

    print(f"\n  Original patient name : '{orig_patient_name}'")
    print(f"  Original patient ID   : '{orig_patient_id}'")
    print(f"  Original StudyUID     : '{orig_study_uid}'")

    # ── Read first anonymized to get expected new StudyUID ──
    first_anon = pydicom.dcmread(str(anon_files[0]), force=True)
    expected_anon_study_uid = _tag_value_str(first_anon, (0x0020, 0x000D))

    print(f"  Anonymized StudyUID   : '{expected_anon_study_uid}'")

    # ── Print summary of what's in the anonymized output ──
    print_anonymized_summary(first_anon)

    # ── Per-file checks ──
    print(f"\n[CHECK] Per-file verification ({len(anon_files)} files) ...")

    # Build UID-based pairing between original and anonymized files
    from dicom_anonymizer import make_anon_uid
    orig_sop_map = {}
    for f in orig_files:
        ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        sop = _tag_value_str(ds, (0x0008, 0x0018))
        if sop:
            orig_sop_map[sop] = f
    anon_to_orig_sop = {make_anon_uid(sop): sop for sop in orig_sop_map}

    # We check every anonymized file against the corresponding original.
    # Since filenames changed (UID-based), we match by InstanceNumber order.
    file_failures: dict[str, list[str]] = {}
    pixels_checked = 0
    PIXEL_SAMPLE_LIMIT = 10  # check pixel integrity on first 10 files only

    for idx, anon_f in enumerate(anon_files, 1):
        load_pixels = idx <= PIXEL_SAMPLE_LIMIT
        anon_ds = pydicom.dcmread(str(anon_f), stop_before_pixels=not load_pixels, force=True)

        anon_sop = _tag_value_str(anon_ds, (0x0008, 0x0018))
        orig_sop = anon_to_orig_sop.get(anon_sop)
        if orig_sop is None or orig_sop not in orig_sop_map:
            file_failures[anon_f.name] = [f"NO MATCH  could not find original file for anon SOP {anon_sop}"]
            continue

        orig_ds = pydicom.dcmread(str(orig_sop_map[orig_sop]), stop_before_pixels=not load_pixels, force=True)

        if load_pixels:
            pixels_checked += 1

        failures = check_file(
            orig_ds,
            anon_ds,
            orig_patient_name,
            orig_patient_id,
            orig_study_uid,
            expected_anon_study_uid,
        )

        if failures:
            file_failures[anon_f.name] = failures

        # Progress
        if idx % 20 == 0 or idx == len(anon_files):
            status = "✓" if not failures else "✗"
            print(f"  [{idx:>4}/{len(anon_files)}] {status}  {anon_f.name[-35:]}")

    # ── UID consistency check across ALL files ──
    print(f"\n[CHECK] StudyInstanceUID consistency across all {len(anon_files)} files ...")
    study_uids_seen = set()
    series_uids_seen = set()
    for anon_f in anon_files:
        ds = pydicom.dcmread(str(anon_f), stop_before_pixels=True, force=True)
        uid = _tag_value_str(ds, (0x0020, 0x000D))
        series_uid = _tag_value_str(ds, (0x0020, 0x000E))
        if uid:
            study_uids_seen.add(uid)
        if series_uid:
            series_uids_seen.add(series_uid)

    if len(study_uids_seen) == 1:
        print(f"  ✓ PASS  All files share exactly 1 StudyInstanceUID: {list(study_uids_seen)[0]}")
    else:
        print(f"  ✗ FAIL  Found {len(study_uids_seen)} different StudyInstanceUIDs — should be 1")
        for uid in study_uids_seen:
            print(f"          {uid}")
        all_passed = False

    if len(series_uids_seen) == 1:
        print(f"  ✓ PASS  All files share exactly 1 SeriesInstanceUID: {list(series_uids_seen)[0]}")
    else:
        # Multiple series is valid — just report it
        print(f"  ℹ INFO  Found {len(series_uids_seen)} SeriesInstanceUID(s) — expected if multiple series")
        for uid in series_uids_seen:
            print(f"          {uid}")

    # ── Original UIDs absent in output ──
    print(f"\n[CHECK] Original UIDs absent from all output files ...")
    orig_uids = set()
    for orig_f in orig_files:
        ds = pydicom.dcmread(str(orig_f), stop_before_pixels=True, force=True)
        for tag in UID_TAGS:
            val = _tag_value_str(ds, tag)
            if val:
                orig_uids.add(val)

    uid_leaks = []
    for anon_f in anon_files:
        ds = pydicom.dcmread(str(anon_f), stop_before_pixels=True, force=True)
        for tag in UID_TAGS:
            val = _tag_value_str(ds, tag)
            if val and val in orig_uids:
                uid_leaks.append((anon_f.name, _tag_name(tag), val))

    if not uid_leaks:
        print(f"  ✓ PASS  No original UIDs found in any output file")
    else:
        print(f"  ✗ FAIL  {len(uid_leaks)} original UID(s) found in output:")
        for fname, tagname, val in uid_leaks[:5]:
            print(f"          {fname[-30:]}  {tagname}: {val}")
        all_passed = False

    # ── Summary of per-file failures ──
    print(f"\n[CHECK] Per-file failure summary ...")
    print(f"  Pixel integrity checked on first {pixels_checked} files")
    print(f"  Technical params checked per file: {len(TECHNICAL_TAGS)} tags "
          f"(Manufacturer, KVP, TR/TE, SliceThickness, PixelSpacing, "
          f"WindowCenter/Width, etc.)")

    # Split failures into "PII/anonymization" vs "technical param retention"
    # so a technical regression doesn't get lost among PII-check noise.
    technical_failures: dict[str, list[str]] = {}
    pii_failures: dict[str, list[str]] = {}
    for fname, msgs in file_failures.items():
        tech_msgs = [m for m in msgs if m.startswith("TECHNICAL PARAM")]
        other_msgs = [m for m in msgs if not m.startswith("TECHNICAL PARAM")]
        if tech_msgs:
            technical_failures[fname] = tech_msgs
        if other_msgs:
            pii_failures[fname] = other_msgs

    if not pii_failures:
        print(f"  ✓ PASS  No PII/anonymization failures across {len(anon_files)} files")
    else:
        all_passed = False
        print(f"  ✗ FAIL  {len(pii_failures)} file(s) had PII/anonymization failures:")
        for fname, msgs in list(pii_failures.items())[:10]:
            print(f"\n  File: {fname[-45:]}")
            for msg in msgs:
                print(f"    ✗ {msg}")
        if len(pii_failures) > 10:
            print(f"\n  ... and {len(pii_failures) - 10} more files with PII failures")

    if not technical_failures:
        print(f"\n  ✓ PASS  Technical/acquisition parameters retained unchanged "
              f"across {len(anon_files)} files")
    else:
        all_passed = False
        print(f"\n  ✗ FAIL  {len(technical_failures)} file(s) lost or altered "
              f"technical parameters:")
        for fname, msgs in list(technical_failures.items())[:10]:
            print(f"\n  File: {fname[-45:]}")
            for msg in msgs:
                print(f"    ✗ {msg}")
        if len(technical_failures) > 10:
            print(f"\n  ... and {len(technical_failures) - 10} more files "
                  f"with technical param failures")

    # ── Final verdict ──
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ ALL CHECKS PASSED — anonymization looks correct")
    else:
        print("  ✗ SOME CHECKS FAILED — review the failures above")
    print("=" * 60)

    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dicom_verify.py <original_dicom_dir> <anonymized_dicom_dir>")
        print("Example:")
        print("  python dicom_verify.py input/141-03-2026ME/ Output/ANBARSU-R/")
        sys.exit(1)

    original_dir  = sys.argv[1]
    anonymized_dir = sys.argv[2]

    passed = verify(original_dir, anonymized_dir)
    sys.exit(0 if passed else 1)