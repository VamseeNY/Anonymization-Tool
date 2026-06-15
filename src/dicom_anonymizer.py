"""
dicom_anonymizer.py
-------------------
Stage 2 of the anonymization pipeline.

Takes a DicomStudy (from dicom_reader.py), anonymizes all PII tags,
remaps all UIDs deterministically, runs openai/privacy-filter on any
free-text tags, then writes anonymized .dcm files to an output folder.

Design decisions
----------------
- Anonymization values (new UIDs, redacted free-text) are computed ONCE
  from the first instance, then stamped identically onto every instance.
  This guarantees consistency across all 160 slices.

- UIDs are remapped deterministically using SHA-256(salt + original_uid).
  Same input always produces the same output — no mapping table needed,
  and cross-file UID references remain valid.

- PatientName is replaced with "ANON-<anon_patient_id>".
  PatientID is replaced with the anon_patient_id.
  PatientSex is retained under PS3.15 E.3.7 (Retain Patient Characteristics Option).
  Table E.1-1 action code for PatientSex is Z/D (replace or zero); retention requires
  explicit invocation of E.3.7, which is declared in DeidentificationMethodCodeSequence
  with code 113108. Sex is retained because it is required for image quantification
  pipelines (e.g. PET SUV body-surface-area corrections).
  All other identity, date, and device tags are wiped (set to empty string).

- Free-text tags (StudyDescription, AdditionalPatientHistory, etc.) are
  run through openai/privacy-filter. Detected spans are replaced with
  [ENTITY_TYPE] tokens. The rest of the text is preserved.

- Nested sequences (SQ tags) are handled recursively. The walker enters
  every sequence at any depth and applies the same wipe/remap/replace
  logic to nested tags (ReferencedStudySequence, RequestAttributesSequence,
  ContentSequence, etc.). This addresses DICOM PS3.15 Annex E requirement
  for de-identification of nested attributes.

- Output filenames use the new (anonymized) SOPInstanceUID so the
  filename itself carries no original identifying information.

Usage
-----
    from dicom_reader import read_and_group
    from dicom_anonymizer import anonymize_study

    study = read_and_group("input/dicom/")
    anonymize_study(study, output_dir="output/dicom/patient_001/")

Requires
--------
    pip install pydicom transformers>=5.6.0 torch
"""

import copy
import hashlib
import os
from pathlib import Path
from typing import Optional

import pydicom
import pydicom.sequence
import pydicom.uid

from dicom_reader import DicomStudy, DicomSeries, DicomInstance
from pixel_scrubber import (
    load_ocr_reader,
    scrub_pixels,
)
import uuid as _uuid_mod 

# ─────────────────────────────────────────────────────────────────────────────
#  Tag lists  (DICOM PS3.15 Annex E — Basic Application Level Confidentiality)
# ─────────────────────────────────────────────────────────────────────────────

# Tags whose values will be set to empty string (or empty sequence for SQ VR).
# PatientName and PatientID are handled separately (replaced, not wiped).
# Stored as a set for O(1) lookup during recursive walking.
WIPE_TAGS: set[tuple[int, int]] = {
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x1010),  # PatientAge
    (0x0010, 0x1020),  # PatientSize
    (0x0010, 0x1030),  # PatientWeight
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
    # Dates and times — wipe to remove temporal re-identification risk
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
    # Additional tags commonly found inside nested sequences (PS3.15 Annex E)
    (0x0040, 0x0006),  # ScheduledPerformingPhysicianName
    (0x0040, 0x0004),  # ScheduledProcedureStepEndDate
    (0x0040, 0x0005),  # ScheduledProcedureStepEndTime
    (0x0040, 0x0001),  # ScheduledStationAETTitle
    (0x0040, 0x0002),  # ScheduledProcedureStepStartDate
    (0x0040, 0x0003),  # ScheduledProcedureStepStartTime
    (0x0040, 0x0007),  # ScheduledProcedureStepDescription
    (0x0040, 0x0009),  # ScheduledProcedureStepID
    (0x0040, 0x1004),  # PatientTransportArrangements
    (0x0040, 0x1400),  # RequestedProcedureComments
    (0x0040, 0x2001),  # ReasonForTheImagingServiceRequest
    (0x0040, 0x2016),  # PlacerOrderNumberImagingServiceRequest
    (0x0040, 0x2017),  # FillerOrderNumberImagingServiceRequest
    (0x0040, 0x2400),  # ImagingServiceRequestComments
    (0x0032, 0x1060),  # RequestedProcedureDescription
    (0x0032, 0x1070),  # RequestedContrastAgent
    (0x0010, 0x21F0),  # PatientReligiousPreference
    (0x0010, 0x1090),  # MedicalRecordLocator
    (0x0010, 0x1080),  # MilitaryRank
    (0x0010, 0x1081),  # BranchOfService
    (0x0010, 0x21D0),  # LastMenstrualDate
    (0x0038, 0x0300),  # CurrentPatientLocation
    (0x0038, 0x0050),  # SpecialNeeds
    (0x0038, 0x0010),  # AdmissionID
    (0x0038, 0x0020),  # AdmittingDate
    (0x0038, 0x0021),  # AdmittingTime
    (0x0038, 0x4000),  # VisitComments
}

# Tags whose string values need NLP PII scanning before writing.
# Key = DICOM tag tuple, Value = keyword name (for logging only).
FREE_TEXT_TAGS: dict[tuple[int, int], str] = {
    (0x0008, 0x1030): "StudyDescription",
    (0x0008, 0x103E): "SeriesDescription",
    (0x0010, 0x21B0): "AdditionalPatientHistory",
    (0x0010, 0x4000): "PatientComments",
    (0x0020, 0x4000): "ImageComments",
    (0x0032, 0x4000): "StudyComments",
}

# UIDs that must be remapped to break cross-reference links.
# Top-level study/series UIDs — kept as list for build_uid_map() iteration.
# Stored as a set for O(1) lookup during recursive walking.
UID_TAGS: list[tuple[int, int]] = [
    (0x0020, 0x000D),  # StudyInstanceUID
    (0x0020, 0x000E),  # SeriesInstanceUID
    (0x0008, 0x0018),  # SOPInstanceUID
    (0x0020, 0x0052),  # FrameOfReferenceUID
    (0x0008, 0x0014),  # InstanceCreatorUID
    (0x0020, 0x0200),  # SynchronizationFrameOfReferenceUID
]

# Expanded UID set — includes UIDs commonly found inside nested sequences.
# Used by the recursive walker to remap any UID found at any depth.
# make_anon_uid() is deterministic so unknown nested UIDs are safely remapped.
UID_TAGS_ALL: set[tuple[int, int]] = {
    (0x0020, 0x000D),  # StudyInstanceUID
    (0x0020, 0x000E),  # SeriesInstanceUID
    (0x0008, 0x0018),  # SOPInstanceUID
    (0x0020, 0x0052),  # FrameOfReferenceUID
    (0x0008, 0x0014),  # InstanceCreatorUID
    (0x0020, 0x0200),  # SynchronizationFrameOfReferenceUID
    (0x0008, 0x1155),  # ReferencedSOPInstanceUID   (inside ReferencedImageSequence etc)
    (0x0008, 0x1150),  # ReferencedSOPClassUID      (inside sequences)
    (0x0020, 0x9164),  # DimensionOrganizationUID   (inside DimensionOrganizationSequence)
    (0x0028, 0x1199),  # PaletteColorLookupTableUID
    (0x0008, 0x0058),  # FailedSOPInstanceUIDList
    (0x0088, 0x0140),  # StorageMediaFileSetUID
    (0x0008, 0x9123),  # CreatorVersionUID
    # NOTE: (0x0018,0xA001) ContributingEquipmentSequence and
    # (0x0008,0x0096) ReferringPhysicianIdentificationSequence are SQ VR tags,
    # not UID tags. They are handled by the SQ recursion branch in
    # _walk_and_anonymize(). Their inner UIDs (e.g. ReferencedSOPInstanceUID)
    # are already caught by entries above. Placing SQ tags here causes the
    # UID remapping branch to call str() on a sequence object — a no-op at best.
}

# Patient identity tag constants for the recursive walker
_TAG_PATIENT_NAME = (0x0010, 0x0010)
_TAG_PATIENT_ID   = (0x0010, 0x0020)

# Tags explicitly retained under PS3.15 E.3.7 (Retain Patient Characteristics Option).
# PatientSex (0010,0040) action code in Table E.1-1 is Z/D (zero or replace).
# Retaining it is permitted only when this option is declared.
# It is retained here because sex is required for image quantification pipelines
# (e.g. PET standardized uptake value corrections).
# DeidentificationMethodCodeSequence must include the E.3.7 option code (113108)
# if this tag is kept.
RETAIN_PATIENT_CHARACTERISTICS_TAGS: set[tuple[int, int]] = {
    (0x0010, 0x0040),  # PatientSex
}


# ─────────────────────────────────────────────────────────────────────────────
#  UID remapping
# ─────────────────────────────────────────────────────────────────────────────

def make_anon_uid(original_uid: str, salt: str = "pipeline-salt-v1") -> str:
    """
    Deterministically derive an anonymized DICOM UID from an original UID.

    Properties:
    - Same input always produces the same output (no state needed).
    - Different inputs always produce different outputs (no collisions in practice).
    - Output is a valid DICOM UID: digits + dots, max 64 chars.
    - Uses the 2.25. prefix per DICOM PS3.5 B.2 / ITU-T X.667.

    Implementation:
    Uses UUID v5 (RFC 4122, name-based SHA-1) with a fixed namespace UUID
    derived from the salt. UUID v5 produces a standards-compliant 128-bit UUID
    whose decimal integer representation is valid under the 2.25. OID arc,
    as required by ITU-T X.667 / ISO IEC 9834-8.

    Parameters
    ----------
    original_uid : str
        The original DICOM UID to remap.
    salt : str
        Change this string if you need to re-anonymize a dataset with
        different output UIDs (e.g. for a second anonymization run).

    Returns
    -------
    str
        A valid DICOM UID string, deterministically derived from original_uid.
    """
    # Derive a stable namespace UUID from the salt using UUID v5 with
    # the DNS namespace (a fixed, well-known UUID per RFC 4122).
    namespace = _uuid_mod.uuid5(_uuid_mod.NAMESPACE_DNS, salt)
    # Derive the output UUID from the original UID within that namespace.
    output_uuid = _uuid_mod.uuid5(namespace, original_uid)
    # Convert to the 2.25. OID arc per ITU-T X.667:
    # treat the 128-bit UUID integer as a decimal number, suppress leading zeros.
    numeric = str(output_uuid.int)  # up to 39 digits, no leading zeros
    uid = f"2.25.{numeric}"
    # DICOM UIDs must not exceed 64 characters (PS3.5 section 9.1).
    assert len(uid) <= 64, f"Generated UID exceeds 64 chars: {uid}"
    return uid


def build_uid_map(ds: pydicom.Dataset) -> dict[str, str]:
    """
    Build a mapping of {original_uid: anonymized_uid} for all UID tags
    present in *ds*. Called once on the first instance; the same map is
    applied to all subsequent instances in the study.
    """
    uid_map: dict[str, str] = {}
    for tag in UID_TAGS:
        if tag in ds:
            orig = str(ds[tag].value).strip()
            if orig and orig not in uid_map:
                uid_map[orig] = make_anon_uid(orig)
    # Also remap MediaStorageSOPInstanceUID in file_meta if present
    if hasattr(ds, "file_meta") and ds.file_meta:
        meta_tag = (0x0002, 0x0003)
        if meta_tag in ds.file_meta:
            orig = str(ds.file_meta[meta_tag].value).strip()
            if orig and orig not in uid_map:
                uid_map[orig] = make_anon_uid(orig)
    return uid_map


# ─────────────────────────────────────────────────────────────────────────────
#  Free-text PII redaction via openai/privacy-filter
# ─────────────────────────────────────────────────────────────────────────────

def load_privacy_filter():
    """
    Load the openai/privacy-filter pipeline.
    Called once and reused across all instances.
    Requires transformers >= 5.6.0.
    """
    from transformers import pipeline as hf_pipeline
    print("[INFO] Loading openai/privacy-filter model ...")
    clf = hf_pipeline(
        task="token-classification",
        model="openai/privacy-filter",
        aggregation_strategy="simple",
    )
    print("[INFO] Model loaded.")
    return clf


def redact_text(text: str, clf) -> str:
    """
    Run *text* through privacy-filter and replace detected PII spans
    with [ENTITY_TYPE] tokens.

    Example
    -------
    "Brain CT for Alice Smith DOB 1990-05-20"
    -> "Brain CT for [private_person] DOB [private_date_time]"
    """
    if not text or not text.strip():
        return text

    entities = clf(text)
    if not entities:
        return text

    # Build redacted string by replacing spans from right to left
    # (right-to-left avoids offset shifts as we modify the string)
    result = text
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        label = ent["entity_group"]
        start = ent["start"]
        end   = ent["end"]
        result = result[:start] + f"[{label}]" + result[end:]

    return result


def build_redacted_free_texts(
    ds: pydicom.Dataset,
    clf,
) -> dict[tuple[int, int], str]:
    """
    For each free-text tag present in *ds*, run redact_text and return
    a mapping of {tag: redacted_value}.
    Called once on the first instance; reused for all subsequent instances.
    """
    redacted: dict[tuple[int, int], str] = {}
    for tag, keyword in FREE_TEXT_TAGS.items():
        if tag in ds:
            original = str(ds[tag].value).strip()
            if original:
                redacted_val = redact_text(original, clf)
                redacted[tag] = redacted_val
                if redacted_val != original:
                    print(f"  [REDACT] {keyword}: '{original}' -> '{redacted_val}'")
            else:
                redacted[tag] = original
    return redacted


# ─────────────────────────────────────────────────────────────────────────────
#  Recursive tag walker
# ─────────────────────────────────────────────────────────────────────────────

def _walk_and_anonymize(
    ds: pydicom.Dataset,
    anon_patient_id: str,
    redacted_free_texts: dict[tuple[int, int], str],
    uid_map: dict[str, str],
    _depth: int = 0,
) -> None:
    """
    Recursively walk all elements in *ds* and apply anonymization actions.

    Handles arbitrarily nested SQ (sequence) tags — each item inside a
    sequence is a Dataset that is walked with the same logic. This ensures
    PII inside nested sequences (ReferencedStudySequence, RequestAttributesSequence,
    ContentSequence, etc.) is handled identically to top-level tags.

    Actions applied at every depth:
    - PatientName → replaced with pseudonym
    - PatientID   → replaced with pseudonym
    - WIPE_TAGS   → set to empty string (or empty Sequence for SQ VR)
    - FREE_TEXT_TAGS → replaced with pre-redacted value
    - UID_TAGS_ALL   → remapped via uid_map or make_anon_uid()
    - SQ elements    → recurse into each item Dataset

    Parameters
    ----------
    ds : pydicom.Dataset
        Dataset to walk. May be a top-level instance dataset or a nested
        sequence item dataset.
    anon_patient_id : str
        Pseudonym to assign to PatientName and PatientID.
    redacted_free_texts : dict
        Pre-computed {tag: redacted_string} for free-text tags.
    uid_map : dict
        Pre-computed {original_uid: anonymized_uid} for known UIDs.
    _depth : int
        Recursion depth — for logging only.
    """
    for elem in ds:
        tag = (elem.tag.group, elem.tag.element)

        # ── Recurse into sequences ──
        if elem.VR == "SQ":
            # Check if this entire sequence should be wiped
            if tag in WIPE_TAGS:
                elem.value = pydicom.sequence.Sequence([])
            else:
                # Recurse into each item in the sequence
                for item in elem.value:
                    _walk_and_anonymize(
                        item,
                        anon_patient_id,
                        redacted_free_texts,
                        uid_map,
                        _depth + 1,
                    )
            continue

        # ── PatientName — replace with pseudonym ──
        if tag == _TAG_PATIENT_NAME:
            elem.value = f"ANON-{anon_patient_id}"
            continue

        # ── PatientID — replace with pseudonym ──
        if tag == _TAG_PATIENT_ID:
            elem.value = anon_patient_id
            continue

        # ── Wipe tags — set to empty ──
        if tag in WIPE_TAGS:
            elem.value = ""
            continue

        # ── Free-text tags — apply pre-redacted value ──
        if tag in FREE_TEXT_TAGS:
            if tag in redacted_free_texts:
                elem.value = redacted_free_texts[tag]
            continue

        # ── UID tags — remap deterministically ──
        if tag in UID_TAGS_ALL:
            orig = str(elem.value).strip()
            if orig:
                # Use pre-built map if available, otherwise derive on the fly
                elem.value = uid_map.get(orig, make_anon_uid(orig))
            continue


# ─────────────────────────────────────────────────────────────────────────────
#  Core per-instance anonymization
# ─────────────────────────────────────────────────────────────────────────────

def _anonymize_dataset(
    ds: pydicom.Dataset,
    anon_patient_id: str,
    redacted_free_texts: dict[tuple[int, int], str],
    uid_map: dict[str, str],
) -> pydicom.Dataset:
    """
    Apply all anonymization transforms to *ds* in-place and return it.

    Parameters
    ----------
    ds : pydicom.Dataset
        A fully loaded (with pixel data) DICOM dataset for one instance.
    anon_patient_id : str
        The pseudonym ID to assign to PatientID and PatientName.
    redacted_free_texts : dict
        Pre-computed {tag: redacted_string} from build_redacted_free_texts().
    uid_map : dict
        Pre-computed {original_uid: anonymized_uid} from build_uid_map().
    """
    # 1–4. Recursively walk all tags at every depth — handles top-level tags
    #       and all nested sequence items in a single pass.
    #       PatientSex (0010,0040) is in RETAIN_PATIENT_CHARACTERISTICS_TAGS
    #       and kept under PS3.15 E.3.7 (Retain Patient Characteristics Option).
    _walk_and_anonymize(ds, anon_patient_id, redacted_free_texts, uid_map)

    # 5. Remap SOPInstanceUID per-instance — unique per file, not in uid_map
    if (0x0008, 0x0018) in ds:
        orig_sop = str(ds[(0x0008, 0x0018)].value).strip()
        ds[(0x0008, 0x0018)].value = make_anon_uid(orig_sop)

    # 6. Keep MediaStorageSOPInstanceUID in sync with new SOPInstanceUID
    #    and clean remaining File Meta Information tags that may carry
    #    device or institution identity (PS3.15 E.1.1 / PS3.10).
    if hasattr(ds, "file_meta") and ds.file_meta:
        meta_tag = (0x0002, 0x0003)
        if meta_tag in ds.file_meta:
            orig_meta = str(ds.file_meta[meta_tag].value).strip()
            ds.file_meta[meta_tag].value = make_anon_uid(orig_meta)

        # ImplementationVersionName (0002,0013) — may identify the acquiring
        # system's software version. Replace with pipeline identifier.
        ds.file_meta[0x0002, 0x0013].value = "ANON-PIPELINE-V1" \
            if (0x0002, 0x0013) in ds.file_meta else None

        # SourceApplicationEntityTitle (0002,0016) — identifies the AE (device)
        # that created the file. Remove to prevent device re-identification.
        if (0x0002, 0x0016) in ds.file_meta:
            del ds.file_meta[0x0002, 0x0016]

        # PrivateInformationCreatorUID (0002,0100) and
        # PrivateInformation (0002,0102) — implementation-specific private data.
        for private_meta_tag in [(0x0002, 0x0100), (0x0002, 0x0102)]:
            if private_meta_tag in ds.file_meta:
                del ds.file_meta[private_meta_tag]

    # 7. Stamp mandatory PS3.15 conformance tags (E.1.1).
    #    PatientIdentityRemoved (0012,0062) — required by the standard; value must be "YES".
    ds.PatientIdentityRemoved = "YES"

    # 8. DeidentificationMethodCodeSequence (0012,0064) — declares which profile
    #    and options were applied, using codes from DICOM CID 7050.
    #    113100 = Basic Application Level Confidentiality Profile
    #    113101 = Clean Pixel Data Option
    #    113105 = Clean Descriptors Option
    deid_item = pydicom.Dataset()
    deid_item.CodeValue = "113100"
    deid_item.CodingSchemeDesignator = "DCM"
    deid_item.CodeMeaning = "Basic Application Level Confidentiality Profile"

    deid_item_pixel = pydicom.Dataset()
    deid_item_pixel.CodeValue = "113101"
    deid_item_pixel.CodingSchemeDesignator = "DCM"
    deid_item_pixel.CodeMeaning = "Clean Pixel Data Option"

    deid_item_desc = pydicom.Dataset()
    deid_item_desc.CodeValue = "113105"
    deid_item_desc.CodingSchemeDesignator = "DCM"
    deid_item_desc.CodeMeaning = "Clean Descriptors Option"

    deid_item_pat = pydicom.Dataset()
    deid_item_pat.CodeValue = "113108"
    deid_item_pat.CodingSchemeDesignator = "DCM"
    deid_item_pat.CodeMeaning = "Retain Patient Characteristics Option"

    ds.DeidentificationMethodCodeSequence = pydicom.Sequence(
        [deid_item, deid_item_pixel, deid_item_desc, deid_item_pat]
    )

    return ds


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def anonymize_study(
    study: DicomStudy,
    output_dir: str,
    anon_patient_id: Optional[str] = None,
    uid_salt: str = "pipeline-salt-v1",
    clf=None,
    ocr_reader=None,
) -> dict:
    """
    Anonymize all DICOM instances in *study* and write them to *output_dir*.

    The anonymization values (UIDs, redacted free-text) are computed ONCE
    from the first instance of the study, then applied identically to every
    instance. This ensures all 160 slices share the same new StudyUID.

    Parameters
    ----------
    study : DicomStudy
        Output of dicom_reader.read_and_group().
    output_dir : str
        Folder to write anonymized .dcm files. Created if it does not exist.
        Files are written flat (no sub-folders) using the new SOPInstanceUID
        as the filename.
    anon_patient_id : str, optional
        The pseudonym ID to use for PatientID and PatientName.
        If not provided, a deterministic ID is derived from the original
        PatientID using the same salt as the UID mapping.
    uid_salt : str
        Salt for deterministic UID generation. Change to produce a different
        set of anonymized UIDs from the same source data.

    Returns
    -------
    dict
        Summary containing:
          - anon_patient_id   : str  — the pseudonym used
          - uid_map           : dict — {original_uid: anonymized_uid}
          - redacted_tags     : dict — {tag: redacted_value}
          - files_written     : int
          - output_dir        : str
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Derive anon_patient_id if not provided ──
    if anon_patient_id is None:
        h = hashlib.sha256(
            f"{uid_salt}:patient:{study.patient_id}".encode()
        ).hexdigest()[:12].upper()
        anon_patient_id = f"ANON-{h}"

    print(f"[INFO] Anonymizing study for patient: '{study.patient_name}'")
    print(f"[INFO] Original PatientID : {study.patient_id}")
    print(f"[INFO] Anon    PatientID  : {anon_patient_id}")
    print(f"[INFO] Total instances    : {study.total_instances}")
    print(f"[INFO] Output dir         : {output_path}")
    print()

    # ── Load privacy-filter once ──
    if clf is None:
        clf = load_privacy_filter()

    # ── Load OCR reader once ──
    if ocr_reader is None:
        ocr_reader = load_ocr_reader()

    # ── Compute anonymization values from the first instance ──
    all_instances = study.all_instances
    first_inst = all_instances[0]

    # Load the first instance WITH pixel data to get a complete dataset
    first_ds = pydicom.dcmread(str(first_inst.path), force=True)

    uid_map = build_uid_map(first_ds)
    redacted_free_texts = build_redacted_free_texts(first_ds, clf)

    # Extract window params from first instance for pixel scrubbing
    wc = getattr(first_ds, "WindowCenter", None)
    ww = getattr(first_ds, "WindowWidth",  None)
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = float(wc[0])
    elif wc is not None:
        wc = float(wc)
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = float(ww[0])
    elif ww is not None:
        ww = float(ww)
    print()
    print(f"[INFO] UID map ({len(uid_map)} UIDs remapped):")
    for orig, anon in uid_map.items():
        print(f"  ...{orig[-20:]:>22}  ->  {anon}")
    print()

    # ── OCR runs on every instance unconditionally ──
    # PS3.15 E.3.1 (Clean Pixel Data Option) requires that identifying
    # information is removed from pixel data regardless of how that is achieved.
    # Sampling only a subset of slices is non-compliant because burned-in
    # annotations may appear on specific frames (e.g. calibration slices,
    # last frame of a cine) not present in any fixed-size sample.
    run_ocr = True
    print("[INFO] OCR pixel scrubbing will run on all slices (PS3.15 E.3.1).")
    print()

    # ── Loop: apply to every instance ──
    files_written = 0
    sop_uid_to_filename: dict[str, str] = {}
    for idx, inst in enumerate(all_instances, 1):
        # Load full dataset including pixel data
        ds = pydicom.dcmread(str(inst.path), force=True)

        # Apply all anonymization in-place
        _anonymize_dataset(ds, anon_patient_id, redacted_free_texts, uid_map)

        #Apply OCR pixel scrubbing if burned-in text was found in sample
        if run_ocr:
            scrub_pixels(ds, ocr_reader, wc, ww)
        # Output filename = new SOPInstanceUID (no original info in filename)
        ds.BurnedInAnnotation = "NO"  # PS3.15 E.3.1 mandatory flag
        new_sop_uid = str(ds.SOPInstanceUID)
        out_filename = f"{new_sop_uid}.dcm"
        out_path = output_path / out_filename

        ds.save_as(str(out_path))
        # Record original → actual filename mapping for NIfTI converter
        sop_uid_to_filename[inst.sop_instance_uid] = out_filename

        files_written += 1

        # Progress indicator every 10 files
        if idx % 10 == 0 or idx == len(all_instances):
            print(f"  [{idx:>4}/{len(all_instances)}] written: {out_filename[-30:]}")

    print()
    print(f"[INFO] Done. {files_written} anonymized files written to: {output_path}")

    return {
        "anon_patient_id":  anon_patient_id,
        "uid_map":          uid_map,
        "redacted_tags":    {str(k): v for k, v in redacted_free_texts.items()},
        "files_written":    files_written,
        "output_dir":       str(output_path),
        "sop_uid_to_filename":   sop_uid_to_filename
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dicom_reader import read_and_group, print_study_tree

    if len(sys.argv) < 3:
        print("Usage: python dicom_anonymizer.py <input_dicom_dir> <output_dicom_dir>")
        print("Example: python dicom_anonymizer.py input/dicom/ output/dicom/patient_001/")
        sys.exit(1)

    input_dir  = sys.argv[1]
    output_dir = sys.argv[2]

    print("=" * 60)
    print("STEP 1: Reading DICOM folder")
    print("=" * 60)
    study = read_and_group(input_dir)
    print_study_tree(study)

    print("=" * 60)
    print("STEP 2: Anonymizing")
    print("=" * 60)
    result = anonymize_study(study, output_dir)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Anon patient ID : {result['anon_patient_id']}")
    print(f"  Files written   : {result['files_written']}")
    print(f"  Output dir      : {result['output_dir']}")
