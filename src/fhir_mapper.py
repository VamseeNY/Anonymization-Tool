"""
fhir_mapper.py
--------------
Stage 3 of the pipeline — FHIR R4 mapping.

Takes the outputs of both the DICOM anonymizer and the PDF extractor
and builds a valid FHIR R4 transaction Bundle containing:

    Patient
    ImagingStudy      (one per DICOM study, with all series and instances)
    DiagnosticReport  (one per PDF report)
    Observation       (one per finding sentence from the PDF)

The bundle is written as a single JSON file to the output directory.

Design decisions
----------------
- No external FHIR library used — resources are built as plain Python dicts
  and serialized with json.dumps. This keeps dependencies minimal and makes
  the structure fully transparent.
- All IDs are either derived from anonymized UIDs (for DICOM resources) or
  generated as deterministic UUIDs (for PDF resources) so re-running the
  pipeline on the same inputs always produces the same output.
- Instance-level SOP UIDs in ImagingStudy are re-derived using make_anon_uid
  from the original SOPInstanceUIDs stored in the DicomStudy object. This
  matches the filenames of the anonymized .dcm files exactly.
- LOINC codes are assigned via a keyword lookup on the study title. If no
  keyword matches, a generic diagnostic imaging LOINC code is used.
- effectiveDateTime is omitted (date was wiped during anonymization).

Usage
-----
    from dicom_reader import read_and_group
    from dicom_anonymizer import anonymize_study, load_privacy_filter
    from pdf_extractor import extract_pdf
    from fhir_mapper import map_to_fhir

    study = read_and_group("input/dicom/")
    anon_result = anonymize_study(study, "output/dicom/")
    clf = load_privacy_filter()
    pdf = extract_pdf("input/report.pdf", clf, anon_result["anon_patient_id"])
    map_to_fhir(study, anon_result, pdf, "output/fhir/")

Requires
--------
    pip install pydicom  (already installed for dicom_reader)
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dicom_reader import read_and_group, DicomStudy
from dicom_anonymizer import anonymize_study, load_privacy_filter, make_anon_uid
from pdf_extractor import extract_pdf, PdfExtract
from dicom_to_nifti import convert_to_nifti


# ─────────────────────────────────────────────────────────────────────────────
#  LOINC lookup — study title keyword → (code, display)
# ─────────────────────────────────────────────────────────────────────────────

_LOINC_LOOKUP: list[tuple[str, str, str]] = [
    # keyword (lowercase)  code         display
    ("brain",              "24725-4",   "CT of head"),
    ("head",               "24725-4",   "CT of head"),
    ("chest",              "24627-2",   "CT of chest"),
    ("thorax",             "24627-2",   "CT of chest"),
    ("abdomen",            "36643-5",   "CT of abdomen"),
    ("pelvis",             "24729-6",   "CT of pelvis"),
    ("spine",              "24775-9",   "CT of spine"),
    ("lumbar",             "24775-9",   "CT of spine"),
    ("cervical",           "24775-9",   "CT of spine"),
    ("liver",              "24815-3",   "CT of liver"),
    ("kidney",             "44136-0",   "CT of kidney"),
    ("neck",               "24746-0",   "CT of neck"),
]

_LOINC_DEFAULT = ("18748-4", "Diagnostic imaging report")
_LOINC_FINDING = ("59776-5", "Findings")


def _get_loinc(study_title: Optional[str]) -> tuple[str, str]:
    """
    Return (loinc_code, display) for the given study title string.
    Matches the first keyword found in the title (case insensitive).
    Falls back to generic diagnostic imaging code if no match.
    """
    if not study_title:
        return _LOINC_DEFAULT
    title_lower = study_title.lower()
    for keyword, code, display in _LOINC_LOOKUP:
        if keyword in title_lower:
            return code, display
    return _LOINC_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
#  Deterministic UUID helper
# ─────────────────────────────────────────────────────────────────────────────

def _det_uuid(seed: str) -> str:
    """
    Generate a deterministic UUID v5 from a seed string.
    Same seed always produces the same UUID — ensures re-runs are idempotent.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


# ─────────────────────────────────────────────────────────────────────────────
#  Gender normalization
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_gender(sex: Optional[str]) -> str:
    """
    Normalize sex string to FHIR R4 gender values:
    male | female | other | unknown
    """
    if not sex:
        return "unknown"
    s = sex.strip().lower()
    if s in ("male", "m"):
        return "male"
    if s in ("female", "f"):
        return "female"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
#  Resource builders
# ─────────────────────────────────────────────────────────────────────────────

def build_patient(
    anon_patient_id: str,
    pdf: PdfExtract,
) -> dict:
    """
    Build a FHIR R4 Patient resource.

    Parameters
    ----------
    anon_patient_id : str
        The bare pseudonym (without ANON- prefix), e.g. "5CDDFE3D5457".
    pdf : PdfExtract
        Anonymized PDF extract — used for gender only.
    """
    return {
        "resourceType": "Patient",
        "id": anon_patient_id,
        "identifier": [
            {
                "system": "urn:oid:1.2.3.4.5.anonymized",
                "value":  anon_patient_id,
            }
        ],
        "name": [
            {
                "use":  "anonymous",
                "text": f"ANON-{anon_patient_id}",
            }
        ],
        "gender": _normalize_gender(pdf.sex),
    }


def build_imaging_study(
    study: DicomStudy,
    anon_result: dict,
    patient_id: str,
) -> dict:
    """
    Build a FHIR R4 ImagingStudy resource including all series and instances.

    Parameters
    ----------
    study : DicomStudy
        Original (pre-anonymization) study object from dicom_reader.
    anon_result : dict
        Output of anonymize_study() — contains uid_map and anon_patient_id.
    patient_id : str
        Bare anonymized patient ID for the subject reference.
    """
    uid_map = anon_result["uid_map"]
    anon_study_uid = uid_map.get(study.study_uid, make_anon_uid(study.study_uid))

    # Collect all modalities across series (deduplicated)
    modalities = list({
        s.modality for s in study.series.values() if s.modality
    })

    # Build series list
    series_list = []
    for orig_series_uid, series in study.series.items():
        anon_series_uid = uid_map.get(orig_series_uid, make_anon_uid(orig_series_uid))

        # Build instance list — re-derive each anon SOP UID from original
        instances = []
        for inst in series.instances:
            anon_sop = make_anon_uid(inst.sop_instance_uid)
            instance_entry = {
                "uid":    anon_sop,
                "number": inst.instance_number,
            }
            # Add SOPClassUID if available in the dataset
            sop_class = getattr(inst.dataset, "SOPClassUID", None)
            if sop_class:
                instance_entry["sopClass"] = {
                    "system": "urn:ietf:rfc:3986",
                    "code":   f"urn:oid:{str(sop_class)}",
                }
            instances.append(instance_entry)

        series_entry = {
            "uid":               anon_series_uid,
            "number":            series.series_number,
            "modality": {
                "system":  "http://dicom.nema.org/resources/ontology/DCM",
                "code":    series.modality,
                "display": series.modality,
            },
            "numberOfInstances": len(series.instances),
            "instance":          instances,
        }

        # Add series description if present
        if series.series_description and series.series_description != "(none)":
            series_entry["description"] = series.series_description

        series_list.append(series_entry)

    resource = {
        "resourceType":       "ImagingStudy",
        "id":                 anon_study_uid,
        "status":             "available",
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
        "numberOfSeries":     len(study.series),
        "numberOfInstances":  study.total_instances,
        "series":             series_list,
    }

    # Add modality at study level
    if modalities:
        resource["modality"] = [
            {
                "system":  "http://dicom.nema.org/resources/ontology/DCM",
                "code":    m,
                "display": m,
            }
            for m in modalities
        ]

    # Add study description if present
    if study.study_description:
        resource["description"] = study.study_description

    return resource


def build_observations(
    pdf: PdfExtract,
    patient_id: str,
    report_id: str,
) -> list[dict]:
    """
    Build one FHIR R4 Observation resource per finding sentence.

    Parameters
    ----------
    pdf : PdfExtract
        Anonymized PDF extract — findings list used.
    patient_id : str
        Bare anonymized patient ID for the subject reference.
    report_id : str
        ID of the DiagnosticReport that these observations belong to.
    """
    observations = []
    loinc_code, loinc_display = _LOINC_FINDING

    for idx, finding_text in enumerate(pdf.findings):
        if not finding_text or not finding_text.strip():
            continue

        obs_id = _det_uuid(f"{report_id}:finding:{idx}")

        observations.append({
            "resourceType": "Observation",
            "id":           obs_id,
            "status":       "final",
            "category": [
                {
                    "coding": [
                        {
                            "system":  "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code":    "imaging",
                            "display": "Imaging",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system":  "http://loinc.org",
                        "code":    loinc_code,
                        "display": loinc_display,
                    }
                ],
                "text": "Imaging Finding",
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
            },
            "valueString": finding_text,
        })

    return observations


def build_diagnostic_report(
    pdf: PdfExtract,
    patient_id: str,
    imaging_study_id: str,
    observation_ids: list[str],
) -> dict:
    """
    Build a FHIR R4 DiagnosticReport resource.

    Parameters
    ----------
    pdf : PdfExtract
        Anonymized PDF extract.
    patient_id : str
        Bare anonymized patient ID.
    imaging_study_id : str
        ID of the linked ImagingStudy resource.
    observation_ids : list[str]
        IDs of all Observation resources to reference as results.
    """
    report_id = _det_uuid(f"{patient_id}:diagnostic-report:{pdf.study_title or 'report'}")
    loinc_code, loinc_display = _get_loinc(pdf.study_title)

    resource = {
        "resourceType": "DiagnosticReport",
        "id":           report_id,
        "status":       "final",
        "category": [
            {
                "coding": [
                    {
                        "system":  "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code":    "RAD",
                        "display": "Radiology",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system":  "http://loinc.org",
                    "code":    loinc_code,
                    "display": loinc_display,
                }
            ],
            "text": pdf.study_title or "Diagnostic Imaging Report",
        },
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
        "imagingStudy": [
            {
                "reference": f"ImagingStudy/{imaging_study_id}",
            }
        ],
        "result": [
            {"reference": f"Observation/{obs_id}"}
            for obs_id in observation_ids
        ],
    }

    # Add conclusion (impression) if present
    if pdf.impression:
        resource["conclusion"] = pdf.impression

    # Add technique as presentedForm text if present
    if pdf.technique:
        resource["extension"] = [
            {
                "url":         "http://hl7.org/fhir/StructureDefinition/imaging-technique",
                "valueString": pdf.technique,
            }
        ]

    return resource

def build_document_references(
    nifti_results: list[dict],
    patient_id: str,
    imaging_study_id: str,
) -> list[dict]:
    """
    Build one FHIR R4 DocumentReference resource per NIfTI file.

    Parameters
    ----------
    nifti_results : list[dict]
        Output of dicom_to_nifti.convert_to_nifti() — one dict per series.
    patient_id : str
        Bare anonymized patient ID.
    imaging_study_id : str
        ID of the linked ImagingStudy resource.
    """
    doc_refs = []

    for nifti in nifti_results:
        doc_id = _det_uuid(f"{patient_id}:docref:{nifti['series_uid']}")

        doc_refs.append({
            "resourceType": "DocumentReference",
            "id":           doc_id,
            "status":       "current",
            "type": {
                "coding": [
                    {
                        "system":  "http://loinc.org",
                        "code":    "18748-4",
                        "display": "Diagnostic imaging report",
                    }
                ]
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
            },
            "content": [
                {
                    "attachment": {
                        "contentType": "application/octet-stream",
                        "url":         Path(nifti["nifti_path"]).resolve().as_uri(),
                        "title":       (
                            f"{nifti['modality']} volume — "
                            f"{nifti['instance_count']} slices — "
                            f"anonymized NIfTI"
                        ),
                    }
                }
            ],
            "context": {
                "related": [
                    {
                        "reference": f"ImagingStudy/{imaging_study_id}",
                    }
                ]
            },
        })

    return doc_refs

def build_bundle(
    patient: dict,
    imaging_study: dict,
    diagnostic_report: dict,
    observations: list[dict],
    document_references: list[dict],
    bundle_id: str,
) -> dict:
    """
    Assemble all resources into a FHIR R4 transaction Bundle.

    Each entry has:
      - fullUrl  : urn:uuid:<resource_id>
      - resource : the resource dict
      - request  : PUT method (idempotent — re-running produces same IDs)
    """
    entries = []

    def _make_entry(resource: dict) -> dict:
        rtype = resource["resourceType"]
        rid   = resource["id"]
        return {
            "fullUrl":  f"urn:uuid:{rid}",
            "resource": resource,
            "request": {
                "method": "PUT",
                "url":    f"{rtype}/{rid}",
            },
        }

    # Order matters for FHIR transaction: Patient first, then dependents
    entries.append(_make_entry(patient))
    entries.append(_make_entry(imaging_study))
    for doc_ref in document_references:
        entries.append(_make_entry(doc_ref))
    for obs in observations:
        entries.append(_make_entry(obs))
    entries.append(_make_entry(diagnostic_report))

    return {
        "resourceType": "Bundle",
        "id":           bundle_id,
        "type":         "transaction",
        "timestamp":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entry":        entries,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def map_to_fhir(
    study: DicomStudy,
    anon_result: dict,
    pdf: PdfExtract,
    nifti_results: list[dict],
    output_dir: str,
) -> dict:
    """
    Build a FHIR R4 transaction Bundle from DICOM and PDF pipeline outputs
    and write it to output_dir as a JSON file.

    Parameters
    ----------
    study : DicomStudy
        Original study object from dicom_reader.read_and_group().
    anon_result : dict
        Output of dicom_anonymizer.anonymize_study().
    pdf : PdfExtract
        Output of pdf_extractor.extract_pdf().
    output_dir : str
        Directory to write the bundle JSON. Created if it does not exist.

    Returns
    -------
    dict
        The assembled FHIR R4 Bundle as a Python dict.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    anon_patient_id = anon_result["anon_patient_id"]
    uid_map         = anon_result["uid_map"]
    anon_study_uid  = uid_map.get(study.study_uid, make_anon_uid(study.study_uid))

    print(f"[INFO] Building FHIR R4 bundle ...")
    print(f"  Patient ID      : {anon_patient_id}")
    print(f"  ImagingStudy ID : {anon_study_uid}")

    # ── Build resources ──
    patient        = build_patient(anon_patient_id, pdf)

    imaging_study  = build_imaging_study(study, anon_result, anon_patient_id)

    # DiagnosticReport ID is needed before building observations
    # (observations reference the report seed for deterministic IDs)
    report_id = _det_uuid(
        f"{anon_patient_id}:diagnostic-report:{pdf.study_title or 'report'}"
    )

    observations   = build_observations(pdf, anon_patient_id, report_id)
    obs_ids        = [o["id"] for o in observations]

    diagnostic_report = build_diagnostic_report(
        pdf,
        anon_patient_id,
        anon_study_uid,
        obs_ids,
    )

    bundle_id = _det_uuid(f"{anon_patient_id}:bundle")
    document_references = build_document_references(
        nifti_results,
        anon_patient_id,
        anon_study_uid,
    )

    bundle_id = _det_uuid(f"{anon_patient_id}:bundle")
    bundle    = build_bundle(
        patient,
        imaging_study,
        diagnostic_report,
        observations,
        document_references,
        bundle_id,
    )

    # ── Write to disk ──
    out_filename = f"patient_{anon_patient_id}_bundle.json"
    out_path     = output_path / out_filename

    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Bundle written: {out_path}")
    print(f"  Resources  : 1 Patient + 1 ImagingStudy + "
          f"{len(document_references)} DocumentReference(s) + "
          f"{len(observations)} Observations + 1 DiagnosticReport")
    print(f"  Total entries in bundle : {len(bundle['entry'])}")
    print(f"  Output file : {out_path}")

    return bundle


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dicom_reader import read_and_group
    from dicom_anonymizer import anonymize_study, load_privacy_filter
    from pdf_extractor import extract_pdf

    if len(sys.argv) < 6:
        print("Usage:")
        print("  python fhir_mapper.py <dicom_input_dir> <dicom_output_dir> "
              "<pdf_path> <anon_patient_id> <fhir_output_dir>")
        print()
        print("Example:")
        print("  python fhir_mapper.py \\")
        print('      "input/141-03-2026ME/" \\')
        print('      "output/dicom/ANON/" \\')
        print('      "input/report.pdf" \\')
        print('      "5CDDFE3D5457" \\')
        print('      "output/fhir/"')
        sys.exit(1)

    dicom_input_dir  = sys.argv[1]
    dicom_output_dir = sys.argv[2]
    pdf_path         = sys.argv[3]
    anon_patient_id  = sys.argv[4]
    fhir_output_dir  = sys.argv[5]

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
    print("STEP 3: Extracting PDF")
    print("=" * 60)
    pdf = extract_pdf(pdf_path, clf, anon_result["anon_patient_id"])

    print("=" * 60)
    print("STEP 4: Converting to NIfTI")          # ← new step
    print("=" * 60)
    nifti_results = convert_to_nifti(study, anon_result, "output/nifti/")

    print("=" * 60)
    print("STEP 5: Building FHIR R4 bundle")       # ← was STEP 4
    print("=" * 60)
    bundle = map_to_fhir(study, anon_result, pdf, nifti_results, fhir_output_dir)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Anonymized DICOM : {dicom_output_dir}")
    print(f"  FHIR bundle      : {fhir_output_dir}/patient_{anon_result['anon_patient_id']}_bundle.json")