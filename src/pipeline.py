"""
pipeline.py
-----------
Full anonymization pipeline runner for a single patient.

Runs all stages in sequence:
  1. Read DICOM folder
  2. Load all models (privacy-filter, EasyOCR) — once, shared across stages
  3. Anonymize DICOM files → output/dicom/
  4. Convert anonymized DICOM to NIfTI → output/nifti/
  5. Extract and anonymize PDF report
  6. Build FHIR R4 bundle → output/fhir/

No stage assumes any prior output exists on disk. Every stage receives
its inputs directly from the previous stage's in-memory return values.
Models are loaded once in Stage 2 and passed explicitly to every stage
that needs them — never loaded twice.

Usage
-----
    python pipeline.py <dicom_input_dir> <pdf_path> <output_dir>

Example
-------
    python pipeline.py \\
        "input/141-03-2026ME/" \\
        "input/anbarasu-r_36_Yrs_male_73861.pdf" \\
        "output/"

Output structure
----------------
    output/
      dicom/
        patient_{anon_id}/
          {anon_sop_uid}.dcm  × N
      nifti/
        patient_{anon_id}/
          series_{modality}_{uid_short}.nii.gz
      fhir/
        patient_{anon_id}_bundle.json
"""

import sys
import time
from pathlib import Path

from dicom_reader import read_and_group, print_study_tree
from dicom_anonymizer import anonymize_study, load_privacy_filter
from dicom_to_nifti import convert_to_nifti
from pdf_extractor import extract_pdf
from fhir_mapper import map_to_fhir
from pixel_scrubber import load_ocr_reader


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _elapsed(start: float) -> str:
    s = time.time() - start
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m {int(s % 60)}s"


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    dicom_input_dir: str,
    pdf_path: str,
    output_dir: str,
) -> dict:
    """
    Run the full anonymization pipeline for one patient.

    Parameters
    ----------
    dicom_input_dir : str
        Path to the flat folder of original .dcm files.
    pdf_path : str
        Path to the radiology or lab report PDF.
    output_dir : str
        Root output folder. Subfolders dicom/, nifti/, fhir/ are created
        inside it automatically.

    Returns
    -------
    dict
        Summary of all outputs:
          - anon_patient_id  : str
          - dicom_output_dir : str
          - nifti_files      : list[str]
          - fhir_bundle_path : str
          - total_elapsed    : str
    """
    pipeline_start = time.time()

    dicom_input_dir = Path(dicom_input_dir)
    pdf_path        = Path(pdf_path)
    output_dir      = Path(output_dir)

    # ── Validate inputs ──
    if not dicom_input_dir.exists():
        raise FileNotFoundError(f"DICOM input dir not found: {dicom_input_dir}")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ── Define output subdirectories ──
    dicom_out = output_dir / "dicom"
    nifti_out = output_dir / "nifti"
    fhir_out  = output_dir / "fhir"

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1 — Read DICOM folder
    # ─────────────────────────────────────────────────────────────────────────
    _banner("STAGE 1 — Reading DICOM folder")
    t = time.time()

    study = read_and_group(str(dicom_input_dir))
    print_study_tree(study)

    print(f"  [Done in {_elapsed(t)}]")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2 — Load all models once
    # Both privacy-filter and EasyOCR are loaded here and passed explicitly
    # to every downstream stage that needs them. Neither is loaded again.
    # ─────────────────────────────────────────────────────────────────────────
    _banner("STAGE 2 — Loading models")
    t = time.time()

    clf        = load_privacy_filter()
    ocr_reader = load_ocr_reader()

    print(f"  [Done in {_elapsed(t)}]")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3 — Anonymize DICOM files
    # Receives: study (from Stage 1), clf + ocr_reader (from Stage 2)
    # Produces: anon_result dict containing anon_patient_id, uid_map,
    #           output_dir of anonymized .dcm files, files_written count
    # ─────────────────────────────────────────────────────────────────────────
    _banner("STAGE 3 — Anonymizing DICOM files")
    t = time.time()

    anon_result = anonymize_study(
        study,
        output_dir=str(dicom_out / f"patient_{study.patient_id}"),
        clf=clf,
        ocr_reader=ocr_reader,
    )

    anon_patient_id = anon_result["anon_patient_id"]
    print(f"  [Done in {_elapsed(t)}]")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4 — Convert anonymized DICOM to NIfTI
    # Receives: study (from Stage 1), anon_result (from Stage 3)
    # anon_result["output_dir"] is the folder written by Stage 3 —
    # this stage reads from it immediately after Stage 3 completes.
    # Produces: list of nifti_results dicts (one per series)
    # ─────────────────────────────────────────────────────────────────────────
    _banner("STAGE 4 — Converting to NIfTI")
    t = time.time()

    nifti_results = convert_to_nifti(
        study,
        anon_result,
        output_dir=str(nifti_out),
    )

    print(f"  [Done in {_elapsed(t)}]")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 5 — Extract and anonymize PDF
    # Receives: clf (from Stage 2), anon_patient_id (from Stage 3)
    # Produces: PdfExtract dataclass with all fields anonymized
    # ─────────────────────────────────────────────────────────────────────────
    _banner("STAGE 5 — Extracting and anonymizing PDF")
    t = time.time()

    pdf_extract = extract_pdf(
        str(pdf_path),
        clf,
        anon_patient_id=anon_patient_id.removeprefix("ANON-"),
    )

    print(f"  [Done in {_elapsed(t)}]")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 6 — Build FHIR R4 bundle
    # Receives: study (Stage 1), anon_result (Stage 3),
    #           pdf_extract (Stage 5), nifti_results (Stage 4)
    # Produces: FHIR R4 bundle JSON written to fhir_out/
    # ─────────────────────────────────────────────────────────────────────────
    _banner("STAGE 6 — Building FHIR R4 bundle")
    t = time.time()

    bundle = map_to_fhir(
        study,
        anon_result,
        pdf_extract,
        nifti_results,
        output_dir=str(fhir_out),
    )

    fhir_bundle_path = (
        fhir_out / f"patient_{anon_patient_id}_bundle.json"
    )

    print(f"  [Done in {_elapsed(t)}]")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    total = _elapsed(pipeline_start)

    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Anon patient ID    : {anon_patient_id}")
    print(f"  DICOM files        : {anon_result['files_written']} files")
    print(f"    → {anon_result['output_dir']}")
    print(f"  NIfTI files        : {len(nifti_results)}")
    for r in nifti_results:
        size_mb = Path(r["nifti_path"]).stat().st_size / (1024 * 1024)
        print(f"    → {Path(r['nifti_path']).name}  ({size_mb:.1f} MB)")
    print(f"  FHIR bundle        : {fhir_bundle_path.name}")
    print(f"    → {fhir_bundle_path}")
    print(f"  Total time         : {total}")
    print("=" * 60)

    return {
        "anon_patient_id":  anon_patient_id,
        "dicom_output_dir": anon_result["output_dir"],
        "nifti_files":      [r["nifti_path"] for r in nifti_results],
        "fhir_bundle_path": str(fhir_bundle_path),
        "total_elapsed":    total,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python pipeline.py <dicom_input_dir> <pdf_path> <output_dir>")
        print()
        print("Example:")
        print("  python pipeline.py \\")
        print('      "input/141-03-2026ME/" \\')
        print('      "input/anbarasu-r_36_Yrs_male_73861.pdf" \\')
        print('      "output/"')
        sys.exit(1)

    run_pipeline(
        dicom_input_dir=sys.argv[1],
        pdf_path=sys.argv[2],
        output_dir=sys.argv[3],
    )