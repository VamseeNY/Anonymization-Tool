# Data Anonymization Tool — Medical Imaging Pipeline

A single-patient anonymization pipeline that processes DICOM imaging files and radiology/lab report PDFs, anonymizes all PII, converts imaging data to NIfTI format, outputs structured clinical data in FHIR R4 JSON format, and exposes a FastAPI backend with a browser-based frontend.

---

## Overview

```
Input:  1 DICOM folder (N .dcm files)  +  1 PDF report
          │                                    │
          ▼                                    ▼
    DICOM Anonymizer                    PDF Extractor
    Pixel Scrubber (OCR)                PII Redaction
    NIfTI Converter                     FHIR DiagnosticReport
    FHIR ImagingStudy                   FHIR Observation(s)
    FHIR DocumentReference                    │
          │                                   │
          └──────────────┬────────────────────┘
                         ▼
                  FHIR R4 Bundle JSON
                         +
                  Anonymized DICOM files
                         +
                  Anonymized NIfTI volume
                         +
                  FastAPI backend + browser frontend
```

---

## Quick Start

### Command line

```bash
pip install pydicom pdfplumber "transformers>=5.6.0" torch fhir.resources \
    Pillow dicom2nifti nibabel matplotlib easyocr fastapi uvicorn \
    python-multipart aiofiles

python pipeline.py \
    "input/141-03-2026ME/" \
    "input/report.pdf" \
    "output/"
```

### Web interface

```bash
python src/api.py
# Open http://localhost:8000 in your browser
```

---

## Project Structure

```
src/
  api.py                  ← FastAPI backend
  pipeline.py             ← End-to-end CLI pipeline runner
  dicom_reader.py         ← Stage 1 — DICOM folder reader
  dicom_anonymizer.py     ← Stage 2 — DICOM PII anonymizer + pixel scrubber
  pixel_scrubber.py       ← Stage 2b — OCR burned-in annotation scrubber
  dicom_to_nifti.py       ← Stage 3 — DICOM to NIfTI converter
  pdf_extractor.py        ← Stage 4 — PDF extractor and anonymizer
  fhir_mapper.py          ← Stage 5 — FHIR R4 bundle builder
  dicom_verify.py         ← Utility — DICOM anonymization verifier
  pdf_inspect.py          ← Utility — PDF raw text inspector
  nifti_viewer.py         ← Utility — NIfTI slice plotter
  frontend/
    index.html            ← Landing page
    login.html            ← Authentication page
    upload.html           ← File upload + live pipeline progress (SSE)
    results.html          ← Download outputs + FHIR bundle preview
    assets/
      style.css           ← Design system
      script.js           ← Shared layout helpers and auth
```

---

## Output Structure

```
output/
  dicom/
    patient_{original_id}/
      {anon_sop_uid}.dcm  ×  N      # anonymized DICOM files
  nifti/
    patient_{anon_id}/
      series_{modality}_{uid}.nii.gz # anonymized NIfTI volume
  fhir/
    patient_{anon_id}_bundle.json    # FHIR R4 transaction bundle
```

---

## Pipeline Stages

### Stage 1 — DICOM Reading (`dicom_reader.py`)

Reads all `.dcm` files from the input folder and organises them into a three-level hierarchy:

```
DicomStudy
  └── DicomSeries (grouped by SeriesInstanceUID)
        └── DicomInstance (sorted by InstanceNumber)
```

**Key decisions:**
- Uses `stop_before_pixels=True` during the scan pass — reads only metadata headers, not pixel data. This keeps the scan fast even for large studies (300+ files).
- Pixel data is loaded lazily per-instance only when needed for anonymization and NIfTI conversion.
- Instances are sorted by `InstanceNumber` within each series. Files with missing `InstanceNumber` sort to the end.

---

### Stage 2 — DICOM Anonymization (`dicom_anonymizer.py`)

Anonymizes all PII-carrying DICOM tags across every instance in the study and writes anonymized `.dcm` files to the output folder.

#### Protocol followed: DICOM PS3.15 Annex E — Basic Application Level Confidentiality Profile

This is the HL7/NEMA standard for DICOM de-identification. Three categories of tags are handled:

**Wiped tags (set to empty string):**
Patient identity, dates/times, device identifiers, and institutional information including:
`PatientBirthDate`, `PatientAge`, `PatientSize`, `PatientWeight`, `PatientAddress`,
`PatientTelephoneNumbers`, `AccessionNumber`, `InstitutionName`, `InstitutionAddress`,
`ReferringPhysicianName`, `PerformingPhysicianName`, `OperatorsName`, `StationName`,
`DeviceSerialNumber`, `StudyDate`, `SeriesDate`, `AcquisitionDate`, `ContentDate`,
`StudyTime`, `SeriesTime`, `AcquisitionTime`, `ContentTime`, scheduling tags,
admission tags, request attribute tags, and others per PS3.15 Annex E.

Conformance declaration tags stamped on every output instance:

PatientIdentityRemoved (0012,0062) → set to "YES" (mandatory per PS3.15 E.1.1)  
DeidentificationMethodCodeSequence (0012,0064) → populated with CID 7050 codes  
declaring the applied profile and options:  
  113100 — Basic Application Level Confidentiality Profile  
  113101 — Clean Pixel Data Option  
  113105 — Clean Descriptors Option  

**Replaced with pseudonym:**
- `PatientName` → `ANON-{anon_patient_id}`
- `PatientID` → `{anon_patient_id}`
- `PatientSex` is kept (not directly identifying alone)

**UIDs remapped deterministically:**
All DICOM UIDs are replaced using UUID v5 (RFC 4122 name-based SHA-1) mapped to the 2.25. OID arc (the registered OID arc for UUID-derived UIDs per ITU-T X.667 / ISO IEC 9834-8 / DICOM PS3.5 B.2).  
A stable namespace UUID is first derived from the pipeline salt using UUID v5 with the IETF DNS namespace. The output UID for each original UID is then UUID v5 (namespace=salt-derived UUID, name=original_uid). The 128-bit UUID integer is converted to decimal and prepended with "2.25." per the ITU-T X.667 encoding rule. This produces a standards-conformant OID under the 2.25. arc, unlike a raw SHA-256 truncation which is not a valid UUID:  
- `StudyInstanceUID` — same new UID across all files in the study
- `SeriesInstanceUID` — same new UID across all files in the series
- `SOPInstanceUID` — unique new UID per instance (remapped per-file in the write loop)
- `FrameOfReferenceUID`, `InstanceCreatorUID`, `SynchronizationFrameOfReferenceUID`
- `MediaStorageSOPInstanceUID` in file meta header
- `ReferencedSOPInstanceUID`, `DimensionOrganizationUID` and others inside nested sequences
- SQ-typed tags (ContributingEquipmentSequence, ReferringPhysicianIdentificationSequence) are not placed in the UID set — they are handled by the recursive SQ walker, which descends into them and remaps their inner UID tags individually.

**Free-text tags** (`StudyDescription`, `SeriesDescription`, `AdditionalPatientHistory`, `PatientComments`, `ImageComments`, `StudyComments`) are run through `openai/privacy-filter` to detect and replace incidental PII with `[ENTITY_TYPE]` tokens.

**Recursive nested sequence handling:**
A recursive walker (`_walk_and_anonymize`) traverses every element at every depth of the DICOM dataset. When it encounters an `SQ` (sequence) tag it either wipes the entire sequence (if it is in the denylist) or recurses into each item dataset. This ensures PII inside `ReferencedStudySequence`, `RequestAttributesSequence`, `ContentSequence`, and other nested structures is handled identically to top-level tags.

**Key design decisions:**
- Anonymization values (new UIDs, redacted free-text) are computed **once from the first instance** then stamped identically across all instances. This ensures consistency — all 160 slices share the same new `StudyInstanceUID`.
- UID generation is **deterministic** — same input always produces the same output UID. Re-running the pipeline on the same data produces identical anonymized UIDs.
- Output filenames use the new anonymized `SOPInstanceUID` — no original identifiers appear in filenames.  
  File Meta Information is partially cleaned per PS3.15 E.1.1 / PS3.10:  
  MediaStorageSOPInstanceUID (0002,0003) is remapped to match the new SOPInstanceUID.  
  ImplementationVersionName (0002,0013) is replaced with "ANON-PIPELINE-V1".  
  SourceApplicationEntityTitle (0002,0016) is removed (identifies the acquiring device).  
  PrivateInformationCreatorUID (0002,0100) and PrivateInformation (0002,0102) are removed. The 128-byte preamble is left as-is by pydicom's save_as(); it does not carry structured PHI in standard DICOM files but should be zeroed in high-assurance deployments.
- A `sop_uid_to_filename` mapping is returned in `anon_result` so downstream stages (NIfTI conversion) can reliably locate anonymized files regardless of UID truncation differences between DICOM implementations.

---

### Stage 2b — OCR Pixel Scrubbing (`pixel_scrubber.py`)

Detects and zero-fills burned-in patient annotations in DICOM pixel data using EasyOCR.

**Why needed:** Some modalities (ultrasound, mammography, secondary capture) burn patient name, MRN, and date directly into the pixel data as rendered text. Tag-level anonymization does not touch pixel data, so a separate OCR pass is required.

**Sampling strategy:** OCR runs on every instance unconditionally. PS3.15 E.3.1 (Clean Pixel Data Option) requires that identifying information is removed from pixel data regardless of how that is achieved — no sampling exemption exists in the standard. Burned-in annotations may appear only on specific frames (e.g. calibration slices, first or last frame of a cine loop) that would not be caught by a fixed-size sample.

**Pixel processing pipeline:**
- Raw DICOM pixel array (any bit depth) is windowed to 8-bit greyscale using `WindowCenter`/`WindowWidth` from the DICOM tags (DICOM PS3.3 C.7.6.3.1.2 linear windowing formula)
- EasyOCR runs on the full 8-bit frame
- Detections with confidence ≥ 0.5 are accepted; each bounding box is expanded by 5px padding
- Zero-fill rectangles are applied to the **original** pixel array (not the 8-bit display image), preserving the native bit depth
- BurnedInAnnotation (0028,0301) is set to "NO" on every instance after scrubbing, as required by PS3.15 E.3.1 (Clean Pixel Data Option), regardless of whether OCR detected any text.
- Multi-frame DICOM (e.g. ultrasound cines) is handled frame by frame

**Integration:** The EasyOCR reader is loaded once at pipeline startup and passed to `anonymize_study()`. Per-instance scrubbing runs inside the anonymizer write loop, after tag anonymization and before `dcmwrite()`.

---

### Stage 3 — NIfTI Conversion (`dicom_to_nifti.py`)

Converts each anonymized DICOM series to a single compressed NIfTI volume (`.nii.gz`).

**Conversion tool:** `dicom2nifti` — handles slice ordering, affine matrix construction, and transfer syntax decoding.

**Slice ordering:** `dicom2nifti` uses `ImagePositionPatient` (0020,0032) as the primary sort key — the actual 3D spatial coordinate of each slice. This is more reliable than `InstanceNumber` or filename order. `SliceLocation` and `InstanceNumber` are used as fallbacks. UID anonymization does not affect slice ordering since no spatial geometry tags are modified.

**Per-series temp folder approach:** `dicom2nifti` requires all slices of a series in a single flat folder. For multi-series studies, temporary per-series subfolders are created using **symlinks** (no disk copy of pixel data), conversion runs, then temp folders are deleted in a `finally` block.

**NIfTI header anonymization:** After conversion, the following NIfTI1 header fields are wiped:
- `descrip` (80 chars) — sometimes populated with `StudyDescription` or patient info
- `db_name` (18 chars) — sometimes populated with `PatientName`
- `aux_file` (24 chars) — auxiliary filename field
- `intent_name` (16 chars) — sometimes populated with protocol info

**Display window:** `WindowCenter` and `WindowWidth` from the DICOM tags are extracted and written into the NIfTI header as `cal_min` and `cal_max`:
```
cal_min = WindowCenter - (WindowWidth / 2)
cal_max = WindowCenter + (WindowWidth / 2)
```
This allows viewers like 3D Slicer to apply the correct display window automatically without manual adjustment.

**Sidecar JSON:** Any `.json` sidecar files produced by `dicom2nifti` (BIDS format) are deleted — all relevant metadata is captured in the FHIR `ImagingStudy` resource.

---

### Stage 4 — PDF Extraction and Anonymization (`pdf_extractor.py`)

Extracts structured clinical fields from a radiology or lab report PDF and anonymizes all PII.

**Text extraction:** Uses `pdfplumber` with two strategies:
- **Word bounding boxes** for the header box — groups words by vertical position (±3px), splits each line into left/right columns at `x0=300`, extracts `label: value` pairs. Handles two-column clinic header layouts robustly across different clinic formats.
- **Raw text lines** for the report body — section keywords (`Findings`, `Impression`, `Technique`, `Conclusion`, `Opinion`, `Results`) are used as optional anchors. If no section keywords are found, all body text is stored as findings.

**Doubled character encoding fix:** Some Indian clinic PDFs use embedded fonts where each glyph maps to two Unicode characters (`AANNBBAARRAASSUU` → `ANBARASU`). A word-level de-duplication function detects and collapses these before any processing.

**Line-wrap joining:** Lines that do not end with sentence terminators (`.`, `!`, `?`) are joined with the following line. This reconstructs sentences that were split across lines due to PDF word-wrap.

**Sentence splitting:** The findings paragraph is split into individual sentences on `. ` followed by a capital letter. Each sentence becomes one FHIR `Observation` resource.

**Anonymization — matching DICOM treatment exactly:**

Identity fields are handled deterministically (not via the NLP model):
- `patient_name` → `ANON-{anon_patient_id}` (matches DICOM `PatientName`)
- `reg_no` → `{anon_patient_id}` (matches DICOM `PatientID` — this is the join key between PDF and DICOM pipelines)
- `referred_by` → empty (matches DICOM `ReferringPhysicianName` wipe)
- `ref_center` → empty (matches DICOM `InstitutionName` wipe)
- `age` → empty (matches DICOM `PatientAge` wipe)
- `date_of_report` → empty (matches DICOM `StudyDate` wipe)

Clinical content fields (`findings`, `impression`, `technique`) are run through `openai/privacy-filter` to catch any incidental PII in the clinical text. `sex` is kept as-is, matching the DICOM `PatientSex` treatment.

---

### Stage 5 — FHIR R4 Mapping (`fhir_mapper.py`)

Builds a FHIR R4 transaction Bundle from the outputs of all previous stages.

#### Standard followed: HL7 FHIR R4 (Release 4)

FHIR R4 is the version mandated by India's ABDM (Ayushman Bharat Digital Mission) under the National Health Authority's Health Data Interchange Specifications.

#### Resources produced

**`Patient`**
```
id              ← anon_patient_id
identifier      ← anon_patient_id (system: urn:oid:1.2.3.4.5.anonymized)
name.text       ← "ANON-{anon_patient_id}"
gender          ← normalized from PDF sex field (male/female/other/unknown)
```

**`ImagingStudy`**
```
id                      ← anonymized StudyInstanceUID (2.25.xxx)
status                  ← "available"
subject                 ← reference to Patient
modality                ← from DICOM series (CT, MR, etc.)
numberOfSeries          ← series count
numberOfInstances       ← total instance count
series[].uid            ← anonymized SeriesInstanceUID
series[].instance[].uid ← make_anon_uid(original SOPInstanceUID)
                           matches anonymized .dcm filenames exactly
```

**`DocumentReference`** (one per NIfTI file — per IHE AIW-I pattern)
```
status                           ← "current"
type                             ← LOINC 18748-4 (Diagnostic imaging report)
subject                          ← reference to Patient
content[].attachment.url         ← file URI of .nii.gz
content[].attachment.contentType ← "application/octet-stream"
context.related                  ← reference to ImagingStudy
```

**`DiagnosticReport`**
```
status        ← "final"
category      ← RAD (Radiology) — v2-0074 code system
code          ← LOINC code from study title keyword lookup (see below)
subject       ← reference to Patient
imagingStudy  ← reference to ImagingStudy
result        ← references to all Observation resources
conclusion    ← impression/conclusion text from PDF
extension     ← technique text
```

**`Observation`** (one per finding sentence)
```
status      ← "final"
category    ← imaging
code        ← LOINC 59776-5 (Findings)
subject     ← reference to Patient
valueString ← individual finding sentence
```

**`Bundle`**
```
type                       ← "transaction"
entry[].request.method     ← "PUT" (idempotent — re-running produces same IDs)
```

Entry order in bundle: `Patient` → `ImagingStudy` → `DocumentReference(s)` → `Observation(s)` → `DiagnosticReport`

#### LOINC code lookup for study type

| Keyword in study title | LOINC code | Display |
|---|---|---|
| brain, head | 24725-4 | CT of head |
| chest, thorax | 24627-2 | CT of chest |
| abdomen | 36643-5 | CT of abdomen |
| pelvis | 24729-6 | CT of pelvis |
| spine, lumbar, cervical | 24775-9 | CT of spine |
| liver | 24815-3 | CT of liver |
| kidney | 44136-0 | CT of kidney |
| neck | 24746-0 | CT of neck |
| *(no match)* | 18748-4 | Diagnostic imaging report |

#### ID generation

All resource IDs are **deterministic** — same inputs always produce the same IDs:
- DICOM-derived IDs use anonymized UIDs directly
- PDF-derived IDs use UUID v5 seeded from `{anon_patient_id}:{resource_type}:{qualifier}`
- This makes the pipeline idempotent — re-running produces an identical bundle

---

## API Backend (`api.py`)

A FastAPI backend that exposes the pipeline over HTTP with Server-Sent Events (SSE) for live progress streaming.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/jobs` | Upload DICOM files + PDF, start pipeline. Returns `job_id`. |
| `GET` | `/jobs/{job_id}/stream` | SSE stream of pipeline progress events |
| `GET` | `/jobs/{job_id}/status` | Current status + download URLs when complete |
| `GET` | `/jobs/{job_id}/download/nifti` | Download anonymized `.nii.gz` file |
| `GET` | `/jobs/{job_id}/download/fhir` | Download FHIR R4 bundle `.json` |
| `DELETE` | `/jobs/{job_id}` | Delete job and all associated files |
| `GET` | `/health` | Health check — confirms models are loaded |

### SSE event format
```json
{ "type": "progress", "stage": "anonymize_dicom", "message": "Anonymized 160 DICOM files", "progress": 40 }
```
`type` is one of `progress`, `complete`, or `error`.

### Design decisions
- **Models loaded once at startup** via FastAPI's `lifespan` context manager. Both `openai/privacy-filter` and EasyOCR are stored on `app.state` and passed to every job thread — never loaded more than once.
- **Background thread per job** — FastAPI remains responsive during pipeline execution. The thread pushes events into a `queue.Queue` which the SSE generator drains asynchronously.
- **Job isolation** — each job gets its own directory under `jobs/{job_id}/` for inputs and outputs.
- **Static files mount** — frontend is served at `/app` to avoid conflicting with API routes at `/jobs`, `/health` etc.

### Running the server
```bash
python src/api.py
# API:      http://localhost:8000/jobs
# Frontend: http://localhost:8000/app
# Docs:     http://localhost:8000/docs
```

---

## Frontend (`frontend/`)

Browser-based interface matching the dark clinical design system.

| Page | File | Description |
|---|---|---|
| Landing | `index.html` | Entry point, redirects to login |
| Sign in | `login.html` | Username/password authentication |
| Upload | `upload.html` | DICOM folder + PDF upload, live SSE pipeline progress |
| Results | `results.html` | Download NIfTI + FHIR bundle, inline FHIR JSON preview |

**Frontend → API flow:**
```javascript
// 1. Upload files
POST /jobs  (multipart: dicom_files[] + pdf_file)
→ { job_id }

// 2. Stream progress
EventSource /jobs/{job_id}/stream
→ { type, stage, message, progress }

// 3. Download outputs
GET /jobs/{job_id}/download/nifti  → .nii.gz
GET /jobs/{job_id}/download/fhir   → .json
```

---

## PII Model: `openai/privacy-filter`

Used for free-text PII detection in DICOM free-text tags and PDF clinical content.

- **Architecture:** Bidirectional token classification model
- **Task:** Named entity recognition for PII spans
- **Library requirement:** `transformers >= 5.6.0`
- **Aggregation strategy:** `simple` — merges consecutive tokens of the same entity type into spans
- **Replacement:** Detected spans are replaced with `[entity_group]` tokens (e.g. `[private_person]`, `[private_date_time]`)

---

## DICOM ↔ FHIR Storage Architecture

DICOM files and FHIR JSON are always stored separately — this follows the global standard and ABDM guidelines:

| Store | Content | Format |
|---|---|---|
| DICOM store (PACS / local) | Pixel data + imaging metadata | `.dcm` files |
| NIfTI store (local) | Anonymized 3D volumes | `.nii.gz` files |
| FHIR server / local | Clinical metadata + references | FHIR R4 JSON |

The `ImagingStudy` resource in the FHIR bundle links to the DICOM store via anonymized UIDs. The `DocumentReference` resource links to the NIfTI store via file URI. No pixel data is embedded in the FHIR bundle.

---

## Dependencies

```bash
pip install \
    pydicom \
    pdfplumber \
    "transformers>=5.6.0" \
    torch \
    fhir.resources \
    Pillow \
    dicom2nifti \
    nibabel \
    matplotlib \
    easyocr \
    fastapi \
    uvicorn \
    python-multipart \
    aiofiles
```

| Library | Version | Purpose |
|---|---|---|
| `pydicom` | latest | DICOM read/write/tag manipulation |
| `pdfplumber` | latest | PDF text and bounding box extraction |
| `transformers` | ≥ 5.6.0 | `openai/privacy-filter` PII detection model |
| `torch` | latest | Model inference backend |
| `fhir.resources` | latest | FHIR R4 resource validation (available for future use) |
| `Pillow` | latest | Image handling for certain DICOM transfer syntaxes |
| `dicom2nifti` | latest | DICOM series to NIfTI volume conversion |
| `nibabel` | latest | NIfTI header read/write |
| `matplotlib` | latest | NIfTI slice visualisation (`nifti_viewer.py`) |
| `easyocr` | latest | OCR for burned-in pixel annotation detection |
| `fastapi` | latest | REST API backend framework |
| `uvicorn` | latest | ASGI server for FastAPI |
| `python-multipart` | latest | Multipart file upload parsing |
| `aiofiles` | latest | Async file I/O for upload handling |

---

## Standards and Protocols Reference

| Standard | Version | Used for |
|---|---|---|
| DICOM PS3.15 Annex E | Current | De-identification tag list and actions — including nested sequences |
| HL7 FHIR | R4 | Clinical data exchange format |
| ABDM Health Data Interchange | R4-based | Indian national health data standard |
| IHE AIW-I | Current | AI imaging workflow — NIfTI via DocumentReference |
| LOINC | Current | Clinical terminology codes for study types and findings |
| ITU-T X.667 / 2.25 OID arc | Current | UID generation for anonymized DICOM UIDs |
| DICOM PS3.3 C.7.6.3.1.2 | Current | Linear windowing formula for pixel display |

---

## Known Limitations

- **Multi-patient batching** — the pipeline processes one patient at a time. Batch processing would require wrapping `pipeline.py` in a loop over a patient list.
- **LOINC code coverage** — the study type lookup covers common CT/MRI studies. Unusual study types fall back to the generic `18748-4` code.
- **PDF format variability** — the header parser uses a fixed `x0=300` threshold to split two-column layouts and a fixed `top < 160` cutoff for the header zone. PDFs with significantly different layouts may require tuning these constants.
- **In-memory job registry** — the API stores job state in a Python dict. If the server restarts, all job history is lost. A production deployment would replace this with Redis or a database.
- **Single-user authentication** — the frontend uses hardcoded demo credentials. Replace with a real authentication backend before any production or multi-user deployment.
