# Anonymization Tool — Codebase Guide

> Describes the code **as it exists in the working tree** (including uncommitted changes), not the README. Where the two disagree, this document follows the code. Issues marked **[verified]** were reproduced against real pipeline output (job `9a10d600…`, 555-slice CT study + its PDF report).

**How to read this:** Part 1 is the high-level picture (≈10 min). Part 2 is the per-module reference — every function, what it does, and why it was built that way. Part 3 lists the assumptions the code relies on. Part 4 lists shortcomings, ordered by severity.

---

## Contents

- **Part 1 — High level**
  - [1.1 What the tool does](#11-what-the-tool-does)
  - [1.2 Module map](#12-module-map)
  - [1.3 End-to-end data flow](#13-end-to-end-data-flow)
  - [1.4 Outputs](#14-outputs)
  - [1.5 Core design decisions](#15-core-design-decisions)
- **Part 2 — In depth**
  - [2.1 dicom_reader.py](#21-dicom_readerpy--stage-1-read--group)
  - [2.2 dicom_anonymizer.py](#22-dicom_anonymizerpy--stage-2-tag-anonymization)
  - [2.3 pixel_scrubber.py](#23-pixel_scrubberpy--stage-2b-burned-in-text-removal)
  - [2.4 dicom_to_nifti.py](#24-dicom_to_niftipy--stage-3-nifti-conversion)
  - [2.5 pdf_extractor.py](#25-pdf_extractorpy--stage-4-pdf-extraction--redaction)
  - [2.6 pdf_template_store.py](#26-pdf_template_storepy--learned-layout-cache)
  - [2.7 pdf_nuextract.py](#27-pdf_nuextractpy--llm-cold-path)
  - [2.8 fhir_mapper.py](#28-fhir_mapperpy--stage-5-fhir-r4-bundle)
  - [2.9 api.py](#29-apipy--http-backend--orchestrator)
  - [2.10 pipeline.py](#210-pipelinepy--cli-orchestrator)
  - [2.11 Utilities](#211-utilities-dicom_verifypy-pdf_analyzerpy-view_niftipy)
  - [2.12 Frontend](#212-frontend)
  - [2.13 Configuration and on-disk state](#213-configuration-and-on-disk-state)
- [Part 3 — Assumptions](#part-3--assumptions)
- [Part 4 — Shortcomings](#part-4--shortcomings)

---

# Part 1 — High level

## 1.1 What the tool does

Takes **one or more patients' imaging studies** (DICOM `.dcm` files) plus **one radiology report PDF per patient**, and produces a de-identified research package:

| Input | Output |
|---|---|
| DICOM files (CT/MR/…, any series layout) | Anonymized DICOM (tags wiped/pseudonymized, UIDs remapped, burned-in text zeroed) |
| — | NIfTI volumes (`.nii.gz`), one per convertible series |
| Report PDF | Visually redacted PDF (same layout, PII boxes removed) + structured, redacted fields |
| — | FHIR R4 transaction Bundle linking patient, imaging study, report, findings, NIfTI files |
| — | QA artifacts: `pii_audit.json` (before/after), `preservation_report.json` |

Two entry points run the same stages: a **FastAPI web app** (`api.py` + `frontend/`) — the primary path — and a **CLI** (`pipeline.py`) that is older and runs a subset.

## 1.2 Module map

| Module | Stage | Responsibility |
|---|---|---|
| `dicom_reader.py` | 1 | Scan `.dcm` files (headers only) → `DicomStudy > DicomSeries > DicomInstance` |
| `dicom_anonymizer.py` | 2 | Wipe/pseudonymize tags, remap UIDs, NER-redact free text, call pixel scrubber, write files |
| `pixel_scrubber.py` | 2b | OCR (EasyOCR) burned-in text in pixel data and zero-fill it |
| `dicom_to_nifti.py` | 3 | Convert each anonymized series to `.nii.gz`, wipe NIfTI header text fields |
| `pdf_extractor.py` | 4 | Extract header/body fields from the PDF, redact them, produce the visually redacted PDF |
| `pdf_template_store.py` | 4 (helper) | Persist and match learned PDF layouts ("templates") |
| `pdf_nuextract.py` | 4 (helper) | LLM (NuExtract3 GGUF via llama.cpp) extraction for unseen PDF layouts |
| `fhir_mapper.py` | 5 | Build the FHIR R4 Bundle JSON |
| `api.py` | orchestrator | HTTP endpoints, background job threads, SSE progress, multi-patient sessions, zip export |
| `pipeline.py` | orchestrator | CLI single-patient run (no redacted PDF, no audits) |
| `dicom_verify.py` | utility | Offline original-vs-anonymized DICOM verifier (not called by the pipeline) |
| `pdf_analyzer.py` | utility | Dump a PDF's raw text / word boxes / tables for parser development |
| `view_nifti.py` | utility | Plot 9 axial slices of a NIfTI for visual QA |
| `frontend/*.html`, `assets/` | UI | Login (demo), upload + live progress, results + zip download |

Shared models, loaded **once** at API startup and passed down explicitly:
- **`openai/privacy-filter`** (HF token-classification NER) → text PII detection.
- **EasyOCR** (English) → burned-in pixel text detection.
- **NuExtract3** (≈2.8 GB GGUF) → loaded **lazily** only on the PDF cold path.

## 1.3 End-to-end data flow

### Web flow (primary)

```
Browser (upload.html)
 │ 1. user picks a folder ──► POST /analyze (all files, with relative paths)
 │                              └─ stage under sessions/<sid>/, read_and_group_multi(),
 │                                 match PDFs to studies by folder proximity
 │ ◄── {mode: single|multi, patients[], errors[]}
 │
 ├─ single mode: user picks PDF ─► POST /jobs (dcm files + pdf, re-uploaded)
 └─ multi  mode: for each patient (sequentially) ─► POST /jobs/from-session (sid, study_uid)
                                                     └─ symlink that study's files into jobs/<jid>/input
 │
 │ 2. GET /jobs/<jid>/stream  (SSE progress events)
 │
 ▼ background thread: _run_pipeline_thread()
   ┌──────────────────────────────────────────────────────────────────────┐
   │ S1 read_and_group(input/dicom)                       → DicomStudy    │
   │ S2 anonymize_study()  → per instance:                                │
   │      _anonymize_dataset (tags, UIDs, free text)                      │
   │      scrub_pixels (only if OCR sample found text AND has pixel data) │
   │      save_as(output/dicom/patient_<origID>/<newSOP>.dcm)             │
   │ S3 convert_to_nifti() → output/nifti/patient_<anonID>/*.nii.gz       │
   │      (series < 3 slices skipped; zero series is a warning, not fatal) │
   │ S4 extract_pdf()      → PdfExtract (structured, redacted fields)     │
   │    redact_pdf_visual()→ output/pdf/…_report_anonymized.pdf           │
   │      both call _resolve_structure(): legacy → template → NuExtract3  │
   │ S5 map_to_fhir()      → output/fhir/patient_<anonID>_bundle.json     │
   │ QA pii_audit.json + preservation_report.json                          │
   └──────────────────────────────────────────────────────────────────────┘
 │
 │ 3. results.html ─► GET /jobs/<jid>/download/pii_audit (before/after view)
 │ 4. "Download All" ─► GET /export?job_ids=… (browser-native download)
 │       └─ zip: Anonymized_data_<ts>/Patient_<anonID>/Study_<uid16>/{dicom,nifti,pdf,fhir}
 ▼
 multi mode only: DELETE /sessions/<sid> after the last patient
```

### Step-by-step (what happens to the data)

1. **Grouping by headers, not folders.** Every `.dcm` is read with `stop_before_pixels=True`; files are grouped by `StudyInstanceUID` → `SeriesInstanceUID`, sorted by `InstanceNumber`. Folder layout only matters for pairing PDFs to patients.
2. **Pseudonym.** `anon_patient_id = "ANON-" + SHA256("pipeline-salt-v1:patient:<PatientID>")[:12]`.
3. **Per-study constants from the first instance.** UID map and NER-redacted free text are computed once and stamped on every instance; OCR window and preservation-check anchor come from the first instance that actually has pixel data.
4. **OCR decision.** Up to 3 image-bearing instances are OCR'd; if any text is found, every image instance in the study is scrubbed.
5. **Per-instance write.** Recursive tag walk → optional pixel scrub → `save_as` under a filename equal to the new SOPInstanceUID.
6. **NIfTI.** Each series (≥3 slices) is symlinked into a temp folder and converted by `dicom2nifti`; header text fields wiped, display window set from DICOM WC/WW.
7. **PDF structure.** Text + word boxes from pdfplumber. Header/body fields resolved by the first tier that succeeds: fixed-layout parser → learned template → NuExtract3 (which then saves a new template).
8. **PDF anonymization.** Identity fields replaced deterministically (not NER); clinical text NER-redacted; a visually redacted copy is produced with PyMuPDF true-redaction (identity boxes + NER hits + sign-off block + signature images).
9. **FHIR.** Patient, ImagingStudy, DocumentReference per NIfTI, Observation per finding sentence, DiagnosticReport — PUT entries with deterministic IDs.
10. **Export.** Zip assembled on disk from the job's outputs, streamed to the browser.

### CLI flow (`pipeline.py`)
Read → load models → anonymize → NIfTI → `extract_pdf` → FHIR. **Does not** produce the redacted PDF, `pii_audit.json`, or `preservation_report.json`, and requires a flat folder with exactly one study.

## 1.4 Outputs

Per job, under `src/jobs/<job_id>/`:

```
input/dicom/*.dcm            uploaded copies (single) or symlinks to the session (multi)
input/<report>.pdf
output/dicom/patient_<ORIGINAL PatientID>/<newSOP>.dcm   ← folder name uses the original ID
output/nifti/patient_<anonID>/series_<MOD>_<uid16>.nii.gz
output/pdf/patient_<anonID>_report_anonymized.pdf
output/fhir/patient_<anonID>_bundle.json
output/pii_audit.json        ← contains ORIGINAL PII values (see Part 4)
output/preservation_report.json
```
The export zip renames everything to anonymized identifiers only; `pii_audit.json` and `preservation_report.json` are **not** included in the zip.

Persistent (outside the repo): learned PDF templates at `~/Library/Application Support/AnonymizationTool/pdf_templates/` (macOS; `%APPDATA%` / `$XDG_DATA_HOME` elsewhere), and the NuExtract3 GGUF in the Hugging Face cache.

## 1.5 Core design decisions

| Decision | Reason |
|---|---|
| Header-only scan first (`stop_before_pixels`) | Grouping 500+ files is fast; pixels are read once, per file, at write time |
| Deterministic SHA-256 UID remap into the `2.25.` arc | No mapping table to store; re-runs reproduce identical UIDs; cross-file references stay consistent |
| Compute anonymization values once per study | Every slice gets the same new Study/Series UIDs and the same redacted descriptions |
| Identity fields replaced by rule, not by NER | NER can miss; known identity fields are wiped/pseudonymized unconditionally |
| NER only for free text | Free text is where PII appears unpredictably |
| OCR sampling (3 images) before full OCR | OCR is the slowest step; most CT/MR has no burned-in text |
| Three-tier PDF layout resolution | Keep the common case free; only pay for the LLM on genuinely new layouts, and only once per layout |
| Hallucination guard on LLM output | Every LLM value must be a verbatim substring of the PDF before its location is trusted for redaction |
| PyMuPDF true redaction (`apply_redactions`) | Removes the underlying text/image, not just paints over it |
| Background thread per job + SSE | Keeps the API responsive; progress streamed to the browser |
| `GET /export` + native browser download | Multi-GB zips can't be buffered through `fetch().blob()` |
| Models on `app.state`, loaded at startup | Never load a model twice per process |

---

# Part 2 — In depth

Each module: purpose → data structures → function table (**Role** / **Why**) → notes. Issues found while reading are cross-referenced to Part 4 (e.g. **[B1]**).

## 2.1 `dicom_reader.py` — Stage 1: read & group

**Data classes**

| Class | Fields | Notes |
|---|---|---|
| `DicomInstance` | `path`, `sop_instance_uid`, `instance_number`, `dataset` | `dataset` is **metadata only** — the docstring's "pixel data loaded lazily via `instance.dataset.pixel_array`" is inaccurate; the anonymizer re-reads each file with `dcmread` to get pixels. |
| `DicomSeries` | `series_uid`, `modality`, `series_number`, `series_description`, `instances` | `first` property = lowest InstanceNumber |
| `DicomStudy` | `study_uid`, `study_date`, `study_description`, `patient_name`, `patient_id`, `patient_dob`, `series{uid→DicomSeries}` | `all_instances` (flat list, series order then instance order), `total_instances` |

**Functions**

| Function | Role | Why |
|---|---|---|
| `_ingest_dataset(ds, fpath, studies)` | Insert one header into the study/series/instance tree, creating levels on first sight. Refuses (skips with warning) a file whose StudyInstanceUID was already seen with a different PatientID. | One shared ingest path for single and multi readers; the PatientID guard prevents silently merging two patients on a UID collision or a mixed upload. |
| `_read_dcm_files(files)` | `dcmread(stop_before_pixels=True, force=True)` each file, ingest, then sort instances by `(InstanceNumber is None, InstanceNumber)`. Unreadable files are counted and skipped. | `force=True` accepts files missing the DICM preamble; the sort key pushes missing InstanceNumbers to the end. |
| `read_and_group(dir)` | Flat `*.dcm` glob → exactly one study; raises if 0 files, 0 readable, or >1 study. | Single-patient entry point used by the job thread (job input is always flat). |
| `read_and_group_multi(root)` | Recursive `rglob("*.dcm")` → list of studies. | Multi-patient entry point (`/analyze`, `/jobs/from-session`); folder layout irrelevant. |
| `print_study_tree(study)` | Human-readable dump. | Debug/CLI. |

**Notes:** a "patient" in multi mode is really a **study** (grouped by StudyInstanceUID). Two studies of one person appear as two "patients" but receive the same pseudonym (it hashes PatientID). Only files ending in `.dcm` are seen anywhere **[R4]**. No SOP-class filtering — Presentation States, SRs, KO documents are grouped like images (handled downstream).

## 2.2 `dicom_anonymizer.py` — Stage 2: tag anonymization

**Tag lists (module constants)**

| Constant | Size | Action |
|---|---|---|
| `WIPE_TAGS` | ~70 tags | Set to `""` (or empty sequence for SQ). Birth date, address, phones, other IDs/names, accession, institution, all physician/operator names, station, device serial, StudyID, all study/series/acquisition/content/creation dates & times, scheduling/request/admission/visit tags. |
| `FREE_TEXT_TAGS` | 6 | StudyDescription, SeriesDescription, AdditionalPatientHistory, PatientComments, ImageComments, StudyComments → NER-redacted |
| `UID_TAGS` | 6 | Study/Series/SOP Instance, FrameOfReference, InstanceCreator, SynchronizationFrameOfReference — used to build the UID map |
| `UID_TAGS_ALL` | 15 | `UID_TAGS` + ReferencedSOPInstanceUID, **ReferencedSOPClassUID** **[B6]**, DimensionOrganizationUID, PaletteColorLUT UID, FailedSOPInstanceUIDList, StorageMediaFileSetUID, CreatorVersionUID, plus two SQ tags that the walker handles as sequences first (dead entries) |
| `_TECHNICAL_TAGS` | 36 | Must survive unchanged (modality, scanner, kVp/TR/TE, geometry, rescale, window, age/size/weight/sex) — QA only |

**Retained by design:** PatientSex, PatientAge, PatientSize, PatientWeight (clinically useful, low re-ID risk alone). **Not handled:** private (odd-group) tags **[P2]**.

**Functions**

| Function | Role | Why |
|---|---|---|
| `make_anon_uid(uid, salt="pipeline-salt-v1")` | `"2.25." + int(sha256(f"{salt}:{uid}")[:32 hex], 16)[:39 digits]` | Deterministic, stateless, valid DICOM UID (digits/dots, ≤64 chars) under the UUID-derived OID arc (ITU-T X.667). |
| `build_uid_map(ds)` | `{orig: make_anon_uid(orig)}` for `UID_TAGS` + file-meta MediaStorageSOPInstanceUID of one dataset. | Logged and returned; the walker falls back to `make_anon_uid` for anything not in the map, so the map is informational more than functional. |
| `load_privacy_filter()` | HF `pipeline("token-classification", "openai/privacy-filter", aggregation_strategy="simple")` | `simple` merges sub-tokens into entity spans with char offsets. |
| `redact_text(text, clf)` | Replace each detected span with `[entity_group]`, iterating right-to-left. | Right-to-left keeps earlier offsets valid while the string changes length. |
| `build_redacted_free_texts(ds, clf)` | Redact each present `FREE_TEXT_TAGS` value of **one** dataset. | NER is expensive; run once per study. Consequence: **[B4]**. |
| `_walk_and_anonymize(ds, …)` | Recursive walk of every element: SQ in `WIPE_TAGS` → emptied, other SQ → recurse into items; PatientName → `f"ANON-{anon_patient_id}"`; PatientID → `anon_patient_id`; wipe; free text → precomputed value (only if the first instance had that tag); UID → map/`make_anon_uid`. | PS3.15 Annex E requires nested attributes (ContentSequence, RequestAttributesSequence…) to be treated like top-level ones. |
| `_anonymize_dataset(ds, …)` | Walk, then re-derive SOPInstanceUID and MediaStorageSOPInstanceUID. | Intended to give each file a unique new SOP UID — but the walker already remapped it, so it is hashed twice **[B1]**. |
| `_build_pii_audit(before, after)` | Per acted-on tag: `{tag, name, original, anonymized, action}`. | Feeds the results page "before vs after". Writes originals to disk via the API **[P1]**. |
| `_build_technical_preservation_check(before, after)` | For each `_TECHNICAL_TAGS` present: preserved = string equality. | Catches over-wiping regressions. |
| `_build_pixel_preservation_check(before, after, scrubbing_expected)` | Shape/dtype equality + fraction of changed pixels; clean if 0% (no scrub expected) or <50% (scrub expected). | Distinguishes targeted text scrubbing from corruption. |
| `anonymize_study(study, output_dir, anon_patient_id=None, uid_salt, clf, ocr_reader)` | Orchestrates Stage 2 (below). Returns `{anon_patient_id, uid_map, redacted_tags, files_written, output_dir, sop_uid_to_filename, pii_audit, preservation_check}`. | `sop_uid_to_filename` lets Stage 3 find files without re-deriving names. |

**`anonymize_study` sequence**
1. Derive pseudonym `ANON-<12 hex>` from `uid_salt + PatientID` (if not supplied).
2. Load NER/OCR if not passed in.
3. `first_ds` = full read of instance 0 → UID map + redacted free text.
4. **Anchor** = first instance with pixel data (instance 0 if it qualifies) → window center/width + "before" snapshot for preservation checks. Avoids anchoring on a non-image object.
5. **OCR sample:** first ≤3 image-bearing instances of the **whole study** → `run_ocr` **[P4]**.
6. Per instance: full read → `_anonymize_dataset` → `scrub_pixels` if `run_ocr and has_pixel_data` → PII audit (instance 0) / preservation check (anchor) → `save_as(<new SOP>.dcm)`.

Every instance is tag-anonymized and written regardless of whether it has pixel data — only pixel scrubbing is skipped for non-image objects.

## 2.3 `pixel_scrubber.py` — Stage 2b: burned-in text removal

Constants: `CONFIDENCE_THRESHOLD = 0.5`, `BBOX_PADDING = 5` px, `SAMPLE_SLICE_COUNT = 3`.

| Function | Role | Why |
|---|---|---|
| `load_ocr_reader()` | `easyocr.Reader(["en"])` | GPU auto-detected; load once. English only **[R5]**. |
| `_get_window_params(ds)` | First WC/WW value, default (127, 256). | Multi-valued window tags are common; fallback = full 8-bit range. |
| `_pixel_array_to_8bit(arr, wc, ww)` | Linear window (PS3.3 C.7.6.3.1.2) → clip → uint8. | OCR needs an 8-bit display image. Applied to raw stored values without Rescale Slope/Intercept **[R6]**. |
| `has_pixel_data(ds)` | Presence of PixelData / FloatPixelData / DoubleFloatPixelData. | Exact, future-proof non-image test; no SOP-class allowlist to maintain. |
| `_get_pixel_frames(ds)` | `ds.pixel_array` → list of 2-D frames (ndim 2 → 1 frame, 3 → split on axis 0). | Multi-frame support (US cines). Treats single-frame RGB `(rows, cols, 3)` as frames **[R7]**. |
| `_expand_bbox(bbox, pad, h, w)` | EasyOCR quad → padded, clipped `(r0, r1, c0, c1)`. | Padding covers anti-aliased glyph edges. |
| `_detect_text_regions(img8, reader)` | `readtext(detail=1)`, keep conf ≥0.5 and non-blank. | Threshold reduces false positives on anatomy. |
| `_zero_fill_regions(frame, regions)` | Copy and set boxes to 0 in the **original-bit-depth** array. | Redacts stored values, not just the display image. |
| `_write_pixel_frames_back(ds, frames)` | If transfer syntax is compressed, switch to Explicit VR Little Endian; write bytes. | No codec re-encoder needed; writing native bytes under a compressed TS is rejected by pydicom. |
| `sample_series_for_burned_in(datasets, reader)` | True if any sampled frame has text; **any read error → True**. | Fail-safe toward scrubbing. |
| `scrub_pixels(ds, reader, wc, ww, force_uncompressed)` | OCR every frame, zero hits, write back if anything changed **or** `force_uncompressed`. | `force_uncompressed` keeps one transfer syntax across a scrubbed study (dicom2nifti dislikes mixed encodings). |

## 2.4 `dicom_to_nifti.py` — Stage 3: NIfTI conversion

| Function | Role | Why |
|---|---|---|
| `_wipe_nifti_header(path, cal_min, cal_max)` | Blank `descrip`, `db_name`, `aux_file`, `intent_name`; set `cal_min/cal_max`. | Converters/firmware sometimes put identifying text in these; the window lets viewers (3D Slicer) open with the right contrast. |
| `_get_window_from_dicom(path)` | First WC/WW → `(wc − ww/2, wc + ww/2)`. | Same window semantics as DICOM. |
| `_check_nifti_geometry_preserved(dcm, nii, n)` | Voxel spacing compared as a sorted set (±5%), window not both 0, `n in img.shape`. | Order-insensitive because `reorient_nifti=True` may permute axes. Uses SliceThickness, which can differ from true slice spacing (overlapping reconstructions) → false failures possible. |
| `_convert_series(series, anon_dir, anon_result, out, id)` | Locate the series' anonymized files via `sop_uid_to_filename`; skip if <3 slices; symlink into a temp dir; `dicom2nifti.dicom_series_to_nifti(reorient=True)` with orientation and slice-increment validation **disabled**; delete sidecar JSON; wipe header; move to `series_<MOD>_<uid16>.nii.gz`; geometry check. Returns `None` on any failure. | dicom2nifti wants one flat folder per series; symlinks avoid copying pixels; validation disabled so tilted/odd scanners don't abort — at the cost of silently accepting missing slices or gantry tilt **[R8]**. |
| `convert_to_nifti(study, anon_result, out)` | Loop all series, collect successes. | An empty list is legal (non-image or tiny series only); the API treats it as a warning. |

## 2.5 `pdf_extractor.py` — Stage 4: PDF extraction & redaction

**`PdfExtract` dataclass:** header (`patient_name, age, sex, reg_no, date_of_birth, date_of_report, referred_by, ref_center`), body (`study_title, technique, findings[], impression`), `full_text_anon`, `raw_values` (pre-redaction snapshot for the audit).

### 2.5.1 Text normalization

| Function | Role | Why |
|---|---|---|
| `_fix_doubled_chars(text)` | Per word: if every adjacent pair is identical, keep one of each (`AANNBB` → `ANB`). | Some clinic PDFs' embedded fonts map each glyph to two chars. Also collapses legitimate doubled tokens like `"11"` → `"1"` **[R9]**. |
| `_clean_line(line)` | Strip + collapse runs of spaces. | — |

### 2.5.2 Tier 0 — fixed-layout ("legacy") parser

Constants: `_HEADER_RIGHT_COL_X0 = 300` pt, `_HEADER_BOTTOM_TOP = 160` pt (A4-oriented absolute coordinates **[R10]**).

| Function | Role | Why |
|---|---|---|
| `_parse_header_from_words(words)` | Words with top <160 → lines (±3 pt) → left/right column split at x=300 → `label: value` on first colon → map labels by keyword (`NAME` not `REF`, `AGE`/`SEX` split on `/`, `REG`/`LAB`, `DOB`, `DATE+REPORT` (text before " at "), `REFER`, `REF+CENTER`). | Two-column header boxes are common; word boxes are more robust than raw text for columns. |
| `_SECTION_PATTERNS` | technique (colon required), findings, impression, conclusion, opinion, results, `end` (`**** End of Report`), `advice`. | `advice` exists only so trailing ADVICE text stops being appended to the open section. |
| `_SKIP_PATTERNS`, `_is_skip_line` | "Page N of M", "Consultant Radiologist", `**…`. | Footer noise. |
| `_is_radiologist_line(line)` | ≥2 doubled-char words → skip. | Doubled-font radiologist name lines. |
| `_join_wrapped_lines(lines)` | Join to previous if it doesn't end in `.!?`. | Reconstruct sentences broken by PDF wrapping. |
| `_split_into_sentences(text)` | Split on `.` + whitespace + capital; drop fragments ≤5 chars. | One FHIR Observation per sentence. |
| `_parse_body_from_raw_text(lines, header_end_line=4, extra_section_patterns=None)` | Body starts at first ALL-CAPS title line; state machine: stop at a sign-off marker (`_SIGNOFF_KEYWORDS`) or `end`; section keyword → flush previous buffer and switch; other lines → current buffer (lines before any section are **dropped**); findings-like sections → findings list, then joined & sentence-split. `extra_section_patterns` merges a template's learned headings. | Section headings are the most reliable body anchors; the sign-off stop keeps the radiologist's name/registration out of `impression`. README's "no keywords → everything is findings" is not what the code does **[R11]**. |

### 2.5.3 Layout resolution (tiers 0–2)

| Function | Role | Why |
|---|---|---|
| `_default_template_store_dir()` | `PDF_TEMPLATE_STORE_DIR` env, else per-OS user-data dir. | Must survive app restarts and PyInstaller one-file temp extraction. |
| `_get_template_store()` | Lazy singleton `TemplateStore`. | — |
| `_section_patterns_from_template(tpl)` | Learned heading per field → `^<heading>\s*:?` regex (skipping the generic field name). | Lets the body parser recognize this layout's own section words ("Protocol", "Observations"). |
| `_apply_template(tpl, words, w, h, v_tol=4, l_tol=6, right_margin=220)` | For each stored header field: collect words in the stored box (denormalized), widened 220 pt rightward; value = joined text; all collected words become identity redaction boxes. age/sex default `None`. | Values differ in length per document, so the box is widened. The widening over-captures neighbouring header fields **[R12]** — harmless for identity fields (they are overwritten) and over-redacts visually (safe direction), but age/sex are lost on this path. |
| `_resolve_structure(words, lines, w, h, clf)` | **Tier 0** legacy parse; accept if ≥3 identity labels and any body section. **Tier 1/2** `find_matching_template`; on match → `_apply_template`; if any stored field comes out empty, that template is dropped and the next candidate is tried; on success → `record_match`, body re-parsed with the template's headings. **Cold path** NuExtract3 → `verify_and_locate` → build template → **saved only if it matches its own document** (and none of its fields are empty) → fields from model output (findings filtered to verified; technique/impression/header values taken even if unverified **[P6]**); identity boxes = verified identity fields. Any NuExtract exception → `legacy_fallback`. | Pay only for what's needed; one shared resolver so extraction and redaction can't disagree. |

### 2.5.4 Extraction & structured anonymization

| Function | Role | Why |
|---|---|---|
| `extract_pdf(path, clf, anon_patient_id)` | pdfplumber page-0 words + all-page text → `_resolve_structure` → `PdfExtract` → snapshot `raw_values` → `_redact_extract` → `full_text_anon` (NER on joined body; unused downstream **[Q3]**). | Header from page 0 only; body may span pages. |
| `_redact(text, clf)` / `_redact_extract(extract, clf, id)` | `patient_name = "ANON-<id>"`, `reg_no = <id>`, DOB/referred_by/ref_center/date_of_report = `""`; age/sex kept; findings/impression/technique NER-redacted. `study_title` is **not** redacted. | Deterministic replacement mirrors DICOM policy and doesn't depend on NER recall. |
| `build_pdf_pii_audit(extract)` | Before/after rows for the results page. | — |

### 2.5.5 Visual redaction (`redact_pdf_visual`)

| Function | Role | Why |
|---|---|---|
| `_group_words_into_lines(words)` | ±3 pt line clustering, sorted by x. | Shared line grouping. |
| `_split_label_value_words(col)` | Split a column at the word containing `:`. | Get boxes for the value only. |
| `_header_identity_value_word_boxes(words)` | Legacy-layout identity value boxes (excludes age/sex). | Tier-0 redaction boxes. |
| `_body_pii_word_boxes(words, clf, include_header_zone)` | NER per line (on doubled-char-fixed text), map entity char spans back to word boxes. Header zone excluded unless no identity box was found at all. | Catch incidental PII anywhere; excluding the header protects retained age/sex from NER false positives. |
| `_SIGNOFF_KEYWORDS` | `reported by`, `electronically signed/verified`, `signed by`, `verified by`. | Marks the start of the radiologist sign-off block. |
| `_signature_zone_top(words)` | `top` of the first line matching a sign-off keyword. | Anchor for the next two. |
| `_signature_image_boxes(page, words)` | Embedded images narrower than 60% of page width, at/below the sign-off line (any narrow image if no marker). | Signatures are images — invisible to text-based checks. Full-width letterhead/footer banners are excluded. |
| `_signoff_zone_word_boxes(words)` | Every word from the sign-off line to page bottom. | Registration numbers (e.g. `DMC/R/21536`) are not recognized by NER; the whole block is wiped like other identity fields. |
| `redact_pdf_visual(pdf, clf, out)` | Resolve structure once, then per page: NER boxes (+ identity boxes on page 0) + signature images + sign-off block → `add_redact_annot` (1.5 pt padding, black fill) → `apply_redactions` → save. | True removal of content. Pages with no extractable words are skipped entirely **[R13]**. |

## 2.6 `pdf_template_store.py` — learned layout cache

| Item | Role | Why |
|---|---|---|
| `HeaderField(label_text, bbox_norm)` | One identity value's box normalized 0–1 + its label text. | Normalized so page size differences don't matter. |
| `HeaderField` | `label_text` (`""` = no label on the page), `bbox_norm` | Position as page fractions; label used for re-verification. |
| `SectionHeading` | `text, x0_norm, top_norm` | A body heading's left edge (checked) and vertical position (used for order only). |
| `PdfTemplate` | `template_id, status (provisional/trusted), created_at, match_count, body_section_keywords{field→[regex]}, header_fields, source, page_size, section_headings` | `status` is informational (promoted to `trusted` at 3 matches); matching ignores it. `page_size`/`section_headings` default to empty, so older templates load but never match. |
| `TemplateStore` | One JSON file per template; `load_all` (skips unreadable), `save`, `record_match`, `create_provisional` (unused). | Human-inspectable, no DB. |
| `keyword_prefilter(text, tpl, min_hits=2)` | ≥2 of the template's body fields have a keyword present. | Cheap document-type check. |
| `positional_verify(words, tpl, w, h, v_tol=5, h_search=140)` | Needs ≥2 labeled fields; **every** labeled field's full label (spaces ignored, case-insensitive) must be found within ±5 pt vertically and up to 140 pt left of the stored value position. Unlabeled fields are skipped. | Safety-critical: reusing boxes from a non-matching layout would misplace redactions. First-word / half-of-fields matching let different layouts match (short words like `Date`, `Ref.` are on most headers). Absolute point tolerances because header rows are ~12–16 pt apart. |
| `page_size_matches(tpl, w, h)` | Width and height within ±2 pt. | Normalized coordinates alone let a different-size layout pass. |
| `heading_verify(words, tpl, w, h, x_tol=12)` | Needs ≥2 stored headings; each must start a page-0 line at the same left edge (±12 pt), and the found headings must keep the stored top-to-bottom order. | Separates layouts that share a header but differ in body. Vertical position isn't checked because findings length moves later headings. |
| `find_matching_template(…)` | Prefilter → sort by `match_count` desc → first that passes page size, labels and headings. | Prefer the most-proven layout. |

Templates are created by `pdf_nuextract.build_template_from_extraction` (2.7). Tested on 18 unique reports: every layout reused on its second pass, no duplicate templates, no missed redactions vs. the model's extraction. The 180 pre-fix templates were deleted.

## 2.7 `pdf_nuextract.py` — LLM cold path

**Configuration (env):** `NUEXTRACT_BACKEND` (`llama_cpp_python` default | `llama_server`), `NUEXTRACT_GGUF_PATH` (else auto-download `numind/NuExtract3-GGUF`, `*Q4_K_M.gguf`), `NUEXTRACT_GPU_LAYERS` (0 = CPU), `NUEXTRACT_CTX_SIZE` (8192), `NUEXTRACT_LLAMA_SERVER_PATH`, `NUEXTRACT_STARTUP_TIMEOUT` (120 s). Sets `HF_HUB_ENABLE_HF_TRANSFER=0` at import (the accelerated downloader can hang behind proxies).

**Schema:** `CANONICAL_TEMPLATE` — every field the pipeline needs to locate is `"verbatim-string"` (a reformatted value can't be found on the page); `sex` is an enum; `findings` a list. `IDENTITY_FIELDS` = name, reg_no, DOB, referred_by, ref_center, date_of_report.

| Function / class | Role | Why |
|---|---|---|
| `_parse_model_json(text)` | `json.loads`, retry after stripping markdown fences. | Defensive. |
| `_LlamaCppPythonBackend` | Lazy, lock-guarded, process-resident `Llama` (local path or `from_pretrained`); reads the GGUF's embedded `tokenizer.chat_template`. | In-process = simplest to package; lock prevents double loads from concurrent jobs. |
| `_build_jinja_env()` | Jinja2 env with a pass-through `{% generation %}` tag. | NuExtract3's HF template uses that non-standard tag. |
| `_render_chat_prompt(tpl, text)` | Render with `messages`, `template` (schema JSON), `instructions`, `enable_thinking=False`, `add_generation_prompt`. | llama-cpp-python doesn't reliably forward extra template variables, so the prompt is rendered manually. |
| `_run_llama_cpp_python(text)` | Completion: `max_tokens=2048`, `temperature=0.2`, ChatML stop tokens → JSON. | Low temperature (not 0 — so not strictly deterministic **[Q4]**). |
| `_LlamaServerBackend`, `_default_llama_server_path`, `_free_local_port`, `_run_llama_server` | Alternative: spawn `llama-server` on a free localhost port, health-poll, call via OpenAI-compatible API with `chat_template_kwargs`. | Crash isolation and swappable hardware backends. Needs the `openai` package. |
| `is_loaded`, `load_nuextract_model`, `run_nuextract_on_text` | Backend-agnostic dispatch. | — |
| `_find_words_for_text(words, target)` | Order words (≈3 pt row buckets, then x), find the first contiguous run whose joined lowercase text contains the target. | LLM returns text only; boxes are recovered by search. First occurrence only. |
| `verify_and_locate(output, words, text)` | Each scalar/finding: `verified` = case-insensitive substring of raw text **and** locatable. | Hallucination guard. A de-doubled value from a doubled-font PDF fails verification and is therefore not visually redacted **[P5]**. |
| `_guess_label_text(value_words, all_words, max_gap=80, other_value_words=None)` | Last 1–2 words left of the value on the same line (≤80 pt gap), cut after any word belonging to another field's value; `""` if it has digits or is a title (`_HONORIFICS`: Mr, Mrs, Dr, …). | Header label for Tier-2 re-verification. Without the cut, a referring doctor's name became the ref-center label (PHI in the template). |
| `_guess_section_label(value_words, all_words, max_gap_lines=4)` → `(text, line_words)` | Nearest short line (≤4 words, ≤40 chars) **above** the value, within 4× the value's text height, that `_looks_like_heading`. | Body headings sit on their own line above the paragraph. A fixed 30 pt missed a heading 31.3 pt away. |
| `_looks_like_heading(text)` | Starts with a letter, no trailing `.`, no digits; ends with `:`, is all caps, or is Title Case. | Rejects bullets (`• DWI`) and wrapped sentence ends (`acute infarcts.`). |
| `build_template_from_extraction(verify, words, w, h)` | Header fields from verified identity fields (normalized box + guessed label, no fake fallback label). Body: anchors each section on its verified words, or on the first 3 words of an unverified value; stores the heading as a keyword (+ generic field name) and as a `SectionHeading`; stores `page_size`. | Only verified identity data enters a template; the loose anchor only locates the heading (bullet lists come back comma-joined and fail verbatim checks). |

## 2.8 `fhir_mapper.py` — Stage 5: FHIR R4 bundle

Plain dicts + `json.dump` (no `fhir.resources` validation).

| Function | Role | Notes |
|---|---|---|
| `_LOINC_LOOKUP`, `_get_loinc(title)` | First keyword hit in the study title → LOINC code (all "CT of …"); default 18748-4. | MR studies get CT codes **[Q5]**. |
| `_det_uuid(seed)` | `uuid5(NAMESPACE_DNS, seed)` | Deterministic IDs. |
| `_normalize_gender(sex)` | male/m, female/f, else other, empty → unknown. | — |
| `build_patient(id, pdf)` | `id` & identifier = pseudonym, `name.text = "ANON-"+id`, gender from PDF. | Receives `"ANON-…"` already, so the name becomes `ANON-ANON-…` **[B2]**. |
| `build_imaging_study(study, anon_result, pid)` | id = anon StudyUID; series (anon UID, number, modality, description); instances `uid = make_anon_uid(orig SOP)`, number, sopClass. | Instance UIDs don't match the files (double hash) **[B1]**; descriptions come from the **original** un-redacted study object **[P3]**; study-level modality list built from a `set` (order not stable). |
| `build_observations(pdf, pid, report_id)` | One Observation per finding sentence (LOINC 59776-5, `valueString`). | — |
| `build_diagnostic_report(pdf, pid, isid, obs_ids)` | RAD category, LOINC from title, imagingStudy ref, results, `conclusion = impression`, technique in a custom extension URL. | Report ID seeded by patient + study title → collisions for same-title reports. |
| `build_document_references(nifti, pid, isid)` | One per NIfTI; `attachment.url` = absolute `file://` path on the server. | Path is meaningless after export **[Q6]**. |
| `build_bundle(...)` | Transaction bundle, `fullUrl: urn:uuid:<id>`, `PUT Type/id`; order Patient → ImagingStudy → DocRefs → Observations → Report. | `urn:uuid:` with non-UUID ids (pseudonym, `2.25.…`) is invalid per spec; `timestamp` changes every run. |
| `map_to_fhir(study, anon_result, pdf, nifti, out)` | Build all, write `patient_<id>_bundle.json`. | Module imports the whole pipeline at import time (heavy). |

## 2.9 `api.py` — HTTP backend & orchestrator

**Startup:** `lifespan` loads privacy-filter and EasyOCR onto `app.state`. Frontend served at `/app` (`/` redirects). CORS `*` with `expose_headers=["Content-Disposition"]`. A global exception handler attaches CORS headers manually so errors reach the browser instead of appearing as "Failed to fetch". Multipart limits raised to 50 000 files/fields (Starlette default is 1 000).

**State:** `_jobs` — in-memory dict `{job_id: {status, created_at, stages, error, outputs, export, event_queue}}`. Lost on restart.

| Function / endpoint | Role | Why / notes |
|---|---|---|
| `_new_job`, `_push_event`, `_get_job_or_404` | Job record; push JSON event onto the job's `queue.Queue`. | Thread → async SSE bridge. |
| `_safe_relpath(raw)` | Drop `.`/`..`/empty segments of browser `webkitRelativePath`. | Prevents writing outside the session dir. |
| `_match_pdfs_to_studies(studies, dir)` | Pair each PDF with the study whose files share the deepest common directory. | No PDF content is read; ties go to all tied studies. |
| `_run_pipeline_thread(...)` | S1→S5 + audits (progress 5→100). Zero NIfTI = warning. On success stores `outputs` and `export` (dicom dir, nifti list, pdf, fhir). Any exception → `status=error`. Always pushes the `None` sentinel. | Output DICOM dir uses the **original** PatientID **[P7]**; `pii_audit.json` stores originals **[P1]**. |
| `POST /jobs` | Single patient: save `.dcm` uploads by **basename** into a flat dir + the PDF, start thread. | Nested series folders with repeated filenames overwrite each other **[B3]**. |
| `POST /analyze` | Stage files under `sessions/<sid>`, group, match PDFs, return per-patient structure + validation errors. Single mode deletes the session (frontend re-uploads via `/jobs`). | Preview before committing; header-based grouping. |
| `POST /jobs/from-session` | Re-scan session, symlink the chosen study's files (named by SOP UID) into the job, resolve PDF (explicit upload wins), start thread. | No re-upload. Re-scans the whole session per patient (O(n²) header reads for big batches). `session_id` is not validated as a UUID **[S2]**. |
| `DELETE /sessions/{sid}` | `rmtree` the session. | Called by the frontend after the batch or on Clear. |
| `GET /export?job_ids=…` | Validate jobs done → build zip in a worker thread → stream 1 MB chunks → delete zip. | Native browser download (no JS buffering). Zip build is fully synchronous before the first byte. |
| `GET /jobs/{id}/stream` | SSE: poll queue in an executor (30 s timeout → `: keepalive`). | — |
| `GET /jobs/{id}/status` | Status + fixed download URLs. | — |
| `GET /jobs/{id}/download/{type}` | `nifti` (first series) / `fhir` / `pii_audit` / `preservation`. | Unauthenticated access to original PII via `pii_audit` **[P1][S1]**. |
| `DELETE /jobs/{id}` | Remove job dir + registry entry. | Never called by the frontend; jobs accumulate **[O3]**. |
| `GET /health` | Liveness + models loaded. | — |
| `__main__` | `uvicorn.run(host="0.0.0.0", port=8000, reload=False)` | Listens on all interfaces **[S1]**. |

## 2.10 `pipeline.py` — CLI orchestrator

`run_pipeline(dicom_dir, pdf, out)`: validate paths → read (flat, single study) → load models → anonymize → NIfTI → `extract_pdf` (bare id) → FHIR → summary. Helpers `_banner`, `_elapsed`. Diverged from the API: no redacted PDF, no `pii_audit.json` / `preservation_report.json`, flat single-study input only. Zero converted series is not an error here either (`convert_to_nifti` returns `[]` and the bundle simply has no DocumentReferences).

## 2.11 Utilities (`dicom_verify.py`, `pdf_analyzer.py`, `view_nifti.py`)

| Module | What it does | Notes |
|---|---|---|
| `dicom_verify.py` — `verify(orig, anon)` | File counts; pairs files via `make_anon_uid(orig SOP)`; per file `check_file`: wipe tags empty, name/ID pseudonymized & no PII substrings, UIDs changed, StudyUID consistent, pixel hash unchanged (first 10), free text, technical tags retained; then study/series UID consistency and original-UID leak scan. Exit code 0/1. | Its `WIPE_TAGS` is a 28-tag subset of the anonymizer's ~70 despite "must match exactly". Pairing uses single-hash SOP UIDs, so against current output **every file reports NO MATCH** (consequence of **[B1]**). Pixel check fails by design when OCR scrubbed. Flat folders only. Not wired into the pipeline. |
| `pdf_analyzer.py` — `inspect_pdf(path)` | Per page: raw text, layout text, page-1 word boxes, tables. | Docstring names it `pdf_inspect.py`. Used to tune the legacy parser thresholds. |
| `view_nifti.py` — `inspect_and_plot(path, png)` | Header info + 9 axial slices using `cal_min/max` or 2–98th percentile. | Docstring names it `nifti_viewer.py`. Stray debug `print(cal_min)`. |

## 2.12 Frontend

Vanilla HTML/JS, served by the API at `/app`. `assets/script.js` holds `API_BASE = 'http://localhost:8000'`, demo auth (`sessionStorage.loggedIn`), header/footer injection, `fmtBytes`, `escapeHtml`, `fmtTimestamp`. `style.css` is presentation only.

| Page | Behaviour |
|---|---|
| `index.html` | Landing; redirects to upload if "logged in". |
| `login.html` | Hard-coded `admin` / `admin123` checked **in the browser** **[S3]**. |
| `upload.html` | Folder pick → debounced `POST /analyze` (stale responses discarded via a token) → structure tree + validation. Single: `POST /jobs` (re-uploads dcm files) → SSE. Multi: **sequential** `POST /jobs/from-session` per patient, each streamed to completion; failures don't stop the batch; `DELETE /sessions` at the end; Clear also deletes a staged session. Results stored in `sessionStorage.pipelineResult` → `results.html`. |
| `results.html` | Meta strip (patients/completed/failed), "Download All" → hidden `<a target=_blank>` to `GET /export?job_ids=…`, per-patient cards with PII before/after from `pii_audit`. The "Outputs Ready" ticks and "Anonymization Summary" are **static text**, shown even when e.g. zero NIfTI volumes were produced **[Q7]**. Preservation report is not displayed. |

## 2.13 Configuration and on-disk state

| Item | Location / default |
|---|---|
| Job workspaces | `src/jobs/<uuid>/` — never auto-deleted (139 present at time of writing) |
| Multi-patient staging | `src/sessions/<uuid>/` — deleted by the frontend; orphaned if the tab closes mid-batch |
| PDF templates | `PDF_TEMPLATE_STORE_DIR` or OS user-data dir |
| NuExtract GGUF | HF cache (or `NUEXTRACT_GGUF_PATH`) |
| UID / pseudonym salt | `"pipeline-salt-v1"`, hard-coded |
| Server | `0.0.0.0:8000`, CORS `*` |
| Python | 3.14 (venv); `requirements.txt` omits `pymupdf`, `llama-cpp-python`, `openai` **[O1]** |

---

# Part 3 — Assumptions

**Input data**
1. DICOM files carry a `.dcm` extension (extension-less files are ignored everywhere).
2. One upload folder ⇒ files are valid DICOM with consistent `PatientID` per `StudyInstanceUID`.
3. "Patient" = `StudyInstanceUID`; a multi-patient folder has one study per patient.
4. Patient-identity tags (name, ID, dates, institution, free text) are identical across all instances of a study — so the first instance is representative.
5. A study's need for OCR can be judged from its first ≤3 image instances.
6. Burned-in text is English, horizontal, and visible after windowing raw stored values with WC/WW.
7. A NIfTI-worthy series has ≥3 slices; anything smaller is a localizer/reference object.
8. Each patient has exactly one report PDF; in multi mode it sits in that patient's folder.
9. PDFs are text-based (not scanned images) and roughly A4: header in the top 160 pt, two columns split at x = 300 pt.
10. Report body uses recognizable section headings, and the sign-off block begins with "Reported by" / "signed by" / "verified by".
11. Signatures/stamps are embedded images narrower than 60% of the page; letterheads are full-width.
12. Header fields are on page 0; body may span pages.

**Policy**
13. Age, sex, size, weight are not identifying on their own and are retained.
14. Deterministic pseudonyms (same input ⇒ same output) are desirable for re-runs and cross-modality joins.
15. Wiping all dates (no date shifting) is acceptable for downstream research.
16. The reporting radiologist's identity is PII and is removed from the PDF.

**Environment**
17. Single user, local machine, trusted network; the API is not exposed.
18. The filesystem supports symlinks.
19. Enough RAM/CPU for EasyOCR + privacy-filter + (on first new layout) a 2.8 GB GGUF on CPU.
20. First NuExtract use has internet access (or `NUEXTRACT_GGUF_PATH` is set).

---

# Part 4 — Shortcomings

Severity: **B** = confirmed correctness bug · **P** = privacy · **S** = security · **R** = robustness/input coverage · **Q** = output quality/standards · **O** = operations/packaging · **D** = documentation drift.

### Confirmed correctness bugs

| ID | Issue | Evidence / location | Impact |
|---|---|---|---|
| **B1** | SOPInstanceUID is hashed **twice**: the walker remaps it (it's in `UID_TAGS_ALL`), then `_anonymize_dataset` hashes the already-remapped value. MediaStorageSOPInstanceUID (file meta, not walked) is hashed once. | `dicom_anonymizer.py:443` **[verified]** — 0/555 FHIR instance UIDs match the files; every file's MediaStorage ≠ SOP UID | DICOM conformance violation (file meta disagrees with dataset); FHIR ImagingStudy references non-existent instances; `ReferencedSOPInstanceUID`s (single hash) point at nothing; `dicom_verify.py` can't pair any file. |
| **B2** | `anon_patient_id` already includes `ANON-`, and the walker and FHIR prepend it again. | `dicom_anonymizer.py:384`, `fhir_mapper.py:162` **[verified]** — DICOM PatientName and FHIR name = `ANON-ANON-07CF…`; PDF says `ANON-07CF…`; PDF `reg_no` = bare hash vs DICOM PatientID `ANON-…` | Identifiers disagree across outputs; the README's "PDF reg_no = DICOM PatientID join key" doesn't hold. |
| **B3** | `POST /jobs` saves uploads by basename into one flat dir. | `api.py:546` | Single-patient uploads organized in series subfolders with repeating filenames (`IM0001.dcm` per series) silently overwrite each other — data loss. The UI advertises this layout as supported. |
| **B4** | Free-text redaction is computed from the first instance and stamped on all. | `dicom_anonymizer.py:399` | Every series gets the **first series'** SeriesDescription; a free-text tag absent from instance 0 but present later is left **unredacted**. |
| **B5** | `UID_TAGS_ALL` includes ReferencedSOPClassUID. | `dicom_anonymizer.py:178` | SOP Class UIDs are standard constants (e.g. CT Image Storage); remapping them corrupts references' meaning. |

### Privacy

| ID | Issue | Impact |
|---|---|---|
| **P1** | `pii_audit.json` stores original name, ID, institution, DOB, referring doctor (DICOM + PDF) in the job folder **[verified]**; downloadable via `/jobs/{id}/download/pii_audit` without auth. Uploaded originals also stay in `jobs/<id>/input/` indefinitely. | Original PHI persists on disk and is network-reachable. |
| **P2** | Private (odd-group) tags are never removed. | Vendors routinely store names/IDs/protocol text in private tags; PS3.15 Basic Profile requires removal. |
| **P3** | FHIR `ImagingStudy.description` and series descriptions come from the original study object, not the redacted values. | `fhir_mapper.py:229,259` — free-text PII bypasses NER into the bundle. |
| **P4** | OCR decision is per **study**, from its first 3 image instances. | A later series with burned-in text (secondary capture, dose report, screenshot) is never scrubbed if the first series is clean. |
| **P5** | NuExtract identity values that fail verification (e.g. model de-doubles `AANNBB…`) produce no redaction box; header-zone NER fallback only triggers when **no** identity box was found. | A patient name can survive in the visually redacted PDF header. |
| **P6** | Cold path uses unverified model values for technique, impression and header fields. | Hallucinated text can enter structured output/FHIR. |
| **P7** | Output DICOM folder is named `patient_<original PatientID>` (also creates nested dirs when the ID contains `/`). | Original MRN in server paths (not in the zip). |
| **P8** | Pseudonym = SHA-256 of PatientID with a **public, hard-coded** salt. | Anyone with the source and a candidate MRN can confirm linkage; MRNs are low-entropy → brute-forceable. Same for UIDs. |
| **P9** | `study_title` (PDF) is never NER-redacted; ages are retained — ages >89 are identifying under HIPAA Safe Harbor. | Policy gaps. |

### Security

| ID | Issue |
|---|---|
| **S1** | No authentication on any endpoint; CORS `*`; server binds `0.0.0.0`. Anyone on the network can upload, read audits (original PHI), export, or delete. |
| **S2** | `session_id` (form field / path param) is used in filesystem paths without UUID validation (e.g. `SESSIONS_DIR / ".."`). Low exploitability in practice but should be validated. |
| **S3** | Login is a client-side string comparison (`admin`/`admin123`) stored in `sessionStorage` — not access control. |

### Robustness / input coverage

| ID | Issue |
|---|---|
| **R1** | Job registry is in memory; a restart orphans all jobs (export then 404s). |
| **R2** | No job cleanup and no session TTL; disk grows unbounded (139 job dirs at time of writing). |
| **R3** | Single-patient mode uploads the whole folder twice (`/analyze`, then `/jobs`). `/jobs/from-session` re-reads every header in the session per patient. |
| **R4** | Extension-less DICOM (common: `IM_0001`, `DICOMDIR` trees) is ignored. |
| **R5** | OCR is English-only. |
| **R6** | OCR windowing ignores RescaleSlope/Intercept (WC/WW are in rescaled units) — e.g. CT stored values are offset by ~1024 from HU, so the OCR image can be badly windowed and text missed. |
| **R7** | Colour images: single-frame RGB `(rows, cols, 3)` is split into "frames"; multi-frame RGB is 4-D. Ultrasound/secondary-capture (exactly the modalities with burned-in text) are the ones affected. Photometric interpretation isn't updated after decompression. |
| **R8** | dicom2nifti orientation/slice-increment validation disabled → missing slices, gantry tilt, mixed-orientation series convert silently with wrong geometry. Series with <3 slices produce no NIfTI. |
| **R9** | `_fix_doubled_chars` also collapses legitimate tokens (`"11"` → `"1"`, `"AA"` → `"A"`). |
| **R10** | Legacy parser thresholds are absolute A4 points (x = 300, top < 160). |
| **R11** | Body lines before the first recognized section heading are dropped; a report with no known headings yields empty findings (README claims the opposite). |
| **R12** | `_apply_template` over-captures neighbouring header text (220 pt right margin) and loses age/sex on the template path. |
| **R13** | Scanned (image-only) PDFs: no text → no extraction, and pages without words are skipped by visual redaction entirely — the PDF passes through unredacted. |
| **R14** | `os.symlink` (job inputs, NIfTI temp dirs) needs Developer Mode/admin on Windows — relevant to the packaging plan. |
| **R15** | Shared `clf`/OCR models are used from multiple threads without locking; safe today only because the frontend runs patients sequentially. Two browser tabs can run jobs concurrently. |

### Output quality / standards

| ID | Issue |
|---|---|
| **Q1** | Single-value UID map from instance 0 is mostly informational (the walker derives everything else on the fly); fine functionally, misleading in logs. |
| **Q2** | ~~~180 templates created before the keyword fix never matched~~ — fixed: matching rewritten, old templates deleted. Remaining: strict matching gives near-identical variants (`Date` vs `Date :`) separate templates. |
| **Q3** | `full_text_anon` costs an extra NER pass and is never used. |
| **Q4** | NuExtract runs at `temperature=0.2` — the docstring calls it deterministic; it isn't. |
| **Q5** | LOINC lookup only has CT codes (MR brain → "CT of head"); keyword-first-match. |
| **Q6** | FHIR: `urn:uuid:` fullUrls with non-UUID ids; DocumentReference `file://` URLs to server paths; custom extension URL; non-deterministic modality order and timestamp (bundle not byte-identical across runs); no `fhir.resources` validation; "ABDM compliant" is asserted, not tested. |
| **Q7** | Results page shows static ticks/summary claims regardless of what actually happened; preservation failures aren't surfaced in the UI. |
| **Q8** | No automated tests anywhere in the repo. |

### Operations / packaging

| ID | Issue |
|---|---|
| **O1** | `requirements.txt` omits `pymupdf` (visual redaction → job **fails** without it), `llama-cpp-python` (cold path silently falls back), `openai` (llama-server backend). |
| **O2** | First new PDF layout blocks its job on a 2.8 GB download + ~20–35 s CPU inference; for an offline executable, ship the GGUF and set `NUEXTRACT_GGUF_PATH`. |
| **O3** | `DELETE /jobs/{id}` exists but nothing calls it. |
| **O4** | `API_BASE` is hard-coded to `http://localhost:8000`; the UI breaks if served from another host/port. |
| **O5** | `pipeline.py` and `api.py` have diverged (CLI lacks redacted PDF and audits); CLI helpers in `dicom_to_nifti.py`/`fhir_mapper.py` load the NER model and don't pass it on. |

### Documentation drift

| ID | Issue |
|---|---|
| **D1** | README lists `pdf_inspect.py` / `nifti_viewer.py` (actual: `pdf_analyzer.py` / `view_nifti.py`), doesn't mention the template store, NuExtract, multi-patient flow, or `/export`, and still lists multi-patient as a limitation. |
| **D2** | README says PatientAge is wiped and that ImagingStudy instance UIDs match filenames — both untrue now. |
| **D3** | `pdf_extractor.py` docstring says "No PDF is saved"; `dicom_reader.py` docstring describes lazy pixel loading that doesn't happen; `dicom_verify.py` claims its tag lists match the anonymizer exactly. |

### Suggested fix order
1. **B1, B2, B5** (UID/identifier integrity — small, contained fixes in `dicom_anonymizer.py` / `fhir_mapper.py`).
2. **P1, S1** (stop persisting/serving original PHI; bind to localhost or add auth).
3. **P2, P3, P4, B4** (private tags, FHIR descriptions, per-series OCR, per-series free text).
4. **B3, O1, R14** (upload collisions, dependency list, Windows symlinks — prerequisites for packaging).
5. Add a regression test suite around the real sample study before further changes.
