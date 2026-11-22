"""
api.py
------
FastAPI backend for the medical imaging anonymization pipeline.

Endpoints
---------
POST /jobs
    Single-patient flow (unchanged). Upload DICOM files + PDF directly,
    start pipeline job. Returns { job_id: str }

POST /analyze
    Preview the structure of an upload (any mix of flat/nested .dcm and
    .pdf files) WITHOUT running the pipeline. Groups .dcm files into
    Patient > Study > Series > Instance by reading real DICOM headers.
    If exactly one patient is detected, the staged files are discarded
    and the frontend should submit via POST /jobs as usual (with its own
    PDF field). If more than one patient is detected, the staged files
    are kept under a session_id and each patient's report .pdf must be
    found nested in that patient's folder — use POST /jobs/from-session
    to run each one.

POST /jobs/from-session
    Multi-patient flow. Start a pipeline job for ONE patient/study already
    staged by a prior POST /analyze call — no re-upload of DICOM files.
    Body: session_id, study_uid, optional pdf_file override.
    Returns { job_id: str } — same shape as POST /jobs.

DELETE /sessions/{session_id}
    Clean up a staged multi-patient session once all its patients have
    been submitted (or the upload was abandoned).

GET /export
    Build and stream a single zip for one or more completed jobs, laid
    out as Anonymized_data_<timestamp>/Patient_<id>/Study_<id>/{dicom,
    nifti,pdf,fhir}/. Query params: job_ids=<id>&job_ids=<id>&...
    A plain GET so the frontend can trigger it as a real browser
    download/navigation rather than fetch()+blob() — see export_zip()'s
    docstring for why that matters for large multi-patient exports.

GET  /jobs/{job_id}/stream
    SSE stream of pipeline progress events.
    Streams stage updates until completion or error.
    Event format: { type, stage, message, progress }

GET  /jobs/{job_id}/status
    Current status and output download URLs once complete.

GET  /jobs/{job_id}/download/{file_type}
    Download output file. file_type: "nifti" | "fhir"

DELETE /jobs/{job_id}
    Clean up job files from disk.

CORS
----
Configured to accept requests from any origin (adjust ALLOWED_ORIGINS
for production).

Usage
-----
    pip install fastapi uvicorn python-multipart aiofiles
    python api.py

    # or with uvicorn directly:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import os
import queue
import shutil
import sys
import threading
import traceback
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
JOBS_DIR      = BASE_DIR / "jobs"          # working directory per job
SESSIONS_DIR  = BASE_DIR / "sessions"      # staged multi-patient uploads, from /analyze
ALLOWED_ORIGINS = ["*"]                    # restrict in production

# Starlette's automatic multipart parser caps a request at 1000 files and
# 1000 form fields by default (a DoS guard) — real DICOM studies routinely
# exceed that (thousands of slices across series/patients). FastAPI's
# File(...)/UploadFile binding always calls request.form() with no
# arguments, so it can't be raised through the decorator; /jobs and
# /analyze instead take the raw Request and call request.form() themselves
# with these higher limits.
FORM_MAX_FILES  = 50_000
FORM_MAX_FIELDS = 50_000

JOBS_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

# In-memory job registry  { job_id: JobState }
_jobs: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
#  App + CORS
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup and share across all jobs
    print("[API] Loading models at startup ...")
    from dicom_anonymizer import load_privacy_filter
    from pixel_scrubber import load_ocr_reader
    app.state.clf        = load_privacy_filter()
    app.state.ocr_reader = load_ocr_reader()
    print("[API] Models loaded. Ready.")
    yield
    # Cleanup on shutdown (nothing needed)


app = FastAPI(
    title="Medical Imaging Anonymization API",
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.staticfiles import StaticFiles
from pathlib import Path

FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    return RedirectResponse(url="/app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    # Content-Disposition isn't in the browser's default CORS-safelisted
    # response headers — without exposing it explicitly, results.html's
    # `res.headers.get('Content-Disposition')` (used to read the
    # suggested download filename from /export) silently returns null
    # for a cross-origin request, even though everything else worked.
    expose_headers=["Content-Disposition"],
)


def _cors_headers_for(request: Request) -> dict:
    """
    Compute the same CORS headers CORSMiddleware would normally add, for
    manual use in the global exception handler below.

    CORSMiddleware does NOT add headers to responses produced by
    exception handlers (verified directly — a bare JSONResponse from an
    @app.exception_handler has no Access-Control-* headers at all,
    regardless of the middleware being registered), so this is the
    necessary workaround. Mirrors ALLOWED_ORIGINS rather than hardcoding
    "*" here, so this stays correct if ALLOWED_ORIGINS is ever
    restricted for production.
    """
    origin = request.headers.get("origin")
    if "*" in ALLOWED_ORIGINS:
        allow_origin = "*"
    elif origin and origin in ALLOWED_ORIGINS:
        allow_origin = origin
    else:
        return {}
    return {
        "Access-Control-Allow-Origin": allow_origin,
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    }


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """
    Any exception that isn't already an HTTPException lands here instead
    of falling through to Starlette's default error handling. This
    matters specifically for CORS: a truly unhandled exception produces
    a response with NO CORS headers at all (see _cors_headers_for's
    docstring), which the browser reports to JS as a generic "Failed to
    fetch" — the real error message never reaches the frontend, just a
    dead end to debug from. Attaching CORS headers here ensures every
    error is actually visible to the frontend instead of looking like a
    network failure.
    """
    print(f"[ERROR] Unhandled exception on {request.method} {request.url.path}: {exc}")
    response = JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )
    response.headers.update(_cors_headers_for(request))
    return response


# ─────────────────────────────────────────────────────────────────────────────
#  Job state helpers
# ─────────────────────────────────────────────────────────────────────────────

def _new_job(job_id: str) -> dict:
    return {
        "job_id":     job_id,
        "status":     "queued",       # queued | running | done | error
        "created_at": datetime.utcnow().isoformat(),
        "stages":     [],             # list of completed stage names
        "error":      None,
        "outputs":    {},             # { "nifti": path, "fhir": path }
        "event_queue": queue.Queue(), # SSE events pushed here
    }


def _push_event(job_id: str, event_type: str, stage: str, message: str, progress: int = 0):
    """Push a progress event into the job's SSE queue."""
    if job_id not in _jobs:
        return
    payload = json.dumps({
        "type":     event_type,   # "progress" | "complete" | "error"
        "stage":    stage,
        "message":  message,
        "progress": progress,     # 0-100
    })
    _jobs[job_id]["event_queue"].put(payload)


def _get_job_or_404(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _jobs[job_id]


# ─────────────────────────────────────────────────────────────────────────────
#  Structure-analysis helpers (used by /analyze and /jobs/from-session)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_relpath(raw: str) -> Path:
    """
    Turn an untrusted client-supplied relative path (from a browser's
    File.webkitRelativePath) into a safe on-disk relative Path.

    Strips any leading slashes and drops '.'/'..' segments so a crafted
    filename can't escape the session's staging directory.
    """
    from pathlib import PurePosixPath
    posix = PurePosixPath(raw.replace("\\", "/"))
    parts = [seg for seg in posix.parts if seg not in ("", ".", "..", "/")]
    if not parts:
        raise ValueError(f"Unusable relative path: {raw!r}")
    return Path(*parts)


def _match_pdfs_to_studies(studies: list, session_dir: Path):
    """
    Associate .pdf files found anywhere under session_dir with the
    DicomStudy(s) whose DICOM files they sit "closest" to, by directory
    path only — no PDF content is ever read for this. A PDF is matched to
    whichever study's DICOM files share the deepest common ancestor
    directory with it. This correctly handles a PDF placed either directly
    alongside a flat dump of .dcm files, or one level above nested
    Study/Series subfolders (both are "inside the patient folder").

    Returns
    -------
    (matches, unassociated)
        matches       : dict[study_uid, list[Path]] — pdfs matched to each study
        unassociated  : list[Path] — pdfs that didn't fall inside any
                        patient's folder (e.g. sitting at the upload root)
    """
    study_roots: dict[str, Path] = {}
    for study in studies:
        dirs = [inst.path.parent for inst in study.all_instances]
        study_roots[study.study_uid] = Path(os.path.commonpath([str(d) for d in dirs]))

    all_pdfs = sorted(session_dir.rglob("*.pdf"))
    session_depth = len(session_dir.parts)

    matches: dict[str, list] = {s.study_uid: [] for s in studies}
    unassociated: list = []

    for pdf in all_pdfs:
        best_depth = session_depth
        best_uids: list[str] = []
        for uid, root in study_roots.items():
            common = os.path.commonpath([str(pdf.parent), str(root)])
            depth = len(Path(common).parts)
            if depth > best_depth:
                best_depth, best_uids = depth, [uid]
            elif depth == best_depth and depth > session_depth:
                best_uids.append(uid)
        if best_uids:
            for uid in best_uids:
                matches[uid].append(pdf)
        else:
            unassociated.append(pdf)

    return matches, unassociated


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline runner (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_thread(
    job_id: str,
    dicom_dir: Path,
    pdf_path: Path,
    output_dir: Path,
    clf,
    ocr_reader,
):
    """
    Run the full pipeline in a background thread.
    Pushes SSE events at each stage for the frontend to consume.
    """
    job = _jobs[job_id]
    job["status"] = "running"

    def push(event_type, stage, message, progress=0):
        _push_event(job_id, event_type, stage, message, progress)

    try:
        # ── Stage 1 — Read DICOM ──
        push("progress", "read_dicom", "Reading DICOM folder...", 5)
        from dicom_reader import read_and_group
        study = read_and_group(str(dicom_dir))
        push("progress", "read_dicom",
             f"Found {study.total_instances} instances across {len(study.series)} series", 10)

        # ── Stage 2 — Anonymize DICOM ──
        push("progress", "anonymize_dicom", "Anonymizing DICOM tags and pixels...", 15)
        from dicom_anonymizer import anonymize_study, make_anon_uid
        anon_dicom_dir = output_dir / "dicom" / f"patient_{study.patient_id}"
        anon_result = anonymize_study(
            study,
            output_dir=str(anon_dicom_dir),
            clf=clf,
            ocr_reader=ocr_reader,
        )
        anon_patient_id = anon_result["anon_patient_id"]
        anon_study_uid  = anon_result["uid_map"].get(
            study.study_uid, make_anon_uid(study.study_uid)
        )
        push("progress", "anonymize_dicom",
             f"Anonymized {anon_result['files_written']} DICOM files", 40)

        # ── Stage 3 — Convert to NIfTI ──
        push("progress", "nifti", "Converting to NIfTI...", 45)
        from dicom_to_nifti import convert_to_nifti
        nifti_results = convert_to_nifti(
            study,
            anon_result,
            output_dir=str(output_dir / "nifti"),
        )
        # Every converted series, not just the first — a study can have
        # more than one series and the export zip needs all of them.
        nifti_paths = [Path(r["nifti_path"]) for r in nifti_results]
        if not nifti_paths:
            # A study can legitimately contain no NIfTI-eligible series —
            # e.g. only localizers/scouts/non-image objects (presentation
            # states, key object docs, structured reports) below
            # dicom2nifti's 3-slice minimum. The anonymized DICOM output
            # (already written above) and the PDF/FHIR stages below are
            # still valid and useful on their own — degrade gracefully
            # instead of failing the whole job and discarding that work.
            msg = (f"No series met the minimum slice count for NIfTI conversion "
                   f"({len(study.series)} series checked) — continuing without a NIfTI volume")
            print(f"[WARN] {msg}")
            push("progress", "nifti", msg, 65)
        else:
            push("progress", "nifti",
                 f"NIfTI written: {len(nifti_paths)} series", 65)

        # ── Stage 4 — Extract and anonymize PDF ──
        push("progress", "pdf",
             "Extracting and anonymizing PDF (first run may download/load a model — can take a while)...",
             70)
        from pdf_extractor import extract_pdf, redact_pdf_visual, build_pdf_pii_audit
        bare_id = anon_patient_id.removeprefix("ANON-")
        pdf_extract = extract_pdf(str(pdf_path), clf, anon_patient_id=bare_id)

        # Redact the ORIGINAL PDF in place — same layout/look, PII regions
        # truly removed (not just covered) and blacked out. This is the
        # deliverable that goes in the export zip's pdf/ subfolder.
        anon_pdf_dir  = output_dir / "pdf"
        anon_pdf_path = anon_pdf_dir / f"patient_{anon_patient_id}_report_anonymized.pdf"
        redact_pdf_visual(str(pdf_path), clf, str(anon_pdf_path))
        push("progress", "pdf", "PDF anonymized", 80)

        # ── Stage 5 — Build FHIR bundle ──
        push("progress", "fhir", "Building FHIR R4 bundle...", 85)
        from fhir_mapper import map_to_fhir
        fhir_dir = output_dir / "fhir"
        map_to_fhir(study, anon_result, pdf_extract, nifti_results, str(fhir_dir))
        fhir_path = fhir_dir / f"patient_{anon_patient_id}_bundle.json"
        push("progress", "fhir", "FHIR R4 bundle written", 92)

        # ── PII audit — before vs after, for the results-page comparison ──
        # Built from work already done above (anon_result + pdf_extract);
        # never written into any clinical output, purely a review artifact.
        pii_audit = {
            "dicom": anon_result.get("pii_audit", []),
            "pdf":   build_pdf_pii_audit(pdf_extract),
        }
        pii_audit_path = output_dir / "pii_audit.json"
        with open(str(pii_audit_path), "w", encoding="utf-8") as f:
            json.dump(pii_audit, f, indent=2, ensure_ascii=False)
        push("progress", "fhir", "PII audit report written", 94)

        # ── Preservation check — technical/scanner/geometry params and
        # pixel data, across both the DICOM and NIfTI outputs. Internal QA
        # step: never blocks the pipeline, but is written to disk and
        # surfaced as a warning (not silently dropped) if anything that
        # should have survived anonymization didn't.
        nifti_geometry_checks = [
            {
                "series_uid": r["series_uid"],
                "modality":   r["modality"],
                **r.get("geometry_check", {}),
            }
            for r in nifti_results
        ]
        dicom_preservation = anon_result.get("preservation_check", {})
        preservation_report = {
            "dicom_technical": dicom_preservation.get("technical", []),
            "dicom_pixel":     dicom_preservation.get("pixel", {}),
            "nifti":           nifti_geometry_checks,
        }

        preservation_failures = [
            e for e in preservation_report["dicom_technical"] if not e.get("preserved", True)
        ]
        if not preservation_report["dicom_pixel"].get("preserved", True):
            preservation_failures.append({"name": "Pixel data", **preservation_report["dicom_pixel"]})
        preservation_failures += [g for g in nifti_geometry_checks if not g.get("preserved", True)]

        preservation_path = output_dir / "preservation_report.json"
        with open(str(preservation_path), "w", encoding="utf-8") as f:
            json.dump(preservation_report, f, indent=2, ensure_ascii=False)

        if preservation_failures:
            msg = (f"Warning: {len(preservation_failures)} technical parameter(s)/geometry "
                   f"check(s) did not match the source — see preservation report")
            print(f"[WARN] {msg}")
            push("progress", "fhir", msg, 96)
        else:
            push("progress", "fhir",
                 "Technical parameters, scanner info, geometry, and pixel data verified preserved", 96)

        # ── Done ──
        job["status"]  = "done"
        job["outputs"] = {
            "fhir":         str(fhir_path),
            "pii_audit":    str(pii_audit_path),
            "preservation": str(preservation_path),
        }
        if nifti_paths:
            # First series only — see /export for all. Omitted entirely
            # (rather than set to None/"") when no series converted, so
            # GET /jobs/{id}/download/nifti's existing "file not found"
            # 404 path handles it without special-casing an empty value.
            job["outputs"]["nifti"] = str(nifti_paths[0])
        # Everything /export needs to assemble the full per-patient zip.
        job["export"] = {
            "anon_patient_id": anon_patient_id,
            "anon_study_uid":  anon_study_uid,
            "dicom_dir":       str(anon_dicom_dir),
            "nifti_files":     [str(p) for p in nifti_paths],
            "pdf_file":        str(anon_pdf_path),
            "fhir_file":       str(fhir_path),
        }
        push("complete", "done",
             f"Pipeline complete. Patient ID: {anon_patient_id}", 100)

    except Exception as e:
        tb = traceback.format_exc()
        job["status"] = "error"
        job["error"]  = str(e)
        push("error", "error", f"Pipeline failed: {e}", 0)
        print(f"[API] Job {job_id} failed:\n{tb}")

    finally:
        # Signal SSE stream to close
        job["event_queue"].put(None)


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/jobs", status_code=202)
async def create_job(request: Request):
    """
    Upload DICOM files and a PDF report, start the anonymization pipeline.

    Returns a job_id which can be used to stream progress and download outputs.

    Parses the multipart body manually (rather than via FastAPI's automatic
    File(...) parameters, which always defers to request.form() with no
    arguments — i.e. Starlette's default 1000-file cap). Real studies can
    easily exceed 1000 slices.

    Example (curl):
        curl -X POST http://localhost:8000/jobs \\
          $(for f in *.dcm; do echo "-F dicom_files=@$f"; done) \\
          -F pdf_file=@report.pdf
    """
    form = await request.form(max_files=FORM_MAX_FILES, max_fields=FORM_MAX_FIELDS)
    dicom_files = [f for f in form.getlist("dicom_files") if hasattr(f, "filename")]
    pdf_file    = form.get("pdf_file")

    if not dicom_files:
        raise HTTPException(status_code=400, detail="No dicom_files provided.")
    if pdf_file is None or not hasattr(pdf_file, "filename"):
        raise HTTPException(status_code=400, detail="No pdf_file provided.")

    job_id  = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    # ── Save uploaded DICOM files ──
    dicom_dir = job_dir / "input" / "dicom"
    dicom_dir.mkdir(parents=True)

    for upload in dicom_files:
        if not upload.filename.endswith(".dcm"):
            raise HTTPException(
                status_code=400,
                detail=f"File {upload.filename} is not a .dcm file",
            )
        dest = dicom_dir / Path(upload.filename).name
        async with aiofiles.open(str(dest), "wb") as f:
            content = await upload.read()
            await f.write(content)

    # ── Save uploaded PDF ──
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="pdf_file must be a .pdf")
    pdf_path = job_dir / "input" / pdf_file.filename
    async with aiofiles.open(str(pdf_path), "wb") as f:
        content = await pdf_file.read()
        await f.write(content)

    # ── Register job ──
    _jobs[job_id] = _new_job(job_id)
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True)

    # ── Start pipeline in background thread ──
    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(
            job_id,
            dicom_dir,
            pdf_path,
            output_dir,
            app.state.clf,
            app.state.ocr_reader,
        ),
        daemon=True,
    )
    thread.start()

    return {
        "job_id":      job_id,
        "status":      "queued",
        "stream_url":  f"/jobs/{job_id}/stream",
        "status_url":  f"/jobs/{job_id}/status",
    }


@app.post("/analyze")
async def analyze_upload(request: Request):
    """
    Preview the structure of an upload WITHOUT running the pipeline.

    Groups .dcm files into Patient > Study > Series > Instance using real
    DICOM header tags (PatientID / StudyInstanceUID / SeriesInstanceUID) —
    never folder names. This works for any layout: one flat folder, or many
    nested per-patient folders.

    If exactly one patient is detected, the staged files are discarded —
    the frontend should submit via POST /jobs as usual, with its own PDF
    field.

    If more than one patient is detected, each patient's report PDF must
    already be nested inside that patient's folder (matched purely by
    directory proximity — PDF content is never inspected). The staged
    files are kept under the returned session_id so POST /jobs/from-session
    can use them without a second upload. Call
    DELETE /sessions/{session_id} once every patient has been submitted.

    Parses the multipart body manually with a raised file-count limit —
    see /jobs for the same fix and rationale (a large study can easily
    exceed Starlette's default 1000-file cap).
    """
    form = await request.form(max_files=FORM_MAX_FILES, max_fields=FORM_MAX_FIELDS)
    files = [f for f in form.getlist("files") if hasattr(f, "filename")]

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    session_id  = str(uuid.uuid4())
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True)

    dcm_count = 0
    pdf_count = 0
    ignored   = 0

    for upload in files:
        try:
            rel = _safe_relpath(upload.filename or "")
        except ValueError:
            ignored += 1
            continue

        lower = rel.name.lower()
        if not (lower.endswith(".dcm") or lower.endswith(".pdf")):
            ignored += 1
            continue

        dest = session_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(str(dest), "wb") as f:
            content = await upload.read()
            await f.write(content)

        if lower.endswith(".dcm"):
            dcm_count += 1
        else:
            pdf_count += 1

    if dcm_count == 0:
        shutil.rmtree(str(session_dir), ignore_errors=True)
        raise HTTPException(status_code=400, detail="No .dcm files found in upload.")

    from dicom_reader import read_and_group_multi
    try:
        studies = read_and_group_multi(str(session_dir))
    except ValueError as e:
        shutil.rmtree(str(session_dir), ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))

    patient_count = len(studies)
    mode = "single" if patient_count == 1 else "multi"

    pdf_matches, unassociated_pdfs = _match_pdfs_to_studies(studies, session_dir)

    patients_json = []
    errors: list[str] = []

    for study in studies:
        matched = pdf_matches[study.study_uid]
        dirs = [inst.path.parent for inst in study.all_instances]
        folder = Path(os.path.commonpath([str(d) for d in dirs])).relative_to(session_dir)

        if mode == "single":
            # Single-patient mode keeps today's separate PDF upload field —
            # a nested pdf isn't required (or used) here.
            pdf_status = "not_required"
        elif len(matched) == 1:
            pdf_status = "ok"
        elif len(matched) == 0:
            pdf_status = "missing"
            errors.append(
                f"Patient '{study.patient_name or study.patient_id}': no PDF "
                f"report found in its folder. Place the report .pdf inside "
                f"the same folder as this patient's DICOM series."
            )
        else:
            pdf_status = "multiple"
            errors.append(
                f"Patient '{study.patient_name or study.patient_id}': found "
                f"{len(matched)} PDF files in its folder — expected exactly 1."
            )

        patients_json.append({
            "study_uid":         study.study_uid,
            "patient_name":      study.patient_name,
            "patient_id":        study.patient_id,
            "study_description": study.study_description,
            "study_date":        study.study_date,
            "folder":            str(folder),
            "series": [
                {
                    "series_uid":         s.series_uid,
                    "modality":           s.modality,
                    "series_description": s.series_description,
                    "instance_count":     len(s.instances),
                }
                for s in study.series.values()
            ],
            "pdf_files":  [str(p.relative_to(session_dir)) for p in matched],
            "pdf_status": pdf_status,
        })

    unassociated_rel = [str(p.relative_to(session_dir)) for p in unassociated_pdfs]
    if mode == "multi" and unassociated_rel:
        errors.append(
            f"{len(unassociated_rel)} PDF file(s) could not be matched to any "
            f"patient folder: {', '.join(unassociated_rel)}. Move each report "
            f"into its patient's folder, alongside that patient's DICOM series."
        )

    # Single-patient sessions aren't needed after analysis — the frontend
    # re-submits via POST /jobs directly, so free the disk immediately.
    if mode == "single":
        shutil.rmtree(str(session_dir), ignore_errors=True)
        session_id = None

    return {
        "session_id":        session_id,
        "mode":              mode,
        "patient_count":     patient_count,
        "patients":          patients_json,
        "unassociated_pdfs": unassociated_rel,
        "errors":            errors,
        "ignored_files":     ignored,
    }


@app.post("/jobs/from-session", status_code=202)
async def create_job_from_session(
    session_id: str = Form(...),
    study_uid: str = Form(...),
    pdf_file: Optional[UploadFile] = File(
        None,
        description="Only needed if this patient's folder didn't already "
                    "contain exactly one PDF",
    ),
):
    """
    Start a pipeline job for ONE patient/study already staged by a prior
    POST /analyze call (multi-patient flow). No re-upload of the DICOM
    files is needed — they're read directly from the session's staging dir.
    """
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found or already cleaned up.",
        )

    from dicom_reader import read_and_group_multi
    try:
        studies = read_and_group_multi(str(session_dir))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    study = next((s for s in studies if s.study_uid == study_uid), None)
    if study is None:
        raise HTTPException(
            status_code=404,
            detail=f"Study {study_uid} not found in session {session_id}",
        )

    job_id  = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    # ── Materialize this study's DICOM files into the job's input dir ──
    # Symlinked (not copied) — cheap and avoids duplicating potentially
    # large pixel data on disk. Named by SOPInstanceUID (globally unique
    # per DICOM) rather than original filename, since files pulled from
    # different nested series subfolders could otherwise collide on name.
    dicom_dir = job_dir / "input" / "dicom"
    dicom_dir.mkdir(parents=True)
    for inst in study.all_instances:
        link = dicom_dir / f"{inst.sop_instance_uid or uuid.uuid4().hex}.dcm"
        os.symlink(inst.path.resolve(), link)

    # ── Resolve the PDF: an explicit upload wins, otherwise use the one
    #    already matched to this patient's folder during /analyze ──
    pdf_path: Optional[Path] = None
    if pdf_file is not None and pdf_file.filename:
        if not pdf_file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="pdf_file must be a .pdf")
        pdf_path = job_dir / "input" / Path(pdf_file.filename).name
        async with aiofiles.open(str(pdf_path), "wb") as f:
            content = await pdf_file.read()
            await f.write(content)
    else:
        matches, _ = _match_pdfs_to_studies(studies, session_dir)
        candidates = matches.get(study_uid, [])
        if len(candidates) == 1:
            pdf_path = candidates[0]

    if pdf_path is None:
        raise HTTPException(
            status_code=400,
            detail="No PDF report found for this patient — upload one via "
                   "pdf_file, or place exactly one .pdf in the patient's folder.",
        )

    # ── Register + run — identical machinery to POST /jobs ──
    _jobs[job_id] = _new_job(job_id)
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True)

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(
            job_id,
            dicom_dir,
            pdf_path,
            output_dir,
            app.state.clf,
            app.state.ocr_reader,
        ),
        daemon=True,
    )
    thread.start()

    return {
        "job_id":     job_id,
        "status":     "queued",
        "stream_url": f"/jobs/{job_id}/stream",
        "status_url": f"/jobs/{job_id}/status",
    }


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """
    Delete a staged multi-patient upload session — call this once every
    patient in the session has been submitted via /jobs/from-session (or
    the upload was abandoned).
    """
    session_dir = SESSIONS_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(str(session_dir), ignore_errors=True)


@app.get("/export")
async def export_zip(job_ids: list[str] = Query(..., min_length=1)):
    """
    Build and stream a single zip containing the complete anonymized output
    for one or more completed jobs, laid out as:

        Anonymized_data_<timestamp>/
          Patient_<anon_patient_id>/
            Study_<anon_study_uid_short>/
              dicom/   — every anonymized .dcm file
              nifti/   — every anonymized .nii.gz file (one per series)
              pdf/     — anonymized report PDF (PII replaced with [ENTITY] tokens)
              fhir/    — FHIR R4 bundle JSON

    All folder and file names use only anonymized identifiers — nothing
    from the original patient name or filenames is ever used in the
    archive layout.

    A plain GET (not POST+JSON) so the frontend can trigger this as a
    real browser navigation/download (an <a> click or window.open)
    rather than fetch()+response.blob(). A multi-patient export can run
    into the hundreds of MB to several GB — buffering that much into a
    JS Blob before it can be saved is exactly the kind of thing that
    produces a browser-side "Failed to fetch" (net::ERR_FAILED) that has
    nothing to do with the server, which built and sent a perfectly
    complete response either way. A real navigation lets the browser
    stream straight to disk instead.

    Query params: job_ids=<id>&job_ids=<id>&...
    """
    jobs = []
    for jid in job_ids:
        job = _jobs.get(jid)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job {jid} not found. Job state is in-memory only and does not "
                       f"survive an API server restart — if the server restarted since "
                       f"this pipeline ran, re-run it and try downloading again.",
            )
        if job["status"] != "done" or "export" not in job:
            raise HTTPException(
                status_code=409, detail=f"Job {jid} is not complete (status: {job['status']})"
            )
        jobs.append(job)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    root_name = f"Anonymized_data_{timestamp}"
    zip_path  = JOBS_DIR / f"_export_{uuid.uuid4().hex}.zip"

    def _build_zip():
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for job in jobs:
                exp = job["export"]
                study_short = exp["anon_study_uid"][-16:]
                base = f"{root_name}/Patient_{exp['anon_patient_id']}/Study_{study_short}"

                dicom_src = Path(exp["dicom_dir"])
                if dicom_src.is_dir():
                    for f in sorted(dicom_src.iterdir()):
                        if f.is_file():
                            zf.write(f, f"{base}/dicom/{f.name}")

                for nifti_str in exp["nifti_files"]:
                    f = Path(nifti_str)
                    if f.exists():
                        zf.write(f, f"{base}/nifti/{f.name}")

                pdf_f = Path(exp["pdf_file"])
                if pdf_f.exists():
                    zf.write(pdf_f, f"{base}/pdf/{pdf_f.name}")

                fhir_f = Path(exp["fhir_file"])
                if fhir_f.exists():
                    zf.write(fhir_f, f"{base}/fhir/{fhir_f.name}")

    try:
        # zipfile is blocking — run it off the event loop so a large multi-
        # patient export doesn't stall other requests being served.
        await asyncio.to_thread(_build_zip)
    except HTTPException:
        raise
    except Exception as e:
        zip_path.unlink(missing_ok=True)
        # Never let this propagate as a raw unhandled exception. Starlette's
        # default handling of an uncaught exception can produce a response
        # that skips CORS header injection depending on middleware
        # ordering — the browser then reports a generic "Failed to fetch"
        # with the real error never reaching the frontend. Converting to a
        # clean HTTPException keeps this a normal, CORS-correct response.
        raise HTTPException(status_code=500, detail=f"Failed to build export zip: {e}")

    def _stream_and_cleanup():
        try:
            with open(zip_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
        finally:
            zip_path.unlink(missing_ok=True)

    return StreamingResponse(
        _stream_and_cleanup(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{root_name}.zip"'},
    )


@app.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """
    SSE endpoint — streams pipeline progress events in real time.

    Connect from the frontend with EventSource:

        const es = new EventSource(`/jobs/${jobId}/stream`);
        es.onmessage = (e) => {
            const event = JSON.parse(e.data);
            console.log(event.stage, event.message, event.progress);
            if (event.type === 'complete' || event.type === 'error') {
                es.close();
            }
        };

    Each event payload:
        { type, stage, message, progress }
    """
    job = _get_job_or_404(job_id)
    q   = job["event_queue"]

    async def event_generator():
        while True:
            # Poll the queue without blocking the event loop
            try:
                payload = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: q.get(timeout=30)
                )
            except Exception:
                # Timeout — send keepalive comment
                yield ": keepalive\n\n"
                continue

            if payload is None:
                # Sentinel — pipeline thread finished
                break

            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection":       "keep-alive",
        },
    )


@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """
    Get current job status and output download URLs once complete.

    Response:
        {
          job_id, status, created_at, stages, error,
          outputs: {
            nifti_url: "/jobs/{id}/download/nifti",
            fhir_url:  "/jobs/{id}/download/fhir",
            pii_audit_url: "/jobs/{id}/download/pii_audit",
            preservation_url: "/jobs/{id}/download/preservation"
          }
        }
    """
    job = _get_job_or_404(job_id)

    outputs = {}
    if job["status"] == "done":
        outputs = {
            "nifti_url":        f"/jobs/{job_id}/download/nifti",
            "fhir_url":         f"/jobs/{job_id}/download/fhir",
            "pii_audit_url":    f"/jobs/{job_id}/download/pii_audit",
            "preservation_url": f"/jobs/{job_id}/download/preservation",
        }

    return {
        "job_id":     job_id,
        "status":     job["status"],
        "created_at": job["created_at"],
        "error":      job["error"],
        "outputs":    outputs,
    }


@app.get("/jobs/{job_id}/download/{file_type}")
async def download_output(job_id: str, file_type: str):
    """
    Download an output file once the pipeline has completed.

    file_type: "nifti" | "fhir" | "pii_audit" | "preservation"

    The NIfTI file is returned as-is (.nii.gz) — first series only; use
    POST /export for the complete multi-series result.
    The FHIR bundle, PII audit, and preservation report are returned
    as-is (.json).
    """
    job = _get_job_or_404(job_id)

    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is not complete yet (status: {job['status']})",
        )

    if file_type not in ("nifti", "fhir", "pii_audit", "preservation"):
        raise HTTPException(
            status_code=400,
            detail="file_type must be 'nifti', 'fhir', 'pii_audit', or 'preservation'",
        )

    file_path = Path(job["outputs"].get(file_type, ""))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{file_type} output file not found")

    media_types = {
        "nifti":        "application/octet-stream",
        "fhir":         "application/fhir+json",
        "pii_audit":    "application/json",
        "preservation": "application/json",
    }

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type=media_types[file_type],
    )


@app.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    """
    Delete a job and all associated files from disk.
    """
    _get_job_or_404(job_id)

    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(str(job_dir))

    del _jobs[job_id]


@app.get("/health")
async def health():
    """Health check — returns 200 if server is running and models are loaded."""
    return {
        "status": "ok",
        "models_loaded": hasattr(app.state, "clf") and app.state.clf is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # disable reload — models are loaded at startup
    )