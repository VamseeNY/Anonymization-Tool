"""
api.py
------
FastAPI backend for the medical imaging anonymization pipeline.

Endpoints
---------
POST /jobs
    Upload DICOM files + PDF, start pipeline job.
    Returns { job_id: str }

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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
JOBS_DIR      = BASE_DIR / "jobs"          # working directory per job
ALLOWED_ORIGINS = ["*"]                    # restrict in production

JOBS_DIR.mkdir(exist_ok=True)

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
)


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
        from dicom_anonymizer import anonymize_study
        anon_dicom_dir = output_dir / "dicom" / f"patient_{study.patient_id}"
        anon_result = anonymize_study(
            study,
            output_dir=str(anon_dicom_dir),
            clf=clf,
            ocr_reader=ocr_reader,
        )
        anon_patient_id = anon_result["anon_patient_id"]
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
        if not nifti_results:
            raise RuntimeError("NIfTI conversion produced no output files.")
        nifti_path = Path(nifti_results[0]["nifti_path"])
        push("progress", "nifti",
             f"NIfTI written: {nifti_path.name}", 65)

        # ── Stage 4 — Extract and anonymize PDF ──
        push("progress", "pdf", "Extracting and anonymizing PDF...", 70)
        from pdf_extractor import extract_pdf
        bare_id = anon_patient_id.removeprefix("ANON-")
        pdf_extract = extract_pdf(str(pdf_path), clf, anon_patient_id=bare_id)
        push("progress", "pdf", "PDF anonymized", 80)

        # ── Stage 5 — Build FHIR bundle ──
        push("progress", "fhir", "Building FHIR R4 bundle...", 85)
        from fhir_mapper import map_to_fhir
        fhir_dir = output_dir / "fhir"
        map_to_fhir(study, anon_result, pdf_extract, nifti_results, str(fhir_dir))
        fhir_path = fhir_dir / f"patient_{anon_patient_id}_bundle.json"
        push("progress", "fhir", "FHIR R4 bundle written", 95)

        # ── Done ──
        job["status"]  = "done"
        job["outputs"] = {
            "nifti": str(nifti_path),
            "fhir":  str(fhir_path),
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
async def create_job(
    dicom_files: list[UploadFile] = File(..., description="All .dcm files for one patient"),
    pdf_file: UploadFile = File(..., description="Radiology or lab report PDF"),
):
    """
    Upload DICOM files and a PDF report, start the anonymization pipeline.

    Returns a job_id which can be used to stream progress and download outputs.

    Example (curl):
        curl -X POST http://localhost:8000/jobs \\
          $(for f in *.dcm; do echo "-F dicom_files=@$f"; done) \\
          -F pdf_file=@report.pdf
    """
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
            fhir_url:  "/jobs/{id}/download/fhir"
          }
        }
    """
    job = _get_job_or_404(job_id)

    outputs = {}
    if job["status"] == "done":
        outputs = {
            "nifti_url": f"/jobs/{job_id}/download/nifti",
            "fhir_url":  f"/jobs/{job_id}/download/fhir",
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

    file_type: "nifti" | "fhir"

    The NIfTI file is returned as-is (.nii.gz).
    The FHIR bundle is returned as-is (.json).
    """
    job = _get_job_or_404(job_id)

    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is not complete yet (status: {job['status']})",
        )

    if file_type not in ("nifti", "fhir"):
        raise HTTPException(status_code=400, detail="file_type must be 'nifti' or 'fhir'")

    file_path = Path(job["outputs"].get(file_type, ""))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{file_type} output file not found")

    media_types = {
        "nifti": "application/octet-stream",
        "fhir":  "application/fhir+json",
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