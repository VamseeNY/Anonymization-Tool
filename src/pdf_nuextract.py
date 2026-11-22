"""
pdf_nuextract.py
-----------------
NuExtract3-based structured extraction — the "cold path" used only when
a PDF report doesn't match any known layout in pdf_template_store.py.

Calls NuMind's official quantized GGUF release (numind/NuExtract3-GGUF).
Two backends are supported, selected via NUEXTRACT_BACKEND:

  "llama_cpp_python" (default) — in-process, via the llama-cpp-python
      pip package. No separate binary to locate/bundle/manage as a
      subprocess — simplest to package, and doesn't lock in a hardware
      backend before you've packaged the app or know the deployment
      hardware. Defaults to CPU-only (n_gpu_layers=0), which runs
      anywhere; raise NUEXTRACT_GPU_LAYERS once the target hardware is
      known and a GPU-enabled wheel is installed.

  "llama_server" — a lazily-launched llama-server (llama.cpp) subprocess,
      called over an OpenAI-compatible HTTP API. More moving parts to
      package (a separate binary, port/health-check management), but
      isolates a crash from the main process and lets you swap hardware
      backends (CPU/CUDA/Vulkan) by swapping the binary, without
      touching the Python bundle at all.

Either way, loaded LAZILY — only the first time the cold path is
actually needed; most documents in steady-state operation hit the
cached-template fast path in pdf_extractor.py and never touch this
module at all. Once loaded, stays resident for the life of the process.

Why llama-cpp-python's chat template needs manual rendering
---------------------------------------------------------------
NuExtract3's structured extraction relies on its Jinja chat template
receiving THREE extra variables beyond the standard `messages`:
`template` (our JSON schema), `instructions`, and `enable_thinking`.
llama-cpp-python's high-level create_chat_completion() does auto-detect
a GGUF's embedded tokenizer.chat_template for basic messages, but
reliably forwarding arbitrary EXTRA template kwargs through to the Jinja
render has been an open gap in that library (see
github.com/abetlen/llama-cpp-python/issues/2023) that I can't verify is
resolved for this exact case without the real model + hardware to test
against. So instead, we extract the template string ourselves from
Llama.metadata["tokenizer.chat_template"] and render it directly with
jinja2 — full control, and it's exactly what apply_chat_template()
would do internally regardless of which library calls it. This is the
one piece of this module most worth validating once you have the real
GGUF loaded on real hardware — see the module-level note in
_render_chat_prompt().

Design notes (same reasoning as the earlier in-process/HTTP versions)
------------------------------------------------------------------------
- We always send our OWN fixed schema (CANONICAL_TEMPLATE below), never
  let the model invent field names.
- Every field we need to LOCATE on the page is requested as
  "verbatim-string" specifically — a reformatted value can't be found
  verbatim in the source text to recover its position.
- Text input, not image — see pdf_extractor.py's module docstring.
- Because NuExtract3 doesn't return bounding boxes, every non-null
  verbatim-string result is checked against the source text before
  being trusted (hallucination guard), then located via a word-list
  search — never assumed correct just because the model said so.

Configuration (environment variables)
----------------------------------------
NUEXTRACT_BACKEND
    "llama_cpp_python" (default) or "llama_server".
NUEXTRACT_GGUF_PATH
    Path to a local .gguf file. If unset, auto-downloads from HF Hub
    (NUEXTRACT_HF_REPO) — fine for development; for an offline bundled
    deployment, pre-download the GGUF and set this so no internet
    access is needed at runtime.
NUEXTRACT_HF_REPO
    Used only when NUEXTRACT_GGUF_PATH is unset. Defaults to
    "numind/NuExtract3-GGUF" with NUEXTRACT_GGUF_FILENAME picking the
    quant (defaults to "*Q4_K_M.gguf").
NUEXTRACT_GPU_LAYERS
    llama_cpp_python backend only. Number of layers to offload to GPU;
    0 (default) = CPU-only, always works regardless of hardware. -1 =
    offload everything if a GPU-enabled wheel is installed.
NUEXTRACT_CTX_SIZE
    Context window. Defaults to 8192.
NUEXTRACT_LLAMA_SERVER_PATH / NUEXTRACT_STARTUP_TIMEOUT
    llama_server backend only — see that backend's class docstring.
"""

import json
import os
import re
import threading
from pathlib import Path

# huggingface_hub's optional accelerated Rust downloader (hf_transfer) is
# a common cause of a download that silently hangs at 0% forever — no
# error, no timeout — on networks with corporate proxies, VPNs, or SSL
# inspection, where the accelerated path stalls instead of falling back.
# Disabled by default; if hf_transfer isn't installed this has no effect
# either way. Only takes effect if set before huggingface_hub is
# imported, so this must stay above the NUEXTRACT_GGUF_PATH auto-download
# path below. Override with HF_HUB_ENABLE_HF_TRANSFER=1 if your network
# is fine with it and you want the speed.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


NUEXTRACT_BACKEND    = os.environ.get("NUEXTRACT_BACKEND", "llama_cpp_python")
NUEXTRACT_MODEL_ID    = "numind/NuExtract3-GGUF"
NUEXTRACT_GGUF_PATH   = os.environ.get("NUEXTRACT_GGUF_PATH")
NUEXTRACT_HF_REPO     = os.environ.get("NUEXTRACT_HF_REPO", "numind/NuExtract3-GGUF")
NUEXTRACT_GGUF_FILENAME = os.environ.get("NUEXTRACT_GGUF_FILENAME", "*Q4_K_M.gguf")
NUEXTRACT_GPU_LAYERS  = int(os.environ.get("NUEXTRACT_GPU_LAYERS", "0"))
NUEXTRACT_CTX_SIZE    = int(os.environ.get("NUEXTRACT_CTX_SIZE", "8192"))

# Fields we need to physically locate on the page (for redaction) must be
# verbatim-string. "sex" is a closed set so an enum is fine — we don't
# need to locate it (age/sex are retained, never redacted).
CANONICAL_TEMPLATE = {
    "patient_name":   "verbatim-string",
    "age":            "verbatim-string",
    "sex":            ["Male", "Female", "Other"],
    "reg_no":         "verbatim-string",
    "date_of_birth":  "verbatim-string",
    "referred_by":    "verbatim-string",
    "ref_center":     "verbatim-string",
    "date_of_report": "verbatim-string",
    "study_title":    "verbatim-string",
    "technique":      "verbatim-string",
    "findings":       ["verbatim-string"],
    "impression":     "verbatim-string",
}

EXTRACTION_INSTRUCTIONS = (
    "Extract only patient-identifying header fields and the report's "
    "clinical sections, exactly as written in the document. Do not "
    "paraphrase, summarize, or reformat any value."
)

# Fields that need a page location recovered (everything except sex,
# which is an enum we never redact/locate).
_VERBATIM_SCALAR_FIELDS = [
    "patient_name", "age", "reg_no", "date_of_birth", "referred_by",
    "ref_center", "date_of_report", "study_title", "technique", "impression",
]

# Identity fields specifically — the subset that gets redacted/wiped.
# (age is verbatim-located like the others, but retained, not redacted —
# same policy as everywhere else in this pipeline. date_of_birth IS
# redacted — more identifying than age alone, matches DICOM
# PatientBirthDate.)
IDENTITY_FIELDS = ["patient_name", "reg_no", "date_of_birth", "referred_by", "ref_center", "date_of_report"]


def _parse_model_json(text: str) -> dict:
    """Model output should be raw JSON, but strip markdown fences defensively."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.IGNORECASE).strip()
        return json.loads(cleaned)


# ─────────────────────────────────────────────────────────────────────────────
#  Backend: llama_cpp_python (default) — in-process, no subprocess to manage
# ─────────────────────────────────────────────────────────────────────────────

class _LlamaCppPythonBackend:
    """
    Lazily-loaded, process-lifetime-resident llama-cpp-python model.
    Thread-safe against two pipeline jobs hitting the cold path at once.

    Defaults to CPU-only (n_gpu_layers=0) — always runs, regardless of
    whether the deployment machine has a GPU. Raise NUEXTRACT_GPU_LAYERS
    once you know the target hardware and have a GPU-enabled wheel
    installed (llama-cpp-python's GPU support is a build-time choice —
    see the module docstring).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._llm = None
        self._chat_template: str | None = None

    def is_loaded(self) -> bool:
        return self._llm is not None

    def get(self):
        with self._lock:
            if self._llm is None:
                self._load()
            return self._llm, self._chat_template

    def _load(self):
        from llama_cpp import Llama

        print(f"[INFO] Loading NuExtract3 (llama-cpp-python, "
              f"n_gpu_layers={NUEXTRACT_GPU_LAYERS}, first use — this can take a while)...")
        if not NUEXTRACT_GGUF_PATH:
            print("[INFO] No NUEXTRACT_GGUF_PATH set — auto-downloading from Hugging Face. "
                  "If the progress bar doesn't move for more than ~30s, your network is "
                  "likely blocking/stalling the download (proxy, VPN, or SSL inspection are "
                  "common causes) — Ctrl+C, download the file manually, and set "
                  "NUEXTRACT_GGUF_PATH to it instead of waiting.")

        if NUEXTRACT_GGUF_PATH:
            self._llm = Llama(
                model_path=NUEXTRACT_GGUF_PATH,
                n_ctx=NUEXTRACT_CTX_SIZE,
                n_gpu_layers=NUEXTRACT_GPU_LAYERS,
                verbose=False,
            )
        else:
            self._llm = Llama.from_pretrained(
                repo_id=NUEXTRACT_HF_REPO,
                filename=NUEXTRACT_GGUF_FILENAME,
                n_ctx=NUEXTRACT_CTX_SIZE,
                n_gpu_layers=NUEXTRACT_GPU_LAYERS,
                verbose=False,
            )

        self._chat_template = self._llm.metadata.get("tokenizer.chat_template")
        if not self._chat_template:
            raise RuntimeError(
                "Loaded GGUF has no embedded chat template "
                "(tokenizer.chat_template metadata key missing) — cannot render "
                "NuExtract3's structured-extraction prompt. Make sure you're using "
                "NuMind's official numind/NuExtract3-GGUF release, which embeds it."
            )
        print("[INFO] NuExtract3 loaded.")

    def stop(self):
        self._llm = None
        self._chat_template = None


_llama_cpp_python_backend = _LlamaCppPythonBackend()


def _build_jinja_env():
    """
    A jinja2 Environment that understands {% generation %}...
    {% endgeneration %} — a HuggingFace-specific Jinja extension used in
    many modern chat templates (including NuExtract3's) to mark the
    assistant-response span for training-time token masking. Plain
    jinja2.Environment() doesn't know this tag and raises
    TemplateSyntaxError ("Encountered unknown tag 'generation'") on any
    template that uses it. We don't need the masking behavior — we only
    want the rendered text — so this extension treats it as a
    transparent pass-through: render the block's contents normally, no
    other effect.
    """
    import jinja2
    import jinja2.ext
    import jinja2.nodes

    class GenerationTagExtension(jinja2.ext.Extension):
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            call = self.call_method("_noop", [])
            return jinja2.nodes.CallBlock(call, [], [], body).set_lineno(lineno)

        def _noop(self, caller):
            return caller()

    return jinja2.Environment(
        extensions=[GenerationTagExtension], trim_blocks=True, lstrip_blocks=True
    )


def _render_chat_prompt(template_str: str, raw_text: str) -> str:
    """
    Render NuExtract3's chat template ourselves, rather than depending on
    llama-cpp-python's built-in chat-templating — see the module
    docstring for why. This mirrors exactly what HF's apply_chat_template
    (used by the transformers/vLLM path) does internally: render the
    model's own Jinja template with `messages` plus the extra variables
    NuExtract3's template specifically expects.

    Uses a jinja2 Environment extended with the {% generation %} tag
    (see _build_jinja_env) since NuExtract3's actual template — like many
    current HF chat templates — uses it, and it isn't standard Jinja2.

    NOTE — the variable names (template/instructions/enable_thinking)
    match NuMind's documented vLLM usage exactly, so they should be
    correct, but this whole function is the piece most worth checking
    first if extraction quality looks wrong: run with verbose=True on
    the Llama object and compare the rendered prompt here against
    NuMind's own examples.
    """
    env = _build_jinja_env()

    try:
        jinja_template = env.from_string(template_str)
        return jinja_template.render(
            messages=[{"role": "user", "content": raw_text}],
            template=json.dumps(CANONICAL_TEMPLATE),
            instructions=EXTRACTION_INSTRUCTIONS,
            enable_thinking=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        # Template rendering is the one part of this module that can't be
        # verified without the real GGUF — surface as much as possible to
        # make a template-compatibility issue diagnosable from the log
        # alone, rather than a bare traceback.
        preview = template_str[:800] + ("..." if len(template_str) > 800 else "")
        raise RuntimeError(
            f"Failed to render NuExtract3's chat template ({type(e).__name__}: {e}). "
            f"This usually means the template uses a Jinja feature this renderer "
            f"doesn't yet handle (the {{% generation %}} tag is already supported — "
            f"if this is a *different* unknown tag, it needs the same treatment in "
            f"_build_jinja_env()). Template preview:\n{preview}"
        ) from e


def _run_llama_cpp_python(raw_text: str) -> dict:
    llm, chat_template = _llama_cpp_python_backend.get()
    prompt = _render_chat_prompt(chat_template, raw_text)

    output = llm(
        prompt,
        max_tokens=2048,
        temperature=0.2,
        # Defensive stop sequences for common ChatML-style end-of-turn
        # markers (NuExtract3 is Qwen-family) — llama.cpp already stops
        # at the model's own EOS token by default, this is a backstop
        # in case the template's turn-end marker differs from it.
        stop=["<|im_end|>", "<|endoftext|>"],
        echo=False,
    )
    text = output["choices"][0]["text"].strip()
    return _parse_model_json(text)


# ─────────────────────────────────────────────────────────────────────────────
#  Backend: llama_server — lazily-launched subprocess, OpenAI-compatible API
# ─────────────────────────────────────────────────────────────────────────────
#
# Alternative to llama_cpp_python: more moving parts to package (a
# separate binary + port/health-check management) but isolates a crash
# from the main process and lets you swap hardware backends by swapping
# the binary. Kept available (NUEXTRACT_BACKEND=llama_server) in case
# it's useful again once packaging is settled and target hardware known.

import atexit as _atexit
import socket as _socket
import subprocess as _subprocess
import sys as _sys
import time as _time
import urllib.error as _urllib_error
import urllib.request as _urllib_request

LLAMA_SERVER_PATH  = os.environ.get("NUEXTRACT_LLAMA_SERVER_PATH")
_STARTUP_TIMEOUT_S = float(os.environ.get("NUEXTRACT_STARTUP_TIMEOUT", "120"))


def _default_llama_server_path() -> str:
    """
    Locate a bundled llama-server binary next to this module, or next to
    the frozen executable when packaged (PyInstaller and similar set
    sys.frozen and point sys.executable at the bundled exe / expose
    sys._MEIPASS as the extraction root). Falls back to "llama-server"
    on PATH if no bundled copy is found.
    """
    exe_name = "llama-server.exe" if os.name == "nt" else "llama-server"

    candidates = []
    if getattr(_sys, "frozen", False):
        candidates.append(Path(_sys.executable).parent / exe_name)
        meipass = getattr(_sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / exe_name)
    candidates.append(Path(__file__).resolve().parent / "bin" / exe_name)
    candidates.append(Path(__file__).resolve().parent / exe_name)

    for c in candidates:
        if c.exists():
            return str(c)

    return exe_name  # fall back to PATH lookup


def _free_local_port() -> int:
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _LlamaServerBackend:
    """Lazily-started, process-lifetime-resident llama-server subprocess."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: _subprocess.Popen | None = None
        self._port: int | None = None
        self._client = None

    def is_loaded(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def client(self):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            return self._client

    def _start(self):
        from openai import OpenAI

        server_path = LLAMA_SERVER_PATH or _default_llama_server_path()
        self._port = _free_local_port()

        cmd = [
            server_path,
            "--port", str(self._port),
            "--host", "127.0.0.1",
            "--ctx-size", str(NUEXTRACT_CTX_SIZE),
            "--no-webui",
        ]
        cmd += ["-m", NUEXTRACT_GGUF_PATH] if NUEXTRACT_GGUF_PATH else \
               ["-hf", f"{NUEXTRACT_HF_REPO}:{NUEXTRACT_GGUF_FILENAME.strip(chr(42))}"]

        print(f"[INFO] Starting NuExtract3 llama-server on port {self._port} "
              f"(first use — model load/download can take a while)...")

        try:
            self._proc = _subprocess.Popen(
                cmd, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT, text=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Could not launch llama-server at '{server_path}'. For a bundled "
                f"deployment, either place the llama-server binary next to this "
                f"module (in a 'bin/' subfolder) or set NUEXTRACT_LLAMA_SERVER_PATH "
                f"to its full path. Get it from "
                f"https://github.com/ggml-org/llama.cpp/releases"
            ) from e

        _atexit.register(self.stop)
        self._wait_until_ready()

        self._client = OpenAI(api_key="not-needed", base_url=f"http://127.0.0.1:{self._port}/v1")
        print("[INFO] NuExtract3 server ready.")

    def _wait_until_ready(self):
        deadline = _time.time() + _STARTUP_TIMEOUT_S
        health_url = f"http://127.0.0.1:{self._port}/health"

        while _time.time() < deadline:
            if self._proc.poll() is not None:
                out = self._proc.stdout.read() if self._proc.stdout else ""
                raise RuntimeError(
                    f"llama-server exited early (code {self._proc.returncode}) "
                    f"while loading NuExtract3:\n{out[-4000:]}"
                )
            try:
                with _urllib_request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (_urllib_error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            _time.sleep(1.0)

        self.stop()
        raise TimeoutError(
            f"llama-server did not become ready within {_STARTUP_TIMEOUT_S}s "
            f"(set NUEXTRACT_STARTUP_TIMEOUT higher if the model is still "
            f"downloading on first run)"
        )

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except _subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._client = None


_llama_server_backend = _LlamaServerBackend()


def _run_llama_server(raw_text: str) -> dict:
    client = _llama_server_backend.client()

    response = client.chat.completions.create(
        model=NUEXTRACT_MODEL_ID,
        temperature=0.2,
        messages=[{"role": "user", "content": [{"type": "text", "text": raw_text}]}],
        extra_body={
            "chat_template_kwargs": {
                "template": json.dumps(CANONICAL_TEMPLATE),
                "instructions": EXTRACTION_INSTRUCTIONS,
                "enable_thinking": False,
            }
        },
    )
    content = response.choices[0].message.content
    return _parse_model_json(content)


# ─────────────────────────────────────────────────────────────────────────────
#  Public dispatch — same interface regardless of backend
# ─────────────────────────────────────────────────────────────────────────────

def is_loaded() -> bool:
    if NUEXTRACT_BACKEND == "llama_server":
        return _llama_server_backend.is_loaded()
    return _llama_cpp_python_backend.is_loaded()


def load_nuextract_model():
    """Ensure the configured backend is loaded/running (starting it on
    first call — only reached from the cold path)."""
    if NUEXTRACT_BACKEND == "llama_server":
        return _llama_server_backend.client()
    return _llama_cpp_python_backend.get()


def run_nuextract_on_text(raw_text: str) -> dict:
    """
    One NuExtract3 inference call against extracted PDF text, using our
    fixed CANONICAL_TEMPLATE. Non-reasoning mode (fast, deterministic) —
    reasoning mode is available in NuExtract3 for genuinely hard
    documents but isn't the default here; see NuMind's own guidance to
    start with non-reasoning for production workloads.

    Returns the parsed JSON dict as returned by the model — NOT yet
    verified against the source text. Call verify_and_locate() next.
    """
    if NUEXTRACT_BACKEND == "llama_server":
        return _run_llama_server(raw_text)
    return _run_llama_cpp_python(raw_text)


# ─────────────────────────────────────────────────────────────────────────────
#  Hallucination guard + bounding-box recovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_words_for_text(words: list[dict], target: str) -> list[dict]:
    """
    Find the contiguous run of words (in reading order: top-to-bottom,
    left-to-right) whose joined text contains `target`
    (case-insensitive, whitespace-normalized). Returns the matching word
    dicts in order, or [] if not found anywhere.
    """
    if not target or not target.strip():
        return []
    target_norm = " ".join(target.split()).lower()
    if not target_norm:
        return []

    ordered = sorted(words, key=lambda w: (round(w["top"] / 3), w["x0"]))
    texts = [w["text"] for w in ordered]
    joined = " ".join(texts).lower()

    idx = joined.find(target_norm)
    if idx == -1:
        return []

    cursor = 0
    start_i, end_i = None, None
    for i, t in enumerate(texts):
        w_start = cursor
        w_end = cursor + len(t)
        if start_i is None and w_end > idx:
            start_i = i
        if w_start < idx + len(target_norm):
            end_i = i
        cursor = w_end + 1

    if start_i is None or end_i is None:
        return []
    return ordered[start_i:end_i + 1]


def verify_and_locate(nuextract_output: dict, words: list[dict], raw_text: str) -> dict:
    """
    For every scalar verbatim-string field NuExtract3 returned, confirm
    it's a literal substring of raw_text (the hallucination guard — this
    is what makes "verbatim" trustworthy rather than assumed), and if it
    verifies, locate the word-level bounding box(es) that contain it.

    Returns
    -------
    dict[field_name] -> {"value": str | None, "verified": bool, "words": list[dict]}
    Plus "findings" -> list of that same shape (findings is an array),
    and "sex" -> {"value": ..., "verified": True, "words": []} (enum,
    never located — age/sex are retained, never redacted).
    """
    result: dict = {}

    for field_name in _VERBATIM_SCALAR_FIELDS:
        value = nuextract_output.get(field_name)
        if not value or not isinstance(value, str) or not value.strip():
            result[field_name] = {"value": None, "verified": False, "words": []}
            continue
        value = value.strip()
        in_text = value.lower() in raw_text.lower()
        matched_words = _find_words_for_text(words, value) if in_text else []
        result[field_name] = {
            "value": value,
            "verified": in_text and bool(matched_words),
            "words": matched_words,
        }

    findings_out = []
    for f in (nuextract_output.get("findings") or []):
        if not f or not isinstance(f, str) or not f.strip():
            continue
        f = f.strip()
        in_text = f.lower() in raw_text.lower()
        matched_words = _find_words_for_text(words, f) if in_text else []
        findings_out.append({
            "value": f,
            "verified": in_text and bool(matched_words),
            "words": matched_words,
        })
    result["findings"] = findings_out

    sex_value = nuextract_output.get("sex")
    result["sex"] = {"value": sex_value, "verified": bool(sex_value), "words": []}

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Template synthesis from a verified extraction
# ─────────────────────────────────────────────────────────────────────────────

_HONORIFICS = re.compile(r"^(mr|mrs|ms|miss|master|baby|dr|smt|shri|sri)\.?$", re.IGNORECASE)


def _guess_label_text(
    value_words: list[dict],
    all_words: list[dict],
    max_gap: float = 80.0,
    other_value_words: list[dict] | None = None,
) -> str:
    """
    Best-effort guess at the label immediately preceding a value on the
    page (e.g. the "NAME:" before "John Doe") — used only to populate a
    template's stored label_text for future Tier-2 re-verification. If no
    plausible label word is found, falls back to an empty string (Tier-2
    verification will then always fail for this field, which correctly
    means the template can never be reused via the fast path for it —
    safe default, just less efficient).
    """
    if not value_words:
        return ""
    first = value_words[0]
    same_line = [
        w for w in all_words
        if abs(w["top"] - first["top"]) <= 3 and w["x1"] <= first["x0"] and (first["x0"] - w["x1"]) <= max_gap
    ]
    if not same_line:
        return ""
    same_line.sort(key=lambda w: w["x0"])
    # Words of ANOTHER field's value are never part of this field's
    # label — e.g. on "Ref. DR.X  <ref center>" the word before the
    # ref-center value is the referring doctor's name. Keep only the
    # words after the last such value word.
    taken = {id(w) for w in (other_value_words or [])}
    for i in range(len(same_line) - 1, -1, -1):
        if id(same_line[i]) in taken:
            same_line = same_line[i + 1:]
            break
    label = " ".join(w["text"] for w in same_line[-2:])  # last word or two before the value
    # A label is fixed layout text — never digits (dates, IDs), and never
    # a title like "Mr." that varies from patient to patient.
    if re.search(r"\d", label) or _HONORIFICS.match(label):
        return ""
    return label


def _guess_section_label(
    value_words: list[dict],
    all_words: list[dict],
    max_gap_lines: float = 4.0,
) -> tuple[str, list[dict]]:
    """
    Best-effort guess at the section-heading LINE immediately above a
    body field's (technique/findings/impression) located value on the
    page — e.g. "Protocol:" above a technique paragraph, "Observations:"
    above findings. Used to populate a template's body_section_keywords
    with THIS document's actual section wording.

    Unlike _guess_label_text() (header fields, where the label sits on
    the SAME line as the value — "NAME: John Doe"), a body section's
    heading sits on its OWN line above the value's first line, so this
    groups words into lines and walks upward from there instead of
    scanning leftward on one line.

    The search distance is max_gap_lines times the value's text height
    rather than a fixed number of points, so it scales with font size
    (a fixed 30pt missed an "OBSERVATIONS" heading 31pt above its text).

    Returns (heading text, heading line words), or ("", []) if no
    plausible heading line is found — callers should fall back to the
    generic field name as the keyword in that case (safe, just less
    precise — the previous, always-hardcoded behavior).
    """
    if not value_words:
        return "", []
    from pdf_extractor import _group_words_into_lines

    lines = _group_words_into_lines(all_words)
    first_value_top = min(w["top"] for w in value_words)
    heights = sorted(w["bottom"] - w["top"] for w in value_words)
    max_vertical_gap = max_gap_lines * heights[len(heights) // 2]

    # Lines strictly above the value, nearest first.
    above = [ln for ln in lines if ln and max(w["top"] for w in ln) < first_value_top]
    above.sort(key=lambda ln: -max(w["top"] for w in ln))

    for ln in above:
        gap = first_value_top - max(w["top"] for w in ln)
        if gap > max_vertical_gap:
            break  # lines only get farther away from here — stop looking
        text = " ".join(w["text"] for w in ln).strip()
        # A heading is short (a label, not a sentence) — skip past any
        # wrapped body text from a prior section that happens to fall
        # within the vertical tolerance.
        if 0 < len(text.split()) <= 4 and len(text) <= 40 and _looks_like_heading(text):
            return text.rstrip(":").strip(), ln
    return "", []


def _looks_like_heading(text: str) -> bool:
    """
    Tell a section heading ("PROTOCOL:", "Observations", "IMPRESSION")
    apart from a short line of body text ("acute infarcts.") or a bullet
    item ("• DWI") — only a heading may become part of a template.
    """
    if not text[0].isalpha():          # bullet glyph or list number
        return False
    if text.endswith("."):             # end of a sentence
        return False
    if re.search(r"\d", text):          # dates, IDs, sequence names
        return False
    if text.endswith(":") or text.isupper():
        return True
    return all(w[0].isupper() for w in text.split() if w[0].isalpha())


def build_template_from_extraction(
    verify_result: dict,
    all_words: list[dict],
    page_w: float,
    page_h: float,
):
    """
    Build a new PdfTemplate from a verified NuExtract3 extraction —
    only VERIFIED fields become part of the template (an unverified
    field is simply absent, so future documents will correctly fall
    through to the cold path again for that specific field rather than
    trusting an unlocated position).
    """
    from pdf_template_store import HeaderField, PdfTemplate, SectionHeading, new_template_id
    from datetime import datetime, timezone

    all_value_words = [
        w for f in IDENTITY_FIELDS
        if verify_result.get(f, {}).get("verified")
        for w in verify_result[f]["words"]
    ]

    header_fields = {}
    for field_name in IDENTITY_FIELDS:
        info = verify_result.get(field_name, {})
        if not info.get("verified") or not info.get("words"):
            continue
        ws = info["words"]
        x0 = min(w["x0"] for w in ws)
        top = min(w["top"] for w in ws)
        x1 = max(w["x1"] for w in ws)
        bottom = max(w["bottom"] for w in ws)
        # No fallback label: a made-up label like "PATIENT_NAME" is never
        # on the page, so it can't be verified — "" marks the field as
        # unlabeled instead (see positional_verify()).
        own = {id(w) for w in ws}
        label_text = _guess_label_text(
            ws, all_words, other_value_words=[w for w in all_value_words if id(w) not in own]
        )
        header_fields[field_name] = HeaderField(
            label_text=label_text,
            bbox_norm=[x0 / page_w, top / page_h, x1 / page_w, bottom / page_h],
        )

    # Derive each body section's keyword(s) from what THIS document
    # actually calls that section (e.g. "Protocol"/"Observations" rather
    # than "Technique"/"Findings") — a template built from a hardcoded
    # literal word only ever matches documents that happen to use that
    # exact word, which defeats the point of caching for any clinic
    # whose template uses a synonym. The generic field name is always
    # kept too, as a floor — never worse than the previous behavior.
    def _locate(info: dict) -> list[dict]:
        # Where a value sits, only to find the heading above it — not to
        # trust the value. A value that failed verification (e.g. a
        # bullet list the model joined into one comma-separated line)
        # can still be placed by its first few words.
        if info.get("verified") and info.get("words"):
            return info["words"]
        value = info.get("value") or ""
        head = " ".join(value.split()[:3])
        return _find_words_for_text(all_words, head) if head else []

    def _section_anchor_words(field_name: str) -> list[dict]:
        if field_name == "findings":
            located = [ws for ws in (_locate(f) for f in verify_result.get("findings", [])) if ws]
            if not located:
                return []
            # Topmost (first-appearing) finding — the one closest to the
            # section heading we're trying to locate.
            return min(located, key=lambda ws: min(w["top"] for w in ws))
        return _locate(verify_result.get(field_name, {}))

    body_section_keywords = {}
    section_headings = {}
    for field_name in ("technique", "findings", "impression"):
        anchor_words = _section_anchor_words(field_name)
        label, heading_words = _guess_section_label(anchor_words, all_words) if anchor_words else ("", [])
        patterns = [re.escape(label)] if label else []
        patterns.append(field_name)
        body_section_keywords[field_name] = patterns
        if label:
            section_headings[field_name] = SectionHeading(
                text=label,
                x0_norm=heading_words[0]["x0"] / page_w,
                top_norm=heading_words[0]["top"] / page_h,
            )

    return PdfTemplate(
        template_id=new_template_id(),
        status="provisional",
        created_at=datetime.now(timezone.utc).isoformat(),
        match_count=1,
        body_section_keywords=body_section_keywords,
        header_fields=header_fields,
        source="learned",
        page_size=[page_w, page_h],
        section_headings=section_headings,
    )