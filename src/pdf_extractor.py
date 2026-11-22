"""
pdf_extractor.py
----------------
Stages 1 and 2 of the PDF pipeline.

Stage 1 — Extract:
    Reads a radiology or lab report PDF and extracts:
    - Header fields (patient name, age, sex, reg no, date, referred by,
      ref center) using word bounding boxes for layout-robust column parsing
    - Body sections (study title, technique, findings, impression) using
      raw text line parsing

Stage 2 — Anonymize:
    Runs openai/privacy-filter on all extracted text fields to detect and
    replace PII spans with [ENTITY_TYPE] tokens. No PDF is saved — all
    processing is in memory.

Design decisions
----------------
- Header parsed via bounding boxes (x0 threshold) to handle two-column
  layouts robustly across different clinic formats.
- Body parsed via raw text lines with optional section anchors (Findings,
  Impression etc). Section keywords are treated as optional — if absent,
  all body text is stored as findings.
- Doubled characters (font encoding artifact common in Indian clinic PDFs)
  are detected and collapsed before PII detection.
- privacy-filter is reused from dicom_anonymizer.py — loaded once and
  passed in to avoid reloading the model.

Usage
-----
    from pdf_extractor import extract_pdf, PdfExtract
    from dicom_anonymizer import load_privacy_filter

    clf = load_privacy_filter()
    result = extract_pdf("input/report.pdf", clf)
    print(result)

Requires
--------
    pip install pdfplumber transformers>=5.6.0 torch
"""

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber


# ─────────────────────────────────────────────────────────────────────────────
#  Output data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PdfExtract:
    """
    All structured fields extracted from one radiology/lab report PDF.
    All string values have been anonymized — no raw PII is stored here.
    Fields are None if not found in the document.
    """
    # ── Header fields ──
    patient_name:   Optional[str] = None   # anonymized
    age:            Optional[str] = None   # e.g. "36 Yrs"
    sex:            Optional[str] = None   # e.g. "Male"
    reg_no:         Optional[str] = None   # reg/lab number — kept as identifier
    date_of_birth:  Optional[str] = None   # wiped — more identifying than age, matches DICOM PatientBirthDate
    date_of_report: Optional[str] = None   # e.g. "23-03-2026"
    referred_by:    Optional[str] = None   # anonymized
    ref_center:     Optional[str] = None   # e.g. "BANGALORE NEURO CENTER"

    # ── Body fields ──
    study_title:    Optional[str] = None   # e.g. "CT IMAGING OF BRAIN"
    technique:      Optional[str] = None   # technique line
    findings:       list[str] = field(default_factory=list)   # one entry per finding paragraph
    impression:     Optional[str] = None   # conclusion / impression text

    # ── Raw anonymized full text (for FHIR presentedForm) ──
    full_text_anon: Optional[str] = None

    # ── Original (pre-redaction) values, keyed by field name ──
    # Populated by extract_pdf() purely to power the "PII before vs after"
    # comparison view on the results page. Never written to disk, never
    # included in the FHIR bundle, never used by anything downstream of
    # the anonymization step.
    raw_values: dict = field(default_factory=dict, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def _fix_doubled_chars(text: str) -> str:
    """
    Collapse doubled-character encoding artifacts common in Indian clinic PDFs.

    Example: "AANNBBAARRAASSUU" -> "ANBARASU"
             "MMrr.." -> "Mr."
             "DDrr.." -> "Dr."

    Detection: if every character in a word is repeated exactly twice
    consecutively, collapse it. Applied word by word to avoid affecting
    normal text that happens to have repeated chars (e.g. "normal").
    """
    def fix_word(word: str) -> str:
        # Must be even length and every pair must be identical
        if len(word) < 2 or len(word) % 2 != 0:
            return word
        pairs = [word[i:i+2] for i in range(0, len(word), 2)]
        if all(p[0] == p[1] for p in pairs):
            return "".join(p[0] for p in pairs)
        return word

    return " ".join(fix_word(w) for w in text.split())


def _clean_line(line: str) -> str:
    """Strip whitespace and collapse internal multiple spaces."""
    return re.sub(r" {2,}", " ", line.strip())


# ─────────────────────────────────────────────────────────────────────────────
#  Header parsing via bounding boxes
# ─────────────────────────────────────────────────────────────────────────────

# x0 threshold separating left column from right column in the header box
_HEADER_RIGHT_COL_X0 = 300

# How far down the page the header box extends (top coordinate cutoff)
# Words below this are part of the body, not the header
_HEADER_BOTTOM_TOP = 160


def _parse_header_from_words(words: list[dict]) -> dict[str, str]:
    """
    Extract header fields from word bounding boxes.

    Groups words by line (similar top coordinate within ±3px), then splits
    each line into left and right columns by x0 threshold. Extracts
    label:value pairs from each column.

    Returns a dict with keys matching PdfExtract header field names.
    """
    # Filter to header zone only
    header_words = [w for w in words if w["top"] < _HEADER_BOTTOM_TOP]

    if not header_words:
        return {}

    # Group into lines by top coordinate (cluster within 3px)
    lines: dict[float, list[dict]] = {}
    for w in header_words:
        top = w["top"]
        matched = None
        for existing_top in lines:
            if abs(existing_top - top) <= 3:
                matched = existing_top
                break
        if matched is not None:
            lines[matched].append(w)
        else:
            lines[top] = [w]

    # For each line, split into left and right columns
    raw_fields: dict[str, str] = {}
    for top in sorted(lines.keys()):
        line_words = sorted(lines[top], key=lambda w: w["x0"])
        left_words  = [w for w in line_words if w["x0"] < _HEADER_RIGHT_COL_X0]
        right_words = [w for w in line_words if w["x0"] >= _HEADER_RIGHT_COL_X0]

        for col_words in [left_words, right_words]:
            if not col_words:
                continue
            col_text = " ".join(w["text"] for w in col_words)
            col_text = _fix_doubled_chars(col_text)
            # Split on first colon to get label and value
            if ":" in col_text:
                label, _, value = col_text.partition(":")
                raw_fields[label.strip().upper()] = value.strip()

    # Map raw labels to PdfExtract field names
    result: dict[str, str] = {}

    # Patient name
    for key in raw_fields:
        if "NAME" in key and "REF" not in key:
            result["patient_name"] = raw_fields[key]
            break

    # Age and Sex — often combined as "36 Yrs / Male"
    for key in raw_fields:
        if "AGE" in key or "SEX" in key:
            age_sex = raw_fields[key]
            parts = [p.strip() for p in age_sex.split("/")]
            if len(parts) >= 1:
                result["age"] = parts[0].strip()
            if len(parts) >= 2:
                result["sex"] = parts[1].strip()
            break

    # Reg / Lab No
    for key in raw_fields:
        if "REG" in key or "LAB" in key:
            result["reg_no"] = raw_fields[key]
            break

    # Date of birth — wiped like DICOM PatientBirthDate; more identifying
    # than age alone, so this is NOT in the retained-fields set.
    for key in raw_fields:
        if "DOB" in key or "D.O.B" in key or ("DATE" in key and "BIRTH" in key) or ("BIRTH" in key and "DATE" in key):
            result["date_of_birth"] = raw_fields[key]
            break

    # Date of report — extract date part only (before "at")
    for key in raw_fields:
        if "DATE" in key and "REPORT" in key:
            date_val = raw_fields[key]
            # Take everything before " at " if present
            result["date_of_report"] = date_val.split(" at ")[0].strip()
            break

    # Referred by
    for key in raw_fields:
        if "REFER" in key:
            result["referred_by"] = raw_fields[key]
            break

    # Ref center
    for key in raw_fields:
        if "REF" in key and "CENTER" in key:
            result["ref_center"] = raw_fields[key]
            break

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Body parsing via raw text lines
# ─────────────────────────────────────────────────────────────────────────────

# Section keyword patterns — case insensitive, optional colon
_SECTION_PATTERNS = {
    "technique":   re.compile(r"^technique\s*:", re.IGNORECASE),
    "findings":    re.compile(r"^findings\s*:?", re.IGNORECASE),
    "impression":  re.compile(r"^impression\s*:?", re.IGNORECASE),
    "conclusion":  re.compile(r"^conclusion\s*:?", re.IGNORECASE),
    "opinion":     re.compile(r"^opinion\s*:?", re.IGNORECASE),
    "results":     re.compile(r"^results\s*:?", re.IGNORECASE),
    "end":         re.compile(r"^\*{2,}\s*end of report", re.IGNORECASE),
    # Not a field we capture — matching it here just stops it (and
    # anything after it) from being silently appended onto whatever
    # section was previously open (typically impression).
    "advice":      re.compile(r"^advice\s*:?", re.IGNORECASE),
}

# Patterns to detect lines that are footers/artifacts to skip
_SKIP_PATTERNS = [
    re.compile(r"^page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"^consultant\s+radiologist", re.IGNORECASE),
    re.compile(r"^\*{2,}"),
]

# Pattern for study title detection — all caps, no colon, short line
_STUDY_TITLE_PATTERN = re.compile(r"^[A-Z][A-Z\s/]{5,}$")


def _is_skip_line(line: str) -> bool:
    return any(p.match(line) for p in _SKIP_PATTERNS)


def _is_radiologist_line(line: str) -> bool:
    """Detect doubled-character radiologist name lines at bottom of report."""
    # If more than 40% of consecutive char pairs are identical, it's a doubled name
    words = line.split()
    if not words:
        return False
    doubled = sum(
        1 for w in words
        if len(w) >= 4 and len(w) % 2 == 0
        and all(w[i] == w[i+1] for i in range(0, len(w)-1, 2))
    )
    return doubled >= 2


def _join_wrapped_lines(lines: list[str]) -> list[str]:
    """
    Join lines that are continuations of the previous line (word-wrapped).
    A line is a continuation if it does not start with a capital letter
    after a sentence-ending punctuation on the previous line, or if
    the previous line doesn't end with a sentence terminator.
    """
    if not lines:
        return lines

    joined = []
    buffer = lines[0]

    for line in lines[1:]:
        # If previous buffer ends without sentence terminator, join
        if buffer and not buffer[-1] in ".!?":
            buffer = buffer + " " + line
        else:
            joined.append(buffer)
            buffer = line

    if buffer:
        joined.append(buffer)

    return joined

def _split_into_sentences(text: str) -> list[str]:
    """
    Split a paragraph into individual sentences.
    Splits on '. ' followed by a capital letter, or end of string.
    Filters out empty strings and very short fragments.
    """
    # Split on period+space followed by capital letter
    raw = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
    # Clean and filter
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 5]


def _parse_body_from_raw_text(
    raw_lines: list[str],
    header_end_line: int = 4,
    extra_section_patterns: Optional[dict[str, re.Pattern]] = None,
) -> dict:
    """
    Parse the body of the report from raw text lines.

    Parameters
    ----------
    raw_lines : list[str]
        All lines from raw text extraction, 0-indexed.
    header_end_line : int
        Line index after which body content starts (skip header lines).
    extra_section_patterns : dict[str, re.Pattern], optional
        Additional field_name -> pattern entries to recognize as section
        starts, layered on top of the hardcoded _SECTION_PATTERNS. Used
        by a matched template (see _resolve_structure()) to recognize
        THIS document's actual section wording (e.g. "Protocol"/
        "Observations") — without this, a document whose layout matches
        a cached template but uses section words _SECTION_PATTERNS
        doesn't know would silently come back with technique/findings
        missing, same as before a template ever matched.

    Returns dict with keys: study_title, technique, findings, impression.
    """
    result = {
        "study_title": None,
        "technique":   None,
        "findings":    [],
        "impression":  None,
    }

    # Determine where body starts — skip header lines
    # Header lines are those before the study title (all-caps short line)
    body_start = header_end_line
    for i, line in enumerate(raw_lines):
        line_clean = _clean_line(line)
        if _STUDY_TITLE_PATTERN.match(line_clean) and len(line_clean) > 5:
            body_start = i
            break

    body_lines = raw_lines[body_start:]

    section_patterns = _SECTION_PATTERNS
    if extra_section_patterns:
        section_patterns = {**_SECTION_PATTERNS, **extra_section_patterns}

    # State machine: track current section
    current_section = None
    section_buffer: list[str] = []

    def flush_buffer(section: str, buf: list[str]):
        """Store buffered lines into result for the given section."""
        text = " ".join(buf).strip()
        if not text:
            return
        if section in ("findings", "conclusion", "opinion", "results"):
            # Split into paragraphs by double-space or sentence boundaries
            result["findings"].append(text)
        elif section == "impression":
            result["impression"] = (result["impression"] + " " + text).strip() \
                if result["impression"] else text
        elif section == "technique":
            result["technique"] = text

    for line in body_lines:
        line_clean = _clean_line(line)

        if not line_clean:
            continue
        if _is_skip_line(line_clean):
            continue
        if _is_radiologist_line(line_clean):
            continue

        # Sign-off block begins — stop body parsing entirely. Nothing
        # past this point is clinical content, so none of it should ever
        # be absorbed into whatever section happened to be open (e.g.
        # impression swallowing "Reported By, Dr. X ... DMC/R/21536" when
        # nothing else terminates that section first). Same boundary the
        # visual redaction path already uses — see _SIGNOFF_KEYWORDS.
        if _SIGNOFF_KEYWORDS.search(line_clean):
            flush_buffer(current_section, section_buffer)
            section_buffer = []
            break

        # Check for study title (all caps, no colon, before any section)
        if (current_section is None
                and _STUDY_TITLE_PATTERN.match(line_clean)
                and ":" not in line_clean):
            result["study_title"] = line_clean
            continue

        # Check for section keyword
        matched_section = None
        for sec_name, pattern in section_patterns.items():
            if pattern.match(line_clean):
                matched_section = sec_name
                break

        if matched_section == "end":
            # End of report — flush and stop
            flush_buffer(current_section, section_buffer)
            section_buffer = []
            break

        if matched_section:
            # Flush previous section
            flush_buffer(current_section, section_buffer)
            section_buffer = []
            current_section = matched_section

            # Check if there's inline content after the section label
            # e.g. "Technique: Non enhanced helical..."
            colon_pos = line_clean.find(":")
            inline = line_clean[colon_pos + 1:].strip() if colon_pos != -1 else ""
            if inline:
                section_buffer.append(inline)
            continue

        # Regular content line — add to current section buffer
        # If no section has started yet, treat as pre-section content (skip)
        if current_section is not None:
            section_buffer.append(line_clean)

    # Flush any remaining buffer
    flush_buffer(current_section, section_buffer)

    # Join wrapped finding lines into complete sentences
    #result["findings"] = _join_wrapped_lines(result["findings"])
    # Join wrapped lines first, then split into individual sentences
    joined = _join_wrapped_lines(result["findings"])
    result["findings"] = []
    for paragraph in joined:
        result["findings"].extend(_split_into_sentences(paragraph))

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  PII redaction
# ─────────────────────────────────────────────────────────────────────────────

def _redact(text: Optional[str], clf) -> Optional[str]:
    """Run privacy-filter on a single text string. Returns None if input is None."""
    if not text:
        return text
    from dicom_anonymizer import redact_text
    return redact_text(text, clf)


def _redact_extract(extract: PdfExtract, clf, anon_patient_id: str) -> PdfExtract:
    # Identity fields — replace with pseudonym or wipe
    extract.patient_name   = f"ANON-{anon_patient_id}"
    extract.reg_no         = anon_patient_id           # same ID as DICOM PatientID
    extract.date_of_birth  = ""                        # wipe — matches DICOM PatientBirthDate
    extract.referred_by    = ""                        # wipe — matches DICOM ReferringPhysicianName
    extract.ref_center     = ""                        # wipe — matches DICOM InstitutionName
    extract.date_of_report = ""                        # wipe — matches DICOM StudyDate
    # extract.age and extract.sex are intentionally left untouched — retained
    # as clinically relevant, low re-identification-risk patient
    # characteristics, matching DICOM PatientAge/PatientSex in dicom_anonymizer.py.

    # Clinical content — run through privacy-filter to catch incidental PII
    extract.findings   = [_redact(f, clf) for f in extract.findings]
    extract.impression = _redact(extract.impression, clf)
    extract.technique  = _redact(extract.technique,  clf)

    return extract


# ─────────────────────────────────────────────────────────────────────────────
#  Layout resolution — legacy parser -> learned templates -> NuExtract3
# ─────────────────────────────────────────────────────────────────────────────
#
# Three progressively more expensive strategies, tried in order, so the
# common case stays fast and the pipeline still works correctly on a
# report layout it has never seen before:
#
#   1. Legacy fixed-layout parser (_parse_header_from_words /
#      _parse_body_from_raw_text) — the original column/keyword logic.
#      Zero cost, zero behavior change for documents that already work.
#   2. Learned template store (pdf_template_store.py) — Tier 1 keyword
#      prefilter + Tier 2 positional/label verification against
#      previously-seen layouts. Cheap, no model call.
#   3. NuExtract3 cold path (pdf_nuextract.py) — only reached when
#      nothing above matched. Verifies every returned value against the
#      source text before trusting it (hallucination guard), locates it
#      on the page, and saves a new template so the SAME layout never
#      needs the model again.
#
# Both extract_pdf() and redact_pdf_visual() call _resolve_structure()
# and use its result — one source of truth for "where is the patient
# name on this page", instead of two independently-maintained copies.

def _default_template_store_dir() -> Path:
    """
    Resolve where learned PDF templates persist.

    Deliberately NOT next to this source file (Path(__file__).parent):
    once this pipeline is packaged (e.g. PyInstaller --onefile), the
    bundle is extracted to a fresh temp directory on every launch —
    anything written next to __file__ would vanish the moment the
    process exits, silently breaking cross-session template persistence,
    which is the entire point of this store.

    Priority:
      1. PDF_TEMPLATE_STORE_DIR env var, if set — explicit override, e.g.
         for a bundled deployment or a shared/network location across
         multiple pipeline instances.
      2. A per-OS persistent user-data directory, which survives app
         restarts, updates, and (for a bundled build) is independent of
         wherever the executable happens to be extracted/installed.
    """
    env_override = os.environ.get("PDF_TEMPLATE_STORE_DIR")
    if env_override:
        return Path(env_override)

    app_name = "AnonymizationTool"
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    return base / app_name / "pdf_templates"


_template_store_singleton = None


def _get_template_store():
    global _template_store_singleton
    if _template_store_singleton is None:
        from pdf_template_store import TemplateStore
        store_dir = _default_template_store_dir()
        print(f"[INFO] PDF template store: {store_dir}")
        _template_store_singleton = TemplateStore(str(store_dir))
    return _template_store_singleton


def _section_patterns_from_template(tpl) -> dict[str, re.Pattern]:
    """
    Turn a matched template's learned body_section_keywords (built by
    build_template_from_extraction() in pdf_nuextract.py — this
    document's actual section wording, e.g. "Protocol" for technique)
    into extra_section_patterns for _parse_body_from_raw_text(). Skips
    any pattern identical to the field name itself — that's the generic
    fallback already present in _SECTION_PATTERNS, nothing new to add.
    """
    extra: dict[str, re.Pattern] = {}
    for field_name, patterns in tpl.body_section_keywords.items():
        for p in patterns:
            if p and p.lower() != field_name.lower():
                extra[field_name] = re.compile(rf"^{p}\s*:?", re.IGNORECASE)
                break
    return extra


def _apply_template(
    tpl,
    page0_words: list[dict],
    page_w: float,
    page_h: float,
    vertical_tolerance_pt: float = 4.0,
    left_tolerance_pt: float = 6.0,
    right_margin_pt: float = 220.0,
) -> tuple[dict, list[dict]]:
    """
    Given a MATCHED template, recover each header field's VALUE TEXT and
    word boxes on THIS document — not just reuse the template's own
    stored value, which came from a different document and is almost
    certainly a different length/content.

    The vertical band is matched with a small ABSOLUTE point tolerance
    (not a page-fraction), since header rows are typically only ~12-16pt
    tall with similarly small gaps between them — a percentage-of-page
    tolerance (e.g. 2% of an 842pt page = ~17pt) is easily wider than the
    gap between two adjacent rows and will pull in a neighboring field's
    words. The horizontal extent is widened generously to the right
    (right_margin_pt) since the value's length varies document to
    document — this is a deliberate simplification: a very long value
    that runs into a neighboring column could theoretically over-capture.
    Acceptable for now; revisit if that shows up in practice.
    """
    header_fields: dict = {}
    identity_word_boxes: list[dict] = []

    for field_name, hf in tpl.header_fields.items():
        x0 = hf.bbox_norm[0] * page_w
        top = hf.bbox_norm[1] * page_h
        bottom = hf.bbox_norm[3] * page_h
        x1 = hf.bbox_norm[2] * page_w + right_margin_pt

        region_words = [
            w for w in page0_words
            if w["x0"] >= x0 - left_tolerance_pt and w["x1"] <= x1
            and w["top"] >= top - vertical_tolerance_pt
            and w["top"] <= bottom + vertical_tolerance_pt
        ]
        region_words.sort(key=lambda w: w["x0"])
        value_text = " ".join(w["text"] for w in region_words).strip()
        header_fields[field_name] = value_text or None
        identity_word_boxes.extend(region_words)

    # age/sex are never stored in a template (never redacted, so never
    # located) — the fast path simply doesn't populate them; that's a
    # data-quality gap, not a privacy one, since they aren't wiped anyway.
    header_fields.setdefault("age", None)
    header_fields.setdefault("sex", None)

    return header_fields, identity_word_boxes


def _resolve_structure(
    page0_words: list[dict],
    all_raw_lines: list[str],
    page_w: float,
    page_h: float,
    clf,
) -> dict:
    """
    Resolve this document's header field values/locations and body
    section text, trying legacy -> learned template -> NuExtract3 in
    that order. See module section header above for the full rationale.

    Returns
    -------
    {
        "header_fields":        {...same shape as _parse_header_from_words()...},
        "body_fields":          {...same shape as _parse_body_from_raw_text()...},
        "identity_word_boxes":  [word dicts to redact on page 0],
        "source":               "legacy" | "template:<id>" | "nuextract" | "legacy_fallback",
    }
    """
    raw_text = "\n".join(all_raw_lines)

    legacy_header = _parse_header_from_words(page0_words)
    legacy_body   = _parse_body_from_raw_text(all_raw_lines)

    identity_keys = ("patient_name", "reg_no", "date_of_birth", "referred_by", "ref_center", "date_of_report")
    legacy_label_count = sum(1 for k in identity_keys if legacy_header.get(k))
    legacy_body_ok = bool(
        legacy_body.get("technique") or legacy_body.get("findings") or legacy_body.get("impression")
    )

    # ── Tier 0: legacy fixed-layout parser ──
    if legacy_label_count >= 3 and legacy_body_ok:
        return {
            "header_fields":       legacy_header,
            "body_fields":         legacy_body,
            "identity_word_boxes": _header_identity_value_word_boxes(page0_words),
            "source":              "legacy",
        }

    # ── Tier 1/2: learned template store ──
    from pdf_template_store import find_matching_template

    store = _get_template_store()
    templates = store.load_all()
    # Last guard: every field the template knows must find text here.
    # An empty field means a box would redact nothing (an unlabeled field
    # like the patient name can't be checked any other way), so skip that
    # template and try the next candidate — a similar layout's template
    # must not hide this layout's own.
    remaining = templates
    while True:
        tpl = find_matching_template(raw_text, page0_words, page_w, page_h, remaining)
        if tpl is None:
            break
        header_fields, identity_word_boxes = _apply_template(tpl, page0_words, page_w, page_h)
        empty = [f for f in tpl.header_fields if not header_fields.get(f)]
        if not empty:
            break
        print(f"[INFO] Template {tpl.template_id} matched but found no text for {empty} — trying next.")
        remaining = [t for t in remaining if t is not tpl]

    if tpl is not None:
        store.record_match(tpl)
        # Re-parse with THIS template's learned section wording merged
        # in, rather than reusing legacy_body/legacy_body_ok (computed
        # above using only the hardcoded defaults) — a document can
        # score legacy_body_ok=True from a single matched keyword (e.g.
        # "impression") while technique/findings, which this layout
        # calls something _SECTION_PATTERNS doesn't know, come back
        # empty. The template knows better for its own matched layout.
        body_fields = _parse_body_from_raw_text(
            all_raw_lines, extra_section_patterns=_section_patterns_from_template(tpl)
        )
        return {
            "header_fields":       header_fields,
            "body_fields":         body_fields,
            "identity_word_boxes": identity_word_boxes,
            "source":              f"template:{tpl.template_id}",
        }

    # ── Cold path: NuExtract3 ──
    try:
        from pdf_nuextract import (
            run_nuextract_on_text, verify_and_locate, build_template_from_extraction,
            IDENTITY_FIELDS,
        )
        model_output  = run_nuextract_on_text(raw_text)
        verify_result = verify_and_locate(model_output, page0_words, raw_text)
    except Exception as e:
        # Model unavailable/failed to load (no GPU, OOM, etc.) — degrade
        # to the legacy parser's best effort rather than crashing the
        # whole pipeline. Whatever it found (possibly nothing) is used;
        # redact_pdf_visual()'s body-text NLP sweep remains the safety
        # net regardless of how header resolution went.
        print(f"[WARN] NuExtract3 cold path unavailable ({e}) — using legacy parser best-effort.")
        return {
            "header_fields":       legacy_header,
            "body_fields":         legacy_body,
            "identity_word_boxes": _header_identity_value_word_boxes(page0_words),
            "source":              "legacy_fallback",
        }

    new_tpl = build_template_from_extraction(verify_result, page0_words, page_w, page_h)
    # Only keep a template that would match the very document it was
    # built from — otherwise it can never be reused, and saving it just
    # piles up a new unusable template on every run of this layout.
    new_fields, _ = _apply_template(new_tpl, page0_words, page_w, page_h)
    if (find_matching_template(raw_text, page0_words, page_w, page_h, [new_tpl]) is not None
            and all(new_fields.get(f) for f in new_tpl.header_fields)):
        store.save(new_tpl)
    else:
        print("[INFO] Learned layout is not reliably reusable — template not saved.")

    header_fields = {
        "patient_name":   verify_result["patient_name"]["value"],
        "age":            verify_result["age"]["value"],
        "sex":            verify_result["sex"]["value"],
        "reg_no":         verify_result["reg_no"]["value"],
        "date_of_birth":  verify_result["date_of_birth"]["value"],
        "date_of_report": verify_result["date_of_report"]["value"],
        "referred_by":    verify_result["referred_by"]["value"],
        "ref_center":     verify_result["ref_center"]["value"],
    }
    body_fields = {
        "study_title": verify_result["study_title"]["value"],
        "technique":   verify_result["technique"]["value"],
        "findings":    [f["value"] for f in verify_result["findings"] if f.get("verified")],
        "impression":  verify_result["impression"]["value"],
    }

    identity_word_boxes = []
    for fname in IDENTITY_FIELDS:
        info = verify_result.get(fname, {})
        if info.get("verified"):
            identity_word_boxes.extend(info["words"])

    return {
        "header_fields":       header_fields,
        "body_fields":         body_fields,
        "identity_word_boxes": identity_word_boxes,
        "source":              "nuextract",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf(pdf_path: str, clf, anon_patient_id: str) -> PdfExtract:    
    """
    Extract and anonymize all structured fields from a radiology/lab PDF.

    Parameters
    ----------
    pdf_path : str
        Path to the input PDF file.
    clf :
        Loaded openai/privacy-filter pipeline from load_privacy_filter().
    anon_patient_id : str
        Pseudonym to replace patient identifiers.

    Returns
    -------
    PdfExtract
        All extracted fields with PII redacted.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    print(f"[INFO] Extracting PDF: {path.name}")

    with pdfplumber.open(str(path)) as pdf:
        if not pdf.pages:
            raise ValueError(f"PDF has no pages: {path}")

        page0 = pdf.pages[0]
        page0_words = page0.extract_words()

        # Collect raw text across all pages (findings/impression can span
        # more than one page) — but header fields are resolved from page 0
        # only, see _resolve_structure().
        all_raw_lines: list[str] = []
        for page in pdf.pages:
            raw_text = page.extract_text() or ""
            all_raw_lines.extend(raw_text.splitlines())

    # ── Stage 1: Resolve header + body structure ──
    # Tries the legacy fixed-layout parser first, then a learned template
    # match, then NuExtract3 as a last resort for a genuinely new layout.
    structure     = _resolve_structure(page0_words, all_raw_lines, page0.width, page0.height, clf)
    header_fields = structure["header_fields"]
    body_fields   = structure["body_fields"]
    print(f"[INFO] PDF structure resolved via: {structure['source']}")

    # ── Assemble raw extract (before anonymization) ──
    extract = PdfExtract(
        patient_name   = header_fields.get("patient_name"),
        age            = header_fields.get("age"),
        sex            = header_fields.get("sex"),
        reg_no         = header_fields.get("reg_no"),
        date_of_birth  = header_fields.get("date_of_birth"),
        date_of_report = header_fields.get("date_of_report"),
        referred_by    = header_fields.get("referred_by"),
        ref_center     = header_fields.get("ref_center"),
        study_title    = body_fields.get("study_title"),
        technique      = body_fields.get("technique"),
        findings       = body_fields.get("findings", []),
        impression     = body_fields.get("impression"),
    )

    # Build full raw text for logging/debugging before redaction
    full_text_parts = []
    for label, val in [
        ("Study", extract.study_title),
        ("Technique", extract.technique),
        ("Findings", "\n".join(extract.findings) if extract.findings else None),
        ("Impression", extract.impression),
    ]:
        if val:
            full_text_parts.append(f"{label}: {val}")
    raw_full_text = "\n".join(full_text_parts)

    print(f"[INFO] Raw extract before anonymization:")
    print(f"  patient_name   : {extract.patient_name}")
    print(f"  age            : {extract.age}")
    print(f"  sex            : {extract.sex}")
    print(f"  reg_no         : {extract.reg_no}")
    print(f"  date_of_birth  : {extract.date_of_birth}")
    print(f"  date_of_report : {extract.date_of_report}")
    print(f"  referred_by    : {extract.referred_by}")
    print(f"  ref_center     : {extract.ref_center}")
    print(f"  study_title    : {extract.study_title}")
    print(f"  technique      : {extract.technique}")
    print(f"  findings       : {extract.findings}")
    print(f"  impression     : {extract.impression}")
    print()

    # ── Stage 2: Anonymize all text fields ──
    print("[INFO] Running PII detection on PDF fields ...")

    # Snapshot every original value BEFORE any redaction happens — this is
    # the only place the pre-anonymization values exist anywhere, and it's
    # kept solely for the results-page PII comparison view.
    raw_values = {
        "patient_name":   extract.patient_name,
        "age":            extract.age,
        "sex":            extract.sex,
        "reg_no":         extract.reg_no,
        "date_of_birth":  extract.date_of_birth,
        "date_of_report": extract.date_of_report,
        "referred_by":    extract.referred_by,
        "ref_center":     extract.ref_center,
        "study_title":    extract.study_title,
        "technique":      extract.technique,
        "findings":       list(extract.findings),
        "impression":     extract.impression,
        "full_text":      raw_full_text,
    }

    extract = _redact_extract(extract, clf, anon_patient_id)
    extract.full_text_anon = _redact(raw_full_text, clf)
    extract.raw_values = raw_values

    print(f"[INFO] Anonymized extract:")
    print(f"  patient_name   : {extract.patient_name}")
    print(f"  referred_by    : {extract.referred_by}")
    print(f"  findings       : {extract.findings}")
    print(f"  impression     : {extract.impression}")
    print()

    return extract


# ─────────────────────────────────────────────────────────────────────────────
#  PII audit — before vs after, for the results-page comparison view
# ─────────────────────────────────────────────────────────────────────────────

# field_name -> (display label, action taken by _redact_extract)
_AUDIT_FIELD_LABELS: dict[str, tuple[str, str]] = {
    "patient_name":   ("Patient Name",   "pseudonymized"),
    "reg_no":         ("Reg/Lab No",     "pseudonymized"),
    "date_of_birth":  ("Date of Birth",  "removed"),
    "referred_by":    ("Referred By",    "removed"),
    "ref_center":     ("Referring Center", "removed"),
    "date_of_report": ("Date of Report", "removed"),
    "technique":      ("Technique",      "redacted"),
    "impression":     ("Impression",     "redacted"),
}


def build_pdf_pii_audit(extract: PdfExtract) -> list[dict]:
    """
    Compare extract.raw_values (captured before redaction) against the
    final anonymized field values, and return a list of
    {field, label, original, anonymized, action} entries — one per field
    that actually had a value in the original document.

    Only for the results-page "PII before vs after" comparison view —
    plays no part in anonymization itself.
    """
    entries = []
    raw = extract.raw_values or {}

    for field_name, (label, action) in _AUDIT_FIELD_LABELS.items():
        original = raw.get(field_name)
        if not original:
            continue
        anonymized = getattr(extract, field_name, "")
        entries.append({
            "field":      field_name,
            "label":      label,
            "original":   str(original),
            "anonymized": str(anonymized or ""),
            "action":     action,
        })

    # Findings is a list — join for display, but only if there was content
    raw_findings = raw.get("findings") or []
    if raw_findings:
        entries.append({
            "field":      "findings",
            "label":      "Findings",
            "original":   "\n".join(raw_findings),
            "anonymized": "\n".join(extract.findings or []),
            "action":     "redacted",
        })

    return entries


# ─────────────────────────────────────────────────────────────────────────────
#  Anonymized PDF file generation
# ─────────────────────────────────────────────────────────────────────────────

def _group_words_into_lines(words: list[dict]) -> list[list[dict]]:
    """
    Group words into lines by top-coordinate clustering (within 3px),
    same convention as _parse_header_from_words()'s line grouping, but
    applied across a whole set of words (any zone of the page).
    """
    lines: dict[float, list[dict]] = {}
    for w in words:
        top = w["top"]
        matched = None
        for existing_top in lines:
            if abs(existing_top - top) <= 3:
                matched = existing_top
                break
        if matched is not None:
            lines[matched].append(w)
        else:
            lines[top] = [w]
    return [sorted(lines[t], key=lambda w: w["x0"]) for t in sorted(lines.keys())]


def _split_label_value_words(col_words: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Given words in one label:value column (sorted by x0), split into
    (label_words, value_words) by locating which word contains the colon.
    Word-level counterpart to the label/value split in
    _parse_header_from_words() — needed here to get bounding boxes for
    just the VALUE portion, not the label.
    """
    for i, w in enumerate(col_words):
        if ":" in w["text"]:
            if w["text"].strip().endswith(":"):
                return col_words[:i + 1], col_words[i + 1:]
            return col_words[:i], col_words[i:]
    return col_words, []


def _header_identity_value_word_boxes(words: list[dict]) -> list[dict]:
    """
    Return word-level bounding boxes for every header VALUE that belongs
    to an identity field — patient name, reg/lab no, date of birth,
    referred-by, referring center, date of report. These are exactly the
    fields _redact_extract() wipes from the extracted text.

    Age and Sex values are deliberately excluded — retained, not PII
    (same policy as DICOM PatientAge/PatientSex).
    """
    header_words = [w for w in words if w["top"] < _HEADER_BOTTOM_TOP]
    if not header_words:
        return []

    lines: dict[float, list[dict]] = {}
    for w in header_words:
        top = w["top"]
        matched = None
        for existing_top in lines:
            if abs(existing_top - top) <= 3:
                matched = existing_top
                break
        if matched is not None:
            lines[matched].append(w)
        else:
            lines[top] = [w]

    redact_words: list[dict] = []

    for top in sorted(lines.keys()):
        line_words = sorted(lines[top], key=lambda w: w["x0"])
        left_words  = [w for w in line_words if w["x0"] < _HEADER_RIGHT_COL_X0]
        right_words = [w for w in line_words if w["x0"] >= _HEADER_RIGHT_COL_X0]

        for col_words in [left_words, right_words]:
            if not col_words:
                continue
            label_words, value_words = _split_label_value_words(col_words)
            if not value_words:
                continue
            label_text = _fix_doubled_chars(" ".join(w["text"] for w in label_words)).upper()

            is_age_sex = "AGE" in label_text or "SEX" in label_text
            is_dob = (
                "DOB" in label_text or "D.O.B" in label_text
                or ("DATE" in label_text and "BIRTH" in label_text)
            )
            is_identity = (
                ("NAME" in label_text and "REF" not in label_text)
                or "REG" in label_text or "LAB" in label_text
                or is_dob
                or ("DATE" in label_text and "REPORT" in label_text)
                or "REFER" in label_text
                or ("REF" in label_text and "CENTER" in label_text)
            )
            if is_identity and not is_age_sex:
                redact_words.extend(value_words)

    return redact_words


def _body_pii_word_boxes(words: list[dict], clf, include_header_zone: bool = False) -> list[dict]:
    """
    Run privacy-filter over every line of body text and return word-level
    bounding boxes for any detected PII span.

    Detection runs on doubled-character-fixed text (garbled font-encoding
    artifacts otherwise confuse the NER model — see _fix_doubled_chars),
    but bounding boxes always come from the ORIGINAL word objects, so
    redaction lands on the actual rendered glyphs regardless of how the
    text extracted.

    By default, does NOT scan the header zone — the structured identity
    fields there are already handled by _resolve_structure(), and
    keeping the NLP sweep out of that zone guarantees a false-positive
    detection can never redact the retained age/sex values sitting right
    next to identity fields.

    Set include_header_zone=True as a fallback when header resolution
    found NOTHING for this document (e.g. NuExtract3 unavailable and the
    layout doesn't match the legacy parser either) — in that situation
    there is no precisely-located identity field to protect age/sex
    from, and leaving the header zone completely unscanned would be
    worse than the small risk of an NLP false positive there.
    """
    body_words = words if include_header_zone else [w for w in words if w["top"] >= _HEADER_BOTTOM_TOP]
    if not body_words:
        return []

    redact_words: list[dict] = []

    for line_words in _group_words_into_lines(body_words):
        fixed_texts = [_fix_doubled_chars(w["text"]) for w in line_words]
        line_text = " ".join(fixed_texts)
        if not line_text.strip():
            continue

        try:
            entities = clf(line_text)
        except Exception:
            continue
        if not entities:
            continue

        # Map each word's [start, end) character range within line_text
        # (words joined by single spaces, matching how line_text was built)
        word_spans = []
        cursor = 0
        for t in fixed_texts:
            word_spans.append((cursor, cursor + len(t)))
            cursor += len(t) + 1

        for ent in entities:
            e_start, e_end = ent["start"], ent["end"]
            for i, (w_start, w_end) in enumerate(word_spans):
                if w_end > e_start and w_start < e_end:
                    redact_words.append(line_words[i])

    return redact_words


# Keywords marking the start of a report's sign-off/signature block —
# scopes signature-image detection (see _signature_image_boxes()) to
# where a signature is actually expected, rather than flagging every
# small embedded image on the page (which could legitimately be a
# clinical reference image rather than a signature/stamp).
_SIGNOFF_KEYWORDS = re.compile(
    r"reported\s*by|electronically\s*(signed|verified)|signed\s*by|verified\s*by",
    re.IGNORECASE,
)

# An embedded image narrower than this fraction of the page is treated
# as a signature/stamp candidate rather than a letterhead or footer
# banner. Calibrated against a real report: its letterhead/footer
# banners each spanned ~100% of the page width, while its actual
# signature image spanned ~20%.
_MAX_SIGNATURE_WIDTH_FRACTION = 0.6


def _signature_zone_top(words: list[dict]) -> Optional[float]:
    """
    Return the y-position (points from page top) of the topmost sign-off
    marker line on this page (e.g. "Reported By,"), or None if no such
    marker is found.
    """
    for line_words in _group_words_into_lines(words):
        line_text = " ".join(w["text"] for w in line_words)
        if _SIGNOFF_KEYWORDS.search(line_text):
            return min(w["top"] for w in line_words)
    return None


def _signature_image_boxes(page, words: list[dict]) -> list[dict]:
    """
    Locate embedded images that are likely a radiologist's signature or
    stamp, so redact_pdf_visual() can black them out alongside identity
    text.

    A signature is purely graphical — pdfplumber's word extraction never
    sees it, so it is completely invisible to both the structured header
    resolution and the privacy-filter NER sweep, which only ever operate
    on extracted text. Neither _header_identity_value_word_boxes() nor
    _body_pii_word_boxes() can protect against it; this is a separate,
    image-based signal.

    Heuristic (validated against a real report that left an un-redacted
    signature — see the PDF signature-handling discussion this was added
    for):
      - Only images narrower than _MAX_SIGNATURE_WIDTH_FRACTION of the
        page are considered — a (near-)full-page-width image is a
        letterhead/footer banner, not a signature.
      - Restricted to images at or below the sign-off zone (see
        _signature_zone_top()) when this page has a recognizable sign-off
        marker line. If no marker is found on the page, every narrow
        image is treated as a candidate instead — better to redact one
        unexpected small graphic than silently leave an unrecognized
        sign-off layout's signature untouched.
    """
    zone_top = _signature_zone_top(words)
    boxes = []
    for im in page.images:
        img_width = im["x1"] - im["x0"]
        if img_width > page.width * _MAX_SIGNATURE_WIDTH_FRACTION:
            continue
        if zone_top is not None and im["top"] < zone_top:
            continue
        boxes.append({"x0": im["x0"], "top": im["top"], "x1": im["x1"], "bottom": im["bottom"]})
    return boxes


def _signoff_zone_word_boxes(words: list[dict]) -> list[dict]:
    """
    Return every word from the sign-off marker line (see
    _signature_zone_top()) to the bottom of the page — the reporting
    radiologist's name, qualifications, title, and registration/license
    number (e.g. "DMC/R/21536").

    Wiped wholesale, the same policy already applied to every other
    identity field in this pipeline (ReferringPhysicianName,
    InstitutionName, ...) rather than picked apart line by line: a
    registration/license number identifies a specific person just as
    reliably as a name does, but it doesn't look like a name, date, or
    any other category a general-purpose PII NER model is trained to
    recognize — _body_pii_word_boxes()'s NER sweep catches the printed
    name in this block (it's a PERSON span) but can't be trusted to
    catch the registration number too, so this block is a distinct,
    rule-based signal rather than a gap to special-case in the NER path.
    """
    zone_top = _signature_zone_top(words)
    if zone_top is None:
        return []
    return [w for w in words if w["top"] >= zone_top]


def redact_pdf_visual(pdf_path: str, clf, out_path: str) -> str:
    """
    Produce a redacted copy of the ORIGINAL PDF at *pdf_path* — same
    layout, same pages, same look — with PII regions TRULY removed (not
    just visually covered) and replaced with solid black boxes.

    Redacted: patient name, reg/lab no, referred-by, referring center,
    date of report (header identity fields — located via the same
    legacy/template/NuExtract3 resolution as extract_pdf(), see
    _resolve_structure(), so extraction and redaction can never disagree
    about where these fields are), plus any PII privacy-filter detects
    anywhere in the body text (findings, impression, technique, etc), plus
    the entire sign-off block (reporting radiologist's name,
    qualifications, and registration/license number — see
    _signoff_zone_word_boxes()) and any embedded signature/stamp image
    near it (see _signature_image_boxes()) — two separate signals,
    since neither a signature image nor a registration-number pattern
    is something the word/NER-based checks above can see or recognize.

    Kept visible: age, sex — not wiped, matching the retention policy
    applied to DICOM PatientAge/PatientSex elsewhere in the pipeline.

    Uses PyMuPDF's redaction annotations (add_redact_annot +
    apply_redactions), which delete the underlying text object within the
    box, not just draw over it — the text is genuinely gone from the
    output file, not merely hidden under a black rectangle that could
    still be selected/extracted/copied.

    Parameters
    ----------
    pdf_path : str
        Path to the ORIGINAL (un-redacted) PDF report.
    clf :
        Loaded privacy-filter pipeline (see load_privacy_filter() in
        dicom_anonymizer.py).
    out_path : str
        Where to write the redacted PDF.

    Returns
    -------
    str
        The path written (same as out_path).
    """
    import pymupdf

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            doc.close()
            return str(out)

        page0 = pdf.pages[0]
        page0_words = page0.extract_words()
        all_raw_lines = []
        for p in pdf.pages:
            all_raw_lines.extend((p.extract_text() or "").splitlines())

        # Same resolution extract_pdf() uses — one source of truth for
        # where the identity fields are, shared between extraction and
        # redaction rather than two independently-maintained copies.
        structure = _resolve_structure(page0_words, all_raw_lines, page0.width, page0.height, clf)
        identity_word_boxes = structure["identity_word_boxes"]
        print(f"[INFO] PDF redaction header resolved via: {structure['source']}")

        # If NOTHING was located for the header (model unavailable AND
        # layout didn't match the legacy parser), there's no precisely-
        # found identity field to protect age/sex from — widen the NLP
        # sweep to cover the header zone too rather than leave it
        # completely unscanned. Some coverage beats none.
        header_zone_fallback = not identity_word_boxes

        for page_index, page in enumerate(pdf.pages):
            words = page0_words if page_index == 0 else page.extract_words()
            if not words:
                continue

            include_header = (page_index == 0 and header_zone_fallback)
            redact_word_dicts = list(_body_pii_word_boxes(words, clf, include_header_zone=include_header))
            if page_index == 0:
                redact_word_dicts += identity_word_boxes
            redact_word_dicts += _signature_image_boxes(page, words)
            redact_word_dicts += _signoff_zone_word_boxes(words)

            if not redact_word_dicts:
                continue

            fitz_page = doc[page_index]
            for w in redact_word_dicts:
                rect = pymupdf.Rect(
                    w["x0"] - 1.5, w["top"] - 1.5, w["x1"] + 1.5, w["bottom"] + 1.5
                )
                fitz_page.add_redact_annot(rect, fill=(0, 0, 0))
            fitz_page.apply_redactions()

    doc.save(str(out))
    doc.close()

    return str(out)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dicom_anonymizer import load_privacy_filter

    if len(sys.argv) < 3:
        print("Usage: python pdf_extractor.py <path_to_pdf> <anon_patient_id>")
        print("Example: python pdf_extractor.py input/report.pdf ANON-5CDDFE3D5457")
        sys.exit(1)

    clf = load_privacy_filter()
    result = extract_pdf(sys.argv[1], clf, anon_patient_id=sys.argv[2])

    print("=" * 60)
    print("FINAL EXTRACTED FIELDS")
    print("=" * 60)
    for fname in result.__dataclass_fields__:
        val = getattr(result, fname)
        if fname == "full_text_anon":
            continue
        print(f"  {fname:<20} : {val}")