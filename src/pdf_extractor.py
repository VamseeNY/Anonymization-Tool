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

import re
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
) -> dict:
    """
    Parse the body of the report from raw text lines.

    Parameters
    ----------
    raw_lines : list[str]
        All lines from raw text extraction, 0-indexed.
    header_end_line : int
        Line index after which body content starts (skip header lines).

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

        # Check for study title (all caps, no colon, before any section)
        if (current_section is None
                and _STUDY_TITLE_PATTERN.match(line_clean)
                and ":" not in line_clean):
            result["study_title"] = line_clean
            continue

        # Check for section keyword
        matched_section = None
        for sec_name, pattern in _SECTION_PATTERNS.items():
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
    extract.referred_by    = ""                        # wipe — matches DICOM ReferringPhysicianName
    extract.ref_center     = ""                        # wipe — matches DICOM InstitutionName
    extract.age            = ""                        # wipe — matches DICOM PatientAge
    extract.date_of_report = ""                        # wipe — matches DICOM StudyDate

    # Clinical content — run through privacy-filter to catch incidental PII
    extract.findings   = [_redact(f, clf) for f in extract.findings]
    extract.impression = _redact(extract.impression, clf)
    extract.technique  = _redact(extract.technique,  clf)

    return extract


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

        # Collect raw text and words across all pages
        all_raw_lines: list[str] = []
        all_words: list[dict]    = []

        for page in pdf.pages:
            raw_text = page.extract_text() or ""
            all_raw_lines.extend(raw_text.splitlines())
            all_words.extend(page.extract_words())

    # ── Stage 1a: Parse header from bounding boxes ──
    header_fields = _parse_header_from_words(all_words)

    # ── Stage 1b: Parse body from raw text ──
    body_fields = _parse_body_from_raw_text(all_raw_lines)

    # ── Assemble raw extract (before anonymization) ──
    extract = PdfExtract(
        patient_name   = header_fields.get("patient_name"),
        age            = header_fields.get("age"),
        sex            = header_fields.get("sex"),
        reg_no         = header_fields.get("reg_no"),
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
    extract = _redact_extract(extract, clf, anon_patient_id)
    extract.full_text_anon = _redact(raw_full_text, clf)

    print(f"[INFO] Anonymized extract:")
    print(f"  patient_name   : {extract.patient_name}")
    print(f"  referred_by    : {extract.referred_by}")
    print(f"  findings       : {extract.findings}")
    print(f"  impression     : {extract.impression}")
    print()

    return extract


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
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