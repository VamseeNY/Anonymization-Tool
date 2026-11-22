"""
pdf_template_store.py
----------------------
Persistent store of learned PDF report layouts ("templates"), so the
expensive NuExtract3 model only has to run once per *distinct layout*,
not once per document.

A template records, for one report layout:
  - which body-section keywords it uses (technique/findings/impression
    and synonyms) — the cheap Tier-1 prefilter signal
  - where each identity header field's VALUE sits on the page, in
    page-fraction coordinates, plus what its label looked like — the
    Tier-2 positional/label verification signal, and the payload a
    matched document reuses for both structured extraction and visual
    redaction

Matching flow (see also pdf_extractor.py's _resolve_structure):
  Tier 1 (prefilter): do the expected body-section keywords appear at
    all? Cheap, but only tells you the document TYPE matches — nothing
    about layout.
  Tier 2 (verify): for a Tier-1 candidate, is each header field's stored
    LABEL text still found near its stored NORMALIZED position in this
    document? Only a template that passes both is trusted for reuse —
    reusing header coordinates from a template whose layout doesn't
    actually match would silently misplace a redaction box, so Tier 2 is
    the safety-critical step, not an optimization. Tier 2 also requires
    the same page size and every stored section heading at the same left
    edge and in the same order (heading_verify), since different layouts
    can share an identical header.

Templates are stored as one JSON file per template under store_dir. No
manual-approval gate: a template is usable for matching from the moment
it's created (status "provisional"), and is simply relabeled "trusted"
once it has matched successfully TRUST_PROMOTION_THRESHOLD times — the
label is informational (useful for monitoring/audit), not a functional
gate; matching behaves identically either way.
"""

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


TRUST_PROMOTION_THRESHOLD = 3

# How close a label must be found to its template-recorded position to
# count as a positional match. Absolute point values, not a page
# fraction — header rows are typically only ~12-16pt tall with similarly
# small gaps between them, so a percentage-of-page tolerance (e.g. 2% of
# an 842pt page = ~17pt) is easily wider than the gap between two
# adjacent rows and would risk matching a neighboring field's label.
_VERTICAL_TOLERANCE_PT = 5.0
_HORIZONTAL_LABEL_SEARCH_PT = 140.0

# A section heading's LEFT edge is fixed by the layout, but its vertical
# position is not — a longer protocol or findings list pushes every later
# heading down. So headings are checked on text + left edge + top-to-bottom
# order, never on absolute vertical position.
_HEADING_X_TOLERANCE_PT = 12.0

# Same layout => same page size. Normalized coordinates alone let a
# different-size layout with a similar header pass the label check.
_PAGE_SIZE_TOLERANCE_PT = 2.0


@dataclass
class HeaderField:
    """Where one identity field's VALUE sits on the page, and what its
    label looked like when this template was captured."""
    label_text: str            # e.g. "NAME", "REG NO" — for re-verification
    bbox_norm: list[float]     # [x0, top, x1, bottom], normalized 0-1 by page size


@dataclass
class SectionHeading:
    """A body-section heading line as it appears in this layout
    (e.g. "PROTOCOL" for technique) and where it starts on page 0."""
    text: str                  # heading text, trailing colon stripped
    x0_norm: float             # left edge, normalized 0-1 by page width
    top_norm: float            # top, normalized 0-1 by page height — used for ORDER only


@dataclass
class PdfTemplate:
    template_id: str
    status: str                                   # "provisional" | "trusted"
    created_at: str
    match_count: int
    body_section_keywords: dict[str, list[str]]   # field -> regex patterns (as strings)
    header_fields: dict[str, HeaderField]          # identity field -> HeaderField
    source: str = "learned"                        # "legacy" for the seeded default, "learned" otherwise
    page_size: list[float] = field(default_factory=list)                     # [width, height] in points
    section_headings: dict[str, SectionHeading] = field(default_factory=dict)  # body field -> heading

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_json(d: dict) -> "PdfTemplate":
        header_fields = {k: HeaderField(**v) for k, v in d.get("header_fields", {}).items()}
        section_headings = {k: SectionHeading(**v) for k, v in d.get("section_headings", {}).items()}
        return PdfTemplate(
            template_id=d["template_id"],
            status=d["status"],
            created_at=d["created_at"],
            match_count=d["match_count"],
            body_section_keywords=d["body_section_keywords"],
            header_fields=header_fields,
            source=d.get("source", "learned"),
            page_size=d.get("page_size", []),
            section_headings=section_headings,
        )


def new_template_id() -> str:
    return f"tmpl_{uuid.uuid4().hex[:12]}"


class TemplateStore:
    """Directory-backed store of PdfTemplate records — one JSON file per template."""

    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, template_id: str) -> Path:
        return self.store_dir / f"{template_id}.json"

    def load_all(self) -> list[PdfTemplate]:
        templates = []
        for p in sorted(self.store_dir.glob("*.json")):
            try:
                templates.append(PdfTemplate.from_json(json.loads(p.read_text(encoding="utf-8"))))
            except Exception as e:
                print(f"[WARN] Skipping unreadable template file {p.name}: {e}")
        return templates

    def save(self, tpl: PdfTemplate) -> None:
        self._path_for(tpl.template_id).write_text(
            json.dumps(tpl.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def create_provisional(
        self,
        body_section_keywords: dict[str, list[str]],
        header_fields: dict[str, HeaderField],
    ) -> PdfTemplate:
        tpl = PdfTemplate(
            template_id=new_template_id(),
            status="provisional",
            created_at=datetime.now(timezone.utc).isoformat(),
            match_count=1,
            body_section_keywords=body_section_keywords,
            header_fields=header_fields,
            source="learned",
        )
        self.save(tpl)
        return tpl

    def record_match(self, tpl: PdfTemplate) -> PdfTemplate:
        """Call when a template was successfully used for a document —
        increments its match count and promotes provisional -> trusted
        once TRUST_PROMOTION_THRESHOLD is reached. No manual approval
        step; the label is informational, matching works the same either
        way."""
        tpl.match_count += 1
        if tpl.status == "provisional" and tpl.match_count >= TRUST_PROMOTION_THRESHOLD:
            tpl.status = "trusted"
        self.save(tpl)
        return tpl


# ─────────────────────────────────────────────────────────────────────────────
#  Matching
# ─────────────────────────────────────────────────────────────────────────────

def keyword_prefilter(raw_text: str, tpl: PdfTemplate, min_hits: int = 2) -> bool:
    """Tier 1 — cheap document-TYPE check: do enough of this template's
    body-section keyword patterns appear anywhere in the text?"""
    hits = 0
    for patterns in tpl.body_section_keywords.values():
        if any(re.search(p, raw_text, re.IGNORECASE) for p in patterns):
            hits += 1
    return hits >= min_hits


def positional_verify(
    words: list[dict],
    tpl: PdfTemplate,
    page_w: float,
    page_h: float,
    vertical_tolerance_pt: float = _VERTICAL_TOLERANCE_PT,
    horizontal_search_pt: float = _HORIZONTAL_LABEL_SEARCH_PT,
) -> bool:
    """Tier 2 — safety-critical LAYOUT check: for each of this template's
    header fields, is a word resembling its stored label still found near
    its stored normalized position in this document? Only templates that
    pass this are trusted for reuse — this is what prevents a body-only
    keyword match from causing a header field to be redacted (or not) in
    the wrong place."""
    # Only fields with a real stored label can be checked. A field whose
    # value has no label on the page (e.g. a patient name printed alone at
    # the left margin) is stored with label_text "" and skipped here — its
    # position is trusted only because the rest of the layout matched.
    labeled = [hf for hf in tpl.header_fields.values() if hf.label_text]
    if len(labeled) < 2:
        return False

    for hf in labeled:
        expected_x0 = hf.bbox_norm[0] * page_w
        expected_top = hf.bbox_norm[1] * page_h
        # Search a window to the left of and around the expected value
        # position — the label sits before the value, so widen the
        # horizontal tolerance rather than the vertical one, which must
        # stay tight to avoid picking up an adjacent row.
        nearby = [
            w for w in words
            if abs(w["top"] - expected_top) <= vertical_tolerance_pt
            and (expected_x0 - horizontal_search_pt) <= w["x0"] <= (expected_x0 + vertical_tolerance_pt)
        ]
        # Compare the FULL label, not just its first word — short first
        # words like "Date" or "Ref." appear on most report headers.
        nearby_text = _norm(" ".join(w["text"] for w in nearby))
        if _norm(hf.label_text) not in nearby_text:
            return False

    return True


def _norm(text: str) -> str:
    """Uppercase and drop spaces, so "Date :" matches "Date:"."""
    return re.sub(r"\s+", "", text).upper()


def page_size_matches(tpl: PdfTemplate, page_w: float, page_h: float) -> bool:
    if len(tpl.page_size) != 2:
        return False
    return (abs(tpl.page_size[0] - page_w) <= _PAGE_SIZE_TOLERANCE_PT
            and abs(tpl.page_size[1] - page_h) <= _PAGE_SIZE_TOLERANCE_PT)


def heading_verify(
    words: list[dict],
    tpl: PdfTemplate,
    page_w: float,
    page_h: float,
    x_tolerance_pt: float = _HEADING_X_TOLERANCE_PT,
) -> bool:
    """
    Tier 2b — body LAYOUT check: is every stored section heading present
    on page 0 as the start of a line, at the same left edge, and in the
    same top-to-bottom order? This is what tells apart two layouts that
    share an identical header but structure the report body differently.
    """
    if len(tpl.section_headings) < 2:
        return False
    from pdf_extractor import _group_words_into_lines

    lines = _group_words_into_lines(words)
    found = []   # (stored top_norm, found top) per heading
    for sh in tpl.section_headings.values():
        expected_x0 = sh.x0_norm * page_w
        match_top = None
        for ln in lines:
            if abs(ln[0]["x0"] - expected_x0) > x_tolerance_pt:
                continue
            if _norm(" ".join(w["text"] for w in ln)).startswith(_norm(sh.text)):
                match_top = ln[0]["top"]
                break
        if match_top is None:
            return False
        found.append((sh.top_norm, match_top))

    found.sort()
    return all(a[1] < b[1] for a, b in zip(found, found[1:]))


def find_matching_template(
    raw_text: str,
    words: list[dict],
    page_w: float,
    page_h: float,
    templates: list[PdfTemplate],
) -> Optional[PdfTemplate]:
    """Run Tier 1 then Tier 2 against every stored template, preferring
    the most-proven (highest match_count) candidate. Returns the first
    template that passes both, or None if nothing matches confidently —
    callers should fall back to the NuExtract3 cold path in that case."""
    candidates = [t for t in templates if keyword_prefilter(raw_text, t)]
    candidates.sort(key=lambda t: -t.match_count)
    for tpl in candidates:
        if (page_size_matches(tpl, page_w, page_h)
                and positional_verify(words, tpl, page_w, page_h)
                and heading_verify(words, tpl, page_w, page_h)):
            return tpl
    return None