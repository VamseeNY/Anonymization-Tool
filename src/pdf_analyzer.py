"""
pdf_inspect.py
--------------
Inspection utility for the PDF pipeline.

Extracts and prints the raw text from a PDF exactly as pdfplumber sees it,
with line numbers and page markers. Run this first on any new PDF format
before writing the parser — the output tells you exactly what structure
you are working with.

Also prints word-level bounding boxes for the first page so you can see
how fields are positioned spatially (useful for header box parsing).

Usage
-----
    python pdf_inspect.py <path_to_pdf>

Example
-------
    python pdf_inspect.py "input/ANBARSU-R_report.pdf"
"""

import sys
from pathlib import Path

import pdfplumber


def inspect_pdf(pdf_path: str) -> None:
    path = Path(pdf_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    print("=" * 60)
    print(f"PDF INSPECTOR")
    print("=" * 60)
    print(f"  File : {path}")
    print(f"  Size : {path.stat().st_size / 1024:.1f} KB")
    print()

    with pdfplumber.open(str(path)) as pdf:
        print(f"  Pages: {len(pdf.pages)}")
        print()

        for page_num, page in enumerate(pdf.pages, 1):

            # ── Raw text extraction ──
            print("=" * 60)
            print(f"PAGE {page_num} — RAW TEXT (extract_text)")
            print("=" * 60)
            raw_text = page.extract_text()
            if not raw_text:
                print("  (no text extracted)")
            else:
                for line_num, line in enumerate(raw_text.splitlines(), 1):
                    print(f"  {line_num:>3} | {line}")

            print()

            # ── Text with layout preserved ──
            print("-" * 60)
            print(f"PAGE {page_num} — LAYOUT-PRESERVED TEXT (extract_text layout=True)")
            print("-" * 60)
            layout_text = page.extract_text(layout=True)
            if not layout_text:
                print("  (no text extracted)")
            else:
                for line_num, line in enumerate(layout_text.splitlines(), 1):
                    print(f"  {line_num:>3} | {line}")

            print()

            # ── Word-level bounding boxes (first page only) ──
            if page_num == 1:
                print("-" * 60)
                print(f"PAGE {page_num} — WORD BOUNDING BOXES (x0, top, x1, bottom, text)")
                print("-" * 60)
                words = page.extract_words()
                if not words:
                    print("  (no words extracted)")
                else:
                    for w in words:
                        print(
                            f"  x0={w['x0']:>6.1f}  top={w['top']:>6.1f}  "
                            f"x1={w['x1']:>6.1f}  bottom={w['bottom']:>6.1f}  "
                            f"text={w['text']}"
                        )
                print()

            # ── Tables (if any) ──
            print("-" * 60)
            print(f"PAGE {page_num} — TABLES (extract_tables)")
            print("-" * 60)
            tables = page.extract_tables()
            if not tables:
                print("  (no tables detected)")
            else:
                for t_idx, table in enumerate(tables, 1):
                    print(f"  Table {t_idx}:")
                    for row in table:
                        print(f"    {row}")
            print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_inspect.py <path_to_pdf>")
        print("Example: python pdf_inspect.py input/report.pdf")
        sys.exit(1)

    inspect_pdf(sys.argv[1])