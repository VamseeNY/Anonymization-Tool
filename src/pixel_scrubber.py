"""
pixel_scrubber.py
-----------------
OCR-based burned-in pixel annotation scrubber.

Detects text burned into DICOM pixel data using EasyOCR and zero-fills
detected regions. Runs on the full image frame for every instance.

Sampling strategy:
    Before processing all instances in a series, OCR is run on the first
    3 slices. If no text is detected in any of them, the remaining slices
    are skipped — the study is assumed clean. If text is found, all slices
    are processed.

Pixel pipeline:
    Raw 16-bit DICOM pixel array
        → window to 8-bit greyscale (using WindowCenter/WindowWidth)
        → EasyOCR on full frame
        → map bounding boxes back to original array coordinates
        → zero-fill rectangles in original 16-bit array
        → write back to ds.PixelData

Design decisions:
    - OCR runs on the 8-bit windowed image for detection, but redaction
      is applied to the original pixel array (before windowing). This
      ensures the actual stored pixel values are zeroed, not just the
      display-space representation.
    - Confidence threshold of 0.5 — reduces false positives on anatomy
      that resembles text (e.g. vertebral edges).
    - Bounding boxes are expanded by BBOX_PADDING pixels on each side
      to ensure full glyph coverage including anti-aliasing.
    - Multi-frame DICOM (e.g. ultrasound cines) — each frame is processed
      independently.

Usage
-----
    from pixel_scrubber import load_ocr_reader, scrub_pixels
    from pixel_scrubber import sample_series_for_burned_in

    reader = load_ocr_reader()

    # Check if series has any burned-in text before processing all slices
    has_text = sample_series_for_burned_in(first_3_datasets, reader)

    if has_text:
        for ds in all_datasets:
            scrub_pixels(ds, reader, window_center, window_width)

Requires
--------
    pip install easyocr numpy
    pip install Pillow  (already in pipeline)
"""

import numpy as np
from pathlib import Path
from typing import Optional

import pydicom


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Minimum OCR confidence to treat a detection as real text
CONFIDENCE_THRESHOLD = 0.5

# Pixels to expand each bounding box on all sides
BBOX_PADDING = 5

# Number of slices to sample before deciding whether to skip the series
SAMPLE_SLICE_COUNT = 3


# ─────────────────────────────────────────────────────────────────────────────
#  OCR reader
# ─────────────────────────────────────────────────────────────────────────────

def load_ocr_reader():
    """
    Load and return an EasyOCR reader for English text detection.
    Call once per pipeline run and pass the reader to all scrub calls.
    GPU is used automatically if available, otherwise falls back to CPU.
    """
    import easyocr
    print("[INFO] Loading EasyOCR reader ...")
    reader = easyocr.Reader(["en"], verbose=False)
    print("[INFO] EasyOCR reader loaded.")
    return reader


# ─────────────────────────────────────────────────────────────────────────────
#  Pixel array helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_window_params(ds: pydicom.Dataset) -> tuple[float, float]:
    """
    Extract WindowCenter and WindowWidth from a DICOM dataset.
    Returns (window_center, window_width).
    Falls back to (127, 256) — full 8-bit range — if tags are absent.
    """
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth",  None)

    if wc is None or ww is None:
        return 127.0, 256.0

    if isinstance(wc, pydicom.multival.MultiValue):
        wc = float(wc[0])
    else:
        wc = float(wc)

    if isinstance(ww, pydicom.multival.MultiValue):
        ww = float(ww[0])
    else:
        ww = float(ww)

    return wc, ww


def _pixel_array_to_8bit(
    pixel_array: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """
    Convert a raw DICOM pixel array to an 8-bit greyscale image suitable
    for OCR input, using the standard DICOM linear windowing transform.

    Formula (DICOM PS3.3 C.7.6.3.1.2):
        output = 255 * (input - (wc - ww/2)) / ww
        clipped to [0, 255]

    Parameters
    ----------
    pixel_array : np.ndarray
        Raw pixel values (any integer dtype, any bit depth).
    window_center : float
    window_width : float

    Returns
    -------
    np.ndarray
        uint8 array, same spatial dimensions as input.
    """
    low  = window_center - window_width / 2.0
    high = window_center + window_width / 2.0

    arr = pixel_array.astype(np.float32)
    arr = (arr - low) / (high - low) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def has_pixel_data(ds: pydicom.Dataset) -> bool:
    """
    True if this dataset actually carries a pixel-data element (Pixel
    Data, Float Pixel Data, or Double Float Pixel Data — the three tags
    ds.pixel_array can decode).

    Many valid DICOM SOP classes carry no pixel data at all —
    Presentation States, Structured Reports, Key Object Selection
    documents, RT Structure Sets/Plans, spatial registrations,
    encapsulated PDFs, waveforms, etc. — and a study can legitimately
    mix these in with real image series. Checking for the tag directly
    is the correct, future-proof test: it works for any object type
    (present or yet to be invented) with no allowlist of non-image SOP
    Class UIDs to maintain, unlike trying to decode pixels and catching
    the failure (noisy — one warning per instance) or keying off
    Modality/SOPClassUID (an extra lookup that still needs a list).
    """
    return any(tag in ds for tag in ("PixelData", "FloatPixelData", "DoubleFloatPixelData"))


def _get_pixel_frames(ds: pydicom.Dataset) -> list[np.ndarray]:
    """
    Return a list of 2D pixel arrays — one per frame.
    Single-frame DICOM returns a list of length 1.
    Multi-frame DICOM (e.g. ultrasound cine) returns one array per frame.
    """
    arr = ds.pixel_array  # shape: (rows, cols) or (frames, rows, cols)
    if arr.ndim == 2:
        return [arr]
    elif arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    else:
        return [arr]


# ─────────────────────────────────────────────────────────────────────────────
#  Bounding box helpers
# ─────────────────────────────────────────────────────────────────────────────

def _expand_bbox(
    bbox: list,
    padding: int,
    max_h: int,
    max_w: int,
) -> tuple[int, int, int, int]:
    """
    Convert EasyOCR bounding box format to (row_min, row_max, col_min, col_max)
    with padding applied and clipped to image bounds.

    EasyOCR bbox format: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
    where (x,y) = (col,row) in image coordinates.
    """
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]

    col_min = max(0,     int(min(xs)) - padding)
    col_max = min(max_w, int(max(xs)) + padding)
    row_min = max(0,     int(min(ys)) - padding)
    row_max = min(max_h, int(max(ys)) + padding)

    return row_min, row_max, col_min, col_max


# ─────────────────────────────────────────────────────────────────────────────
#  Core scrub functions
# ─────────────────────────────────────────────────────────────────────────────

def _detect_text_regions(
    image_8bit: np.ndarray,
    reader,
) -> list[tuple[int, int, int, int]]:
    """
    Run EasyOCR on a full 8-bit greyscale image and return a list of
    bounding boxes for all detections above CONFIDENCE_THRESHOLD.

    Returns
    -------
    list of (row_min, row_max, col_min, col_max) tuples
    """
    h, w = image_8bit.shape[:2]

    results = reader.readtext(image_8bit, detail=1)

    boxes = []
    for (bbox, text, confidence) in results:
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        if not text.strip():
            continue
        row_min, row_max, col_min, col_max = _expand_bbox(
            bbox, BBOX_PADDING, h, w
        )
        boxes.append((row_min, row_max, col_min, col_max))

    return boxes


def _zero_fill_regions(
    pixel_frame: np.ndarray,
    regions: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Zero-fill (black rectangle) each detected text region in the pixel array.

    Parameters
    ----------
    pixel_frame : np.ndarray
        2D pixel array (original bit depth — not the 8-bit display image).
    regions : list of (row_min, row_max, col_min, col_max)

    Returns
    -------
    np.ndarray
        Modified pixel array with text regions zeroed.
    """
    result = pixel_frame.copy()
    for (row_min, row_max, col_min, col_max) in regions:
        result[row_min:row_max, col_min:col_max] = 0
    return result


def _write_pixel_frames_back(
    ds: pydicom.Dataset,
    frames: list[np.ndarray],
) -> None:
    """
    Write modified pixel frames back to the DICOM dataset's PixelData.
    Handles both single-frame and multi-frame cases.

    If the source file uses a COMPRESSED transfer syntax (JPEG2000,
    JPEG-LS, RLE, etc. — common for MR, less so for CT, hence this only
    showing up on some modalities), ds.pixel_array already transparently
    decompressed it to build `frames` — but writing that decompressed
    data back as PixelData while file_meta.TransferSyntaxUID still
    claims "compressed" produces a dataset pydicom's writer correctly
    rejects ("hasn't been encapsulated as required for a compressed
    transfer syntax"). Re-encoding to the original codec would need a
    codec-specific encoder we don't have (and don't want to add just for
    this); instead, switch to an uncompressed transfer syntax and write
    native pixel bytes — always writable, and a strictly safe operation
    for a de-identification pipeline (larger files, but identical pixel
    values; NIfTI conversion and everything downstream reads pixel
    values, not the transfer syntax).

    Only touches the transfer syntax when pixel data actually changed —
    an instance with no burned-in text detected keeps its original
    (possibly compressed) encoding untouched, so the vast majority of
    files aren't affected by this at all.
    """
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        from pydicom.uid import ExplicitVRLittleEndian
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = False

    if len(frames) == 1:
        ds.PixelData = frames[0].tobytes()
    else:
        # Stack frames back into (frames, rows, cols) and flatten
        stacked = np.stack(frames, axis=0)
        ds.PixelData = stacked.tobytes()

    # Ensure the dataset knows pixel data has changed
    if hasattr(ds, "_pixel_array_object"):
        del ds._pixel_array_object


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def sample_series_for_burned_in(
    sample_datasets: list[pydicom.Dataset],
    reader,
) -> bool:
    """
    Run OCR on up to SAMPLE_SLICE_COUNT datasets and return True if any
    burned-in text is detected in any of them.

    Used to decide whether to run full OCR on the entire series or skip it.

    Parameters
    ----------
    sample_datasets : list[pydicom.Dataset]
        First N fully-loaded datasets from the series.
    reader :
        EasyOCR reader from load_ocr_reader().

    Returns
    -------
    bool
        True if any text detected in the sample — process full series.
        False if no text found — skip remaining slices.
    """
    for ds in sample_datasets[:SAMPLE_SLICE_COUNT]:
        try:
            wc, ww = _get_window_params(ds)
            frames = _get_pixel_frames(ds)
            for frame in frames:
                img_8bit = _pixel_array_to_8bit(frame, wc, ww)
                regions = _detect_text_regions(img_8bit, reader)
                if regions:
                    return True
        except Exception:
            # If pixel reading fails on sample, assume text may be present
            return True
    return False


def scrub_pixels(
    ds: pydicom.Dataset,
    reader,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    force_uncompressed: bool = False,
) -> bool:
    """
    Detect and zero-fill burned-in text annotations in a DICOM instance.

    Modifies ds.PixelData in-place. If no text is detected, the dataset
    is unchanged UNLESS force_uncompressed is set.

    Parameters
    ----------
    ds : pydicom.Dataset
        Fully loaded DICOM dataset (with pixel data).
    reader :
        EasyOCR reader from load_ocr_reader().
    window_center : float, optional
        If provided, used for 8-bit conversion. If None, extracted from
        ds.WindowCenter.
    window_width : float, optional
        If provided, used for 8-bit conversion. If None, extracted from
        ds.WindowWidth.
    force_uncompressed : bool, optional
        If True, normalize to an uncompressed transfer syntax even when
        THIS instance has nothing to scrub. The scrub/no-scrub decision
        is normally made per-series (sample_series_for_burned_in), so
        some instances in a scrubbed series may individually have
        nothing detected — without this, those would keep their
        original (possibly compressed) transfer syntax while their
        scrubbed siblings become uncompressed, leaving the series with
        MIXED transfer syntaxes that tools expecting one consistent
        encoding per series (e.g. dicom2nifti) can mishandle. Pass
        force_uncompressed=run_ocr from the caller so a whole series
        that gets scrubbed comes out consistently encoded.

    Returns
    -------
    bool
        True if any text regions were found and zeroed (independent of
        whether force_uncompressed also normalized the encoding).
        False if the image was clean.
    """
    # Use provided window params or extract from dataset
    if window_center is None or window_width is None:
        wc, ww = _get_window_params(ds)
    else:
        wc, ww = float(window_center), float(window_width)

    try:
        frames = _get_pixel_frames(ds)
    except Exception as e:
        print(f"  [WARN] Could not read pixel data: {e}")
        return False

    any_scrubbed = False
    modified_frames = []

    for frame in frames:
        # Convert to 8-bit for OCR
        img_8bit = _pixel_array_to_8bit(frame, wc, ww)

        # Detect text regions on full frame
        regions = _detect_text_regions(img_8bit, reader)

        if regions:
            # Zero-fill in the original (pre-window) pixel array
            frame = _zero_fill_regions(frame, regions)
            any_scrubbed = True

        modified_frames.append(frame)

    if any_scrubbed or force_uncompressed:
        _write_pixel_frames_back(ds, modified_frames)

    return any_scrubbed