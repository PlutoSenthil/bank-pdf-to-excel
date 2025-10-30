# app_streamlit.py
# Bank PDF -> Excel via Camelot (lattice) using your exact logic, wrapped in Streamlit UI.
# Features:
# - Upload PDF
# - Preview first N pages (1–3) as images
# - Enter page ranges like "1-50" or "all"
# - Convert -> Download Excel
# - Cr/Dr + commas + whitespace handling (same as your notebook)
# - Excel formatting: freeze top row, auto-filter, auto column width
# - Clean logging and clear error messages
# - Streamlit deprecation fixed (use width='stretch'), Arrow JSON-safe preview
# - UI tweaks:
#     - Convert button appears after preview
#     - Preview count default = 1
#     - Logs hidden by default, shown only if content exists
#     - After Convert: per-page extraction summary (tables/rows/cols), totals, and warnings

import io
import os
import re
import sys
import time
import tempfile
import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import camelot
import fitz  # PyMuPDF
from PIL import Image
import streamlit as st


# ----------------------------
# Logging setup (buffered)
# ----------------------------
log_stream = io.StringIO()
logger = logging.getLogger("bank_pdf_to_excel")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(log_stream)
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
# Avoid duplicate handlers on reruns
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(handler)


# ----------------------------
# Utility functions (your logic)
# ----------------------------
def parse_pages_arg(pages: str) -> List[int]:
    """
    Parse a pages string like "3,5-7" into a sorted list: [3,5,6,7].
    """
    pgs = set()
    for part in pages.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            for x in range(int(a), int(b) + 1):
                pgs.add(x)
        else:
            pgs.add(int(part))
    return sorted(pgs)


def clean_and_convert_to_numeric(series: pd.Series) -> pd.Series:
    """
    Clean Debits/Credits-like columns:
    - strip, remove commas
    - remove trailing Cr/Dr (case-insensitive)
    - convert to numeric
    """
    series = series.astype(str).str.strip()
    series = series.str.replace(",", "", regex=False)
    series = series.str.replace(r"\s*(Cr|Dr)\s*$", "", regex=True, flags=re.IGNORECASE)
    return pd.to_numeric(series, errors="coerce")


def detect_scanned_pages(pdf_path: str, pages_list: List[int]) -> Tuple[List[int], int]:
    """
    Uses PyMuPDF to check if pages likely scanned (by text length).
    Returns (scanned_pages, total_pages)
    """
    doc = fitz.open(pdf_path)
    total = len(doc)
    scanned = []
    for p in pages_list:
        if not (1 <= p <= total):
            continue
        page_idx = p - 1
        text_len = len((doc.load_page(page_idx).get_text("text") or "").strip())
        if text_len < 60:
            scanned.append(p)
    return scanned, total


def extract_tables_with_meta(pdf_path: str, pages_str: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    EXACT SAME Camelot parameters as before; we only add meta instrumentation.
    Returns:
        - concatenated DataFrame across all tables
        - meta dict with per-page stats and timings
    """
    meta: Dict[str, Any] = {
        "pages_str": pages_str,
        "tables_count": 0,
        "per_page": {},   # page -> {"tables": int, "shapes": [(r,c), ...]}
        "duration_sec": 0.0,
        "warnings": [],
        "info": [],
    }

    logger.info(f"Attempting to read tables from: {pdf_path}")
    t0 = time.time()
    try:
        tables = camelot.read_pdf(
            pdf_path,
            pages=pages_str,
            flavor="lattice",
            table_areas=None,   # auto-detect
            strip_text="\n",
            line_scale=40,
            use_fallback=True,
            joint_tol=3,
            line_tol=2,
            backend="ghostscript",
        )
        if not tables or tables.n == 0:
            w = "No tables detected by Camelot. PDF may be scanned (image)."
            logger.warning(w)
            meta["warnings"].append(w)
            meta["duration_sec"] = time.time() - t0
            return pd.DataFrame(), meta
    except Exception as e:
        err = f"Camelot extraction error: {e}"
        logger.error(err)
        if "No module named 'ghostscript'" in str(e):
            gs_msg = "Ghostscript is required by Camelot lattice backend but missing."
            logger.error(gs_msg)
            meta["warnings"].append(gs_msg)
        meta["warnings"].append(str(e))
        meta["duration_sec"] = time.time() - t0
        return pd.DataFrame(), meta

    meta["duration_sec"] = time.time() - t0
    meta["tables_count"] = tables.n
    logger.info(f"Detected {tables.n} table(s) in document.")

    all_dataframes = []
    # Collect per-page stats
    per_page: Dict[int, Dict[str, Any]] = {}

    for i, table in enumerate(tables):
        df = table.df
        all_dataframes.append(df)

        # table.page may be int or str depending on version; coerce to int when possible
        try:
            page_no = int(getattr(table, "page", None) or table.parsing_report.get("page", 0) or 0)
        except Exception:
            page_no = 0

        shape = (df.shape[0], df.shape[1])
        page_bucket = per_page.setdefault(page_no, {"tables": 0, "shapes": []})
        page_bucket["tables"] += 1
        page_bucket["shapes"].append(shape)

        logger.info(f"Processed table {i+1} (page {page_no}). Shape: {df.shape}")

    meta["per_page"] = per_page

    if not all_dataframes:
        return pd.DataFrame(), meta

    final_df = pd.concat(all_dataframes, ignore_index=True)
    return final_df, meta


def finalize_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same post-processing steps as your notebook:
    - drop duplicates
    - replace '' with NA
    - drop empty rows
    - use first row as header
    - convert numeric-ish columns (Debit/Credit/Balance/Withdrawal/Deposit)
    """
    if raw_df.empty:
        return raw_df

    df = raw_df.copy()
    df = df.drop_duplicates(keep="first")
    df = df.replace("", pd.NA)
    df = df.dropna(axis=0, how="all")

    # Guard: ensure we have at least 1 row for header
    if len(df) == 0:
        return df

    # First row is header
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)

    # Convert numeric columns by name contains any of these keywords
    numeric_cols = ["Debit", "Credit", "Balance", "Withdrawal", "Deposit"]
    for col in df.columns:
        if any(word.lower() in str(col).lower() for word in numeric_cols):
            df.loc[:, col] = clean_and_convert_to_numeric(df[col])

    return df


def df_to_excel_bytes(
    df: pd.DataFrame,
    sheet_name: str = "Sheet1",
    freeze_panes: Tuple[int, int] = (1, 0),
    add_autofilter: bool = True,
    min_width: int = 8,
    max_width: int = 60,
    padding: int = 2,
) -> bytes:
    """
    Write DataFrame to Excel in-memory with:
    - freeze top row
    - auto-filter
    - auto column widths based on content length
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # Freeze panes (1st row)
        if freeze_panes:
            worksheet.freeze_panes(*freeze_panes)

        # Auto filter on header row
        if add_autofilter and not df.empty:
            last_row = len(df)
            last_col = len(df.columns) - 1
            worksheet.autofilter(0, 0, last_row, last_col)

        # Auto-fit widths (approximate, based on string length)
        for idx, col in enumerate(df.columns):
            series_as_str = df[col].astype(str).replace("nan", "")
            max_len = max([len(str(col))] + [len(x) for x in series_as_str.tolist()]) + padding
            width = max(min_width, min(max_len, max_width))
            worksheet.set_column(idx, idx, width)

    output.seek(0)
    return output.read()


def render_page_image(pdf_path: str, page_number: int, dpi: int = 150) -> Image.Image:
    """
    Render a PDF page to a PIL Image using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    if not (1 <= page_number <= len(doc)):
        raise ValueError(f"Page {page_number} out of range (1..{len(doc)})")
    page = doc.load_page(page_number - 1)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def make_display_safe(df: pd.DataFrame, preview_rows: int = 50) -> pd.DataFrame:
    """
    Return a preview DataFrame that is safe for Streamlit/PyArrow JSON metadata:
    - Ensure column names are strings
    - Convert preview rows to strings (display only)
    """
    if df is None or df.empty:
        return df
    preview = df.head(preview_rows).copy()
    preview.columns = [str(c) for c in preview.columns]
    preview = preview.astype(str)
    return preview
def get_log_text() -> str:
    """Return current logs text (if any)."""
    return log_stream.getvalue().strip()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PDF → Excel (Camelot)", layout="centered")

st.title("PDF → Excel")
st.caption("Camelot (lattice) + PyMuPDF • Streamlit UI")

with st.expander("How it works", expanded=False):
    st.markdown(
        """
- **Upload** a bank statement PDF.
- **Preview** the first 1–3 pages as images.
- Enter **pages** like `1-50`, `3,5-7`, or `all`.
- Click **Convert** to generate an Excel with:
  - Top row frozen
  - Header auto-filter
  - Column widths auto-fit to content
- **Cr/Dr**, commas, and whitespace in numeric-like columns are **cleaned** exactly as in your notebook.
        """
    )

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

# Default preview_count set to 1 per your request
preview_count = st.number_input("Preview pages", min_value=1, max_value=3, value=1, step=1)

pages_input_value = "all"  # will render input after preview
convert_clicked = False

if uploaded:
    # Persist the upload to a temp file so libraries can read it by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name

    # Basic PDF info
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        st.info(f"Detected **{total_pages}** page(s) in the PDF.")
    except Exception as e:
        st.error(f"Could not open PDF: {e}")
        logs = get_log_text()
        if logs:
            with st.expander("Logs (optional)", expanded=False):
                st.code(logs or "Logs will appear here...", language="text")
        st.stop()

    # ----- Preview first N pages -----
    st.subheader("Preview")
    pages_to_show = list(range(1, min(preview_count, total_pages) + 1))
    for p in pages_to_show:
        try:
            img = render_page_image(pdf_path, p, dpi=150)
            st.image(img, caption=f"Page {p}", width='stretch')
        except Exception as e:
            st.warning(f"Preview failed for page {p}: {e}")

    st.markdown("---")

    # ----- Pages input and Convert button placed **after** preview -----
    pages_input_value = st.text_input("Pages to extract (e.g., 1-50, 3,5-7, or all)", value="all")
    convert_clicked = st.button("Convert")

    if convert_clicked:
        with st.spinner("Converting…"):
            # Resolve pages_str for Camelot
            if pages_input_value.strip().lower() == "all":
                pages_list = list(range(1, total_pages + 1))
                pages_str = "all"
            else:
                try:
                    pages_list = parse_pages_arg(pages_input_value)
                    pages_list = [p for p in pages_list if 1 <= p <= total_pages]
                    if not pages_list:
                        st.error("No valid page numbers in range.")
                        logs = get_log_text()
                        if logs:
                            with st.expander("Logs (optional)", expanded=False):
                                st.code(logs or "Logs will appear here...", language="text")
                        st.stop()
                    pages_str = ",".join(map(str, pages_list))
                except Exception as e:
                    st.error(f"Invalid pages input: {e}")
                    logs = get_log_text()
                    if logs:
                        with st.expander("Logs (optional)", expanded=False):
                            st.code(logs or "Logs will appear here...", language="text")
                    st.stop()

            # Scanned page detection (like your assert logic)
            try:
                pages_for_scan = pages_list if pages_input_value.strip().lower() != "all" else list(range(1, total_pages + 1))
                scanned_pages, total = detect_scanned_pages(pdf_path, pages_for_scan)
                if scanned_pages:
                    st.error(
                        f"⚠️ These pages may be scanned (very little text detected): {scanned_pages}\n"
                        f"Camelot (lattice) will likely fail on scanned images. "
                        f"Use OCR (e.g., Tesseract) first or upload a text-based PDF."
                    )
                    logs = get_log_text()
                    if logs:
                        with st.expander("Logs (optional)", expanded=False):
                            st.code(logs or "Logs will appear here...", language="text")
                    st.stop()
            except Exception as e:
                logger.warning(f"Scan detection skipped due to error: {e}")

            # Extract tables with metadata (same Camelot logic, extra instrumentation only)
            raw_df, meta = extract_tables_with_meta(pdf_path, pages_str)

            if raw_df.empty:
                st.error("No tables extracted. If the PDF is scanned, please OCR it first.")
                # Show summary block if any meta exists
                with st.expander("Extraction summary (debug)", expanded=False):
                    st.markdown(f"**Pages requested:** `{meta.get('pages_str','')}`")
                    st.markdown(f"**Tables detected:** `{meta.get('tables_count',0)}`")
                    st.markdown(f"**Extraction time:** `{meta.get('duration_sec',0):.2f}s`")
                    if meta.get("per_page"):
                        for pg, info in meta["per_page"].items():
                            st.markdown(f"- Page **{pg}** → tables: {info.get('tables',0)}, shapes: {info.get('shapes',[])}")
                    if meta.get("warnings"):
                        st.warning("Warnings:\n\n- " + "\n- ".join(meta["warnings"]))
                    if meta.get("info"):
                        st.info("Info:\n\n- " + "\n- ".join(meta["info"]))
                logs = get_log_text()
                if logs:
                    with st.expander("Logs (optional)", expanded=False):
                        st.code(logs or "Logs will appear here...", language="text")
                st.stop()

            # Finalize DF (same steps as notebook)
            final_df = finalize_dataframe(raw_df)
            final_df.columns = [str(c) for c in final_df.columns]  # ensure headers are strings

            if final_df.empty:
                st.error("Conversion produced an empty table after cleaning.")
                # Show meta since raw had content
                with st.expander("Extraction summary (debug)", expanded=False):
                    st.markdown(f"**Pages requested:** `{meta.get('pages_str','')}`")
                    st.markdown(f"**Tables detected:** `{meta.get('tables_count',0)}`")
                    st.markdown(f"**Extraction time:** `{meta.get('duration_sec',0):.2f}s`")
                    if meta.get("per_page"):
                        for pg, info in meta["per_page"].items():
                            st.markdown(f"- Page **{pg}** → tables: {info.get('tables',0)}, shapes: {info.get('shapes',[])}")
                logs = get_log_text()
                if logs:
                    with st.expander("Logs (optional)", expanded=False):
                        st.code(logs or "Logs will appear here...", language="text")
                st.stop()

            st.success("✅ Conversion successful!")

            # ---- Extraction Summary (per-page rows/cols, totals) ----
            with st.expander("Extraction summary (debug)", expanded=True):
                # Meta from Camelot
                st.markdown(f"**Pages requested:** `{meta.get('pages_str','')}`")
                st.markdown(f"**Tables detected:** `{meta.get('tables_count',0)}`")
                st.markdown(f"**Extraction time:** `{meta.get('duration_sec',0):.2f}s`")

                # Per-page shapes
                if meta.get("per_page"):
                    st.markdown("**Per-page table shapes:**")
                    for pg in sorted(meta["per_page"].keys()):
                        info = meta["per_page"][pg]
                        st.markdown(f"- Page **{pg}** → tables: `{info.get('tables',0)}`, shapes: `{info.get('shapes',[])}`")
                else:
                    st.markdown("_No per-page breakdown available from Camelot._")

                # Totals before/after cleaning
                total_raw_rows = int(raw_df.shape[0]) if hasattr(raw_df, "shape") else 0
                total_raw_cols = int(raw_df.shape[1]) if hasattr(raw_df, "shape") else 0
                total_final_rows = int(final_df.shape[0]) if hasattr(final_df, "shape") else 0
                total_final_cols = int(final_df.shape[1]) if hasattr(final_df, "shape") else 0

                st.markdown("**Totals:**")
                st.markdown(f"- Raw concat DF: **{total_raw_rows}** rows × **{total_raw_cols}** cols")
                st.markdown(f"- Final DF (after header + cleaning): **{total_final_rows}** rows × **{total_final_cols}** cols")

                # Header preview & numeric columns detected
                st.markdown("**Detected headers (final):**")
                st.code(", ".join(map(str, list(final_df.columns))), language="text")

                numeric_cols = [c for c in final_df.columns if any(
                    w.lower() in str(c).lower() for w in ["Debit", "Credit", "Balance", "Withdrawal", "Deposit"]
                )]
                st.markdown(f"**Numeric-cleaned columns:** {numeric_cols if numeric_cols else 'None'}")

                # Any warnings/info captured
                if meta.get("warnings"):
                    st.warning("Warnings:\n\n- " + "\n- ".join(meta["warnings"]))
                if meta.get("info"):
                    st.info("Info:\n\n- " + "\n- ".join(meta["info"]))

            # ---- SAFE PREVIEW (stringified only for display) ----
            display_df = make_display_safe(final_df, preview_rows=50)
            st.dataframe(display_df, width='stretch')

            # Build Excel bytes with formatting (use original df)
            try:
                xlsx_bytes = df_to_excel_bytes(final_df)
            except Exception as e:
                st.error(f"Error while writing Excel: {e}")
                logs = get_log_text()
                if logs:
                    with st.expander("Logs (optional)", expanded=False):
                        st.code(logs or "Logs will appear here...", language="text")
                st.stop()

            out_name = os.path.splitext(uploaded.name)[0] + ".xlsx"
            st.download_button(
                "⬇️ Download Excel",
                data=xlsx_bytes,
                file_name=out_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        # After a successful run, show logs only if any
        logs = get_log_text()
        if logs:
            with st.expander("Logs (optional)", expanded=False):
                st.code(logs or "Logs will appear here...", language="text")

else:
    st.info("Upload a PDF to begin.")