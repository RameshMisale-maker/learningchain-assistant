"""Document loaders for PDF, DOCX, TXT/MD, CSV, XLSX and JSON.

Each loader returns a `LoadedDocument` made of ordered `Section`s:
  * kind="text"   - prose with an optional page number / heading
  * kind="table"  - a table rendered as Markdown (kept intact during chunking)
  * kind="record" - one structured row (CSV/XLSX/JSON record) kept as a single chunk

Scanned PDFs: pages with almost no extractable text are OCR'd with Tesseract
(via pytesseract) when OCR_ENABLED=true; otherwise they are flagged with
`needs_ocr=True` so administrators can see what was skipped.
"""
from __future__ import annotations

import re

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".json", ".html", ".htm"}


@dataclass
class Section:
    text: str
    kind: str = "text"            # text | table | record
    page: int | None = None
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    path: Path
    name: str
    file_type: str
    sections: list[Section]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


class UnsupportedDocumentError(ValueError):
    """Raised for file types the loaders do not support."""


def detect_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(f"Unsupported file type '{ext}' for {path.name}")
    return ext.lstrip(".")


def load_document(path: str | Path) -> LoadedDocument:
    path = Path(path)
    file_type = detect_type(path)
    loader = {
        "pdf": _load_pdf, "docx": _load_docx, "txt": _load_text, "md": _load_text,
        "csv": _load_csv, "xlsx": _load_xlsx, "json": _load_json, "html": _load_html, "htm": _load_html,
    }[file_type]
    doc = loader(path)
    doc.file_type = file_type
    if not any(s.text.strip() for s in doc.sections):
        doc.warnings.append("No extractable text found")
    return doc


# ------------------------------------------------------------------ PDF
def _table_to_markdown(rows: list[list[Any]]) -> str:
    clean = [["" if c is None else str(c).replace("\n", " ").strip() for c in row] for row in rows if row]
    if not clean:
        return ""
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    header, body = clean[0], clean[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _ocr_page(page) -> str:
    settings = get_settings()
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning("OCR requested but pytesseract/Pillow is not installed")
        return ""
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    pix = page.get_pixmap(dpi=300)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(image, lang=settings.ocr_language)
    except Exception as exc:  # tesseract binary missing / language pack missing
        logger.warning("OCR failed on page %s: %s", page.number + 1, exc)
        return ""


def _pdf_table_rows(page, table, fitz) -> list[list[str]]:
    """Read each cell via its (slightly shrunken) rectangle - keeps word spacing intact."""
    rows: list[list[str]] = []
    for row in table.rows:
        rows.append([page.get_textbox(fitz.Rect(cell) + (0, 2, 0, -2)).strip() if cell else "" for cell in row.cells])
    if not any(any(c for c in r) for r in rows):
        return table.extract()
    return rows


def _load_pdf(path: Path) -> LoadedDocument:
    try:
        import pymupdf as fitz
    except ImportError:  # older PyMuPDF releases
        import fitz

    settings = get_settings()
    sections: list[Section] = []
    warnings: list[str] = []
    with fitz.open(path) as pdf:
        meta = {k: v for k, v in (pdf.metadata or {}).items() if v}
        for page in pdf:
            page_no = page.number + 1
            table_texts: list[str] = []
            table_rects = []
            try:
                for table in page.find_tables().tables:
                    md = _table_to_markdown(_pdf_table_rows(page, table, fitz))
                    if md:
                        table_texts.append(md)
                        table_rects.append(fitz.Rect(table.bbox))
            except Exception as exc:  # table detection is best-effort
                logger.debug("Table extraction failed on %s p%s: %s", path.name, page_no, exc)
            if table_rects:
                blocks = page.get_text("blocks")
                text = "\n".join(b[4] for b in blocks
                                 if not any(fitz.Rect(b[:4]).intersects(r) for r in table_rects))
            else:
                text = page.get_text("text")
            needs_ocr = len(text.strip()) < 25 and not table_texts
            if needs_ocr and page.get_images():
                if settings.ocr_enabled:
                    text = _ocr_page(page)
                    if text.strip():
                        warnings.append(f"Page {page_no} extracted with OCR")
                else:
                    warnings.append(f"Page {page_no} appears scanned; enable OCR_ENABLED to extract it")
            if text.strip():
                sections.append(Section(text=text, page=page_no, metadata={"ocr": needs_ocr}))
            for md in table_texts:
                sections.append(Section(text=md, kind="table", page=page_no))
    return LoadedDocument(path=path, name=path.name, file_type="pdf", sections=sections,
                          metadata={"title": meta.get("title", ""), "author": meta.get("author", ""),
                                    "pdf_metadata": meta}, warnings=warnings)


# ------------------------------------------------------------------ DOCX
def _load_docx(path: Path) -> LoadedDocument:
    from docx import Document as DocxDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    docx = DocxDocument(str(path))
    sections: list[Section] = []
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            sections.append(Section(text="\n".join(buffer), section=current_heading))
            buffer.clear()

    for child in docx.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, docx)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            if style.lower().startswith("heading") or style.lower() == "title":
                flush()
                current_heading = text
                level = "".join(ch for ch in style if ch.isdigit()) or "1"
                buffer.append("#" * min(int(level), 4) + " " + text)
            elif style.lower().startswith("list"):
                buffer.append("- " + text)
            else:
                buffer.append(text)
        elif tag == "tbl":
            flush()
            table = Table(child, docx)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            md = _table_to_markdown(rows)
            if md:
                sections.append(Section(text=md, kind="table", section=current_heading))
    flush()
    props = docx.core_properties
    meta = {"title": props.title or "", "author": props.author or "", "subject": props.subject or ""}
    return LoadedDocument(path=path, name=path.name, file_type="docx", sections=sections, metadata=meta)


# ------------------------------------------------------------------ TXT / MD
def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _load_text(path: Path) -> LoadedDocument:
    text = _read_text_file(path)
    return LoadedDocument(path=path, name=path.name, file_type=path.suffix.lstrip("."),
                          sections=[Section(text=text)])


def _load_html(path: Path) -> LoadedDocument:
    """HTML page/export: main content only (navigation, scripts, ads removed), headings and tables kept as Markdown
    so structure-aware chunking still splits on the page's own headings."""
    from ingestion.web import extract_main_content, page_to_document

    raw = _read_text_file(path)
    page = extract_main_content(raw)
    url_match = re.search(r"""<link[^>]+rel=["']canonical["'][^>]+href=["']([^"']+)""", raw, re.IGNORECASE)
    text = page_to_document(page, url_match.group(1) if url_match else path.name)
    return LoadedDocument(path=path, name=path.name, file_type="html", sections=[Section(text=text)])


# ------------------------------------------------------------------ tabular
def _record_text(record: dict[str, Any]) -> str:
    lines = []
    for key, value in record.items():
        if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
            continue
        lines.append(f"{str(key).replace('_', ' ').strip().title()}: {value}")
    return "\n".join(lines)


def records_to_sections(records: list[dict[str, Any]], section_name: str = "") -> list[Section]:
    sections = []
    for idx, rec in enumerate(records, start=1):
        clean = {str(k): (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in rec.items()}
        text = _record_text(clean)
        if text:
            sections.append(Section(text=text, kind="record", section=section_name or f"Row {idx}",
                                    metadata={"row": idx, "record": clean}))
    return sections


def read_tabular(path: Path) -> dict[str, pd.DataFrame]:
    """Return {sheet_name: DataFrame} for CSV/XLSX (CSV has a single 'Sheet1')."""
    if path.suffix.lower() == ".csv":
        text = _read_text_file(path)
        return {"Sheet1": pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)}
    return pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)


def _load_csv(path: Path) -> LoadedDocument:
    frames = read_tabular(path)
    df = frames["Sheet1"]
    return LoadedDocument(path=path, name=path.name, file_type="csv",
                          sections=records_to_sections(df.to_dict(orient="records")),
                          metadata={"columns": list(df.columns), "rows": len(df)})


def _load_xlsx(path: Path) -> LoadedDocument:
    sections: list[Section] = []
    sheets = read_tabular(path)
    for sheet_name, df in sheets.items():
        df = df.dropna(how="all")
        for sec in records_to_sections(df.to_dict(orient="records"), section_name=""):
            sec.section = f"{sheet_name} / Row {sec.metadata['row']}"
            sec.metadata["sheet"] = sheet_name
            sections.append(sec)
    return LoadedDocument(path=path, name=path.name, file_type="xlsx", sections=sections,
                          metadata={"sheets": list(sheets)})


def _flatten_json(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            lines += _flatten_json(v, f"{prefix}{k}." if not isinstance(v, (str, int, float, bool)) else f"{prefix}{k}")
    elif isinstance(value, list):
        if all(isinstance(v, (str, int, float, bool)) for v in value):
            lines.append(f"{prefix.rstrip('.')}: {', '.join(str(v) for v in value)}")
        else:
            for i, v in enumerate(value, 1):
                lines += _flatten_json(v, f"{prefix}{i}.")
    elif value is not None:
        lines.append(f"{prefix.rstrip('.')}: {value}")
    return lines


def _load_json(path: Path) -> LoadedDocument:
    data = json.loads(_read_text_file(path))
    records: list[dict[str, Any]] | None = None
    meta: dict[str, Any] = {}
    if isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
        records = data
    elif isinstance(data, dict):
        list_keys = [k for k, v in data.items() if isinstance(v, list) and v and all(isinstance(r, dict) for r in v)]
        if len(list_keys) == 1:
            records = data[list_keys[0]]
            meta = {k: v for k, v in data.items() if k != list_keys[0] and isinstance(v, (str, int, float, bool))}
    if records is not None:
        flat_records = [{k: (", ".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items()}
                        for r in records]
        sections = records_to_sections(flat_records)
        for sec, original in zip(sections, records):
            title = original.get("title") or original.get("name") or original.get("question")
            if title:
                sec.section = str(title)[:120]
    else:
        sections = [Section(text="\n".join(_flatten_json(data)))]
    return LoadedDocument(path=path, name=path.name, file_type="json", sections=sections, metadata=meta)
