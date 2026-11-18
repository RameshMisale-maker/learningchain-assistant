"""Secure validation of uploaded knowledge documents.

Checks: extension allow-list, size limit, safe file name, magic bytes / structure per type,
zip-bomb protection for DOCX/XLSX, text decodability for TXT/MD/CSV/JSON and macro-free Office files.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass

from config.settings import get_settings
from ingestion.document_loader import SUPPORTED_EXTENSIONS

MAX_UNCOMPRESSED_RATIO = 100
MAX_UNCOMPRESSED_BYTES = 300 * 1024 * 1024


class UploadValidationError(ValueError):
    """Raised when an uploaded file fails validation (message is shown to the administrator)."""


@dataclass
class ValidatedUpload:
    filename: str
    extension: str
    size: int


def safe_filename(name: str) -> str:
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._\- ]+", "_", name).strip(" .")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    if not name or name.startswith(("_", ".")):
        name = "upload" + (name if name.startswith(".") else "_" + name)
    return name[:150]


def _check_zip(content: bytes, required: str, forbidden_prefix: tuple[str, ...]) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            if "[Content_Types].xml" not in names or not any(n.startswith(required) for n in names):
                raise UploadValidationError("File is not a valid Office document")
            if any(n.lower().startswith(forbidden_prefix) or n.lower().endswith("vbaproject.bin") for n in names):
                raise UploadValidationError("Macro-enabled or embedded-binary Office files are not allowed")
            total = sum(i.file_size for i in zf.infolist())
            if total > MAX_UNCOMPRESSED_BYTES or total > max(len(content), 1) * MAX_UNCOMPRESSED_RATIO:
                raise UploadValidationError("Archive expands too much (possible zip bomb)")
    except zipfile.BadZipFile as exc:
        raise UploadValidationError("Corrupted Office document") from exc


def _check_text(content: bytes) -> str:
    if b"\x00" in content[:4096]:
        raise UploadValidationError("Binary content is not allowed in a text file")
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UploadValidationError("Text file encoding not recognised")


def validate_upload(filename: str, content: bytes) -> ValidatedUpload:
    settings = get_settings()
    name = safe_filename(filename or "")
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError(f"File type '{ext or 'unknown'}' is not allowed. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    size = len(content)
    if size == 0:
        raise UploadValidationError("File is empty")
    if size > settings.max_upload_mb * 1024 * 1024:
        raise UploadValidationError(f"File exceeds the {settings.max_upload_mb} MB limit")
    if ext == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise UploadValidationError("File content is not a PDF")
        if re.search(rb"/(JavaScript|JS|Launch|EmbeddedFile)\b", content[:5_000_000]):
            raise UploadValidationError("PDF contains active content (JavaScript/Launch/embedded files)")
    elif ext == ".docx":
        _check_zip(content, "word/", ("word/activex", "word/embeddings/oleobject"))
    elif ext == ".xlsx":
        _check_zip(content, "xl/", ("xl/activex", "xl/embeddings/oleobject"))
    elif ext == ".json":
        try:
            json.loads(_check_text(content))
        except json.JSONDecodeError as exc:
            raise UploadValidationError(f"Invalid JSON: {exc.msg}") from exc
    elif ext in (".html", ".htm"):
        text = _check_text(content)
        if not re.search(r"<\s*(html|body|div|p|h[1-6]|article|main|section)\b", text[:200_000], re.IGNORECASE):
            raise UploadValidationError("File content is not an HTML document")
        # scripts are never executed (text is extracted server-side), but reject obvious active payloads anyway
        if re.search(r"<\s*(object|embed|applet)\b", text, re.IGNORECASE):
            raise UploadValidationError("HTML contains embedded active content (object/embed/applet)")
    else:  # .txt .md .csv
        _check_text(content)
    return ValidatedUpload(filename=name, extension=ext, size=size)
