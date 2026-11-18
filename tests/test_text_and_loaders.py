"""Document loaders (PDF, DOCX, XLSX, CSV, JSON, TXT, MD, HTML) and structure-aware chunking.

Test documents are generated with the same libraries real files come from, so every parser is exercised."""
from __future__ import annotations

import json

import pytest

from ingestion.chunking import chunk_document
from ingestion.document_loader import load_document


@pytest.fixture(scope="module")
def docs(tmp_path_factory):
    import docx
    import fitz
    import openpyxl

    d = tmp_path_factory.mktemp("docs")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "LearningChain Datasheet", fontsize=18)
    page.insert_text((72, 110), "Certificates are secured on blockchain and verified with QR codes.")
    page2 = pdf.new_page()
    page2.insert_text((72, 72), "Page two: bulk issuance supports CSV student data uploads.")
    pdf.save(d / "sheet.pdf")

    w = docx.Document()
    w.add_heading("Admin Handbook", 0)
    w.add_heading("Revoking certificates", 1)
    w.add_paragraph("Open the Reports module and select Generated Certificates. Choose the certificate type, find the "
                    "student in the list and select Revoke under Actions, then confirm with Go. A revoked certificate "
                    "can no longer be verified by employers or other third parties.")
    w.add_heading("Recalling certificates", 1)
    w.add_paragraph("Select the student and choose Recall under Actions. Recalling withdraws the issued certificate "
                    "so that it can be corrected and issued again; the student is informed by email and the old QR "
                    "code stops resolving once the recall has been processed by the platform.")
    t = w.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Role", "Can revoke"
    t.cell(1, 0).text, t.cell(1, 1).text = "Admin", "Yes"
    w.save(d / "handbook.docx")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["feature", "description"])
    ws.append(["QR verification", "Scan to verify a certificate instantly"])
    wb.save(d / "features.xlsx")

    (d / "faq.csv").write_text("question,answer\nWhich formats?,Only PDF is supported\n", encoding="utf-8")
    (d / "items.json").write_text(json.dumps([{"name": "Student wallet", "detail": "Share credentials"}]),
                                  encoding="utf-8")
    (d / "notes.txt").write_text("LearningChain stores data on Microsoft Azure.", encoding="utf-8")
    (d / "guide.md").write_text("# Guide\n\n## Login\n\nEnter username, password and captcha.\n", encoding="utf-8")
    (d / "page.html").write_text("<html><body><nav>Home</nav><article><h1>Verify</h1><h2>Steps</h2><p>Open the "
                                 "verification URL from the dashboard and upload the original certificate file."
                                 "</p></article><footer>c</footer></body></html>", encoding="utf-8")
    return d


@pytest.mark.parametrize("name,needle", [
    ("sheet.pdf", "verified with QR codes"), ("handbook.docx", "Recall under Actions"),
    ("features.xlsx", "Scan to verify"), ("faq.csv", "Only PDF"), ("items.json", "Share credentials"),
    ("notes.txt", "Microsoft Azure"), ("guide.md", "captcha"), ("page.html", "verification URL"),
])
def test_loaders_extract_text(docs, name, needle):
    doc = load_document(docs / name)
    text = " ".join(s.text for s in doc.sections)
    assert needle in text, f"{name}: {text[:300]}"


def test_pdf_pages_preserved(docs):
    doc = load_document(docs / "sheet.pdf")
    assert {s.page for s in doc.sections} >= {1, 2}


def test_docx_headings_and_tables_preserved(docs):
    doc = load_document(docs / "handbook.docx")
    text = "\n".join(s.text for s in doc.sections)
    assert "Revoking certificates" in text and "Recalling certificates" in text
    assert "Admin" in text and "Can revoke" in text


def test_html_removes_boilerplate_keeps_headings(docs):
    text = load_document(docs / "page.html").sections[0].text
    assert "## Steps" in text and "Home" not in text


def test_chunking_respects_headings_size_and_context_header(docs):
    doc = load_document(docs / "handbook.docx")
    chunks = chunk_document(doc, "doc1", chunk_size=200, chunk_overlap=40, title="Admin Handbook")
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(c.embed_text.startswith("Document: Admin Handbook") for c in chunks)
    assert any("Revoking" in c.section for c in chunks) and any("Recalling" in c.section for c in chunks)


def test_fixed_strategy_ignores_headings(docs):
    doc = load_document(docs / "guide.md")
    fixed = chunk_document(doc, "doc2", chunk_size=900, chunk_overlap=0, strategy="fixed")
    assert len(fixed) == 1


def test_parent_child_strategy_links_children_to_parent():
    from pathlib import Path
    from ingestion.document_loader import LoadedDocument, Section
    para = "LearningChain issues tamper-proof certificates on a blockchain. " * 6
    body = "\n\n".join(f"Paragraph {i}. {para}" for i in range(6))
    doc = LoadedDocument(path=Path("guide.md"), name="guide.md", file_type="md", sections=[Section(text=f"## Issuing certificates\n\n{body}", kind="text",
                                                            section="", page=None)])
    children = chunk_document(doc, "doc3", chunk_size=1800, chunk_overlap=0, title="Admin Guide",
                              strategy="parent_child", child_size=400, child_overlap=80)
    assert len(children) > 2
    assert all(len(c.text) <= 400 for c in children)
    assert all(c.embed_text.startswith("Document: Admin Guide | Section: Issuing certificates") for c in children)
    parents = {c.metadata["parent_id"]: c.metadata["parent_text"] for c in children}
    assert len(parents) < len(children)                       # several children share one parent
    for c in children:                                        # every child's text lies inside its parent
        assert c.text[:60] in c.metadata["parent_text"]
    assert len({c.chunk_id for c in children}) == len(children)


def test_select_context_widens_child_to_parent_and_dedupes():
    from backend.generation.prompt_builder import select_context
    from backend.retrieval.hybrid import Candidate
    from ingestion.vector_store import StoredChunk
    parent = "Full section about verifying certificates. " * 20
    c1 = Candidate(chunk=StoredChunk(id="a", text="child one", metadata={"parent_id": "d:p0", "parent_text": parent}),
                   kb="products", rerank_score=0.9)
    c2 = Candidate(chunk=StoredChunk(id="b", text="child two", metadata={"parent_id": "d:p0", "parent_text": parent}),
                   kb="products", rerank_score=0.8)
    c3 = Candidate(chunk=StoredChunk(id="c", text="standalone chunk", metadata={}), kb="products", rerank_score=0.7)
    refs = select_context([c1, c2, c3], max_chars=9000)
    assert [r.sid for r in refs] == ["S1", "S2"]
    assert refs[0].text == parent and refs[0].matched_text == "child one"
    assert refs[1].text == "standalone chunk"
    # parent too large for the remaining budget -> falls back to the child alone
    small = select_context([c3, c1], max_chars=len("standalone chunk") + 20)
    assert small[1].text == "child one"


def test_recursive_split_prefers_coarse_boundaries_and_respects_size():
    from ingestion.chunking import recursive_split
    para = "Admins can revoke a certificate from the Reports module. The student is notified by email. "
    text = "\n\n".join(f"Step {i}. {para * 2}" for i in range(8))
    pieces = recursive_split(text, size=400, overlap=0)
    assert all(len(p) <= 400 for p in pieces)
    assert all(p.startswith("Step ") for p in pieces)            # each paragraph fits: split only at paragraphs
    assert " ".join(pieces).count("Step ") == 8                   # nothing lost
    one_long = "word " * 500                                      # no paragraph/sentence breaks -> word level
    words = recursive_split(one_long, size=120, overlap=20)
    assert all(len(p) <= 120 for p in words) and all(not p.startswith(" ") for p in words)
    assert recursive_split("x" * 250, size=100) == ["x" * 100, "x" * 100, "x" * 50]   # last-resort hard split
    tail = recursive_split("First line is long enough to stand alone here.\n" * 7 + "Short end.", size=200)
    assert all(len(p) >= 50 for p in tail)                         # a tiny last line is absorbed, not left alone


def test_recursive_overlap_uses_whole_sentences():
    from ingestion.chunking import recursive_split
    sentences = [f"Sentence number {i} explains one verification step clearly." for i in range(12)]
    pieces = recursive_split(" ".join(sentences), size=200, overlap=60)
    assert len(pieces) > 2 and all(len(p) <= 200 for p in pieces)
    for a, b in zip(pieces, pieces[1:]):                          # next chunk starts with the previous last sentence
        assert b.split(".")[0] + "." in a
        assert b[0].isupper()


def test_structure_recursive_strategy_keeps_heading_path():
    from pathlib import Path
    from ingestion.document_loader import LoadedDocument, Section
    body = "Open the Reports module and choose Generated Certificates. " * 12
    text = (f"# Admin Guide\n\nOverview of the admin console. {body}\n\n## Certificates\n\n{body}\n\n"
            f"### Revoke a certificate\n\n{body}\n\n## Users\n\n{body}")
    doc = LoadedDocument(path=Path("admin.md"), name="admin.md", file_type="md",
                         sections=[Section(text=text, kind="text", section="", page=None)])
    chunks = chunk_document(doc, "doc4", chunk_size=500, chunk_overlap=80, title="Admin Guide",
                            strategy="structure_recursive")
    assert all(len(c.text) <= 500 for c in chunks)
    revoke = [c for c in chunks if c.section == "Revoke a certificate"]
    # breadcrumb of enclosing headings; the leading H1 equal to the document title is not repeated
    assert revoke and all(c.embed_text.startswith("Document: Admin Guide | Section: Certificates > Revoke a certificate")
                          for c in revoke)
    users = [c for c in chunks if c.section == "Users"]
    assert users and all(c.embed_text.split("\n")[0] == "Document: Admin Guide | Section: Users"
                         for c in users)                          # sibling heading replaces "Certificates"
