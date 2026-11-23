from __future__ import annotations

import csv
import json
from pathlib import Path

import openpyxl
import pytest

from local_rag.config import Settings
from local_rag.loaders import load_csv, load_docx, load_pdf, load_sqlite, load_text, load_xlsx
from local_rag.service import RAGService
from local_rag.table_parsing import (
    TABLE_PARSER_VERSION,
    clean_cell_value,
    detect_header_row,
    get_parser_version,
    make_unique_headers,
    parse_csv_lines,
    parse_sheet_rows,
)
from tests.test_incremental_index import TrackingFakeClient


# 1. Standard Excel file with headers in row 1
def test_excel_standard_row1_header(tmp_path: Path):
    path = tmp_path / "standard.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Employees"
    ws.append(["ID", "Name", "Department", "Salary"])
    ws.append([101, "Alice", "Engineering", 95000])
    ws.append([102, "Bob", "Marketing", 80000])
    wb.save(path)
    wb.close()

    docs = load_xlsx(path, tmp_path)
    assert len(docs) >= 1
    # Summary document
    summary_doc = docs[0]
    assert "Total data rows: 2" in summary_doc.text
    # Data document
    data_doc = docs[1]
    assert data_doc.locator == "sheet Employees; rows 2–3"
    assert "ID=101; Name=Alice; Department=Engineering; Salary=95000" in data_doc.text
    assert "ID=102; Name=Bob; Department=Marketing; Salary=80000" in data_doc.text


# 2. Excel file with title in row 1 and headers in row 2
def test_excel_title_row1_header_row2(tmp_path: Path):
    path = tmp_path / "title_banner.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Claims"
    ws.append(["CONFIDENTIAL — Internal Use Only"])  # Row 1: banner
    ws.append(["Claim_ID", "Status", "Amount"])       # Row 2: header
    ws.append(["C-001", "open", 1500])               # Row 3: data
    ws.append(["C-002", "closed", 3200])             # Row 4: data
    wb.save(path)
    wb.close()

    docs = load_xlsx(path, tmp_path)
    summary_doc = docs[0]
    assert "Total data rows: 2" in summary_doc.text
    data_doc = docs[1]
    assert data_doc.locator == "sheet Claims; rows 3–4"
    assert "Claim_ID=C-001; Status=open; Amount=1500" in data_doc.text
    assert "CONFIDENTIAL" not in data_doc.text  # Banner excluded from data records!


# 3. Excel file with multiple introductory rows
def test_excel_multiple_introductory_rows(tmp_path: Path):
    path = tmp_path / "multi_intro.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Report"
    ws.append(["Company XYZ Annual Review"])
    ws.append(["Prepared by Financial Planning Team"])
    ws.append(["Data as of 2024-12-31"])
    ws.append([])  # blank row 4
    ws.append(["Region", "Target", "Actual", "Variance"])  # Row 5: header
    ws.append(["North", 1000, 1100, 100])                 # Row 6: data
    ws.append(["South", 800, 750, -50])                   # Row 7: data
    wb.save(path)
    wb.close()

    docs = load_xlsx(path, tmp_path)
    summary_doc = docs[0]
    assert "Total data rows: 2" in summary_doc.text
    data_doc = docs[1]
    assert data_doc.locator == "sheet Report; rows 6–7"
    assert "Region=North; Target=1000; Actual=1100; Variance=100" in data_doc.text


# 4. Worksheet with blank rows before the table
def test_excel_blank_rows_before_table(tmp_path: Path):
    path = tmp_path / "blank_prefix.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append([])
    ws.append([])
    ws.append(["Code", "Description"])  # Row 3: header
    ws.append(["P-1", "Standard Widget"])  # Row 4: data
    wb.save(path)
    wb.close()

    docs = load_xlsx(path, tmp_path)
    assert "Total data rows: 1" in docs[0].text
    assert docs[1].locator == "sheet Sheet; row 4"
    assert "Code=P-1; Description=Standard Widget" in docs[1].text


# 5. Multiple worksheets with different header positions
def test_excel_multiple_worksheets_different_headers(tmp_path: Path):
    path = tmp_path / "multi_sheet.xlsx"
    wb = openpyxl.Workbook()
    # Sheet 1: header row 1
    ws1 = wb.active
    ws1.title = "SheetA"
    ws1.append(["A1", "B1"])
    ws1.append(["valA", "valB"])

    # Sheet 2: title row 1, header row 2
    ws2 = wb.create_sheet(title="SheetB")
    ws2.append(["Sheet B Title Banner"])
    ws2.append(["ColX", "ColY", "ColZ"])
    ws2.append([10, 20, 30])
    wb.save(path)
    wb.close()

    docs = load_xlsx(path, tmp_path)
    # 2 docs per sheet (summary + data)
    assert len(docs) == 4
    sheet_a_data = [d for d in docs if "SheetA" in d.title and "summary" not in d.title][0]
    sheet_b_data = [d for d in docs if "SheetB" in d.title and "summary" not in d.title][0]

    assert sheet_a_data.locator == "sheet SheetA; row 2"
    assert "A1=valA; B1=valB" in sheet_a_data.text

    assert sheet_b_data.locator == "sheet SheetB; row 3"
    assert "ColX=10; ColY=20; ColZ=30" in sheet_b_data.text


# 6. Excel data containing zero values and empty cells
def test_excel_zero_values_and_empty_cells(tmp_path: Path):
    path = tmp_path / "zeros.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Item", "Paid", "Reserve", "Notes"])
    ws.append(["Claim A", 0, 5000, None])
    ws.append(["Claim B", 1200, 0, ""])
    wb.save(path)
    wb.close()

    docs = load_xlsx(path, tmp_path)
    data_doc = docs[1]
    assert "Item=Claim A; Paid=0; Reserve=5000" in data_doc.text
    assert "Item=Claim B; Paid=1200; Reserve=0" in data_doc.text
    assert "Notes=" not in data_doc.text  # Empty notes omitted cleanly


# 7. One-column spreadsheet with a legitimate header
def test_excel_one_column_legitimate_header(tmp_path: Path):
    path = tmp_path / "one_col.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Policy_Number"])
    ws.append(["POL-001"])
    ws.append(["POL-002"])
    ws.append(["POL-003"])
    wb.save(path)
    wb.close()

    docs = load_xlsx(path, tmp_path)
    assert "Total data rows: 3" in docs[0].text
    data_doc = docs[1]
    assert "Policy_Number=POL-001" in data_doc.text
    assert "Policy_Number=POL-003" in data_doc.text


# 8. Duplicate and missing column headings
def test_excel_duplicate_and_missing_headings(tmp_path: Path):
    path = tmp_path / "headers_edge.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Category", "", "Category", None])
    ws.append(["Books", "value2", "Fiction", "value4"])
    wb.save(path)
    wb.close()

    docs = load_xlsx(path, tmp_path)
    data_doc = docs[1]
    # Deduplicated column names: Category, column_2, Category_2, column_4
    assert "Category=Books; column_2=value2; Category_2=Fiction; column_4=value4" in data_doc.text


# 9. Correct physical worksheet row locators with blank rows in-between
def test_excel_physical_row_locators_with_blanks(tmp_path: Path):
    path = tmp_path / "with_blanks.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Title Banner"])           # Row 1
    ws.append(["ID", "Name"])             # Row 2 (header)
    ws.append(["1", "A"])                 # Row 3
    ws.append(["2", "B"])                 # Row 4
    ws.append([])                         # Row 5 (blank)
    ws.append(["3", "C"])                 # Row 6
    wb.save(path)
    wb.close()

    docs = load_xlsx(path, tmp_path)
    data_doc = docs[1]
    # First record row 3, last record row 6
    assert data_doc.locator == "sheet Sheet; rows 3–6"
    assert "Total data rows: 3" in docs[0].text


# 10. Correct Excel summary counts and distributions on real sample files
def test_real_excel_files_counts_and_structure():
    root = Path(__file__).resolve().parents[1] / "data" / "confluence"
    attachments = root / "attachments" / "425986"

    # Test loss-run.xlsx
    loss_path = attachments / "loss-run.xlsx"
    loss_docs = load_xlsx(loss_path, root)
    summary = loss_docs[0]
    assert "Total data rows: 80" in summary.text
    assert "Counts by Status: closed=46, denied=11, open=23" in summary.text
    assert "Counts by Line: Commercial=26, Household=27, Motor=27" in summary.text

    # First data batch starts at physical row 3
    first_batch = loss_docs[1]
    assert first_batch.locator == "sheet Loss Run; rows 3–14"
    assert "Claim=C-1000; Policy=COM-0000077; Line=Commercial" in first_batch.text
    assert "Reserve=0; Paid=22950" in first_batch.text

    # Last batch ends at physical row 82
    last_batch = loss_docs[-1]
    assert last_batch.locator == "sheet Loss Run; rows 75–82"

    # Test premium-register.xlsx
    prem_path = attachments / "premium-register.xlsx"
    prem_docs = load_xlsx(prem_path, root)
    prem_summary = prem_docs[0]
    assert "Total data rows: 120" in prem_summary.text
    assert "Counts by Line: Commercial=45, Household=41, Motor=34" in prem_summary.text
    assert prem_docs[1].locator == "sheet Premium Register; rows 3–14"
    assert prem_docs[-1].locator == "sheet Premium Register; rows 111–122"


# 11. Standard CSV with first-line headers
def test_csv_standard_first_line(tmp_path: Path):
    path = tmp_path / "standard.csv"
    path.write_text("ID,Name,Role\n10,Sara,Analyst\n20,Tom,Director\n", encoding="utf-8")
    docs = load_csv(path, tmp_path)
    assert len(docs) == 1
    assert docs[0].locator == "rows 2–3"
    assert "ID=10; Name=Sara; Role=Analyst" in docs[0].text
    assert "ID=20; Name=Tom; Role=Director" in docs[0].text


# 12. CSV with a banner before the header
def test_csv_banner_before_header(tmp_path: Path):
    path = tmp_path / "banner.csv"
    content = "SYNTHETIC DEMO BANNER\nAgent,Region,Sales\nA-1,North,500\nA-2,South,700\n"
    path.write_text(content, encoding="utf-8")
    docs = load_csv(path, tmp_path)
    assert len(docs) == 1
    assert docs[0].locator == "rows 3–4"
    assert "Agent=A-1; Region=North; Sales=500" in docs[0].text
    assert "SYNTHETIC" not in docs[0].text


# 13. CSV with quoted commas and multiline fields
def test_csv_quoted_commas_and_multiline_fields(tmp_path: Path):
    path = tmp_path / "quoted.csv"
    content = 'Name,Address,Notes\n"Smith, John","123 Main St, Suite 4","Line one\nLine two"\n'
    path.write_text(content, encoding="utf-8")
    docs = load_csv(path, tmp_path)
    assert len(docs) == 1
    assert "Name=Smith, John" in docs[0].text
    assert "Address=123 Main St, Suite 4" in docs[0].text


# 14. CSV with UTF-8 BOM and blank fields
def test_csv_utf8_bom_and_blank_fields(tmp_path: Path):
    path = tmp_path / "bom.csv"
    content = "\ufeffColA,ColB,ColC\nvalA,,valC\n"
    path.write_text(content, encoding="utf-8-sig")
    docs = load_csv(path, tmp_path)
    assert len(docs) == 1
    assert "ColA=valA; ColC=valC" in docs[0].text


# 15. Correct CSV record counts on real sample file commission-summary.csv
def test_real_commission_summary_csv():
    root = Path(__file__).resolve().parents[1] / "data" / "confluence"
    path = root / "attachments" / "425986" / "commission-summary.csv"
    docs = load_csv(path, root)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.locator == "rows 3–17"
    lines = doc.text.splitlines()
    assert len(lines) == 15
    assert lines[0] == "Agent=AG-001; Name=Mr. Alister Kinnear; Agency=Hyman Insurance Group; Region=Ireland; Policies=8; Total premium=17540.0; Commission=2180.1"
    assert lines[-1] == "Agent=AG-015; Name=Calvin Hastings; Agency=Hession Versicherung; Region=Ireland; Policies=7; Total premium=19980.0; Commission=2808.15"


# 16. Malformed or empty spreadsheet returns diagnostic warning without crashing
def test_empty_spreadsheet_returns_empty():
    res = parse_sheet_rows([], filename="empty.xlsx")
    assert res.headers == []
    assert res.records == []
    assert len(res.warnings) >= 1


# 17. Confluence attachment metadata and source URLs remain preserved
def test_confluence_attachment_metadata_preserved_with_new_loader():
    root = Path(__file__).resolve().parents[1] / "data" / "confluence"
    path = root / "attachments" / "425986" / "commission-summary.csv"
    docs = load_csv(path, root)
    from local_rag.loaders import _apply_confluence_attachment_metadata
    docs_with_meta = _apply_confluence_attachment_metadata(path, docs)
    assert docs_with_meta[0].source_url != ""
    assert "Confluence attachment" in docs_with_meta[0].locator


# 18. Existing non-spreadsheet loaders continue working
def test_existing_non_spreadsheet_loaders_continue_working(tmp_path: Path):
    root = Path(__file__).resolve().parents[1] / "data" / "confluence"
    attachments = root / "attachments" / "425986"

    # Test docx
    docx_docs = load_docx(attachments / "claims-handling-manual.docx", root)
    assert len(docx_docs) >= 1
    assert "Claim Lifecycle" in docx_docs[0].text

    # Test pdf
    pdf_docs = load_pdf(attachments / "C-1000-fnol.pdf", root)
    assert any("Notice of Loss" in d.text or "FNOL" in d.text for d in pdf_docs)

    # Test text
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("Sample plain text.", encoding="utf-8")
    assert len(load_text(txt_path, tmp_path)) == 1


# 19. Incremental indexing detects the parser-version change
def test_incremental_indexing_detects_parser_version_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "index"
    data_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)

    # Put a docx (non-spreadsheet) and an xlsx (spreadsheet)
    docx_file = data_dir / "notes.txt"
    docx_file.write_text("Unrelated document notes.", encoding="utf-8")

    xlsx_file = data_dir / "table.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Banner Line"])
    ws.append(["Metric", "Val"])
    ws.append(["M1", 100])
    wb.save(xlsx_file)
    wb.close()

    client = TrackingFakeClient()
    settings = Settings(
        ollama_base_url="http://localhost:11434",
        chat_model="fake-chat",
        embed_model="fake-embed",
        data_dir=data_dir,
        index_dir=index_dir,
        top_k=4,
        min_relevance=0.01,
    )
    service = RAGService(settings, client=client)

    # 1. Build initial index
    res1 = service.update_index()
    assert res1.total_files == 2

    # 2. Simulate an index built with an older parser version:
    # Set table.xlsx's parser_version in sources.json to "" (legacy)
    tracked = service.store.load_sources()
    for sid, rec in tracked.items():
        if "table.xlsx" in rec.source_path:
            rec.parser_version = ""  # old parser version
    service.store.save(
        service._chunks,
        service._embeddings,
        service.settings.embed_model,
        [],
        sources=tracked,
    )

    client.embed_calls.clear()
    # 3. Run update_index: should detect parser_version change on table.xlsx ONLY!
    res2 = service.update_index()
    assert res2.status == "updated"
    assert len(res2.modified_files) == 1
    assert "table.xlsx" in res2.modified_files[0]
    assert len(res2.unchanged_files) == 1
    assert "notes.txt" in res2.unchanged_files[0]  # Unaffected file remained untouched!

    # 4. Running update_index again should report up to date with zero embeds
    client.embed_calls.clear()
    res3 = service.update_index()
    assert res3.status == "up_to_date"
    assert len(client.embed_calls) == 0


# 20. Search can retrieve corrected spreadsheet data after re-indexing
def test_search_can_retrieve_corrected_spreadsheet_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "index"
    data_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)

    csv_file = data_dir / "agents.csv"
    csv_file.write_text("CONFIDENTIAL BANNER\nAgent,Name,Region\nAG-101,John Doe,East\nAG-102,Jane Smith,West\n", encoding="utf-8")

    client = TrackingFakeClient()
    settings = Settings(
        ollama_base_url="http://localhost:11434",
        chat_model="fake-chat",
        embed_model="fake-embed",
        data_dir=data_dir,
        index_dir=index_dir,
        top_k=4,
        min_relevance=0.01,
    )
    service = RAGService(settings, client=client)
    service.update_index()

    # Retrieve
    results = service.retrieve("John Doe")
    assert len(results) > 0
    top_chunk = results[0].chunk
    assert "Agent=AG-101" in top_chunk.text
    assert "Name=John Doe" in top_chunk.text
    assert "CONFIDENTIAL BANNER" not in top_chunk.text
