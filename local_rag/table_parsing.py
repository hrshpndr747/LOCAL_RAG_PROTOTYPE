from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

TABLE_PARSER_VERSION = "table_v2"
DEFAULT_PARSER_VERSION = ""

# Optional explicit header row overrides (1-based physical row index).
# Key can be lowercased filename (e.g., "special.xlsx") or "filename:sheet_name".
HEADER_ROW_OVERRIDES: dict[str, int] = {}


def get_parser_version(path: Path) -> str:
    """Return the active parser version for a given file type."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".csv"}:
        return TABLE_PARSER_VERSION
    return DEFAULT_PARSER_VERSION


def clean_cell_value(val: Any) -> str:
    """Convert an arbitrary cell value to clean text while preserving 0 and dates."""
    if val is None:
        return ""
    if isinstance(val, (datetime, date)):
        if isinstance(val, datetime) and (val.hour != 0 or val.minute != 0 or val.second != 0):
            return val.isoformat()
        return val.strftime("%Y-%m-%d")
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return f"{val:.4f}".rstrip("0").rstrip(".") if val == int(val) else str(val)
    if isinstance(val, int):
        return str(val)
    return str(val).strip()


def make_unique_headers(raw_headers: list[Any]) -> list[str]:
    """Normalize and deduplicate header labels deterministically."""
    seen: dict[str, int] = {}
    cleaned: list[str] = []
    for i, h in enumerate(raw_headers, start=1):
        name = clean_cell_value(h)
        if not name:
            name = f"column_{i}"
        name_lower = name.lower()
        if name_lower in seen:
            seen[name_lower] += 1
            unique_name = f"{name}_{seen[name_lower]}"
        else:
            seen[name_lower] = 1
            unique_name = name
        cleaned.append(unique_name)
    return cleaned


@dataclass
class HeaderDetectionResult:
    header_row_index: int            # 1-based physical row/line index
    candidate_list_index: int        # Index within candidate rows list
    headers: list[str]               # Deduplicated column names
    skipped_rows: list[int]          # 1-based row numbers skipped before header
    confidence: float                # Confidence score (0.0 to 1.0)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParsedTable:
    headers: list[str]
    records: list[dict[str, str]]    # [{col: val, ...}, ...]
    row_numbers: list[int]           # Physical 1-based row number for each record
    detection: HeaderDetectionResult
    warnings: list[str] = field(default_factory=list)


_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")
_DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
_CURRENCY_RE = re.compile(r"^[\$€£₹]\s*-?\d+")
_BANNER_KEYWORDS = {
    "synthetic", "fictional", "confidential", "internal use only",
    "proprietary", "copyright", "disclaimer", "all rights reserved",
}


def _is_data_like(val: str) -> bool:
    """Return True if cell text strongly resembles numeric/date/currency data rather than a header label."""
    if not val:
        return False
    if _NUMERIC_RE.match(val) or _DATE_RE.match(val) or _CURRENCY_RE.match(val):
        return True
    return False


def _is_banner_like(val: str) -> bool:
    """Return True if single cell resembles a document title or disclaimer banner."""
    val_lower = val.lower()
    if any(keyword in val_lower for keyword in _BANNER_KEYWORDS):
        return True
    if len(val) > 50 and (" " in val):
        return True
    return False


def detect_header_row(
    rows: list[tuple[int, list[Any]]],
    filename: str = "",
    sheet_name: str = "",
    max_inspect_rows: int = 20,
) -> HeaderDetectionResult:
    """Detect the true table header row using deterministic heuristics."""
    warnings: list[str] = []

    # 1. Check for explicit overrides
    override_key_sheet = f"{filename.lower()}:{sheet_name.lower()}"
    override_key_file = filename.lower()
    override_row = HEADER_ROW_OVERRIDES.get(override_key_sheet) or HEADER_ROW_OVERRIDES.get(override_key_file)

    if not rows:
        return HeaderDetectionResult(
            header_row_index=1,
            candidate_list_index=0,
            headers=[],
            skipped_rows=[],
            confidence=0.0,
            warnings=["Worksheet is completely empty."],
        )

    # Clean rows and preserve physical row numbers
    cleaned_rows: list[tuple[int, list[str]]] = []
    for phys_row, row_vals in rows:
        c_vals = [clean_cell_value(v) for v in row_vals]
        # Trim trailing empty cells
        while c_vals and not c_vals[-1]:
            c_vals.pop()
        if any(c_vals):
            cleaned_rows.append((phys_row, c_vals))

    if not cleaned_rows:
        return HeaderDetectionResult(
            header_row_index=1,
            candidate_list_index=0,
            headers=[],
            skipped_rows=[],
            confidence=0.0,
            warnings=["No non-empty rows found in sheet."],
        )

    # If override was specified, locate that physical row
    if override_row is not None:
        for idx, (phys_row, c_vals) in enumerate(cleaned_rows):
            if phys_row == override_row:
                skipped = [r[0] for r in cleaned_rows[:idx]]
                return HeaderDetectionResult(
                    header_row_index=phys_row,
                    candidate_list_index=idx,
                    headers=make_unique_headers(c_vals),
                    skipped_rows=skipped,
                    confidence=1.0,
                    warnings=[f"Using explicit header override for row {phys_row}."],
                )

    inspect_pool = cleaned_rows[:max_inspect_rows]

    # Compute width distribution across inspection pool to understand typical table width
    subsequent_widths = [len(r[1]) for r in inspect_pool]
    width_counts = Counter(subsequent_widths)
    # Exclude 1-cell width from modal table width if multi-column rows exist
    multi_widths = [w for w in subsequent_widths if w > 1]
    modal_table_width = Counter(multi_widths).most_common(1)[0][0] if multi_widths else width_counts.most_common(1)[0][0]

    scores: list[float] = []
    earliest_non_banner_seen = False
    for idx, (phys_row, cells) in enumerate(inspect_pool):
        score = 0.0
        n_cells = len(cells)
        non_empty = [c for c in cells if c]
        n_non_empty = len(non_empty)

        if n_non_empty == 0:
            scores.append(-999.0)
            continue

        is_banner = False
        # Rule 1: Table width matching and banner detection
        if modal_table_width > 1:
            if n_non_empty == 1:
                is_banner = True
                score -= 100.0
                if _is_banner_like(non_empty[0]):
                    score -= 50.0
            elif n_non_empty >= modal_table_width - 1:
                score += 45.0
            else:
                score -= 15.0
        else:
            # 1-column table
            if n_non_empty == 1:
                score += 30.0

        # Rule 2: Header cell text types (headers should be text, not numbers/dates)
        data_count = sum(1 for c in non_empty if _is_data_like(c))
        if data_count == 0:
            score += 35.0  # All cells are non-numeric labels
        else:
            score -= (data_count * 20.0)

        # Rule 3: Uniqueness of column names
        unique_count = len(set(c.lower() for c in non_empty))
        if n_non_empty > 1 and unique_count == n_non_empty:
            score += 25.0
        elif n_non_empty > 1 and unique_count < n_non_empty / 2:
            score -= 20.0

        # Rule 4: Label lengths
        avg_len = sum(len(c) for c in non_empty) / n_non_empty
        if 2 <= avg_len <= 35:
            score += 15.0
        elif any(len(c) > 60 for c in non_empty):
            score -= 25.0

        # Rule 5: Subsequent row contrast (does next row contain data?)
        if idx + 1 < len(inspect_pool):
            next_row_cells = inspect_pool[idx + 1][1]
            next_data_count = sum(1 for c in next_row_cells if _is_data_like(c))
            if next_data_count > 0 and data_count == 0:
                score += 35.0  # Classic header followed by data row
            if len(next_row_cells) == n_non_empty:
                score += 15.0

        # Rule 6: Earliest non-banner row matching width gets strong prior
        if not is_banner and not earliest_non_banner_seen:
            earliest_non_banner_seen = True
            score += 50.0

        scores.append(score)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_phys_row, best_cells = inspect_pool[best_idx]
    best_score = scores[best_idx]

    # Calculate confidence bounded in [0.0, 1.0]
    confidence = max(0.1, min(1.0, (best_score + 50.0) / 200.0))
    skipped = [r[0] for r in inspect_pool[:best_idx]]

    if best_score < 0:
        warnings.append(
            f"Low confidence header detection ({confidence:.2f}) on physical row {best_phys_row} for {filename} {sheet_name}."
        )

    headers = make_unique_headers(best_cells)
    return HeaderDetectionResult(
        header_row_index=best_phys_row,
        candidate_list_index=best_idx,
        headers=headers,
        skipped_rows=skipped,
        confidence=confidence,
        warnings=warnings,
    )


def parse_sheet_rows(
    rows: list[tuple[int, list[Any]]],
    filename: str = "",
    sheet_name: str = "",
    max_inspect_rows: int = 20,
) -> ParsedTable:
    """Parse Excel rows into records with true header detection and physical row tracking."""
    detection = detect_header_row(rows, filename=filename, sheet_name=sheet_name, max_inspect_rows=max_inspect_rows)
    headers = list(detection.headers)
    warnings = list(detection.warnings)

    records: list[dict[str, str]] = []
    row_numbers: list[int] = []

    if not headers:
        return ParsedTable(headers=[], records=[], row_numbers=[], detection=detection, warnings=warnings)

    # Ensure headers cover the maximum width of data rows
    max_width = len(headers)
    for phys_row, raw_cells in rows:
        c_vals = [clean_cell_value(v) for v in raw_cells]
        while c_vals and not c_vals[-1]:
            c_vals.pop()
        if len(c_vals) > max_width:
            max_width = len(c_vals)

    if len(headers) < max_width:
        for i in range(len(headers) + 1, max_width + 1):
            headers.append(f"column_{i}")

    # Locate the header row index in raw rows
    header_found = False
    for phys_row, raw_cells in rows:
        if not header_found:
            if phys_row == detection.header_row_index:
                header_found = True
            continue

        # Data row
        cell_values = [clean_cell_value(v) for v in raw_cells]
        # Ignore completely blank records
        if not any(cell_values):
            continue

        record: dict[str, str] = {}
        for col_idx, header_name in enumerate(headers):
            val = cell_values[col_idx] if col_idx < len(cell_values) else ""
            record[header_name] = val

        records.append(record)
        row_numbers.append(phys_row)

    return ParsedTable(
        headers=headers,
        records=records,
        row_numbers=row_numbers,
        detection=detection,
        warnings=warnings,
    )


def parse_csv_lines(
    handle: Iterable[str],
    filename: str = "",
    max_inspect_rows: int = 20,
) -> ParsedTable:
    """Parse CSV lines into records with true header detection and physical line tracking."""
    reader = csv.reader(handle)
    raw_rows: list[tuple[int, list[str]]] = []

    # csv.reader.line_num tracks the 1-based physical line number of the end of the current row
    curr_line = 1
    for row in reader:
        raw_rows.append((curr_line, row))
        curr_line = reader.line_num + 1

    detection = detect_header_row(raw_rows, filename=filename, max_inspect_rows=max_inspect_rows)
    headers = list(detection.headers)
    warnings = list(detection.warnings)

    records: list[dict[str, str]] = []
    row_numbers: list[int] = []

    if not headers:
        return ParsedTable(headers=[], records=[], row_numbers=[], detection=detection, warnings=warnings)

    max_width = len(headers)
    for phys_line, cells in raw_rows:
        c_vals = [clean_cell_value(v) for v in cells]
        while c_vals and not c_vals[-1]:
            c_vals.pop()
        if len(c_vals) > max_width:
            max_width = len(c_vals)

    if len(headers) < max_width:
        for i in range(len(headers) + 1, max_width + 1):
            headers.append(f"column_{i}")

    header_found = False
    for phys_line, cells in raw_rows:
        if not header_found:
            if phys_line == detection.header_row_index:
                header_found = True
            continue

        clean_cells = [clean_cell_value(c) for c in cells]
        if not any(clean_cells):
            continue

        record: dict[str, str] = {}
        for col_idx, header_name in enumerate(headers):
            val = clean_cells[col_idx] if col_idx < len(clean_cells) else ""
            record[header_name] = val

        records.append(record)
        row_numbers.append(phys_line)

    return ParsedTable(
        headers=headers,
        records=records,
        row_numbers=row_numbers,
        detection=detection,
        warnings=warnings,
    )
