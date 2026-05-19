import pandas as pd
import numpy as np
import io
import chardet
import streamlit as st


ENCODINGS_TO_TRY = ["utf-8", "latin-1", "latin-8", "iso-8859-1", "cp1252", "utf-16", "ascii"]


def detect_encoding(raw_bytes: bytes) -> str:
    """Detect file encoding using chardet."""
    result = chardet.detect(raw_bytes[:50000])
    return result.get("encoding") or "utf-8"


def _try_read(raw_bytes: bytes, encoding: str, **kwargs) -> pd.DataFrame | None:
    try:
        return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, **kwargs)
    except Exception:
        return None


def _is_metadata_row(row: pd.Series) -> bool:
    """
    Heuristic: a row is a metadata/info row (not real headers) if it has many
    NaN/empty cells, or if parsing it as the header produces column names that
    are entirely numeric (like '506', '13', '').
    """
    non_null = row.dropna()
    if len(non_null) == 0:
        return True
    # If more than half the values are empty/NaN, treat as metadata
    if row.isnull().sum() > len(row) * 0.5:
        return True
    return False


def _infer_header_row(df_raw: pd.DataFrame) -> int | None:
    """
    Scan the first few rows to find which one looks like a proper header
    (i.e. mostly strings, not numbers). Returns 0-based index into df_raw.
    """
    for i, row in df_raw.head(5).iterrows():
        vals = row.dropna().astype(str).tolist()
        if not vals:
            continue
        # Count how many are clearly non-numeric strings
        non_num = sum(1 for v in vals if not _is_numeric_str(v))
        if non_num >= len(vals) * 0.5:
            return i
    return None  # give up, use default


def _is_numeric_str(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def smart_load_csv(uploaded_file) -> tuple[pd.DataFrame | None, dict]:
    """
    Robust CSV loader. Returns (df, info_dict).
    Handles:
      - Multiple encodings (utf-8, latin-1, cp1252, ...)
      - Double/metadata header rows (like Boston dataset)
      - No header row (generates Col_0, Col_1, ...)
      - Separator detection (comma, semicolon, tab, pipe)
    """
    raw_bytes = uploaded_file.read()
    info = {"encoding": None, "sep": ",", "header_row": 0, "issues": []}

    # 1. Detect encoding
    detected_enc = detect_encoding(raw_bytes)
    encodings = [detected_enc] + [e for e in ENCODINGS_TO_TRY if e != detected_enc]
    info["encoding"] = detected_enc

    # 2. Detect separator
    sample = raw_bytes[:4096].decode(detected_enc, errors="replace")
    sep = _detect_sep(sample)
    info["sep"] = sep

    df = None
    used_encoding = None

    # 3. Try each encoding
    for enc in encodings:
        df_raw = _try_read(raw_bytes, enc, sep=sep, header=None, dtype=str, nrows=10)
        if df_raw is None:
            continue
        used_encoding = enc
        info["encoding"] = enc

        # 4. Find header row
        header_row_idx = _infer_header_row(df_raw)

        if header_row_idx is None:
            # No good header found — generate column names
            df = _try_read(raw_bytes, enc, sep=sep, header=None)
            if df is not None:
                df.columns = [f"Col_{i}" for i in range(df.shape[1])]
                info["header_row"] = "auto-generated"
                info["issues"].append("No header detected — auto-generated column names.")
        elif header_row_idx == 0:
            # Normal: first row is header
            df = _try_read(raw_bytes, enc, sep=sep, header=0)
            info["header_row"] = 0
        else:
            # Skip rows before header_row_idx
            df = _try_read(raw_bytes, enc, sep=sep, header=header_row_idx, skiprows=range(header_row_idx))
            info["header_row"] = header_row_idx
            info["issues"].append(
                f"Detected {header_row_idx} metadata/extra row(s) before headers — skipped automatically."
            )

        if df is not None:
            break

    if df is None:
        return None, info

    # 5. Clean up
    # Drop fully-empty columns / rows
    df.dropna(how="all", axis=1, inplace=True)
    df.dropna(how="all", axis=0, inplace=True)
    # Strip whitespace from string column names
    df.columns = [str(c).strip() for c in df.columns]
    # Strip whitespace from object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # 6. Auto-infer numeric columns
    for col in df.columns:
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > df[col].notna().sum() * 0.5:
                df[col] = converted
        except Exception:
            pass

    info["shape"] = df.shape
    return df, info


def _detect_sep(sample: str) -> str:
    counts = {",": sample.count(","), ";": sample.count(";"),
              "\t": sample.count("\t"), "|": sample.count("|")}
    return max(counts, key=counts.get)


def get_column_types(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical
