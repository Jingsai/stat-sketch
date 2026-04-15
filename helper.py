import io
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ENCODINGS = ["utf-8", "cp1252", "latin1"]

# User-facing labels for sidebar separator values
SEP_LABELS = {",": "comma (,)", "\t": "Tab", ";": "semicolon (;)", "|": "pipe (|)"}


def warn_if_likely_wrong_separator(df: pd.DataFrame, selected_sep: str) -> None:
    """
    If the file was parsed with the wrong delimiter, pandas often yields a single wide
    text column whose values contain many characters of the true delimiter (e.g. ';').
    Show a gentle warning suggesting another separator from the sidebar.
    """
    if df is None or df.empty or len(df.columns) != 1:
        return

    col = df.columns[0]
    as_str = df[col].dropna().astype(str)
    if as_str.empty:
        return

    selected_label = SEP_LABELS.get(selected_sep, repr(selected_sep))
    best_label = None
    best_score = 0.0

    for delim, label in SEP_LABELS.items():
        if delim == selected_sep:
            continue
        pat = re.escape(delim)
        counts = as_str.str.count(pat)
        frac_any = float((counts > 0).mean())
        mean_ct = float(counts.mean())
        # Strong signal: most rows contain this delimiter more than once on average
        if frac_any < 0.45 or mean_ct < 1.25:
            continue
        score = frac_any * mean_ct
        if score > best_score:
            best_score = score
            best_label = label

    if best_label:
        st.warning(
            "The file loaded as one column, but many cell values contain characters "
            f"typical of another field separator. You currently have {selected_label} "
            f"selected. Try {best_label} under Separator in the sidebar and reload."
        )

# Display label for datasets/penguins.csv (shown first when that file exists).
EXAMPLE_DATASET_PALMER_PENGUINS = "Palmer penguins"


def datasets_directory() -> Path:
    """Directory containing bundled example CSV files (next to this package)."""
    return Path(__file__).resolve().parent / "datasets"


def list_example_csv_filenames() -> list[str]:
    """Basenames of *.csv files in datasets/, sorted."""
    d = datasets_directory()
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.glob("*.csv") if p.is_file())


@st.cache_data(show_spinner=False)
def load_example_csv(filename: str, sep: str = ",") -> pd.DataFrame:
    """Load a CSV from datasets/ by basename only (no path components)."""
    if not filename or "/" in filename or "\\" in filename or filename.startswith("."):
        raise ValueError("Invalid dataset filename.")
    root = datasets_directory().resolve()
    path = (root / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"Example dataset not found: {filename}") from None
    if not path.is_file():
        raise ValueError(f"Example dataset not found: {filename}")
    last_error = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, sep=sep, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
    raise ValueError(
        f"Could not decode file with any of {ENCODINGS}. Last error: {last_error}"
    )


def is_numeric(series: pd.Series) -> bool:
    """True if series has a pandas numeric dtype."""
    return pd.api.types.is_numeric_dtype(series)


def is_categorical(series: pd.Series, max_unique_ratio: float = 0.2, max_unique: int = 5) -> bool:
    """
    Heuristic: treat as categorical if:
      - dtype is object/category/bool, OR
      - numeric but low unique count/ratio (like 0/1, 1-5 ratings, zip-like codes)
    """
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
        return True

    nunique = series.dropna().nunique()
    n = series.dropna().shape[0]
    if n == 0:
        return False
    return (nunique <= max_unique) and ((nunique / n) <= max_unique_ratio)


def ensure_categorical(df: pd.DataFrame, col: str, context: str = "the plot") -> None:
    """
    Ensure a column is treated as categorical. If it is numeric dtype, show a warning
    and convert to string; otherwise normalize to string for consistent labels.
    Modifies df in place.
    """
    if pd.api.types.is_numeric_dtype(df[col]):
        st.warning(
            f"'{col}' is numeric. Treating it as a categorical variable for {context}."
        )
    df[col] = df[col].astype(str)


def ensure_numeric(df: pd.DataFrame, col: str, threshold=0.8) -> None:
    """
    Ensure a column is numeric. If not, attempt conversion with coerce; on failure
    show error and stop the app. Modifies df in place.
    """
    if pd.api.types.is_numeric_dtype(df[col]):
        return
    st.warning(
        f"'{col}' does not appear to be numeric. Attempting to convert it to numbers."
    )
    converted = pd.to_numeric(df[col], errors="coerce")
    nan_fraction = converted.isna().sum() / len(df[col])
    if nan_fraction >= threshold:
        st.error(
            f"'{col}' cannot be safely converted to numeric values. "
            "Please select a numeric variable."
        )
        st.stop()
    df[col] = converted


def compute_axis_breaks(values, max_labels: int):
    """
    Given a 1D sequence/Series of values and an approximate maximum number of labels,
    return a subset of unique values to use as axis breaks to avoid overcrowding.
    """
    if max_labels is None or max_labels <= 0:
        return list(pd.unique(values))

    uniq = pd.unique(values)
    uniq = [v for v in uniq if pd.notna(v)]
    n = len(uniq)
    if n <= max_labels:
        return list(uniq)

    step = int(np.ceil(n / max_labels))
    return [uniq[i] for i in range(0, n, step)]


@st.cache_data(show_spinner=False)
def load_csv_from_upload(uploaded_file, sep=","):
    """Load CSV from Streamlit file uploader. Tries ENCODINGS in order; resets file position before each try."""
    last_error = None
    for encoding in ENCODINGS:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=sep, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
    raise ValueError(
        f"Could not decode file with any of {ENCODINGS}. Last error: {last_error}"
    )


@st.cache_data(show_spinner=False)
def load_csv_from_url(url: str, sep=","):
    """Fetch URL content and load as CSV. Tries ENCODINGS in order."""
    with urllib.request.urlopen(url) as resp:
        content = resp.read()
    last_error = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
    raise ValueError(
        f"Could not decode URL content with any of {ENCODINGS}. Last error: {last_error}"
    )
