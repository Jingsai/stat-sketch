"""
StatSketch: No-code Data Visualizer

Run: streamlit run StatSketch.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from plotnine.data import penguins

from helper import (
    EXAMPLE_DATASET_PALMER_PENGUINS,
    is_numeric,
    is_categorical,
    list_example_csv_filenames,
    load_csv_from_upload,
    load_csv_from_url,
    load_example_csv,
)
from visualization import THEME_MAP, render_visualization_tab
from distribution import DISTRIBUTION_TOOLS_WIDGET_KEYS_TO_CLEAR, render_distribution_tools
from infer import render_inference_tab
from use_cases import render_use_cases

st.set_page_config(page_title="StatSketch", layout="wide")
st.title("StatSketch: No-code Stats Lab")

st.sidebar.header("App mode")
app_mode = st.sidebar.radio(
    "Choose mode",
    ["Data Visualization & Inference", "Distribution Tools", "Use Cases"],
    horizontal=False,
    key="app_mode",
)

if app_mode == "Distribution Tools":
    render_distribution_tools()
    st.stop()

# When Distribution Tools is not shown, Streamlit may keep widget session keys with
# stale values. distribution._sync_widget_from_saved only restores from *_saved when
# the widget key is missing. Clear distribution input keys when leaving this mode so
# return visits restore from *_saved (see DISTRIBUTION_TOOLS_WIDGET_KEYS_TO_CLEAR).
for _k in DISTRIBUTION_TOOLS_WIDGET_KEYS_TO_CLEAR:
    st.session_state.pop(_k, None)

if app_mode == "Use Cases":
    render_use_cases()
    st.stop()

st.sidebar.header("Load data")
source = st.sidebar.radio(
    "Data source",
    ["Upload file", "File URL", "Example dataset"],
    horizontal=False,
)

df = None
if source == "Upload file":
    sep = st.sidebar.selectbox("Separator", [",", "\t", ";", "|"], index=0, key="csv_sep")
    uploaded = st.sidebar.file_uploader("Upload a data file (CSV, TXT, etc.)")
    if uploaded is not None:
        try:
            df = load_csv_from_upload(uploaded, sep=sep)
        except Exception as e:
            st.sidebar.error(f"Could not load data from file: {e}")
            st.stop()
elif source == "File URL":
    sep = st.sidebar.selectbox("Separator", [",", "\t", ";", "|"], index=0, key="csv_sep")
    url = st.sidebar.text_input("Paste a direct CSV URL")
    if url.strip():
        try:
            df = load_csv_from_url(url.strip(), sep=sep)
        except Exception as e:
            st.sidebar.error(f"Could not load data from URL: {e}")
            st.stop()
else:
    example_files = list_example_csv_filenames()
    example_display_names = [Path(f).stem for f in example_files]
    example_options = [EXAMPLE_DATASET_PALMER_PENGUINS] + example_display_names
    choice = st.sidebar.selectbox(
        "Example dataset",
        example_options,
        index=0,
        key="example_dataset_choice",
        label_visibility="collapsed",
    )
    sep = st.sidebar.selectbox("Separator", [",", "\t", ";", "|"], index=0, key="csv_sep")
    try:
        if choice == EXAMPLE_DATASET_PALMER_PENGUINS:
            df = penguins.copy()
        else:
            df = load_example_csv(f"{choice}.csv", sep=sep)
    except Exception as e:
        st.sidebar.error(f"Could not load example dataset: {e}")
        st.stop()

if df is None:
    st.info("Upload a data file, provide a URL, or use the example dataset to begin.")
    st.stop()

st.sidebar.header("Basic options")
drop_na_rows = st.sidebar.checkbox("Drop rows with missing values in selected columns (recommended)", value=True)

theme_name = st.sidebar.selectbox("Plot theme", list(THEME_MAP.keys()), index=0)
selected_theme = THEME_MAP[theme_name]()

try:
    has_groq_secret = bool(st.secrets.get("GROQ_API_KEY"))
except Exception:
    has_groq_secret = False
if not has_groq_secret:
    st.sidebar.text_input(
        "Groq API key (optional, for chat)",
        type="password",
        key="groq_api_key_input",
        help="Paste your key to enable chat. Or set GROQ_API_KEY in .streamlit/secrets.toml",
    )

col_info = []
for c in df.columns:
    s = df[c]
    ctype = "numeric" if is_numeric(s) and not is_categorical(s) else "categorical"
    col_info.append((c, ctype, int(s.isna().sum()), int(s.dropna().nunique())))
info_df = pd.DataFrame(col_info, columns=["column", "inferred_type", "missing", "unique_non_na"])

main_tabs = st.tabs(["Data Preview", "Visualization", "Inference"])

with main_tabs[0]:
    st.subheader("Data Preview")
    preview_rows = st.slider("Preview rows", 5, 50, 10, key="preview_rows")
    st.dataframe(df.head(preview_rows), width="stretch", hide_index=True)

    st.subheader("Inferred Types")
    st.caption("Inferred types are heuristic; you can still pick any column you want.")
    st.dataframe(info_df, width="stretch", hide_index=True)

    st.subheader("Data Statistics")
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] == 0:
        st.info("No numeric columns available for summary statistics.")
    else:
        stats = numeric_df.describe(percentiles=[0.25, 0.5, 0.75]).T
        stats = stats.rename(
            columns={
                "25%": "Q1",
                "50%": "Median",
                "75%": "Q3",
                "std": "Standard Deviation",
                "min": "Min",
                "max": "Max",
                "mean": "Mean",
            }
        )
        stats = stats[["Mean", "Standard Deviation", "Min", "Q1", "Median", "Q3", "Max"]]
        st.dataframe(stats, width="stretch")

with main_tabs[1]:
    render_visualization_tab(df=df, drop_na_rows=drop_na_rows, selected_theme=selected_theme)

with main_tabs[2]:
    render_inference_tab(df, drop_na_rows=drop_na_rows)
