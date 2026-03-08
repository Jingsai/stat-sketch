"""
StatSketch: No-code Data Visualizer

Streamlit app for building common statistical plots (bar, histogram, boxplot,
scatter, line, pie) from CSV/URL or example data. Supports custom title/axis labels
and an optional per-plot chat (Groq LLM) to ask questions about the current plot.

Run: streamlit run StatSketch.py
"""
import io
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

try:
    from groq import Groq
except ImportError:
    Groq = None

from plotnine import (
    ggplot,
    aes,
    geom_bar,
    geom_histogram,
    geom_col,
    geom_point,
    geom_line,
    geom_boxplot,
    facet_grid,
    labs,
    theme,
    theme_bw,
    theme_gray,
    theme_matplotlib,
    theme_seaborn,
    theme_minimal,
    element_text,
    xlim,
    coord_flip,
    geom_smooth,
    scale_x_datetime,
)
from plotnine.data import penguins
import matplotlib.pyplot as plt

# -----------------------------
# Helpers: column type detection and coercion
# -----------------------------
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

    # numeric but "code-like"
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
    # Calculate what percentage of the data is now numbers
    nan_fraction = converted.isna().sum() / len(df[col])
    # If more than 80% (threshold) is numerical, treat as numerical
    if nan_fraction >= threshold:
        st.error(
            f"'{col}' cannot be safely converted to numeric values. "
            "Please select a numeric variable."
        )
        st.stop()
    df[col] = converted

# X-axis label density: rotate or rely on coord_flip at call site
def theme_x_labels(df: pd.DataFrame, x: str):
    """Return a plotnine theme that rotates x-axis labels when there are 5–9 categories (45°). Used for bar/box with categorical x."""
    n = int(df[x].nunique(dropna=True))
    if 4 < n < 10:
        return theme(axis_text_x=element_text(rotation=45, ha="right", size=8))
    return theme()

def render_plotnine(p, theme=None, data: pd.DataFrame | None = None, x_col: str | None = None, n_x_categories: int | None = None):
    """Draw a plotnine figure and display it in Streamlit. Optionally apply theme, x-label rotation (from data/x_col), and stretched height for horizontal bar/box when n_x_categories > 10."""
    if theme is not None:
        p = p + theme
    if data is not None and x_col is not None and x_col in data.columns:
        p = p + theme_x_labels(data, x_col)

    fig = p.draw()

    # Use responsive size
    width = 6
    if n_x_categories is not None and n_x_categories > 10:
        # Horizontal bar/box with many categories: stretch height = width * (n / 25)
        height = width * (n_x_categories / 25)
    else:
        height = width * 0.6  # keep aspect ratio

    fig.set_size_inches(width, height)

    st.pyplot(fig, clear_figure=True, use_container_width=True)

ENCODINGS = ["utf-8", "cp1252", "latin1"]

# -----------------------------
# Data loading (with encoding fallback)
# -----------------------------
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


# Plotnine theme options (sidebar selectbox)
THEME_MAP = {
    "Gray": theme_gray,
    "Matplotlib": theme_matplotlib,
    "Seaborn": theme_seaborn,
    "Black & white": theme_bw,
    "Minimal": theme_minimal,
}

def get_groq_api_key() -> str | None:
    """Get Groq API key from secrets or sidebar fallback."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return st.session_state.get("groq_api_key_input") or None

def get_plot_labels(default_title: str, default_x: str, default_y: str) -> tuple[str, str, str]:
    """Return (title, x_label, y_label). Uses session-state overrides (custom_plot_title etc.) when non-empty, else defaults."""
    t = st.session_state.get("custom_plot_title", "") or default_title
    x = st.session_state.get("custom_plot_x", "") or default_x
    y = st.session_state.get("custom_plot_y", "") or default_y
    return (t, x, y)

def render_label_customizer_expander():
    """Expander: choose Title / X axis label / Y axis label, then one text input. Sync: only overwrite the text input when the dropdown selection changes (so we don't overwrite what the user just typed). Persist happens at start of run into *previous* selection slot."""
    with st.expander("Customize title & axis labels", expanded=False):
        label_which = st.selectbox(
            "Edit",
            ["Title", "X axis label", "Y axis label"],
            key="label_which",
        )
        key_map = {"Title": "custom_plot_title", "X axis label": "custom_plot_x", "Y axis label": "custom_plot_y"}
        current_key = key_map[label_which]
        if st.session_state.get("label_which_prev") != label_which:
            st.session_state["label_input"] = st.session_state.get(current_key, "")
        st.session_state["label_which_prev"] = label_which
        st.text_input("Value", key="label_input", placeholder="Leave blank for default")
        st.session_state[current_key] = st.session_state.get("label_input", "")


def build_plot_context(plot_type: str, params: dict, data: pd.DataFrame) -> str:
    """Build a text summary of plot type, params, and data (describe + value_counts) for the LLM chat context."""
    if data.empty or len(data) == 0:
        return f"Plot type: {plot_type}. Parameters: {params}. Data: empty (no rows)."

    lines = [
        f"Plot type: {plot_type}",
        f"Parameters: {params}",
        "",
        "Data summary:",
    ]
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in data.columns if c not in numeric_cols]
    if numeric_cols:
        lines.append("Numeric columns (describe):")
        lines.append(data[numeric_cols].describe().to_string())
    if cat_cols:
        lines.append("")
        lines.append("Categorical columns (value counts):")
        for col in cat_cols:
            vc = data[col].value_counts(dropna=False).head(20)
            lines.append(f"  {col}: {vc.to_dict()}")
    return "\n".join(lines)


def call_groq(user_message: str, context: str, messages: list[dict]) -> str:
    """Call Groq API with context and conversation history. Returns assistant reply or error message."""
    if Groq is None:
        return "The 'groq' package is not installed. Run: pip install groq"
    api_key = get_groq_api_key()
    if not api_key:
        return "To use the chat, set GROQ_API_KEY in Streamlit secrets (.streamlit/secrets.toml) or enter it in the sidebar."

    try:
        client = Groq(api_key=api_key)
        system_content = (
            "You are a helpful data visualization assistant. The user is viewing a plot. "
            "Use the following context (plot type, parameters, and data summary) to answer their questions. "
            "Be concise and relevant.\n\nContext:\n" + context
        )
        api_messages = [{"role": "system", "content": system_content}]
        for m in messages:
            api_messages.append({"role": m["role"], "content": m["content"]})
        api_messages.append({"role": "user", "content": user_message})

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=api_messages,
        )
        return completion.choices[0].message.content or "(No response)"
    except Exception as e:
        err = str(e).lower()
        if "rate" in err or "limit" in err:
            return "Rate limit exceeded. Please wait a moment and try again."
        return f"API error: {e}"


def render_plot_chat(plot_key: str, context_text: str) -> None:
    """Show chat UI under a plot. plot_key scopes session state so each plot has its own history; rerun after each user message to append assistant reply."""
    chat_key = f"plot_chat_{plot_key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    messages = st.session_state[chat_key]
    api_key = get_groq_api_key()
    with st.expander("Ask about this plot (such as 'Is the relationship linear?' or 'What are the Q1, median, and Q3 of this plot?'):", 
    expanded=len(messages) > 0):
        for m in messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        # Streamlit renders all tabs each run, so multiple chat inputs can exist at once.
        # Use a unique key per plot to avoid DuplicateWidgetID collisions.
        prompt = st.chat_input("Ask about this plot...", key=f"{chat_key}_input")
        if prompt:
            with st.chat_message("user"):
                st.write(prompt)
            messages.append({"role": "user", "content": prompt})
            reply = call_groq(prompt, context_text, messages[:-1])
            messages.append({"role": "assistant", "content": reply})
            st.rerun()


# def safe_top_categories(s: pd.Series, top_k: int):
#     vc = s.value_counts(dropna=False)
#     if top_k is None or top_k <= 0 or len(vc) <= top_k:
#         return s
#     top = vc.index[:top_k]
#     return s.where(s.isin(top), other="Other")

# -----------------------------
# App: page config, sidebar, data load
# -----------------------------
st.set_page_config(page_title="StatSketch", layout="wide")
st.title("StatSketch: No-code Data Visualizer")

st.sidebar.header("Load data")
source = st.sidebar.radio(
    "Data source",
    ["Upload file", "File URL", "Example dataset (penguins)"],
    horizontal=False,
)
sep = st.sidebar.selectbox("Separator", [",", "\t", ";", "|"], index=0)

df = None
if source == "Upload file":
    uploaded = st.sidebar.file_uploader(
        "Upload a data file (CSV, TXT, etc.)"
    )
    if uploaded is not None:
        try:
            df = load_csv_from_upload(uploaded, sep=sep)
        except Exception as e:
            st.sidebar.error(f"Could not load data from file: {e}")
            st.stop()
elif source == "CSV URL":
    url = st.sidebar.text_input("Paste a direct CSV URL")
    if url.strip():
        try:
            df = load_csv_from_url(url.strip(), sep=sep)
        except Exception as e:
            st.sidebar.error(f"Could not load data from URL: {e}")
            st.stop()
else:
    # Example dataset so users can explore the app without uploading data
    df = penguins.copy()

if df is None:
    st.info("Upload a data file, provide a URL, or use the example dataset to begin.")
    st.stop()

# Basic cleaning controls
st.sidebar.header("Basic options")
# drop_first_id_column = st.sidebar.checkbox(
#     "Drop first column (e.g. ID)",
#     value=False,
#     help="Remove the first column from the data if it looks like an index/ID column.",
# )
drop_na_rows = st.sidebar.checkbox("Drop rows with missing values in selected columns (recommended)", value=True)
preview_rows = st.sidebar.slider("Preview rows", 5, 50, 10)

theme_name = st.sidebar.selectbox(
    "Plot theme",
    list(THEME_MAP.keys()),
    index=0,
)
selected_theme = THEME_MAP[theme_name]()

# Groq API key for LLM chat (from secrets or optional sidebar input)
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

# if drop_first_id_column and len(df.columns) > 1:
#     df = df.iloc[:, 1:].copy()
# elif drop_first_id_column and len(df.columns) == 1:
#     st.sidebar.warning("Only one column in data; cannot drop first column.")

# Infer variable types (used for defaults; plots allow any column choice)
col_info = []
for c in df.columns:
    s = df[c]
    ctype = "numeric" if is_numeric(s) and not is_categorical(s) else "categorical"
    col_info.append((c, ctype, int(s.isna().sum()), int(s.dropna().nunique())))
info_df = pd.DataFrame(col_info, columns=["column", "inferred_type", "missing", "unique_non_na"])
categorical_cols = info_df.query("inferred_type == 'categorical'")["column"].tolist()
numeric_cols = info_df.query("inferred_type == 'numeric'")["column"].tolist()

main_tabs = st.tabs(["Visualization & Chat", "Data Preview & Inference"])

# Data Preview tab rendered first so its content is registered even if first tab has st.stop() somewhere
with main_tabs[1]:
    st.subheader("Data Preview")
    st.dataframe(df.head(preview_rows), use_container_width=True, hide_index=True)
    st.subheader("Inferred Types")
    st.caption("Inferred types are heuristic; you can still pick any column you want.")
    st.dataframe(info_df, use_container_width=True, hide_index=True)

with main_tabs[0]:
    mode = st.radio(
        "Variables",
        [
            "One categorical column",
            "Two categorical columns",
            "One numeric column",
            "Numeric vs categorical columns",
            "Two numeric columns",
        ],
        horizontal=True,
    )

    # Session state for optional title/axis overrides (Option A)
    for key in ("custom_plot_title", "custom_plot_x", "custom_plot_y"):
        if key not in st.session_state:
            st.session_state[key] = ""
    # Persist text input into the *previously* selected slot so we don't overwrite the new selection when user switches dropdown
    _key_map = {"Title": "custom_plot_title", "X axis label": "custom_plot_x", "Y axis label": "custom_plot_y"}
    if "label_input" in st.session_state:
        _prev = st.session_state.get("label_which_prev") or st.session_state.get("label_which")
        _current = _key_map.get(_prev) if _prev else None
        if _current:
            st.session_state[_current] = st.session_state.get("label_input", "")

    # --- Visualization branches: one per variable type (1 cat, 2 cat, 1 num, num vs cat, 2 num) ---
    if mode == "One categorical column":
        r1 = st.columns(2)
        with r1[0]:
            x = st.selectbox("Categorical column", options=df.columns.tolist(), index=0, key="v_cat1_x")
        with r1[1]:
            chart_type = st.selectbox("Plot type", ["Bar chart", "Pie chart"], index=0, key="v_cat1_pt")
        show_table = st.checkbox("Show frequency table", value=False, key="v_cat1_table")
        d = df[[x]].copy()
        if drop_na_rows:
            d = d.dropna()
        ensure_categorical(d, x, context="the bar chart")

        if chart_type == "Bar chart":
            # Decide default orientation and label rotation based on how many categories there are.
            n_x = int(d[x].nunique(dropna=True))
            r2 = st.columns(2)
            with r2[0]:
                show_percent = st.checkbox("Plot proportions instead of counts", value=False, key="v_cat1_pct")
            with r2[1]:
                flip_default = n_x > 10
                flip = st.checkbox("Flip coordinates (horizontal bars)", value=flip_default, key="v_cat1_flip")

        if show_table:
            freq = (
                d[x].value_counts(dropna=False)
                .rename("count")
                .to_frame()
                .assign(percent=lambda t: (t["count"] / t["count"].sum()) * 100)
            )
            st.caption("Frequency table")
            st.dataframe(freq, use_container_width=True)

        if chart_type == "Bar chart":
            if show_percent:
                tit, x_lab, y_lab = get_plot_labels(f"Distribution of {x}", x, "Proportion")
                p = (
                    ggplot(d, aes(x=x, y="..count../sum(..count..)", group=1))
                    + geom_bar()
                    + labs(title=tit, x=x_lab, y=y_lab)
                )
            else:
                tit, x_lab, y_lab = get_plot_labels(f"Distribution of {x}", x, "Count")
                p = (
                    ggplot(d, aes(x=x))
                    + geom_bar()
                    + labs(title=tit, x=x_lab, y=y_lab)
                )
            if flip:
                p = p + coord_flip()
            n_cats = n_x if flip and n_x > 10 else None
            render_label_customizer_expander()
            render_plotnine(p, selected_theme, data=d, x_col=x, n_x_categories=n_cats)
            plot_key = f"cat_one_bar_{x}_{show_percent}_{flip}"
            context = build_plot_context("bar_chart", {"x": x, "show_percent": show_percent, "flip": flip}, d)
            render_plot_chat(plot_key, context)
        else:
            freq_df = (
                d[x].value_counts(dropna=False)
                .rename("count")
                .to_frame()
                .reset_index()
                .rename(columns={"index": x})
            )
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(
                freq_df["count"],
                labels=freq_df[x],
                autopct="%1.1f%%",
                startangle=90,
                radius=0.6,
            )
            render_label_customizer_expander()
            tit, _, _ = get_plot_labels(f"Distribution of {x}", x, "")
            ax.set_title(tit)
            ax.axis("equal")
            left, center, right = st.columns([0.3, 2, 0.5])
            with center:
                st.pyplot(fig, use_container_width=False)
            plot_key = f"cat_one_pie_{x}"
            context = build_plot_context("pie_chart", {"x": x}, d)
            render_plot_chat(plot_key, context)

    elif mode == "Two categorical columns":
        r1 = st.columns(2)
        with r1[0]:
            x = st.selectbox("X (categorical)", options=df.columns.tolist(), index=0, key="v_cat2_x")
        with r1[1]:
            fill = st.selectbox("Group / Fill (categorical)", options=df.columns.tolist(), index=min(1, len(df.columns) - 1), key="v_cat2_fill")
        r2 = st.columns(2)
        with r2[0]:
            style = st.selectbox("Bar style", ["Side-by-side (dodge)", "Stacked", "Proportional (fill)"], index=0, key="v_cat2_style")
        with r2[1]:
            n_x = int(df[x].nunique(dropna=True))
            flip_default = n_x > 10
            flip = st.checkbox("Flip coordinates (horizontal bars)", value=flip_default, key="v_cat2_flip")

        if x == fill:
            st.error("Please select two different variables.")
            st.stop()

        d = df[[x, fill]].copy()
        if drop_na_rows:
            d = d.dropna()
        ensure_categorical(d, x, context="the bar plot")
        ensure_categorical(d, fill, context="grouping")

        if style == "Side-by-side (dodge)":
            tit, x_lab, y_lab = get_plot_labels(f"{fill} by {x}", x, "Count")
            p = (
                ggplot(d, aes(x=x, fill=fill))
                + geom_bar(position="dodge")
                + labs(title=tit, x=x_lab, y=y_lab, fill=fill)
            )
        elif style == "Proportional (fill)":
            tit, x_lab, y_lab = get_plot_labels(f"{fill} by {x} (Proportions)", x, "Proportion")
            p = (
                ggplot(d, aes(x=x, fill=fill))
                + geom_bar(position="fill")
                + labs(title=tit, x=x_lab, y=y_lab, fill=fill)
            )
        else:
            tit, x_lab, y_lab = get_plot_labels(f"{fill} by {x} (Stacked)", x, "Count")
            p = (
                ggplot(d, aes(x=x, fill=fill))
                + geom_bar()
                + labs(title=tit, x=x_lab, y=y_lab, fill=fill)
            )
        if flip:
            p = p + coord_flip()
        render_label_customizer_expander()
        n_cats = n_x if flip and n_x > 10 else None
        render_plotnine(p, selected_theme, data=d, x_col=x, n_x_categories=n_cats)
        plot_key = f"cat_two_bar_{x}_{fill}_{style}_{flip}"
        context = build_plot_context("grouped_bar", {"x": x, "fill": fill, "style": style, "flip": flip}, d)
        render_plot_chat(plot_key, context)

    elif mode == "One numeric column":
        r1 = st.columns(2)
        with r1[0]:
            x = st.selectbox("Numeric column", options=df.columns.tolist(), index=0, key="v_num1_x")
        with r1[1]:
            plot_type = st.selectbox("Plot type", ["Histogram", "Boxplot"], index=0, key="v_num1_pt")

        d = df[[x]].copy()
        if drop_na_rows:
            d = d.dropna()
        ensure_numeric(d, x)

        if plot_type == "Histogram":
            r2 = st.columns(3)
            with r2[0]:
                binwidth = st.number_input("Binwidth (0 = auto)", min_value=0.0, value=0.0, step=1.0, key="v_num1_bw")
            with r2[1]:
                boundary = st.number_input("Boundary (optional)", value=float("nan"), key="v_num1_bd")
            with r2[2]:
                density = st.checkbox("Scale y to density", value=False, key="v_num1_dens")

            if density:
                p = ggplot(d, aes(x=x, y="..density.."))
                ylab = "Density"
            else:
                p = ggplot(d, aes(x=x))
                ylab = "Count"
            hist_kwargs = {}
            if binwidth and binwidth > 0:
                hist_kwargs["binwidth"] = binwidth
            if not np.isnan(boundary):
                hist_kwargs["boundary"] = boundary
            tit, x_lab, y_lab = get_plot_labels(f"Histogram of {x}", x, ylab)
            p = p + geom_histogram(**hist_kwargs) + labs(title=tit, x=x_lab, y=y_lab)
            render_label_customizer_expander()
            render_plotnine(p, selected_theme)
            plot_key = f"num_one_hist_{x}_{binwidth}_{boundary}_{density}"
            context = build_plot_context("histogram", {"x": x, "binwidth": binwidth, "boundary": boundary, "density": density}, d)
            render_plot_chat(plot_key, context)
        else:
            tit, x_lab, y_lab = get_plot_labels(f"Boxplot of {x}", "", x)
            p = (
                ggplot(d, aes(x=1, y=x))
                + geom_boxplot(width=0.3)
                + xlim(0.5, 1.5)
                + labs(title=tit, x=x_lab, y=y_lab)
            )
            render_label_customizer_expander()
            render_plotnine(p, selected_theme)
            plot_key = f"num_one_box_{x}"
            context = build_plot_context("boxplot", {"x": x}, d)
            render_plot_chat(plot_key, context)

    elif mode == "Numeric vs categorical columns":
        r1 = st.columns(2)
        with r1[0]:
            y = st.selectbox("Numeric column", options=df.columns.tolist(), index=0, key="v_numcat_y")
        with r1[1]:
            x = st.selectbox("Categorical column", options=df.columns.tolist(), index=min(1, len(df.columns) - 1), key="v_numcat_x")
        plot_type = st.selectbox(
            "Plot type",
            ["Side-by-side boxplot", "Faceted histogram", "Bar chart", "Pie chart", "Line plot"],
            index=0,
            key="v_numcat_pt",
        )

        if x == y:
            st.error("Please select two different variables.")
            st.stop()

        d = df[[x, y]].copy()
        if drop_na_rows:
            d = d.dropna()
        ensure_numeric(d, y)
        ensure_categorical(d, x, context="grouping")

        if plot_type == "Side-by-side boxplot":
            # For many categories, default to horizontal layout
            n_x = int(d[x].nunique(dropna=True))
            tit, x_lab, y_lab = get_plot_labels(f"{y} by {x}", x, y)
            p = (
                ggplot(d, aes(x=x, y=y))
                + geom_boxplot()
                + labs(title=tit, x=x_lab, y=y_lab)
            )
            if n_x > 10:
                p = p + coord_flip()
            n_cats = n_x if n_x > 10 else None
            render_label_customizer_expander()
            render_plotnine(p, selected_theme, data=d, x_col=x, n_x_categories=n_cats)
            plot_key = f"num_cat_box_{x}_{y}"
            context = build_plot_context("grouped_boxplot", {"x": x, "y": y}, d)
            render_plot_chat(plot_key, context)
        elif plot_type == "Faceted histogram":
            binwidth = st.number_input("Histogram binwidth (0 for default)", min_value=0.0, value=0.0, step=1.0, key="v_numcat_bw")
            hist_kwargs = {}
            if binwidth and binwidth > 0:
                hist_kwargs["binwidth"] = binwidth
            tit, x_lab, y_lab = get_plot_labels(f"Histogram of {y}, faceted by {x}", y, "Count")
            p = (
                ggplot(d, aes(x=y))
                + geom_histogram(**hist_kwargs)
                + facet_grid(f"{x} ~ .")
                + labs(title=tit, x=x_lab, y=y_lab)
            )
            render_label_customizer_expander()
            render_plotnine(p, selected_theme)
            plot_key = f"num_cat_facethist_{x}_{y}_{binwidth}"
            context = build_plot_context("faceted_histogram", {"x": x, "y": y, "binwidth": binwidth}, d)
            render_plot_chat(plot_key, context)
        elif plot_type == "Line plot":
            d_line = d.copy()
            # check if x is a time series. If so, we must convert x to time series. Otherwise, plotnine does not work. 
            x_parsed = pd.to_datetime(d_line[x], errors="coerce")
            if x_parsed.notna().mean() >= 0.5:
                d_line["_x_time"] = x_parsed
                d_line = d_line.dropna(subset=["_x_time"]).sort_values("_x_time")
                tit, x_lab, y_lab = get_plot_labels(f"{y} by {x}", x, y)
                p = (
                    ggplot(d_line, aes(x="_x_time", y=y, group=1))
                    + geom_line()
                    + labs(title=tit, x=x_lab, y=y_lab)
                    + scale_x_datetime(date_breaks="1 year", date_labels="%Y", expand=(0,0))
                )
            else:
                tit, x_lab, y_lab = get_plot_labels(f"{y} by {x}", x, y)
                p = (
                    ggplot(d_line, aes(x=x, y=y, group=1))
                    + geom_line()
                    + labs(title=tit, x=x_lab, y=y_lab)
                )
            render_label_customizer_expander()
            render_plotnine(p, selected_theme, data=d_line, x_col=x if x_parsed.notna().mean() < 0.5 else None)
            plot_key = f"num_cat_line_{x}_{y}"
            context = build_plot_context("line_plot", {"x": x, "y": y}, d)
            render_plot_chat(plot_key, context)
        else:
            # Bar chart and Pie chart: use numeric column directly (no value_counts)
            # Data assumed: one row per category, numeric col = value (percentage by default, or count)
            if plot_type == "Bar chart":
                # change option from percentage to count only changes the y label. 
                value_type = st.radio(
                    "Numerical column represents",
                    ["Percentage", "Count"],
                    index=0,
                    horizontal=True,
                    key="v_numcat_valtype",
                )
                ylab = "Percentage" if value_type == "Percentage" else "Count"

                n_x = int(d[x].nunique(dropna=True))
                flip_default = n_x > 10
                flip = st.checkbox("Flip coordinates (horizontal bars)", value=flip_default, key="v_numcat_bar_flip")
                tit, x_lab, y_lab = get_plot_labels(f"{y} by {x}", x, ylab)
                p = (
                    ggplot(d, aes(x=x, y=y))
                    + geom_col()
                    + labs(title=tit, x=x_lab, y=y_lab)
                )
                if flip:
                    p = p + coord_flip()
                n_cats = n_x if flip and n_x > 10 else None
                render_label_customizer_expander()
                render_plotnine(p, selected_theme, data=d, x_col=x, n_x_categories=n_cats)
                plot_key = f"num_cat_bar_{x}_{y}_{value_type}_{flip}"
                context = build_plot_context("bar_chart_preagg", {"x": x, "y": y, "value_type": value_type, "flip": flip}, d)
                render_plot_chat(plot_key, context)
            else:  # Pie chart
                # pie shows the percentage calculated based on y column no matter y is percentage or count. 
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie(
                    d[y],
                    labels=d[x],
                    autopct="%1.1f%%",
                    startangle=90,
                    radius=0.6,
                )
                render_label_customizer_expander()
                tit, _, _ = get_plot_labels(f"{y} by {x}", x, "")
                ax.set_title(tit)
                ax.axis("equal")
                left, center, right = st.columns([0.3, 2, 0.5])
                with center:
                    st.pyplot(fig, use_container_width=False)
                plot_key = f"num_cat_pie_{x}_{y}"
                context = build_plot_context("pie_chart_preagg", {"x": x, "y": y}, d)
                render_plot_chat(plot_key, context)

    else:  # Two numeric columns
        r1 = st.columns(2)
        with r1[0]:
            x = st.selectbox("X (numeric)", options=df.columns.tolist(), index=0, key="v_num2_x")
        with r1[1]:
            y = st.selectbox("Y (numeric)", options=df.columns.tolist(), index=min(1, len(df.columns) - 1), key="v_num2_y")
        add_regression = st.checkbox("Add linear regression line", value=False, key="v_num2_reg")

        if x == y:
            st.error("Please select two different variables.")
            st.stop()

        d = df[[x, y]].copy()
        if drop_na_rows:
            d = d.dropna()
        for col in [x, y]:
            ensure_numeric(d, col)

        tit, x_lab, y_lab = get_plot_labels(f"{y} vs {x}", x, y)
        p = (
            ggplot(d, aes(x=x, y=y))
            + geom_point()
            + labs(title=tit, x=x_lab, y=y_lab)
        )
        if add_regression:
            p = p + geom_smooth(method="lm", se=False, color="red")
        render_label_customizer_expander()
        render_plotnine(p, selected_theme)
        plot_key = f"num_scatter_{x}_{y}_{add_regression}"
        context = build_plot_context("scatter", {"x": x, "y": y, "add_regression": add_regression}, d)
        render_plot_chat(plot_key, context)


# st.caption("Tip: If a column is numeric-but-actually-categorical (like 0/1 or ratings), you can still treat it as categorical by choosing it in the categorical tab.")