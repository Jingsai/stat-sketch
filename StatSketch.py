import io
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from plotnine import (
    ggplot, aes, geom_bar, geom_histogram, geom_col, geom_point, geom_boxplot,
    facet_grid, labs, theme_bw, coord_flip
)
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
def is_numeric(series: pd.Series) -> bool:
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


def render_plotnine(p):
    """Render a plotnine plot into Streamlit."""
    fig = p.draw()
    
    # Use responsive size
    width = 6
    height = width * 0.6   # keep aspect ratio

    fig.set_size_inches(width, height)

    st.pyplot(fig, clear_figure=True, use_container_width=True)

@st.cache_data(show_spinner=False)
def load_csv_from_upload(uploaded_file, sep=","):
    return pd.read_csv(uploaded_file, sep=sep)


@st.cache_data(show_spinner=False)
def load_csv_from_url(url: str, sep=","):
    with urllib.request.urlopen(url) as resp:
        content = resp.read()
    return pd.read_csv(io.BytesIO(content), sep=sep)


# def safe_top_categories(s: pd.Series, top_k: int):
#     vc = s.value_counts(dropna=False)
#     if top_k is None or top_k <= 0 or len(vc) <= top_k:
#         return s
#     top = vc.index[:top_k]
#     return s.where(s.isin(top), other="Other")

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="StatSketch", layout="wide")
st.title("StatSketch: No-code Data Visualizer")

st.sidebar.header("Load a CSV File")
source = st.sidebar.radio("Data source", ["Upload file", "CSV URL"], horizontal=False)
sep = st.sidebar.selectbox("Separator", [",", "\t", ";", "|"], index=0)

df = None
if source == "Upload file":
    uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if uploaded is not None:
        df = load_csv_from_upload(uploaded, sep=sep)
else:
    url = st.sidebar.text_input("Paste a direct CSV URL")
    if url.strip():
        try:
            df = load_csv_from_url(url.strip(), sep=sep)
        except Exception as e:
            st.sidebar.error(f"Could not load CSV from URL: {e}")

if df is None:
    st.info("Upload a CSV or provide a CSV URL to begin.")
    st.stop()

# Basic cleaning controls
st.sidebar.header("Basic options")
drop_na_rows = st.sidebar.checkbox("Drop rows with NA in selected columns (recommended)", value=True)
preview_rows = st.sidebar.slider("Preview rows", 5, 50, 10)

st.subheader("Data preview")
st.dataframe(df.head(preview_rows), use_container_width=True)

# Infer variable types
col_info = []
for c in df.columns:
    s = df[c]
    ctype = "numeric" if is_numeric(s) and not is_categorical(s) else "categorical"
    col_info.append((c, ctype, int(s.isna().sum()), int(s.dropna().nunique())))
info_df = pd.DataFrame(col_info, columns=["column", "inferred_type", "missing", "unique_non_na"])
st.caption("Inferred types are heuristic; you can still pick any column you want.")
st.dataframe(info_df, use_container_width=True, hide_index=True)

categorical_cols = info_df.query("inferred_type == 'categorical'")["column"].tolist()
numeric_cols = info_df.query("inferred_type == 'numeric'")["column"].tolist()

tabs = st.tabs(["Categorical Data", "Numerical Data"])


# -----------------------------
# Categorical tab
# -----------------------------
with tabs[0]:
    st.header("Categorical Data Visualizations")

    mode = st.radio("How many variables?", ["One categorical variable", "Two categorical variables"], horizontal=True)

    if mode == "One categorical variable":
        x = st.selectbox("Categorical column", options=df.columns.tolist(), index=0)

        # top_k = st.slider("Show top-k categories (others → 'Other')", 0, 30, 0)
        show_table = st.checkbox("Show frequency table", value=True)
        show_percent = st.checkbox("Plot proportions instead of counts", value=False)
        flip = st.checkbox("Flip coordinates (horizontal bars)", value=False)

        d = df[[x]].copy()
        if drop_na_rows:
            d = d.dropna()

        # check type 
        if pd.api.types.is_numeric_dtype(d[x]):
            st.warning("Numeric variable detected. Treating values as categories.")
            d[x] = d[x].astype(str)

        # d[x] = safe_top_categories(d[x], top_k)

        if show_table:
            freq = (
                d[x].value_counts(dropna=False)
                .rename("count")
                .to_frame()
                .assign(percent=lambda t: (t["count"] / t["count"].sum()) * 100)
            )
            st.subheader("Frequency table")
            st.dataframe(freq, use_container_width=True)

        st.subheader("Bar chart")
        if show_percent:
            # plotnine: use after_stat for proportions via geom_bar + computed prop
            p = (
                ggplot(d, aes(x=x, y="..count../sum(..count..)", group=1))
                + geom_bar()
                + labs(title=f"Distribution of {x}", x=x, y="Proportion")
                + theme_bw()
            )
        else:
            p = (
                ggplot(d, aes(x=x))
                + geom_bar()
                + labs(title=f"Distribution of {x}", x=x, y="Count")
                + theme_bw()
            )

        if flip:
            p = p + coord_flip()

        render_plotnine(p)

    else:
        x = st.selectbox("X (categorical)", options=df.columns.tolist(), index=0)
        fill = st.selectbox("Group / Fill (categorical)", options=df.columns.tolist(), index=min(1, len(df.columns)-1))

        if x == fill:
            st.error("Please select two different variables.")
            st.stop()

        style = st.selectbox("Bar style", ["Side-by-side (dodge)", "Stacked", "Proportional (fill)"], index=0)
        flip = st.checkbox("Flip coordinates (horizontal bars)", value=False)

        d = df[[x, fill]].copy()
        if drop_na_rows:
            d = d.dropna()

        # Check x
        if pd.api.types.is_numeric_dtype(d[x]):
            st.warning(f"'{x}' is numeric. Treating it as a categorical variable for the bar plot.")
            d[x] = d[x].astype(str)

        # Check fill
        if pd.api.types.is_numeric_dtype(d[fill]):
            st.warning(f"'{fill}' is numeric. Treating it as a categorical variable for grouping.")
            d[fill] = d[fill].astype(str)

        st.subheader("Bar chart")
        if style == "Side-by-side (dodge)":
            p = (
                ggplot(d, aes(x=x, fill=fill))
                + geom_bar(position="dodge")
                + labs(title=f"{fill} by {x}", x=x, y="Count", fill=fill)
                + theme_bw()
            )
        elif style == "Proportional (fill)":
            p = (
                ggplot(d, aes(x=x, fill=fill))
                + geom_bar(position="fill")
                + labs(title=f"{fill} by {x} (Proportions)", x=x, y="Proportion", fill=fill)
                + theme_bw()
            )
        else:
            p = (
                ggplot(d, aes(x=x, fill=fill))
                + geom_bar()
                + labs(title=f"{fill} by {x} (Stacked)", x=x, y="Count", fill=fill)
                + theme_bw()
            )

        if flip:
            p = p + coord_flip()

        render_plotnine(p)


# -----------------------------
# Numerical tab
# -----------------------------
with tabs[1]:
    st.header("Numerical Data Visualizations")

    mode = st.radio("What are you plotting?", ["One numeric variable", "Numeric vs categorical", "Two numeric variables"], horizontal=True)

    if mode == "One numeric variable":
        x = st.selectbox("Numeric column", options=df.columns.tolist(), index=0)

        # Choose plot type for one numeric variable
        plot_type = st.selectbox("Plot type", ["Histogram", "Boxplot"], index=0)

        # Build dataframe for plotting
        d = df[[x]].copy()
        if drop_na_rows:
            d = d.dropna()

        # If not numeric, warn and try converting to numeric
        if not pd.api.types.is_numeric_dtype(d[x]):
            st.warning(
                f"'{x}' does not appear to be numeric. Attempting to convert it to numbers."
            )
            converted = pd.to_numeric(d[x], errors="coerce")

            # If any non-missing values become NaN after conversion, treat as failure
            # (i.e., there were values that cannot be converted)
            if converted.isna().any():
                st.error(
                    f"'{x}' cannot be safely converted to numeric values. "
                    "Please select a numeric variable."
                )
                st.stop()

            d[x] = converted

        # Now we are guaranteed numeric
        if plot_type == "Histogram":
            binwidth = st.number_input(
                "Histogram binwidth (leave as 0 for auto bins)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )
            boundary = st.number_input(
                "Histogram boundary (optional)",
                value=float("nan"),
            )
            density = st.checkbox(
                "Scale y-axis to density",
                value=False,
                help="Useful when comparing distributions; for a single variable, count is usually fine.",
            )

            # Build plot
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

            p = (
                p
                + geom_histogram(**hist_kwargs)
                + labs(title=f"Histogram of {x}", x=x, y=ylab)
                + theme_bw()
            )
            render_plotnine(p)

        else:  # Boxplot
            p = (
                ggplot(d, aes(y=x))
                + geom_boxplot(width=0.3)
                + labs(title=f"Boxplot of {x}", x="", y=x)
                + theme_bw()
            )
            render_plotnine(p)

    elif mode == "Numeric vs categorical":
        y = st.selectbox("Numeric column", options=df.columns.tolist(), index=0)
        x = st.selectbox("Categorical column", options=df.columns.tolist(), index=min(1, len(df.columns)-1))

        if x == y:
            st.error("Please select two different variables.")
            st.stop()

        plot_type = st.selectbox("Plot type", ["Side-by-side boxplot", "Faceted histogram"], index=0)

        d = df[[x, y]].copy()
        if drop_na_rows:
            d = d.dropna()

        # ---- Type checking ----
        # 1) y should be numeric: if not, warn and try convert; if cannot, stop.
        if not pd.api.types.is_numeric_dtype(d[y]):
            st.warning(f"'{y}' does not appear to be numeric. Attempting to convert it to numbers.")
            converted = pd.to_numeric(d[y], errors="coerce")

            # if any values could not be converted, stop
            if converted.isna().any():
                st.error(f"'{y}' cannot be safely converted to numeric values. Please select a numeric variable.")
                st.stop()

            d[y] = converted

        # 2) x should be categorical: if numeric, warn and treat as categories (string)
        if pd.api.types.is_numeric_dtype(d[x]):
            st.warning(f"'{x}' is numeric. Treating it as a categorical variable for grouping.")
            d[x] = d[x].astype(str)
        else:
            # keep categorical clean for labels / legends
            d[x] = d[x].astype(str)

        if plot_type == "Side-by-side boxplot":
            st.subheader("Boxplot")
            p = (
                ggplot(d, aes(x=x, y=y))
                + geom_boxplot()
                + labs(title=f"{y} by {x}", x=x, y=y)
                + theme_bw()
            )
            render_plotnine(p)
        else:
            st.subheader("Faceted histogram")
            binwidth = st.number_input("Histogram binwidth (0 for default)", min_value=0.0, value=0.0, step=1.0)
            hist_kwargs = {}
            if binwidth and binwidth > 0:
                hist_kwargs["binwidth"] = binwidth

            p = (
                ggplot(d, aes(x=y))
                + geom_histogram(**hist_kwargs)
                + facet_grid(f"{x} ~ .")
                + labs(title=f"Histogram of {y}, faceted by {x}", x=y, y="Count")
                + theme_bw()
            )
            render_plotnine(p)

    else:
        x = st.selectbox("X (numeric)", options=df.columns.tolist(), index=0)
        y = st.selectbox("Y (numeric)", options=df.columns.tolist(), index=min(1, len(df.columns)-1))

        if x == y:
            st.error("Please select two different variables.")
            st.stop()

        d = df[[x, y]].copy()
        if drop_na_rows:
            d = d.dropna()

        # ---- Type checking ----
        for col in [x, y]:
            if not pd.api.types.is_numeric_dtype(d[col]):
                st.warning(
                    f"'{col}' does not appear to be numeric. Attempting to convert it to numbers."
                )

                converted = pd.to_numeric(d[col], errors="coerce")

                if converted.isna().any():
                    st.error(
                        f"'{col}' cannot be safely converted to numeric values. "
                        "Please select a numeric variable."
                    )
                    st.stop()

                d[col] = converted

        st.subheader("Scatter plot")
        p = (
            ggplot(d, aes(x=x, y=y))
            + geom_point()
            + labs(title=f"{y} vs {x}", x=x, y=y)
            + theme_bw()
        )
        render_plotnine(p)


st.caption("Tip: If a column is numeric-but-actually-categorical (like 0/1 or ratings), you can still treat it as categorical by choosing it in the categorical tab.")