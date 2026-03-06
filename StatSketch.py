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
    geom_boxplot,
    facet_grid,
    labs,
    theme_bw,
    theme_gray,
    theme_matplotlib,
    theme_seaborn,
    theme_minimal,
    xlim,
    coord_flip,
    geom_smooth,
)
from plotnine.data import penguins
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


def ensure_numeric(df: pd.DataFrame, col: str) -> None:
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
    if converted.isna().any():
        st.error(
            f"'{col}' cannot be safely converted to numeric values. "
            "Please select a numeric variable."
        )
        st.stop()
    df[col] = converted


def render_plotnine(p, theme=None):
    """Render a plotnine plot into Streamlit."""
    if theme is not None:
        p = p + theme

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


def build_plot_context(plot_type: str, params: dict, data: pd.DataFrame) -> str:
    """Build text summary: plot type, parameters, and statistical summary for the LLM."""
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
    """Render chat UI below a plot. plot_key resets chat when plot changes (per-plot history)."""
    chat_key = f"plot_chat_{plot_key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    messages = st.session_state[chat_key]
    api_key = get_groq_api_key()
    with st.expander("Ask about this plot", expanded=len(messages) > 0):
        for m in messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        prompt = st.chat_input("Ask about this plot...")
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
# App
# -----------------------------
st.set_page_config(page_title="StatSketch", layout="wide")
st.title("StatSketch: No-code Data Visualizer")

st.sidebar.header("Load data")
source = st.sidebar.radio(
    "Data source",
    ["Upload file", "CSV URL", "Example dataset (penguins)"],
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
drop_first_id_column = st.sidebar.checkbox(
    "Drop first column (e.g. ID)",
    value=False,
    help="Remove the first column from the data if it looks like an index/ID column.",
)
drop_na_rows = st.sidebar.checkbox("Drop rows with NA in selected columns (recommended)", value=True)
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

if drop_first_id_column and len(df.columns) > 1:
    df = df.iloc[:, 1:].copy()
elif drop_first_id_column and len(df.columns) == 1:
    st.sidebar.warning("Only one column in data; cannot drop first column.")

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
        chart_type = st.selectbox("Plot type", ["Bar chart", "Pie chart"], index=0)
        show_table = st.checkbox("Show frequency table", value=True)

        if chart_type == "Bar chart":
            show_percent = st.checkbox("Plot proportions instead of counts", value=False)
            flip = st.checkbox("Flip coordinates (horizontal bars)", value=False)

        d = df[[x]].copy()
        if drop_na_rows:
            d = d.dropna()

        ensure_categorical(d, x, context="the bar chart")

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

        if chart_type == "Bar chart":
            st.subheader("Bar chart")
            if show_percent:
                # plotnine: use after_stat for proportions via geom_bar + computed prop
                p = (
                    ggplot(d, aes(x=x, y="..count../sum(..count..)", group=1))
                    + geom_bar()
                    + labs(title=f"Distribution of {x}", x=x, y="Proportion")
                )
            else:
                p = (
                    ggplot(d, aes(x=x))
                    + geom_bar()
                    + labs(title=f"Distribution of {x}", x=x, y="Count")
                )

            if flip:
                p = p + coord_flip()

            render_plotnine(p, selected_theme)
            plot_key = f"cat_one_bar_{x}_{show_percent}_{flip}"
            context = build_plot_context("bar_chart", {"x": x, "show_percent": show_percent, "flip": flip}, d)
            render_plot_chat(plot_key, context)

        else:
            st.subheader("Pie chart")
            freq_df = (
                d[x]
                .value_counts(dropna=False)
                .rename("count")
                .to_frame()
                .reset_index()
                .rename(columns={"index": x})
            )
            # pie chart is not supported by plotnine, so we use matplotlib instead
            # p = (
            #     ggplot(freq_df, aes(x="''", y="count", fill=x))
            #     + geom_col(width=1)
            #     + coord_polar(theta="y")
            #     + labs(title=f"Distribution of {x}", x="", y="")
            # )
            # render_plotnine(p, selected_theme)

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(
                freq_df["count"],
                labels=freq_df[x],
                autopct="%1.1f%%",
                startangle=90,
                radius=0.6
            )
            ax.set_title(f"Distribution of {x}")
            ax.axis("equal")
            
            left, center, right = st.columns([0.3,2,0.5])
            with center:
                st.pyplot(fig, use_container_width=False)

            plot_key = f"cat_one_pie_{x}"
            context = build_plot_context("pie_chart", {"x": x}, d)
            render_plot_chat(plot_key, context)

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

        ensure_categorical(d, x, context="the bar plot")
        ensure_categorical(d, fill, context="grouping")

        st.subheader("Bar chart")
        if style == "Side-by-side (dodge)":
            p = (
                ggplot(d, aes(x=x, fill=fill))
                + geom_bar(position="dodge")
                + labs(title=f"{fill} by {x}", x=x, y="Count", fill=fill)
            )
        elif style == "Proportional (fill)":
            p = (
                ggplot(d, aes(x=x, fill=fill))
                + geom_bar(position="fill")
                + labs(title=f"{fill} by {x} (Proportions)", x=x, y="Proportion", fill=fill)
            )
        else:
            p = (
                ggplot(d, aes(x=x, fill=fill))
                + geom_bar()
                + labs(title=f"{fill} by {x} (Stacked)", x=x, y="Count", fill=fill)
            )

        if flip:
            p = p + coord_flip()

        render_plotnine(p, selected_theme)
        plot_key = f"cat_two_bar_{x}_{fill}_{style}_{flip}"
        context = build_plot_context("grouped_bar", {"x": x, "fill": fill, "style": style, "flip": flip}, d)
        render_plot_chat(plot_key, context)


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

        ensure_numeric(d, x)

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
            )
            render_plotnine(p, selected_theme)
            plot_key = f"num_one_hist_{x}_{binwidth}_{boundary}_{density}"
            context = build_plot_context("histogram", {"x": x, "binwidth": binwidth, "boundary": boundary, "density": density}, d)
            render_plot_chat(plot_key, context)

        else:  # Boxplot
            p = (
                ggplot(d, aes(x=1, y=x))
                + geom_boxplot(width=0.3)
                + xlim(0.5, 1.5)
                + labs(title=f"Boxplot of {x}", x="", y=x)
            )
            render_plotnine(p, selected_theme)
            plot_key = f"num_one_box_{x}"
            context = build_plot_context("boxplot", {"x": x}, d)
            render_plot_chat(plot_key, context)

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

        ensure_numeric(d, y)
        ensure_categorical(d, x, context="grouping")

        if plot_type == "Side-by-side boxplot":
            st.subheader("Boxplot")
            p = (
                ggplot(d, aes(x=x, y=y))
                + geom_boxplot()
                + labs(title=f"{y} by {x}", x=x, y=y)
            )
            render_plotnine(p, selected_theme)
            plot_key = f"num_cat_box_{x}_{y}"
            context = build_plot_context("grouped_boxplot", {"x": x, "y": y}, d)
            render_plot_chat(plot_key, context)
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
            )
            render_plotnine(p, selected_theme)
            plot_key = f"num_cat_facethist_{x}_{y}_{binwidth}"
            context = build_plot_context("faceted_histogram", {"x": x, "y": y, "binwidth": binwidth}, d)
            render_plot_chat(plot_key, context)

    else:
        x = st.selectbox("X (numeric)", options=df.columns.tolist(), index=0)
        y = st.selectbox("Y (numeric)", options=df.columns.tolist(), index=min(1, len(df.columns)-1))

        if x == y:
            st.error("Please select two different variables.")
            st.stop()

        d = df[[x, y]].copy()
        if drop_na_rows:
            d = d.dropna()

        for col in [x, y]:
            ensure_numeric(d, col)

        st.subheader("Scatter plot")
        add_regression = st.checkbox("Add linear regression line", value=False)
        p = (
            ggplot(d, aes(x=x, y=y))
            + geom_point()
            + labs(title=f"{y} vs {x}", x=x, y=y)
        )
        if add_regression:
            p = p + geom_smooth(method="lm", se=False, color="red")

        render_plotnine(p, selected_theme)
        plot_key = f"num_scatter_{x}_{y}_{add_regression}"
        context = build_plot_context("scatter", {"x": x, "y": y, "add_regression": add_regression}, d)
        render_plot_chat(plot_key, context)


# st.caption("Tip: If a column is numeric-but-actually-categorical (like 0/1 or ratings), you can still treat it as categorical by choosing it in the categorical tab.")