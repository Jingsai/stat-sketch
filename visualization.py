import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from groq_chat import GROQ_SYSTEM_VISUALIZATION, call_groq

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
    scale_x_discrete,
)

from helper import (
    is_numeric,
    is_categorical,
    ensure_numeric,
    ensure_categorical,
    compute_axis_breaks,
)


THEME_MAP = {
    "Gray": theme_gray,
    "Matplotlib": theme_matplotlib,
    "Seaborn": theme_seaborn,
    "Black & white": theme_bw,
    "Minimal": theme_minimal,
}


def get_plot_labels(default_title: str, default_x: str, default_y: str, plot_key: str) -> tuple[str, str, str]:
    """Return (title, x_label, y_label). Uses per-plot session-state overrides when non-empty, else defaults."""
    labels = st.session_state.get("custom_plot_labels", {}).get(plot_key, {})
    t = (labels.get("title") or "") or default_title
    x = (labels.get("x") or "") or default_x
    y = (labels.get("y") or "") or default_y
    return (t, x, y)


def render_label_customizer_expander(plot_key: str, allowed_fields: tuple[str, ...] = ("Title", "X axis label", "Y axis label")):
    if "custom_plot_labels" not in st.session_state:
        st.session_state["custom_plot_labels"] = {}
    if plot_key not in st.session_state["custom_plot_labels"]:
        st.session_state["custom_plot_labels"][plot_key] = {"title": "", "x": "", "y": ""}
    labels = st.session_state["custom_plot_labels"][plot_key]
    slot_map = {"Title": "title", "X axis label": "x", "Y axis label": "y"}
    options = [opt for opt in allowed_fields if opt in slot_map] or ["Title"]
    if st.session_state.get("label_edit_plot_key") != plot_key:
        current_which = st.session_state.get("label_which", options[0])
        if current_which not in options:
            current_which = options[0]
        current_slot = slot_map.get(current_which, "title")
        st.session_state["label_input"] = labels.get(current_slot, "")
    with st.expander("Customize title & axis labels", expanded=False):
        label_which = st.selectbox("Edit", options, key="label_which")
        current_slot = slot_map[label_which]
        if st.session_state.get("label_which_prev") != label_which:
            st.session_state["label_input"] = labels.get(current_slot, "")
        st.session_state["label_which_prev"] = label_which
        st.text_input("Value", key="label_input", placeholder="Leave blank for default")
        labels[current_slot] = st.session_state.get("label_input", "")
    st.session_state["label_edit_plot_key"] = plot_key


def theme_x_labels(df: pd.DataFrame, x: str):
    n = int(df[x].nunique(dropna=True))
    if 4 < n < 10:
        return theme(axis_text_x=element_text(rotation=45, ha="right", size=8))
    return theme()


def render_plotnine(p, selected_theme=None, data: pd.DataFrame | None = None, x_col: str | None = None, n_x_categories: int | None = None):
    if selected_theme is not None:
        p = p + selected_theme
    if data is not None and x_col is not None and x_col in data.columns:
        p = p + theme_x_labels(data, x_col)
    fig = p.draw()
    width = 6
    height = width * (n_x_categories / 25) if (n_x_categories is not None and n_x_categories > 10) else width * 0.6
    fig.set_size_inches(width, height)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def build_plot_context(plot_type: str, params: dict, data: pd.DataFrame) -> str:
    if data.empty or len(data) == 0:
        return f"Plot type: {plot_type}. Parameters: {params}. Data: empty (no rows)."
    lines = [f"Plot type: {plot_type}", f"Parameters: {params}", "", "Data summary:"]
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


def render_plot_chat(plot_key: str, context_text: str) -> None:
    chat_key = f"plot_chat_{plot_key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    messages = st.session_state[chat_key]
    with st.expander(
        "Ask about this plot (such as 'Is the relationship linear?' or 'What are the Q1, median, and Q3 of this plot?'):",
        expanded=len(messages) > 0,
    ):
        for m in messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])
        prompt = st.chat_input("Ask about this plot...", key=f"{chat_key}_input")
        if prompt:
            with st.chat_message("user"):
                st.write(prompt)
            messages.append({"role": "user", "content": prompt})
            reply = call_groq(prompt, context_text, messages[:-1], system_preamble=GROQ_SYSTEM_VISUALIZATION)
            messages.append({"role": "assistant", "content": reply})
            st.rerun()


def render_visualization_tab(df: pd.DataFrame, drop_na_rows: bool, selected_theme) -> None:
    mode_help = (
        "- **One categorical column**: visualize the distribution of a single categorical column "
        "(e.g., bar or pie chart for age group, region, etc.).\n"
        "- **Two categorical columns**: compare how two categorical columns are related "
        "(e.g., side-by-side or stacked bar charts of gender by region).\n"
        "- **One numeric column**: visualize the distribution of one numeric column "
        "(e.g., histogram or boxplot of age, income, etc.).\n"
        "- **Numeric vs categorical columns**: compare numeric values across groups defined by a categorical column "
        "(e.g., average income by region, or a numeric time series grouped by category).\n"
        "- **Two numeric columns**: explore the relationship between two numeric columns "
        "(e.g., scatterplot of height vs weight with an optional regression line)."
    )
    mode = st.radio(
        "Variables",
        ["One categorical column", "Two categorical columns", "One numeric column", "Numeric vs categorical columns", "Two numeric columns"],
        horizontal=True,
        help=mode_help,
    )

    if "custom_plot_labels" not in st.session_state:
        st.session_state["custom_plot_labels"] = {}
    _slot_map = {"Title": "title", "X axis label": "x", "Y axis label": "y"}
    if "label_input" in st.session_state and st.session_state.get("label_edit_plot_key"):
        _prev_plot = st.session_state["label_edit_plot_key"]
        _prev_which = st.session_state.get("label_which_prev") or st.session_state.get("label_which")
        _slot = _slot_map.get(_prev_which) if _prev_which else None
        if _slot and _prev_plot:
            if _prev_plot not in st.session_state["custom_plot_labels"]:
                st.session_state["custom_plot_labels"][_prev_plot] = {"title": "", "x": "", "y": ""}
            st.session_state["custom_plot_labels"][_prev_plot][_slot] = st.session_state.get("label_input", "")

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
            n_x = int(d[x].nunique(dropna=True))
            r2 = st.columns(2)
            with r2[0]:
                show_percent = st.checkbox("Plot proportions instead of counts", value=False, key="v_cat1_pct")
            with r2[1]:
                flip_default = n_x > 10
                flip = st.checkbox("Flip coordinates (horizontal bars)", value=flip_default, key=f"v_cat1_flip_{x}")

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
                plot_key = f"cat_one_bar_{x}_{show_percent}_{flip}"
                tit, x_lab, y_lab = get_plot_labels(f"Distribution of {x}", x, "Proportion", plot_key)
                p = (
                    ggplot(d, aes(x=x, y="..count../sum(..count..)", group=1))
                    + geom_bar()
                    + labs(title=tit, x=x_lab, y=y_lab)
                )
            else:
                plot_key = f"cat_one_bar_{x}_{show_percent}_{flip}"
                tit, x_lab, y_lab = get_plot_labels(f"Distribution of {x}", x, "Count", plot_key)
                p = (
                    ggplot(d, aes(x=x))
                    + geom_bar()
                    + labs(title=tit, x=x_lab, y=y_lab)
                )
            if flip:
                p = p + coord_flip()
            n_cats = n_x if flip and n_x > 10 else None
            render_label_customizer_expander(plot_key)
            render_plotnine(p, selected_theme, data=d, x_col=x, n_x_categories=n_cats)
            context = build_plot_context("bar_chart", {"x": x, "show_percent": show_percent, "flip": flip}, d)
            render_plot_chat(plot_key, context)
        else:
            plot_key = f"cat_one_pie_{x}"
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
            render_label_customizer_expander(plot_key, allowed_fields=("Title",))
            tit, _, _ = get_plot_labels(f"Distribution of {x}", x, "", plot_key)
            ax.set_title(tit)
            ax.axis("equal")
            left, center, right = st.columns([0.3, 2, 0.5])
            with center:
                st.pyplot(fig, use_container_width=False)
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
            flip = st.checkbox("Flip coordinates (horizontal bars)", value=flip_default, key=f"v_cat2_flip_{x}_{fill}")

        if x == fill:
            st.error("Please select two different variables.")
            st.stop()

        d = df[[x, fill]].copy()
        if drop_na_rows:
            d = d.dropna()
        ensure_categorical(d, x, context="the bar plot")
        ensure_categorical(d, fill, context="grouping")

        plot_key = f"cat_two_bar_{x}_{fill}_{style}_{flip}"
        if style == "Side-by-side (dodge)":
            tit, x_lab, y_lab = get_plot_labels(f"{fill} by {x}", x, "Count", plot_key)
            p = (
                ggplot(d, aes(x=x, fill=fill))
                + geom_bar(position="dodge")
                + labs(title=tit, x=x_lab, y=y_lab, fill=fill)
            )
        elif style == "Proportional (fill)":
            tit, x_lab, y_lab = get_plot_labels(f"{fill} by {x} (Proportions)", x, "Proportion", plot_key)
            p = (
                ggplot(d, aes(x=x, fill=fill))
                + geom_bar(position="fill")
                + labs(title=tit, x=x_lab, y=y_lab, fill=fill)
            )
        else:
            tit, x_lab, y_lab = get_plot_labels(f"{fill} by {x} (Stacked)", x, "Count", plot_key)
            p = (
                ggplot(d, aes(x=x, fill=fill))
                + geom_bar()
                + labs(title=tit, x=x_lab, y=y_lab, fill=fill)
            )
        if flip:
            p = p + coord_flip()
        render_label_customizer_expander(plot_key)
        n_cats = n_x if flip and n_x > 10 else None
        render_plotnine(p, selected_theme, data=d, x_col=x, n_x_categories=n_cats)
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
            if "v_num1_bw" not in st.session_state:
                st.session_state["v_num1_bw"] = 0.0
            if "v_num1_bd" not in st.session_state:
                st.session_state["v_num1_bd"] = float("nan")
            if "v_num1_dens" not in st.session_state:
                st.session_state["v_num1_dens"] = False
            df_signature = (tuple(df.columns.tolist()), int(df.shape[0]), int(df.shape[1]))
            hist_sig = (x, df_signature)
            if st.session_state.get("v_num1_hist_sig") != hist_sig:
                st.session_state["v_num1_bw"] = 0.0
                st.session_state["v_num1_bd"] = float("nan")
                st.session_state["v_num1_dens"] = False
                st.session_state["v_num1_hist_sig"] = hist_sig

            r2 = st.columns(3)
            with r2[0]:
                binwidth = st.number_input("Binwidth (0 = auto)", min_value=0.0, step=1.0, key="v_num1_bw")
            with r2[1]:
                boundary = st.number_input("Boundary (optional)", key="v_num1_bd")
            with r2[2]:
                density = st.checkbox("Scale y to density", key="v_num1_dens")

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
            plot_key = f"num_one_hist_{x}_{binwidth}_{boundary}_{density}"
            tit, x_lab, y_lab = get_plot_labels(f"Histogram of {x}", x, ylab, plot_key)
            p = p + geom_histogram(**hist_kwargs) + labs(title=tit, x=x_lab, y=y_lab)
            render_label_customizer_expander(plot_key)
            render_plotnine(p, selected_theme)
            context = build_plot_context("histogram", {"x": x, "binwidth": binwidth, "boundary": boundary, "density": density}, d)
            render_plot_chat(plot_key, context)
        else:
            plot_key = f"num_one_box_{x}"
            tit, x_lab, y_lab = get_plot_labels(f"Boxplot of {x}", "", x, plot_key)
            p = (
                ggplot(d, aes(x=1, y=x))
                + geom_boxplot(width=0.3)
                + xlim(0.5, 1.5)
                + labs(title=tit, x=x_lab, y=y_lab)
            )
            render_label_customizer_expander(plot_key)
            render_plotnine(p, selected_theme)
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
            n_x = int(d[x].nunique(dropna=True))
            plot_key = f"num_cat_box_{x}_{y}"
            tit, x_lab, y_lab = get_plot_labels(f"{y} by {x}", x, y, plot_key)
            p = (
                ggplot(d, aes(x=x, y=y))
                + geom_boxplot()
                + labs(title=tit, x=x_lab, y=y_lab)
            )
            if n_x > 10:
                p = p + coord_flip()
            n_cats = n_x if n_x > 10 else None
            render_label_customizer_expander(plot_key)
            render_plotnine(p, selected_theme, data=d, x_col=x, n_x_categories=n_cats)
            context = build_plot_context("grouped_boxplot", {"x": x, "y": y}, d)
            render_plot_chat(plot_key, context)
        elif plot_type == "Faceted histogram":
            binwidth = st.number_input("Histogram binwidth (0 for default)", min_value=0.0, value=0.0, step=1.0, key="v_numcat_bw")
            hist_kwargs = {}
            if binwidth and binwidth > 0:
                hist_kwargs["binwidth"] = binwidth
            plot_key = f"num_cat_facethist_{x}_{y}_{binwidth}"
            tit, x_lab, y_lab = get_plot_labels(f"Histogram of {y}, faceted by {x}", y, "Count", plot_key)
            p = (
                ggplot(d, aes(x=y))
                + geom_histogram(**hist_kwargs)
                + facet_grid(f"{x} ~ .")
                + labs(title=tit, x=x_lab, y=y_lab)
            )
            render_label_customizer_expander(plot_key)
            render_plotnine(p, selected_theme)
            context = build_plot_context("faceted_histogram", {"x": x, "y": y, "binwidth": binwidth}, d)
            render_plot_chat(plot_key, context)
        elif plot_type == "Line plot":
            d_line = d.copy()
            x_parsed = pd.to_datetime(d_line[x], errors="coerce")
            plot_key = f"num_cat_line_{x}_{y}"
            max_x_labels = st.selectbox(
                "Customize number of x-axis labels:",
                options=[5, 8, 10, 15, 20],
                index=2,
                key="v_numcat_line_maxlabels",
            )

            if x_parsed.notna().mean() >= 0.5:
                d_line["_x_time"] = x_parsed
                d_line = d_line.dropna(subset=["_x_time"]).sort_values("_x_time")
                tit, x_lab, y_lab = get_plot_labels(f"{y} by {x}", x, y, plot_key)
                breaks = compute_axis_breaks(d_line["_x_time"], max_x_labels)
                p = (
                    ggplot(d_line, aes(x="_x_time", y=y, group=1))
                    + geom_line()
                    + labs(title=tit, x=x_lab, y=y_lab)
                    + scale_x_datetime(breaks=breaks, date_labels="%Y", expand=(0, 0))
                )
            else:
                tit, x_lab, y_lab = get_plot_labels(f"{y} by {x}", x, y, plot_key)
                breaks = compute_axis_breaks(d_line[x], max_x_labels)
                p = (
                    ggplot(d_line, aes(x=x, y=y, group=1))
                    + geom_line()
                    + labs(title=tit, x=x_lab, y=y_lab)
                    + scale_x_discrete(breaks=breaks)
                )
            render_label_customizer_expander(plot_key)
            render_plotnine(p, selected_theme, data=d_line, x_col=x if x_parsed.notna().mean() < 0.5 else None)
            context = build_plot_context("line_plot", {"x": x, "y": y}, d)
            render_plot_chat(plot_key, context)
        else:
            if plot_type == "Bar chart":
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
                flip = st.checkbox("Flip coordinates (horizontal bars)", value=flip_default, key=f"v_numcat_bar_flip_{x}_{y}")
                plot_key = f"num_cat_bar_{x}_{y}_{value_type}_{flip}"
                tit, x_lab, y_lab = get_plot_labels(f"{y} by {x}", x, ylab, plot_key)
                p = (
                    ggplot(d, aes(x=x, y=y))
                    + geom_col()
                    + labs(title=tit, x=x_lab, y=y_lab)
                )
                if flip:
                    p = p + coord_flip()
                n_cats = n_x if flip and n_x > 10 else None
                render_label_customizer_expander(plot_key)
                render_plotnine(p, selected_theme, data=d, x_col=x, n_x_categories=n_cats)
                context = build_plot_context("bar_chart_preagg", {"x": x, "y": y, "value_type": value_type, "flip": flip}, d)
                render_plot_chat(plot_key, context)
            else:
                plot_key = f"num_cat_pie_{x}_{y}"
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie(
                    d[y],
                    labels=d[x],
                    autopct="%1.1f%%",
                    startangle=90,
                    radius=0.6,
                )
                render_label_customizer_expander(plot_key, allowed_fields=("Title",))
                tit, _, _ = get_plot_labels(f"{y} by {x}", x, "", plot_key)
                ax.set_title(tit)
                ax.axis("equal")
                left, center, right = st.columns([0.3, 2, 0.5])
                with center:
                    st.pyplot(fig, use_container_width=False)
                context = build_plot_context("pie_chart_preagg", {"x": x, "y": y}, d)
                render_plot_chat(plot_key, context)

    else:
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

        plot_key = f"num_scatter_{x}_{y}_{add_regression}"
        tit, x_lab, y_lab = get_plot_labels(f"{y} vs {x}", x, y, plot_key)

        p = (
            ggplot(d, aes(x=x, y=y))
            + geom_point()
            + labs(title=tit, x=x_lab, y=y_lab)
        )
        if add_regression:
            xv = d[x].to_numpy(dtype=float)
            yv = d[y].to_numpy(dtype=float)
            ok = np.isfinite(xv) & np.isfinite(yv)
            xv, yv = xv[ok], yv[ok]
            if len(xv) >= 2 and np.ptp(xv) > 0:
                slope, intercept = np.polyfit(xv, yv, 1)
                sign = "+" if intercept >= 0 else "−"
                abs_b = abs(intercept)
                eq = f"{y_lab} = {slope:.4g}·{x_lab} {sign} {abs_b:.4g}"
                st.write(f"##### Linear fit: {eq}")
            p = p + geom_smooth(method="lm", se=False, color="red")

        render_label_customizer_expander(plot_key)
        render_plotnine(p, selected_theme)
        context = build_plot_context("scatter", {"x": x, "y": y, "add_regression": add_regression}, d)
        render_plot_chat(plot_key, context)
