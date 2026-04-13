"""Example use-case copy for the Use Cases app mode."""

from pathlib import Path

import streamlit as st

_USE_CASE_IMAGE_DIR = Path(__file__).resolve().parent / "images"

# Optional screenshots: (topic slug, example title) -> filename under images/
_EXAMPLE_IMAGES: dict[tuple[str, str], str] = {
    ("viz", "Histogram"): "histogram_one_num.png",
}

_USE_CASE_TOPICS = (
    "Data visualization",
    "Data inference",
    "Data preview",
    "Distribution tools",
    "Ask the AI",
)

_TOPIC_SLUG = {
    "Data visualization": "viz",
    "Data inference": "infer",
    "Data preview": "preview",
    "Distribution tools": "dist",
}

_DATA_VIZ_EXAMPLES: list[tuple[str, str]] = [
    (
        "Getting started",
        """
**Getting started**

In the sidebar, choose **Data Visualization & Inference**, load a dataset (upload, URL, or example), then open the **Visualization** tab.

Use **Variables** to pick how many columns you are plotting (one categorical, two numeric, and so on). Change **Plot theme** in the sidebar for a different look, and expand **Customize title & axis labels** under a figure to edit the title and axis text.
""",
    ),
    (
        "Histogram",
        """
**Histogram (one numeric column)**

Under **Variables**, choose **One numeric column**, select the column, set **Plot type** to **Histogram**, and tune bin width, boundary, or density if you want.
""",
    ),
    (
        "Box plot",
        """
**Box plot (numeric vs categorical)**

Choose **Numeric vs categorical columns**, assign the **Numeric** and **Categorical** columns, set **Plot type** to **Side-by-side boxplot**.
""",
    ),
    (
        "Scatter plot",
        """
**Scatter plot (two numeric columns)**

Choose **Two numeric columns**, pick **X** and **Y**. The app draws a scatter of points.
""",
    ),
    (
        "Bar charts",
        """
**Bar charts with many categories**

For counts of one variable, use **One categorical column** and **Bar chart**. For two categorical variables, use **Two categorical columns** and a bar style you prefer.

With several categories on the x-axis, labels may be tilted automatically. For many categories, turn on **Flip coordinates (horizontal bars)**; the app often enables this by default when there are more than ten categories.
""",
    ),
    (
        "Pie chart",
        """
**Pie chart**

For a **single categorical** column, choose **One categorical column** and set **Plot type** to **Pie chart** to show category shares as slices (with optional **Show frequency table** to see counts).

If your table already has one category column and one numeric column of counts or percentages, use **Numeric vs categorical columns**, set **Plot type** to **Pie chart**, and map them to the categorical and numeric fields.
""",
    ),
    (
        "Linear regression",
        """
**Linear fit and equation**

Under **Two numeric columns**, turn on **Add linear regression line** to overlay a least-squares line on the scatter plot. The app shows the **linear equation** (slope and intercept) above the chart.
""",
    ),
    (
        "Line plot",
        """
**Line plot and crowded x-axis labels**

Under **Numeric vs categorical columns**, set **Plot type** to **Line plot**. Use **Customize number of x-axis labels** (e.g. 5, 8, 10, 15, 20) to show fewer ticks so labels do not overlap. If most x values parse as dates, the axis is treated as a time scale.
""",
    ),
]

_DATA_INFER_EXAMPLES: list[tuple[str, str]] = [
    (
        "Getting started",
        """
**Getting started**

Stay in **Data Visualization & Inference** with data loaded, then open the **Inference** tab. Pick a **Hypothesis Test** mode (buttons along the top), fill in variables and null settings, and run the workflow.

For a short note on where the assistant lives in the app, pick the **Ask the AI** topic above.
""",
    ),
    (
        "One proportion",
        """
**One proportion**

Choose **One proportion**. Pick a **Categorical variable** and which category counts as **Success**. Set the null proportion, direction, and confidence level, then run the test.

The app reports the test statistic, p-value, and confidence interval in context of your table.
""",
    ),
    (
        "Two proportions",
        """
**Two proportions**

Choose **Two proportions**. Select two categorical columns (or one column split by another), define success, null, and direction, then calculate.

Useful for comparing proportions between two groups.
""",
    ),
    (
        "Chisq goodness of fit",
        """
**Chi-square goodness of fit**

Choose **Chisq for goodness of fit**. Pick a categorical column and specify the null distribution (e.g. equal frequencies or custom probabilities), then run the test.

Checks whether observed category counts match a claimed distribution.
""",
    ),
    (
        "Chisq independence",
        """
**Chi-square test of independence**

Choose **Chisq for indenpedence** (as labeled in the app). Select two categorical columns to test whether they are associated in the population.

Review expected counts and the chi-square statistic in the output.
""",
    ),
    (
        "One mean",
        """
**One mean**

Choose **One mean**. Select a numeric column and the null mean μ₀, direction, and confidence level for a one-sample *t*-test.

Checks whether the sample mean differs from the hypothesized mean.
""",
    ),
    (
        "Two means",
        """
**Two means (independent samples)**

Choose **Two means**. Pick a numeric response and a two-level categorical grouping, set hypotheses and options, then run the independent two-sample *t*-test.

Compares average outcome between the two groups.
""",
    ),
    (
        "Paired means",
        """
**Paired means**

Choose **Paired means**. Select two numeric columns measured on the same rows (e.g. before and after), set direction and confidence level, then run the paired *t*-test.

Tests whether the mean difference between the two columns is zero.
""",
    ),
    (
        "ANOVA",
        """
**One-way ANOVA**

Choose **ANOVA**. Pick a numeric response and a categorical factor with **three or more** levels, then run the analysis.

Tests whether at least one group mean differs from the others.
""",
    ),
]

_DATA_PREVIEW_EXAMPLES: list[tuple[str, str]] = [
    (
        "Getting started",
        """
**Getting started**

In **Data Visualization & Inference**, load your file or an example dataset, then open the **Data Preview** tab. Use the examples below to focus on one part of that tab.
""",
    ),
    (
        "Table preview",
        """
**Preview the table**

Use the **Preview rows** slider to show more or fewer rows from the top of the dataset. Scan for typos, wrong encodings, or surprising values before you plot or test.
""",
    ),
    (
        "Inferred types",
        """
**Inferred types**

Read the **Inferred types** table: each column is labeled heuristic **numeric** or **categorical** with missing and unique counts.

If a column is misclassified, you can still choose it manually in Visualization or Inference; this table is a quick sanity check.
""",
    ),
    (
        "Data statistics",
        """
**Data statistics**

The **Data Statistics** section summarizes numeric columns (mean, spread, quartiles, min/max). If there are no numeric columns, the app says so.

Use this to spot skewed variables, odd ranges, or zeros before modeling.
""",
    ),
]

_DIST_EXAMPLES: list[tuple[str, str]] = [
    (
        "Getting started",
        """
**Getting started**

In the sidebar, switch to **Distribution Tools**. Each family has its own tab; enter parameters and either a quantile (for tail probability *p*) or a probability (for quantile *q*). A shaded density plot shows the region you asked about.
""",
    ),
    (
        "Normal",
        """
**Normal distribution**

Open the **Normal** tab. Set mean μ and standard deviation σ, then use the *p*- or *q*-style inputs for one tail, two tails, or an interval.

Matches the usual “pnorm / qnorm” style calculations with a picture.
""",
    ),
    (
        "Student's t",
        """
**Student's *t* distribution**

Open the **Student's t** tab, set degrees of freedom, and compute tail probabilities or quantiles the same way as for the normal.

Useful for small-sample reference when you already know the df.
""",
    ),
    (
        "Chi-square",
        """
**Chi-square distribution**

Open the **Chi-square** tab, set df, then request probabilities or quantiles. The plot shades the region on the density.

Handy for critical values or tail areas linked to chi-square tests.
""",
    ),
    (
        "F distribution",
        """
***F* distribution**

Open the **F** tab, set numerator and denominator degrees of freedom, then use *p* or *q* inputs.

Supports *F*-test style tail and quantile lookups with visualization.
""",
    ),
]

_ASK_AI_BODY = """
**Ask the AI**

This app supports an **AI dialog** at the bottom of the **Visualization** tab and the **Inference** tab (expand the section when you are ready to chat). The panel titles include **example questions** you can ask—such as how to read a plot or how to interpret a test result—and you can type your own questions in the same way.
"""

_TOPIC_EXAMPLES: dict[str, list[tuple[str, str]]] = {
    "Data visualization": _DATA_VIZ_EXAMPLES,
    "Data inference": _DATA_INFER_EXAMPLES,
    "Data preview": _DATA_PREVIEW_EXAMPLES,
    "Distribution tools": _DIST_EXAMPLES,
}


def _maybe_show_example_image(slug: str, choice: str) -> None:
    filename = _EXAMPLE_IMAGES.get((slug, choice))
    if not filename:
        return
    path = _USE_CASE_IMAGE_DIR / filename
    if path.is_file():
        st.image(str(path), use_container_width=True)
    else:
        st.caption(f"_(Screenshot not found: add `images/{filename}` next to `use_cases.py`.)_")


def _render_example_radio(slug: str, examples: list[tuple[str, str]]) -> None:
    titles = [t[0] for t in examples]
    state_key = f"use_case_sub_{slug}"
    if state_key not in st.session_state or st.session_state[state_key] not in titles:
        st.session_state[state_key] = titles[0]

    by_title = dict(examples)
    choice = st.radio(
        "Pick an example",
        options=titles,
        key=state_key,
        horizontal=True,
    )
    st.markdown(by_title[choice])
    _maybe_show_example_image(slug, choice)


def render_use_cases() -> None:
    st.markdown(
        "Choose a **topic** below to see a short example of how you might use it. "
        "To try it yourself, switch mode or tab in the sidebar and main view as noted."
    )
    topic = st.radio(
        "Pick a topic",
        options=list(_USE_CASE_TOPICS),
        horizontal=True,
        key="use_case_topic",
    )
    if topic == "Ask the AI":
        st.markdown(_ASK_AI_BODY)
    else:
        slug = _TOPIC_SLUG[topic]
        examples = _TOPIC_EXAMPLES[topic]
        _render_example_radio(slug, examples)
