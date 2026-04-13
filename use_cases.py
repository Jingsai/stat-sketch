"""Example use-case copy for the Use Cases app mode."""

from pathlib import Path

import streamlit as st

_USE_CASE_IMAGE_DIR = Path(__file__).resolve().parent / "images"

# Optional screenshots: (topic slug, example title) -> filename under images/
_EXAMPLE_IMAGES: dict[tuple[str, str], str] = {
    ("viz", "Histogram"): "viz_histogram_num.png",
    ("viz", "Box plot"): "viz_boxplot_num_cat.png",
    ("viz", "Scatter plot"): "viz_scatter_num_num.png",
    ("viz", "Bar charts"): "viz_bar_cat_tilt.png",
    ("viz", "Pie chart"): "viz_pie_num_cat.png",
    ("viz", "Linear regression"): "viz_linear_regression.png",
    ("viz", "Line plot"): "viz_line_plot.png",
}

# Shown above the matching screenshot (how the figure was produced)
_EXAMPLE_IMAGE_INTRO: dict[tuple[str, str], str] = {
    ("viz", "Histogram"): """The histogram below was made by following steps:

1. In the sidebar, set Data source to Example dataset and choose Palmer penguins.
2. Open the Visualization tab. Under Variables, select One numeric column.
3. Set Numeric column to bill_depth_mm and Plot type to Histogram (adjust bins if you like).""",
    ("viz", "Box plot"): """The box plot below was made by following steps:

1. In the sidebar, set Data source to Example dataset and choose Palmer penguins.
2. Open the Visualization tab. Under Variables, select Numeric vs categorical columns.
3. Set Numeric column to bill_length_mm, Categorical column to island, and Plot type to Side-by-side boxplot.
4. Expand Customize title & axis labels below the plot and edit the title, x-axis label, and y-axis label.""",
    ("viz", "Scatter plot"): """The scatter plot below was made by following steps:

1. In the sidebar, set Data source to Example dataset and choose Palmer penguins.
2. Open the Visualization tab. Under Variables, select Two numeric columns.
3. Set X (numeric) to bill_length_mm and Y (numeric) to bill_depth_mm.""",
    ("viz", "Bar charts"): """The bar chart below was made by following steps:

1. In the sidebar, set Data source to Upload file and upload police_killings.csv with the file uploader (comma separator if prompted).
2. Open the Visualization tab. Under Variables, select One categorical column.
3. Set Categorical column to month and Plot type to Bar chart.""",
    ("viz", "Pie chart"): """The pie chart below was made by following steps:

1. In the sidebar, set Data source to Upload file and upload eg10-01.csv with the file uploader (comma separator if prompted).
2. Open the Visualization tab. Under Variables, select Numeric vs categorical columns.
3. Set Numeric column to Percent, Categorical column to Level of education, and Plot type to Pie chart.""",
    ("viz", "Linear regression"): """The scatter plot with regression line below was made by following steps:

1. In the sidebar, set Data source to Example dataset and choose birthwt.
2. Open the Visualization tab. Under Variables, select Two numeric columns.
3. Set X (numeric) to bwt and Y (numeric) to lwt.
4. Turn on Add linear regression line.
5. Expand Customize title & axis labels below the plot and edit the title, x-axis label, and y-axis label.""",
    ("viz", "Line plot"): """The line plot below was made by following steps:

1. In the sidebar, set Data source to Upload file and upload ex10-02.csv with the file uploader (comma separator if prompted).
2. Open the Visualization tab. Under Variables, select Numeric vs categorical columns.
3. Set Numeric column to price, Categorical column to date, and Plot type to Line plot.
4. Use Customize number of x-axis labels to pick how many x-axis tick marks to aim for (default 10; choices 5, 8, 10, 15, or 20).""",
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
Bar charts

1. Three ways to draw a bar chart in this app:

   1. One categorical column — each row is an observation; the app counts how often each category appears and draws bars for those counts (or proportions if you choose that option).

   2. Two categorical columns — pick an X category and a fill/group category; choose a bar style such as side-by-side (dodge), stacked, or proportional (fill).

   3. Numeric vs categorical columns with Plot type Bar chart — your table already has one category column and one numeric column giving the bar height (counts or percentages per category); the app plots those values directly.

2. Tilted x-axis labels and horizontal bars

   If the variable on the x-axis has strictly more than 4 and strictly fewer than 10 distinct categories (that is, 5 through 9 unique values), the app rotates the x-axis category labels for readability.

   If there are more than 10 distinct categories on the x-axis, the Flip coordinates (horizontal bars) option defaults to on so bars run horizontally; you can still turn it off if you want vertical bars.

3. Example screenshot and steps to reproduce it follow.
""",
    ),
    (
        "Pie chart",
        """
**Pie chart (numeric vs categorical)**

Choose **Numeric vs categorical columns**, assign the **Categorical** column (slice labels) and **Numeric** column (counts or percentages that determine slice size), and set **Plot type** to **Pie chart**.

You can also make a pie from a single raw categorical column under **One categorical column** and **Pie chart**; the app counts rows per category. The example screenshot and steps below use the numeric-vs-categorical path with pre-tabulated Percent and Level of education.
""",
    ),
    (
        "Linear regression",
        """
**Linear fit and equation**

Under **Two numeric columns**, pick **X** and **Y**, then turn on **Add linear regression line** to draw a least-squares line on the scatter plot. The app prints the fitted **linear equation** above the chart using your axis labels (default column names until you change them under **Customize title & axis labels**).

Example: with the **birthwt** example dataset, **X (numeric)** = **bwt** (baby’s birth weight) and **Y (numeric)** = **lwt** (mother’s weight), after customizing the axis wording the line matches **Mother's weight = 0.007789·Baby's birth weight + 106.9**. A screenshot and step-by-step setup follow.
""",
    ),
    (
        "Line plot",
        """
**Line plot and crowded x-axis labels**

Under **Numeric vs categorical columns**, set **Plot type** to **Line plot**. Put the series you want on the vertical axis in **Numeric column** (for example **price**) and the values that should run along the horizontal axis in **Categorical column** (for example **date**). When most values in that column parse as dates, the app uses a **datetime** x scale.

Line plots let you **ease overcrowded x-axis ticks** with **Customize number of x-axis labels**: you pick a **target maximum** number of tick positions (default **10**; choices **5, 8, 10, 15, 20**). If there are **more** distinct x values than that number, the app **subsamples** them so you get roughly up to your choice (never more than **20** from the menu). If there are **fewer** distinct x values—e.g. only **10** unique dates when you chose **20**—you get **10** ticks: **you cannot have more ticks than distinct x values**; the control is only an upper cap, not a guarantee to draw that many.

The example screenshot and steps below use **ex10-02.csv** with **date** on x and **price** on y.
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
    intro = _EXAMPLE_IMAGE_INTRO.get((slug, choice))
    if intro:
        st.markdown(intro)
    path = _USE_CASE_IMAGE_DIR / filename
    if path.is_file():
        st.image(str(path), width="stretch")
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
