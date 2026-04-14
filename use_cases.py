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
    ("infer", "One proportion"): "infer_one_prop.png",
    ("infer", "Two proportions"): "infer_two_props.png",
    ("infer", "Chisq goodness of fit"): "infer_chisq_gof.png",
    ("infer", "Chisq independence"): "infer_chisq_ind.png",
    ("infer", "One mean"): "infer_one_mean.png",
    ("infer", "Two means"): "infer_two_means.png",
    ("infer", "Paired means"): "infer_paired_means.png",
    ("infer", "ANOVA"): "infer_anova.png",
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
    ("infer", "One proportion"): """The one-proportion result below was produced by following steps:

1. In the sidebar, choose Data Visualization & Inference, set Data source to Example dataset, and pick heart_transplant2.
2. Open the Inference tab and select One proportion under Hypothesis Test.
3. Set Categorical variable to survived and Success value to alive.
4. Set Null proportion (H0) to 0.5, Alternative hypothesis to less, and Confidence level to 0.95 (adjust if you want a different test).""",
    ("infer", "Two proportions"): """The two-proportion result below was produced by following steps:

1. In the sidebar, choose Data Visualization & Inference, set Data source to Example dataset, and pick Melanoma.
2. Open the Inference tab and select Two proportions under Hypothesis Test.
3. Set Response variable to status_fct, Explanatory variable (two groups) to sex_fct, and Success value (in response) to died from melanoma.
4. Set Alternative hypothesis to two-sided and Confidence level to 0.95 (adjust if you want a different test). The null is that the two group proportions are equal (difference 0).""",
    ("infer", "Chisq goodness of fit"): """The chi-square goodness-of-fit result below was produced by following steps:

1. In the sidebar, choose Data Visualization & Inference, set Data source to Example dataset, and pick mtcars.
2. Open the Inference tab and select Chisq for goodness of fit under Hypothesis Test.
3. Set Categorical variable to cyl (number of cylinders).
4. Under H0, edit the P(category under H0) column so the rows match sorted category order in the table: probability 0.38 for cyl 4, 0.38 for cyl 6, and 0.24 for cyl 8 (these sum to 1).""",
    ("infer", "Chisq independence"): """The chi-square independence result below was produced by following steps:

1. In the sidebar, choose Data Visualization & Inference, set Data source to Example dataset, and pick birthwt.
2. Open the Inference tab and select Chisq for indenpedence under Hypothesis Test (spelling matches the app).
3. Set Response variable to low_fct and Explanatory variable to race_fct.""",
    ("infer", "One mean"): """The one-sample mean result below was produced by following steps:

1. In the sidebar, choose Data Visualization & Inference, set Data source to Example dataset, and pick teacher2.
2. Open the Inference tab and select One mean under Hypothesis Test.
3. Set Response variable to total and μ₀ (null mean) to 63024.
4. Set Alternative hypothesis to two-sided and Confidence level to 0.95 (adjust if you want a different test).""",
    ("infer", "Two means"): """The two-sample mean result below was produced by following steps:

1. In the sidebar, choose Data Visualization & Inference, set Data source to Example dataset, and pick birthwt.
2. Open the Inference tab and select Two means under Hypothesis Test.
3. Set Response variable to bwt and Explanatory variable (two groups) to low_fct.
4. Set Alternative hypothesis to two-sided and Confidence level to 0.95 (adjust if you want a different test). The null is that the two group means are equal (difference 0).""",
    ("infer", "Paired means"): """The paired mean result below was produced by following steps:

1. In the sidebar, choose Data Visualization & Inference, set Data source to Example dataset, and pick textbooks.
2. Open the Inference tab and select Paired means under Hypothesis Test.
3. Set Column 1 to amaz_new (Amazon new price) and Column 2 to ucla_new (UCLA bookstore new price).
4. Set Alternative hypothesis to two-sided and Confidence level to 0.95 (adjust if you want a different test). The null is that the mean paired difference is zero.""",
    ("infer", "ANOVA"): """The one-way ANOVA result below was produced by following steps:

1. In the sidebar, choose Data Visualization & Inference, set Data source to Example dataset, and pick uis.
2. Open the Inference tab and select ANOVA under Hypothesis Test.
3. Set Response variable to BECK and Explanatory variable (factor) to IV_fct.""",
}

_USE_CASE_TOPICS = (
    "Data visualization",
    "Data inference",
    "Data overview",
    "Distribution tools",
    "Ask the AI",
)

_TOPIC_SLUG = {
    "Data visualization": "viz",
    "Data inference": "infer",
    "Data overview": "preview",
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

The tests in the examples below follow **Chapters 15–22** of open textbook [*Introduction to Statistics: an integrated textbook and workbook using R*](https://vectorposse.github.io/intro_stats/) (datasets and research questions are drawn from that book’s narrative).

For a short note on where the assistant lives in the app, pick the **Ask the AI** topic above.
""",
    ),
    (
        "One proportion",
        """
**One proportion**

Choose **One proportion**. Pick a **Categorical variable** and which category counts as **Success**. Set **Null proportion (H0)**, **Alternative hypothesis**, and **Confidence level**; the app updates the results table, interval, and null plot as you change inputs.

The app reports the test statistic, p-value, and confidence interval in context of your table.

This use case parallels the **Research question** in [Chapter 15 — Inference for one proportion](https://vectorposse.github.io/intro_stats/15-inference_for_one_proportion-web.html#research-question) in that book.

An example on the bundled **heart_transplant2** data (testing the proportion **alive** in **survived**) is shown in the screenshot and numbered steps below.
""",
    ),
    (
        "Two proportions",
        """
**Two proportions**

Choose **Two proportions**. Pick a categorical **Response variable** (outcome), an **Explanatory variable** with exactly **two** groups, and which category counts as **Success (in response)**. Set **Alternative hypothesis** and **Confidence level**; the app compares the success proportion in each group (null is no difference).

Useful for comparing proportions between two groups.

This use case parallels the **Research question** in [Chapter 16 — Inference for two proportions](https://vectorposse.github.io/intro_stats/16-inference_for_two_proportions-web.html#research-question) in that book.

An example on the bundled **Melanoma** data (response **status_fct**, groups **sex_fct**, success **died from melanoma**) is shown in the screenshot and numbered steps below.
""",
    ),
    (
        "Chisq goodness of fit",
        """
**Chi-square goodness of fit**

Choose **Chisq for goodness of fit** in the Inference tab. Pick a **Categorical variable**; the app shows observed counts per category and a table where you set **P(category under H₀)** for each category (non-negative, must sum to 1). Use **Reset to uniform 1/k** for equal null probabilities across *k* categories, or type your own null distribution.

The right panel shows the χ² statistic, p-value, observed vs expected counts, and a plot of the null distribution.

This use case parallels the **Research question** in [Chapter 17 — Chi-square goodness-of-fit test](https://vectorposse.github.io/intro_stats/17-chi_square_goodness_of_fit-web.html#research-question) in that book.

An example on the **mtcars** example dataset with **cyl** as the variable and null probabilities **4 → 0.38**, **6 → 0.38**, **8 → 0.24** is shown in the screenshot and numbered steps below.
""",
    ),
    (
        "Chisq independence",
        """
**Chi-square test of independence**

Choose **Chisq for indenpedence** (as labeled in the app). Pick a categorical **Response variable** and a categorical **Explanatory variable**; the app tests whether the two factors are independent in the population and shows the χ² statistic, p-value, expected counts, and a visualization.

This use case parallels the **Research question** in [Chapter 18 — Chi-square test for independence](https://vectorposse.github.io/intro_stats/18-chi_square_test_for_independence-web.html#research-question) in that book.

An example on the **birthwt** example dataset with **low_fct** as response and **race_fct** as explanatory is shown in the screenshot and numbered steps below.
""",
    ),
    (
        "One mean",
        """
**One mean**

Choose **One mean**. Select a numeric **Response variable**, the null mean **μ₀**, **Alternative hypothesis**, and **Confidence level** for a one-sample *t*-test. The app reports the statistic, p-value, and confidence interval for the mean.

Checks whether the sample mean differs from the hypothesized mean.

This use case parallels the **Research question** in [Chapter 19 — Inference for one mean](https://vectorposse.github.io/intro_stats/19-inference_for_one_mean-web.html#research-question) in that book.

An example on the **teacher2** example dataset with response **total** and **μ₀ = 63024** is shown in the screenshot and numbered steps below.
""",
    ),
    (
        "Two means",
        """
**Two means (independent samples)**

Choose **Two means**. Pick a numeric **Response variable** and a categorical **Explanatory variable** with exactly **two** groups, then set **Alternative hypothesis** and **Confidence level** for the independent two-sample *t*-test. The null is that the two group means are equal.

Compares average outcome between the two groups.

This use case parallels the **Research question** in [Chapter 21 — Inference for two independent means](https://vectorposse.github.io/intro_stats/21-inference_for_two_independent_means-web.html#research-question) in that book.

An example on the **birthwt** example dataset with response **bwt** and grouping **low_fct** is shown in the screenshot and numbered steps below.
""",
    ),
    (
        "Paired means",
        """
**Paired means**

Choose **Paired means**. Select **Column 1** and **Column 2** as two numeric measurements on the **same rows** (each row is a pair), then set **Alternative hypothesis** and **Confidence level** for the paired *t*-test.

Tests whether the mean difference between the two columns is zero.

This use case parallels the **Research question** in [Chapter 20 — Inference for paired data](https://vectorposse.github.io/intro_stats/20-inference_for_paired_data-web.html#research-question) in that book.

An example on the **textbooks** example dataset with **amaz_new** and **ucla_new** is shown in the screenshot and numbered steps below.
""",
    ),
    (
        "ANOVA",
        """
**One-way ANOVA**

Choose **ANOVA**. Pick a numeric **Response variable** and a categorical **Explanatory variable (factor)** with **three or more** levels, then run the analysis. The app reports the ANOVA table and a diagnostic plot.

Tests whether at least one group mean differs from the others.

This use case parallels the **Research question** in [Chapter 22 — ANOVA](https://vectorposse.github.io/intro_stats/22-anova-web.html#research-question) in that book.

An example on the **uis** example dataset with response **BECK** and factor **IV_fct** is shown in the screenshot and numbered steps below.
""",
    ),
]

_DATA_PREVIEW_EXAMPLES: list[tuple[str, str]] = [
    (
        "Getting started",
        """
**Getting started**

In **Data Visualization & Inference**, load your file or an example dataset, then open the **Data Overview** tab, then choose **Table Preview** in the section selector. Use the examples below to focus on one part of that tab.
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

The **Data Statistics** section summarizes numeric columns (mean, spread, quartiles, min/max).

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

Common questions you might try (copy the idea, not necessarily the exact wording):

- **Histogram:** “Is this histogram skewed left or right, and what does that mean for the data?”
- **ANOVA:** “What are df1 and df2 in this ANOVA table, and how do I read them?”
- **Inference:** “Explain the p-value I’m seeing in plain language.”
- **Plots:** “What does this boxplot tell me about spread and outliers between groups?”
"""

_TOPIC_EXAMPLES: dict[str, list[tuple[str, str]]] = {
    "Data visualization": _DATA_VIZ_EXAMPLES,
    "Data inference": _DATA_INFER_EXAMPLES,
    "Data overview": _DATA_PREVIEW_EXAMPLES,
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
