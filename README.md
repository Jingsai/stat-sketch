# StatSketch: No-code statistics lab

**Live app:** [https://stat-sketch.streamlit.app/](https://stat-sketch.streamlit.app/)

StatSketch is a **browser-based, no-code statistics lab** geared to **introductory statistics**. Explore data, visualize relationships, run common inference procedures, and use reference distributions **without writing R or Python**—through point-and-click choices, tables, and plots.

---

## Goals

- Support **hands-on workflows**: load a dataset, inspect it, plot relationships, and run standard tests with immediate feedback.
- Keep the barrier low: **no programming** and **no statistical software install** when using the hosted app (open the link in a browser).
- **Ask the AI** panels (Visualization and Inference) for short explanations of plots and results.
- Fit **statistics education** and day-to-day use: a strong option if you have **no coding background**—everything stays in the GUI. If you **already write code** in R or Python, StatSketch can still act as a **quick check** (plots, summaries, and standard tests) **without drafting a script**, to verify an idea or a result faster.

---

## Why StatSketch

- **Free** to use on the public deployment (subject to Streamlit Community Cloud limits).
- **Lightweight** interface focused on common intro-stats tasks.
- **No account or registration** required to try the app.
- **No local or self-managed cloud install** needed when using the hosted URL—only a modern **web browser** (including **smartphones**; the layout is usable on small screens, with horizontal controls where Streamlit allows).
- **Ask the AI** embedded help on plots and inference output. (Availability and depth of answers are limited by the free Groq’s rate limits.)
- **Use Cases** mode in the sidebar: topic-based, step-by-step copy that walks through features and example datasets.

---

## Workflow (typical session)

1. In the sidebar, choose **Data Visualization & Inference** (or **Distribution Tools** for calculator-style distributions).
2. **Load data**: upload a delimited file (**Upload file**), paste a direct **File URL** to a CSV-like file, or pick an **Example dataset** (bundled CSVs, including Palmer penguins as `penguins.csv`). Choose a column **separator** if needed.
3. Open **Data Overview** to preview rows, check **Inferred Types**, or scan **Data Statistics** for numeric summaries.
4. Open **Visualization** or **Inference**, pick the variable layout or test type, assign **one or two (or more) columns** as prompted, adjust options, and read the plot or numeric output.
5. Optionally expand **Ask the AI** at the bottom of Visualization or Inference and ask a question in plain language.

---

## App modes (sidebar)


| Mode                               | What it contains                                                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Data Visualization & Inference** | Main workflow: **Data Overview**, **Visualization**, and **Inference** tabs (see below).                                       |
| **Distribution Tools**             | Standalone **Normal**, **Student’s t**, **Chi-square**, and **F** calculators with shaded density plots (no dataset required). |
| **Use Cases**                      | Guided text (and screenshots where provided) organized by topic.                                                               |


---

## 1. Data Overview

*(Available under **Data Visualization & Inference** after data is loaded.)*


| Section             | What it does                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| **Table Preview**   | Show the first *n* rows of the dataset (slider for row count).                                            |
| **Inferred Types**  | Table of columns with heuristic **numeric** vs **categorical** labels, missing counts, and unique counts. |
| **Data Statistics** | `describe`-style summaries for numeric columns (mean, spread, quartiles, min/max).                        |


---

## 2. Visualization

*(Same mode; **Visualization** tab.)*

Choose a **Variables** layout, then columns and **Plot type** where applicable.


| Variables layout                   | Plot types / options                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **One categorical column**         | (1) **Bar chart** — counts or proportions; **tilted x-axis labels** when the x-axis has **5–9** distinct categories; **horizontal bars** default **on** when there are more than **10** categories (toggle with **Flip coordinates**). (2) **Pie chart**. (3) Optional **frequency table** (with bar or pie).                                                                                                               |
| **Two categorical columns**        | (1) **Side-by-side (dodge)** grouped bars. (2) **Stacked** grouped bars. (3) **Proportional (fill)** grouped bars.                                                                                                                                                                                                                                                                                                          |
| **One numeric column**             | (1) **Histogram** — bin width, optional boundary, density scaling. (2) **Boxplot**.                                                                                                                                                                                                                                                                                                                                         |
| **Numeric vs categorical columns** | (1) **Side-by-side boxplot** — auto **horizontal** layout when there are more than **10** x groups. (2) **Faceted histogram**. (3) **Bar chart** — same **tilted labels** when categorical x has **5–9** categories; **horizontal bars** default **on** when there are more than **10** categories on x (**Flip coordinates**). (4) **Pie chart**. (5) **Line plot** — optional cap on x-axis label count for crowded axes. |
| **Two numeric columns**            | (1) **Scatter plot** — optional **linear regression line** and displayed equation.                                                                                                                                                                                                                                                                                                                                          |


Sidebar options include **Plot theme**, optional **Drop rows with missing values in selected columns**, and per-plot **Customize title & axis labels**.

---

## 3. Inference

*(Same mode; **Inference** tab.)*

Pick a **Hypothesis Test** type, assign variables and null settings as prompted, then read tables, intervals, and diagnostic plots.


| Hypothesis test               | Explanation                                                               |
| ----------------------------- | ------------------------------------------------------------------------- |
| **One proportion**            | Single categorical column vs a null proportion.                           |
| **Two proportions**           | Compare two groups on a binary outcome.                                   |
| **Chisq for goodness of fit** | One categorical column vs a user-specified null category probabilities.   |
| **Chisq for indenpedence**    | Independence of two categorical columns (label spelling matches the app). |
| **One mean**                  | One-sample *t*-test vs null mean μ₀.                                      |
| **Two means**                 | Independent two-sample *t*-test (numeric response, two-level factor).     |
| **Paired means**              | Paired *t*-test on two numeric columns (same rows).                       |
| **ANOVA**                     | One-way ANOVA (numeric response, factor with three or more levels).       |


---

## 4. Distribution tools

*(Sidebar: **Distribution Tools**.)*

Each distribution is its own tab. Enter parameters and either a probability for a **quantile** (*q*) or a quantile for a **tail / interval probability** (*p*); a shaded density plot shows the region.


| Tab             | Parameters (typical)                                     |
| --------------- | -------------------------------------------------------- |
| **Normal**      | Mean μ, standard deviation σ.                            |
| **Student's t** | Degrees of freedom.                                      |
| **Chi-square**  | Degrees of freedom.                                      |
| **F**           | Numerator and denominator degrees of freedom (df1, df2). |


---

## Data privacy

Uploaded data is held **in memory for your session** to draw plots and compute statistics. Review your institution’s policies before pasting sensitive data or API keys. The hosted app’s privacy terms are those of the hosting provider you use (e.g. Streamlit Community Cloud).