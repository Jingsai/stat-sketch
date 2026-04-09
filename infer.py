import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import streamlit as st
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, confint_proportions_2indep

from helper import is_categorical

class Infer:
    def __init__(self, data):
        self.data = data.copy()
        self.response = None
        self.explanatory = None
        self.success = None
        self.null_type = None
        self.null_val = 0
        self.test_type = None
        self.direction = "two-sided"
        self.results = pd.DataFrame()
        self.ci = None
        self.stat_val = None

    def specify(self, formula, success=None):
        if "~" in formula:
            self.response, self.explanatory = formula.replace(" ", "").split("~")
        else:
            self.response = formula.strip()
            self.explanatory = None
        self.success = success
        return self

    def hypothesize(self, null="point", p=0):
        self.null_type = null
        self.null_val = p
        return self

    def calculate(self, direction="two-sided", conf_level=0.95, paired=False):
        self.direction = direction
        y = self.data[self.response]
        n = len(self.data)
        
        # --- 1. CATEGORICAL RESPONSE (Ch 15, 16, 17, 18) ---
        if self.success is not None or y.dtype == 'object':
            # Chapter 15: One-Sample Proportion
            if self.explanatory is None:
                self.test_type = 'z'
                count = (y == self.success).sum()
                
                # Map 'infer' directions to 'statsmodels'
                alt = {'two-sided': 'two-sided', 'greater': 'larger', 'less': 'smaller'}[direction]
                
                """
                https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportions_ztest.html
                prop_var if not set up, the model uses sample proportaions, not the null value, 
                in calculation the z score of p hat in the hypothesis testing. 
                """
                # HYPOTHESIS TEST: Using prop_var=self.null_val to use p0 in denominator
                self.stat_val, p_val = proportions_ztest(
                    count, n, value=self.null_val, 
                    alternative=alt, prop_var=self.null_val
                )
                
                # CONFIDENCE INTERVAL: Uses p-hat by default (Chapter 15 requirement)
                self.ci = proportion_confint(count, n, alpha=1-conf_level, method='normal')
                
                self.results = pd.DataFrame({
                    'stat': [self.stat_val], 
                    'p_val': [p_val], 
                    'p_hat': [count/n],
                    'null_p': [self.null_val]
                })

            # Chapter 16: Two-Sample Proportions
            elif self.data[self.explanatory].nunique() == 2:
                self.test_type = 'z'
                
                # Get counts and nobs for both groups
                group_data = self.data.groupby(self.explanatory)[self.response]
                counts = group_data.apply(lambda x: (x == self.success).sum()).values
                nobs = group_data.count().values
                
                # HYPOTHESIS TEST: Using pooled proportion for the Z-stat
                alt = {'two-sided': 'two-sided', 'greater': 'larger', 'less': 'smaller'}[direction]
                self.stat_val, p_val = proportions_ztest(counts, nobs, alternative=alt)

                # CONFIDENCE INTERVAL: Using the specialized 2-independent function
                # method='wald' matches the standard (p1-p2) +/- Z*SE formula in the book
                low, high = confint_proportions_2indep(
                    counts[0], nobs[0], counts[1], nobs[1], 
                    method='wald', compare='diff', alpha=1-conf_level
                )
                self.ci = (low, high)

                self.results = pd.DataFrame({
                    'stat': [self.stat_val], 
                    'p_val': [p_val], 
                    'p_hat_1': [counts[0]/nobs[0]], 
                    'p_hat_2': [counts[1]/nobs[1]],
                    'diff': [(counts[0]/nobs[0]) - (counts[1]/nobs[1])]
                })
            
            # Chapter 17: Chi-Square Goodness of Fit
            # Triggered if self.explanatory is None AND self.null_type is "point" with multiple probabilities
            elif self.explanatory is None and isinstance(self.null_val, (list, np.ndarray, dict)):
                self.test_type = 'chisq'
                
                # Get observed counts
                observed_counts = y.value_counts().sort_index()
                n = len(y)
                
                # Align expected probabilities with observed categories
                if isinstance(self.null_val, dict):
                    expected_probs = np.array([self.null_val[cat] for cat in observed_counts.index])
                else:
                    expected_probs = np.array(self.null_val)
                
                expected_counts = expected_probs * n
                
                # Run Chi-Square Goodness of Fit
                chisq_stat, p_val = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
                
                self.results = pd.DataFrame({
                    'chi2': [chisq_stat], 
                    'p_val': [p_val], 
                    'dof': [len(observed_counts) - 1]
                })
                self.stat_val = chisq_stat
                
            # Chapter 18: Chi-Square
            else:
                self.test_type = 'chisq'
                expected, observed, stats_df = pg.chi2_independence(self.data, x=self.explanatory, y=self.response)
                self.results = stats_df[stats_df['test'] == 'pearson']
                self.stat_val = self.results['chi2'].iloc[0]

        # --- 2. NUMERICAL RESPONSE (Ch 19, 20, 21, 22) ---
        else:
            # Chapter 19: One-Sample T-test
            if self.explanatory is None:
                self.test_type = 't'
                res = pg.ttest(y, y0=self.null_val, confidence=conf_level, alternative=direction)
                self.results, self.ci, self.stat_val = res, res['CI95%'].values[0], res['T'].iloc[0]

            else:
                groups = self.data[self.explanatory].nunique()
                # Chapter 20/21: Two-Sample or Paired T-test
                if groups == 2:
                    self.test_type = 't'
                    g1_n, g2_n = self.data[self.explanatory].unique()[:2]
                    g1 = y[self.data[self.explanatory] == g1_n]
                    g2 = y[self.data[self.explanatory] == g2_n]
                    res = pg.ttest(g1, g2, paired=paired, confidence=conf_level, alternative=direction)
                    self.results, self.ci, self.stat_val = res, res['CI95%'].values[0], res['T'].iloc[0]
                
                # Chapter 22: ANOVA
                else:
                    self.test_type = 'f'
                    self.results = pg.anova(dv=self.response, between=self.explanatory, data=self.data)
                    self.stat_val = self.results['F'].iloc[0]
                    # CI for ANOVA group means via bootstrapping
                    self.ci = pg.compute_bootci(y, func='mean', confidence=conf_level)

        return self

    def visualize(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        
        # Distribution Selection
        if self.test_type == 'z':
            dist, x = stats.norm, np.linspace(-4, 4, 1000)
        elif self.test_type == 't':
            df = self.results['dof'].iloc[0]
            dist, x = stats.t(df), np.linspace(-4, 4, 1000)
        elif self.test_type == 'chisq':
            df = self.results['dof'].iloc[0]
            dist, x = stats.chi2(df), np.linspace(0, self.stat_val + 5, 1000)
        elif self.test_type == 'f':
            # Df calculation for visualization
            dfn = self.results['SS'].size - 1
            dfd = len(self.data) - (dfn + 1)
            dist, x = stats.f(dfn, dfd), np.linspace(0, self.stat_val + 2, 1000)

        pdf = dist.pdf(x)
        ax.plot(x, pdf, 'k-', lw=2)
        ax.axvline(self.stat_val, color='blue', ls='--', label=f'Observed Stat: {self.stat_val:.3f}')
        
        # Shading
        if self.test_type in ['z', 't']:
            if self.direction == "two-sided":
                ax.fill_between(x, 0, pdf, where=(np.abs(x) > np.abs(self.stat_val)), color='red', alpha=0.3)
            elif self.direction == "less":
                ax.fill_between(x, 0, pdf, where=(x < self.stat_val), color='red', alpha=0.3)
            elif self.direction == "greater":
                ax.fill_between(x, 0, pdf, where=(x > self.stat_val), color='red', alpha=0.3)
        else:
            # Chi-square and F are always right-tailed
            ax.fill_between(x, 0, pdf, where=(x > self.stat_val), color='red', alpha=0.3)
            
        ax.set_title(f"Null Distribution ({self.test_type.upper()})")
        ax.legend()
        return fig


def _categorical_columns_for_inference(df: pd.DataFrame) -> list[str]:
    """Columns suitable as a single categorical response (one proportion)."""
    out: list[str] = []
    for c in df.columns:
        s = df[c]
        if is_categorical(s):
            out.append(c)
    return out


def _two_level_categorical_columns(df: pd.DataFrame, exclude: str | None = None) -> list[str]:
    """Categorical columns with exactly two non-null categories (e.g. grouping variable for two proportions)."""
    out: list[str] = []
    for c in df.columns:
        if exclude is not None and c == exclude:
            continue
        s = df[c]
        if is_categorical(s) and s.dropna().nunique() == 2:
            out.append(c)
    return sorted(out, key=str.lower)


def render_inference_tab(df: pd.DataFrame, drop_na_rows: bool) -> None:
    mode_help = (
        "- **One proportion**: Test the proportion of a single categorical column against a null hypothesis.\n"
        "- **Two proportions**: Test the proportion of two categorical columns against a null hypothesis.\n"
        "- **Chisq for indenpedence**: Test the independence of two categorical columns.\n"
        "- **Chisq for goodness of fit**: Test the goodness of fit of a categorical column against a null hypothesis.\n"
        "- **One mean**: Test the mean of a single numeric column against a null hypothesis.\n"
        "- **Two means**: Test the mean of two numeric columns against a null hypothesis.\n"
        "- **Paired means**: Test the mean of two numeric columns against a null hypothesis.\n"
        "- **ANOVA**: Test the mean of three or more numeric columns against a null hypothesis.\n"
    )
    mode = st.radio(
        "Hypothesis Test",
        [
            "One proportion",
            "Two proportions",
            "Chisq for indenpedence",
            "Chisq for goodness of fit",
            "One mean",
            "Two means",
            "Paired means",
            "ANOVA",
        ],
        horizontal=True,
        help=mode_help,
    )

    if mode == "One proportion":
        cat_cols = _categorical_columns_for_inference(df)
        if not cat_cols:
            st.warning(
                "No categorical columns found for a one-proportion test. Use columns inferred as categorical in Data Preview, or upload different data."
            )
            return

        left, right = st.columns(2, gap="large")

        with left:
            row1a, row1b = st.columns(2)
            with row1a:
                col = st.selectbox(
                    "Categorical variable",
                    cat_cols,
                    key="infer_one_prop_col",
                )
            s = df[col]
            if not is_categorical(s):
                st.error("Selected column is not categorical. Pick a different variable.")
                return

            work = df[[col]].copy()
            if drop_na_rows:
                work = work.dropna(subset=[col])
            if work.empty:
                st.error("No rows left after dropping missing values in the selected column.")
                return

            y = work[col]
            uniques = list(pd.unique(y.dropna()))
            try:
                uniques = sorted(uniques, key=lambda x: (str(type(x).__name__), str(x)))
            except TypeError:
                pass
            with row1b:
                success = st.selectbox(
                    "Success value",
                    uniques,
                    key="infer_one_prop_success",
                    help="Category counted as a success for the proportion.",
                )

            row2a, row2b = st.columns(2)
            with row2a:
                null_p = st.number_input(
                    "Null proportion (H₀)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Hypothesized population proportion under the null (point null).",
                    key="infer_one_prop_null_p",
                )
            with row2b:
                direction = st.selectbox(
                    "Alternative hypothesis",
                    ["two-sided", "less", "greater"],
                    index=0,
                    key="infer_one_prop_direction",
                )

            if (y == success).sum() == 0:
                st.warning("No rows equal the chosen success value; the sample proportion is 0.")
            if (y == success).sum() == len(y):
                st.warning("Every row equals the chosen success value; the sample proportion is 1.")

        with right:
            test = (
                Infer(work)
                .specify(formula=col, success=success)
                .hypothesize(null="point", p=null_p)
                .calculate(direction=direction)
            )
            st.dataframe(test.results, use_container_width=True, hide_index=True)
            if test.ci is not None and len(test.ci) >= 2:
                lo, hi = float(test.ci[0]), float(test.ci[1])
                ci_table = pd.DataFrame(
                    {
                        "Lower bound (95% CI)": [lo],
                        "Upper bound (95% CI)": [hi],
                    }
                )
                # st.caption("95% confidence interval for the population proportion (normal approximation).")
                st.dataframe(ci_table, use_container_width=True, hide_index=True)
            fig = test.visualize()
            st.pyplot(fig)
            plt.close(fig)

    elif mode == "Two proportions":
        cat_cols = _categorical_columns_for_inference(df)
        if not cat_cols:
            st.warning(
                "No categorical columns found. Use columns inferred as categorical in Data Preview, or upload different data."
            )
            return

        left, right = st.columns(2, gap="large")

        with left:
            row1a, row1b = st.columns(2)
            with row1a:
                response = st.selectbox(
                    "Response variable",
                    cat_cols,
                    key="infer_two_prop_response",
                    help="Outcome column; success is defined within this variable.",
                )
            expl_candidates = _two_level_categorical_columns(df, exclude=response)
            if not expl_candidates:
                st.error(
                    "No second categorical column with exactly two levels (for grouping). Add a binary grouping variable or pick another response."
                )
                return
            with row1b:
                explanatory = st.selectbox(
                    "Explanatory variable (two groups)",
                    expl_candidates,
                    key="infer_two_prop_explanatory",
                    help="Must have exactly two categories (e.g. treatment vs control).",
                )

            cols_needed = [response, explanatory]
            work = df[cols_needed].copy()
            if drop_na_rows:
                work = work.dropna(subset=cols_needed)
            if work.empty:
                st.error("No rows left after dropping missing values in the selected columns.")
                return
            if work[explanatory].nunique() != 2:
                st.error("Explanatory variable must have exactly two non-null categories in the filtered data.")
                return
            if not is_categorical(work[response]):
                st.error("Response must be categorical.")
                return
            if not is_categorical(work[explanatory]):
                st.error("Explanatory must be categorical.")
                return

            yr = work[response]
            uniques = list(pd.unique(yr.dropna()))
            try:
                uniques = sorted(uniques, key=lambda x: (str(type(x).__name__), str(x)))
            except TypeError:
                pass
            row2a, row2b = st.columns(2)
            with row2a:
                success = st.selectbox(
                    "Success value (in response)",
                    uniques,
                    key="infer_two_prop_success",
                    help="Category counted as a success when computing proportions within each group.",
                )
            with row2b:
                direction = st.selectbox(
                    "Alternative hypothesis",
                    ["two-sided", "less", "greater"],
                    index=0,
                    key="infer_two_prop_direction",
                )

            g_success = work.groupby(explanatory)[response].apply(lambda x: (x == success).sum())
            g_n = work.groupby(explanatory)[response].count()
            if (g_success == 0).any():
                st.warning("At least one group has no successes; proportions may be 0.")
            if (g_success == g_n).all():
                st.warning("Every row is a success within both groups; proportions are 1.")

        with right:
            formula = f"{response}~{explanatory}"
            test = (
                Infer(work)
                .specify(formula=formula, success=success)
                .hypothesize(null="point", p=0.0)
                .calculate(direction=direction)
            )
            st.dataframe(test.results, use_container_width=True, hide_index=True)
            if test.ci is not None and len(test.ci) >= 2:
                lo, hi = float(test.ci[0]), float(test.ci[1])
                ci_table = pd.DataFrame(
                    {
                        "Lower bound (95% CI)": [lo],
                        "Upper bound (95% CI)": [hi],
                    }
                )
                # st.caption("95% CI for the difference in proportions (p₁ − p₂), Wald method.")
                st.dataframe(ci_table, use_container_width=True, hide_index=True)
            fig = test.visualize()
            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info(
            "This test type is not implemented yet. Choose **One proportion** or **Two proportions** to run inference."
        )