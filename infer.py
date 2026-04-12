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
        self.null_val = 0
        self.test_type = None
        self.direction = "two-sided"
        self.results = pd.DataFrame()
        self.ci = None
        self.stat_val = None
        self._viz_dist = None

    def specify(self, response, explanatory=None, success=None, test_type=None):
        self.response = response
        self.explanatory = explanatory
        self.success = success
        self.test_type = test_type
        return self

    def hypothesize(self, p=0):
        # We only need the value (p0, mu0, or the dict for GOF)
        self.null_val = p
        return self

    def calculate(self, direction="two-sided", conf_level=0.95, paired=False):
        self.direction = direction
        
        # Dispatch Map
        engines = {
            'one prop':  lambda: self._engine_one_prop(direction, conf_level),
            'two props': lambda: self._engine_two_props(direction, conf_level),
            'chisq_gof':       lambda: self._engine_chisq_gof(),
            'chisq_indep':     lambda: self._engine_chisq_indep(),
            'one mean':  lambda: self._engine_one_mean(direction, conf_level),
            'two means': lambda: self._engine_paired(direction, conf_level) if paired else self._engine_two_means(direction, conf_level),
            'anova':     lambda: self._engine_anova()
        }
        
        if self.test_type not in engines:
            raise ValueError(f"Unknown test_type. Use: {list(engines.keys())}")
            
        return engines[self.test_type]()

    # --- ENGINES ---

    def _engine_one_prop(self, direction, conf_level):
        y = self.data[self.response]
        count, n = (y == self.success).sum(), len(y)
        alt = {'two-sided': 'two-sided', 'greater': 'larger', 'less': 'smaller'}[direction]
        self.stat_val, p_val = proportions_ztest(count, n, value=self.null_val, alternative=alt, prop_var=self.null_val)
        self.ci = proportion_confint(count, n, alpha=1-conf_level, method='normal')
        self.results = pd.DataFrame({'stat': [self.stat_val], 'p_val': [p_val], 'p_hat': [count/n]})
        self._viz_dist = 'z'
        return self

    def _engine_two_props(self, direction, conf_level):
        groups = self.data.groupby(self.explanatory)[self.response]
        counts = groups.apply(lambda x: (x == self.success).sum()).values
        nobs = groups.count().values
        alt = {'two-sided': 'two-sided', 'greater': 'larger', 'less': 'smaller'}[direction]
        self.stat_val, p_val = proportions_ztest(counts, nobs, alternative=alt)
        low, high = confint_proportions_2indep(counts[0], nobs[0], counts[1], nobs[1], method='wald', alpha=1-conf_level)
        self.ci = (low, high)
        self.results = pd.DataFrame({'stat': [self.stat_val], 'p_val': [p_val], 'diff': [(counts[0]/nobs[0]) - (counts[1]/nobs[1])]})
        self._viz_dist = 'z'
        return self

    def _engine_chisq_gof(self):
        obs = self.data[self.response].value_counts().sort_index()
        exp_p = np.array([self.null_val[cat] for cat in obs.index]) if isinstance(self.null_val, dict) else np.array(self.null_val)
        self.stat_val, p_val = stats.chisquare(f_obs=obs, f_exp=exp_p * len(self.data))
        self.results = pd.DataFrame({'chi2': [self.stat_val], 'p_val': [p_val], 'dof': [len(obs)-1]})
        self._viz_dist = 'chisq'
        return self

    def _engine_chisq_indep(self):
        _, _, stats_df = pg.chi2_independence(self.data, x=self.explanatory, y=self.response)
        res = stats_df[stats_df['test'] == 'pearson']
        self.stat_val, p_val = res['chi2'].iloc[0], res['pval'].iloc[0]
        self.results = pd.DataFrame({'chi2': [self.stat_val], 'p_val': [p_val], 'dof': [res['dof'].iloc[0]]})
        self._viz_dist = 'chisq'
        return self

    def _engine_one_mean(self, direction, conf_level):
        res = pg.ttest(self.data[self.response], y=self.null_val, confidence=conf_level, alternative=direction)
        self.results, self.ci, self.stat_val = res, res['CI95%'].values[0], res['T'].iloc[0]
        self.results = self.results[['T', 'dof', 'p-val']]
        self._viz_dist = 't'
        return self

    def _engine_two_means(self, direction, conf_level):
        names = self.data[self.explanatory].unique()
        g1 = self.data[self.data[self.explanatory] == names[0]][self.response]
        g2 = self.data[self.data[self.explanatory] == names[1]][self.response]
        res = pg.ttest(g1, g2, confidence=conf_level, alternative=direction)
        self.results, self.ci, self.stat_val = res, res['CI95%'].values[0], res['T'].iloc[0]
        self.results = self.results[['T', 'dof', 'p-val']]
        self._viz_dist = 't'
        return self

    def _engine_paired(self, direction, conf_level):
        # Specific for Ch 20 (e.g., UCLA vs Amazon columns)
        res = pg.ttest(self.data[self.response], self.data[self.explanatory], paired=True, confidence=conf_level, alternative=direction)
        self.results, self.ci, self.stat_val = res, res['CI95%'].values[0], res['T'].iloc[0]
        self.results = self.results[['T', 'dof', 'p-val']]
        self._viz_dist = 't'
        return self

    def _engine_anova(self):
        self.results = pg.anova(dv=self.response, between=self.explanatory, data=self.data)
        self.stat_val, self._viz_dist = self.results['F'].iloc[0], 'f'
        self.results = self.results[["ddof1", "ddof2", "F", "p-unc"]]
        return self

    def visualize(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        if self._viz_dist == 'z':
            dist, x = stats.norm, np.linspace(-4, 4, 1000)
        elif self._viz_dist == 't':
            dist, x = stats.t(self.results['dof'].iloc[0]), np.linspace(-4, 4, 1000)
        elif self._viz_dist == 'chisq':
            dist, x = stats.chi2(self.results['dof'].iloc[0]), np.linspace(0, self.stat_val + 10, 1000)
        elif self._viz_dist == 'f':
            dfn, dfd = self.results['ddof1'], self.results['ddof2']
            dist, x = stats.f(dfn, dfd), np.linspace(0, self.stat_val + 2, 1000)

        pdf = dist.pdf(x)
        ax.plot(x, pdf, 'k-', lw=2)
        ax.axvline(self.stat_val, color='blue', ls='--', label=f'Stat: {self.stat_val:.3f}')
        
        if self._viz_dist in ['z', 't']:
            if self.direction == "two-sided":
                ax.fill_between(x, 0, pdf, where=(np.abs(x) > np.abs(self.stat_val)), color='red', alpha=0.3)
            else:
                shading = (x < self.stat_val) if self.direction == "less" else (x > self.stat_val)
                ax.fill_between(x, 0, pdf, where=shading, color='red', alpha=0.3)
        else:
            ax.fill_between(x, 0, pdf, where=(x > self.stat_val), color='red', alpha=0.3)
            
        ax.set_title(f"Null Distribution: {self.test_type.upper()}")
        ax.legend()
        fig.tight_layout()
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


def _categorical_columns_excluding(df: pd.DataFrame, exclude: str) -> list[str]:
    """Other categorical columns (any number of levels), for chi-square independence."""
    out: list[str] = []
    for c in df.columns:
        if c == exclude:
            continue
        s = df[c]
        if is_categorical(s):
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
            "Chisq for goodness of fit",
            "Chisq for indenpedence",
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
                .specify(response=col, success=success, test_type="one prop")
                .hypothesize(p=null_p)
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
            test = (
                Infer(work)
                .specify(
                    response=response,
                    explanatory=explanatory,
                    success=success,
                    test_type="two props",
                )
                .hypothesize(p=0)
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

    elif mode == "Chisq for indenpedence":
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
                    key="infer_chisq_indep_response",
                    help="One categorical variable in the contingency table.",
                )
            expl_candidates = _categorical_columns_excluding(df, exclude=response)
            if not expl_candidates:
                st.error(
                    "Need at least two categorical columns. Pick a different dataset or mark another column as categorical."
                )
                return
            with row1b:
                explanatory = st.selectbox(
                    "Explanatory variable",
                    expl_candidates,
                    key="infer_chisq_indep_explanatory",
                    help="Second categorical variable; independence is tested between these two.",
                )

            cols_needed = [response, explanatory]
            work = df[cols_needed].copy()
            if drop_na_rows:
                work = work.dropna(subset=cols_needed)
            if work.empty:
                st.error("No rows left after dropping missing values in the selected columns.")
                return
            if not is_categorical(work[response]):
                st.error("Response must be categorical.")
                return
            if not is_categorical(work[explanatory]):
                st.error("Explanatory must be categorical.")
                return
            if work[response].nunique() < 2:
                st.error("Response needs at least two categories in the filtered data for this test.")
                return
            if work[explanatory].nunique() < 2:
                st.error("Explanatory needs at least two categories in the filtered data for this test.")
                return

        with right:
            test = (
                Infer(work)
                .specify(
                    response=response,
                    explanatory=explanatory,
                    test_type="chisq_indep",
                )
                .hypothesize(p=0)
                .calculate()
            )
            st.dataframe(test.results, use_container_width=True, hide_index=True)
            fig = test.visualize()
            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info(
            "This test type is not implemented yet. Choose **One proportion**, **Two proportions**, or **Chisq for indenpedence** to run inference."
        )