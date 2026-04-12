import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import streamlit as st
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, confint_proportions_2indep

from helper import is_categorical, is_numeric

"""
In general, this class is designed to mimic the infer package in R, using:

Infer(data)
    .specify(..., test_type="one prop")
    .hypothesize(...)
    .calculate(...)
    
The key difference from infer in R is that this class requires the test type to 
be passed explicitly, rather than letting the package determine which test to perform automatically.

We use specify, hypothesize, and calculate to build hypothesis tests and confidence intervals 
in a hierarchical way. This approach may seem redundant because all parameters could be 
passed directly into __init__. However, the advantage of this hierarchical design is that 
it allows users to set up a hypothesis test step by step more easily.

Check Stat Note.md for the detailed explanation of the design of this class. 
"""
class Infer:
    def __init__(self, data):
        self.data = data.copy()
        self.response = None
        self.explanatory = None
        self.success = None # success group in proportional testings
        self.test_type = None
        self.null_val = 0 # null value(s) in H0 
        self.direction = "two-sided"
        self.conf_level = 0.95
        self.paired = False # for paired two means
        self.results = pd.DataFrame()
        self.ci = None # confidence interval
        self.stat_val = None # testing statistics. this is not p hat. 
        self._viz_dist = None # distribution type for visualization

    # specify the variables, relationships, and testing type
    def specify(self, response, explanatory=None, success=None, test_type=None):
        self.response = response
        self.explanatory = explanatory
        self.success = success
        self.test_type = test_type
        return self
    
    # set up parameters in hypothesis 
    def hypothesize(self, p=0):
        # We only need the value (p0, mu0, or the dict for GOF)
        self.null_val = p
        return self

    # calculate p value and confidence interval from the testing type
    def calculate(self, direction="two-sided", conf_level=0.95, paired=False):
        self.direction = direction
        self.conf_level = conf_level
        self.paired = paired
        
        # Dispatch Map
        engines = {
            'one prop':  lambda: self._engine_one_prop(),
            'two props': lambda: self._engine_two_props(),
            'chisq_gof':       lambda: self._engine_chisq_gof(),
            'chisq_indep':     lambda: self._engine_chisq_indep(),
            'one mean':  lambda: self._engine_one_mean(),
            'two means': lambda: self._engine_paired() if paired else self._engine_two_means(),
            'anova':     lambda: self._engine_anova()
        }
        
        if self.test_type not in engines:
            raise ValueError(f"Unknown test_type. Use: {list(engines.keys())}")
            
        return engines[self.test_type]()

    # --- ENGINES ---

    def _engine_one_prop(self):
        y = self.data[self.response]
        count, n = (y == self.success).sum(), len(y)
        alt = {'two-sided': 'two-sided', 'greater': 'larger', 'less': 'smaller'}[self.direction]
        self.stat_val, p_val = proportions_ztest(count, n, value=self.null_val, alternative=alt, prop_var=self.null_val)
        self.ci = proportion_confint(count, n, alpha=1-self.conf_level, method='normal')
        self.results = pd.DataFrame({'stat': [self.stat_val], 'p_val': [p_val], 'p_hat': [count/n]})
        self._viz_dist = 'z'
        return self

    def _engine_two_props(self):
        groups = self.data.groupby(self.explanatory)[self.response]
        counts = groups.apply(lambda x: (x == self.success).sum()).values
        nobs = groups.count().values
        alt = {'two-sided': 'two-sided', 'greater': 'larger', 'less': 'smaller'}[self.direction]
        self.stat_val, p_val = proportions_ztest(counts, nobs, alternative=alt)
        low, high = confint_proportions_2indep(counts[0], nobs[0], counts[1], nobs[1], method='wald', alpha=1-self.conf_level)
        self.ci = (low, high)
        self.results = pd.DataFrame({'stat': [self.stat_val], 'p_val': [p_val], 'diff': [(counts[0]/nobs[0]) - (counts[1]/nobs[1])]})
        self._viz_dist = 'z'
        return self

    def _engine_chisq_gof(self):
        obs = self.data[self.response].value_counts().sort_index()
        n = float(obs.sum())
        if isinstance(self.null_val, dict):
            exp_p = np.array([float(self.null_val[cat]) for cat in obs.index], dtype=float)
        else:
            exp_p = np.asarray(self.null_val, dtype=float).ravel()
        if exp_p.shape[0] != len(obs):
            raise ValueError(
                "Null probabilities must have one value per category, in sorted category order, or use a dict keyed by category."
            )
        self.stat_val, p_val = stats.chisquare(f_obs=obs, f_exp=exp_p * n)
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

    def _engine_one_mean(self):
        res = pg.ttest(self.data[self.response], y=self.null_val, confidence=self.conf_level, alternative=self.direction)
        ci_col = _pingouin_ci_column_name(self.conf_level)
        # pingouin version 0.5.5 uses CI95% as column name, but newer version uses CI95 as column name.
        # so we specify 0.5.5 in the requirements.txt file to ensure the correct column name is used.
        self.ci = tuple(np.asarray(res[ci_col].values[0]).ravel()[:2])
        self.stat_val = res['T'].iloc[0]
        self.results = res[['T', 'dof', 'p-val']].rename(columns={'p-val': 'p_val'})
        self._viz_dist = 't'
        return self

    def _engine_two_means(self):
        # sort the names to ensure the order of the groups
        # this is for other direction handling such as less and greater
        # so that we know group 1 > group 2 in the case of greater
        names = np.sort(self.data[self.explanatory].unique())
        g1 = self.data[self.data[self.explanatory] == names[0]][self.response]
        g2 = self.data[self.data[self.explanatory] == names[1]][self.response]
        res = pg.ttest(g1, g2, confidence=self.conf_level, alternative=self.direction)
        ci_col = _pingouin_ci_column_name(self.conf_level)
        self.ci = tuple(np.asarray(res[ci_col].values[0]).ravel()[:2])
        self.stat_val = res['T'].iloc[0]
        self.results = res[['T', 'dof', 'p-val']].rename(columns={'p-val': 'p_val'})
        self._viz_dist = 't'
        return self

    def _engine_paired(self):
        res = pg.ttest(
            self.data[self.response],
            self.data[self.explanatory],
            paired=True,
            confidence=self.conf_level,
            alternative=self.direction,
        )
        ci_col = _pingouin_ci_column_name(self.conf_level)
        self.ci = tuple(np.asarray(res[ci_col].values[0]).ravel()[:2])
        self.stat_val = res['T'].iloc[0]
        self.results = res[['T', 'dof', 'p-val']].rename(columns={'p-val': 'p_val'})
        self._viz_dist = 't'
        return self

    def _engine_anova(self):
        self.results = pg.anova(dv=self.response, between=self.explanatory, data=self.data)
        self.stat_val, self._viz_dist = self.results['F'].iloc[0], 'f'
        self.results = self.results[["ddof1", "ddof2", "F", "p-unc"]].rename(
            columns={"ddof1": "dof1", "ddof2": "dof2", "p-unc": "p_val"}
        )
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
            dfn = float(self.results['dof1'].iloc[0])
            dfd = float(self.results['dof2'].iloc[0])
            dist, x = stats.f(dfn, dfd), np.linspace(0, float(self.stat_val) + 2, 1000)

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


def _pingouin_ci_column_name(conf_level: float) -> str:
    """Column name for pingouin t-test CI (e.g. CI95% for conf_level=0.95)."""
    pct = int(round(conf_level * 100))
    return f"CI{pct}%"


def _numeric_columns_for_inference(df: pd.DataFrame) -> list[str]:
    """Numeric columns treated as quantitative (not inferred as categorical)."""
    out: list[str] = []
    for c in df.columns:
        s = df[c]
        if is_numeric(s) and not is_categorical(s):
            out.append(c)
    return sorted(out, key=str.lower)


def _numeric_columns_excluding(df: pd.DataFrame, exclude: str) -> list[str]:
    return [c for c in _numeric_columns_for_inference(df) if c != exclude]


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


def _parse_confidence_level(raw: str, *, lo: float = 0.80, hi: float = 0.999) -> float | None:
    """Parse confidence level from user text; return None if invalid or out of range."""
    try:
        v = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if not (lo <= v <= hi):
        return None
    return v


# Decimal places shown for p-values and CI bounds in Streamlit tables (avoids long float rendering).
_INFER_DISPLAY_DECIMALS = 4


def _infer_display_df(results: pd.DataFrame) -> pd.DataFrame:
    """Copy of results with p_val rounded for display."""
    out = results.copy()
    if "p_val" in out.columns:
        out["p_val"] = pd.to_numeric(out["p_val"], errors="coerce").round(_INFER_DISPLAY_DECIMALS)
    return out


def _infer_display_ci(lo: float, hi: float) -> tuple[float, float]:
    """Round confidence interval endpoints for display."""
    return (
        round(float(lo), _INFER_DISPLAY_DECIMALS),
        round(float(hi), _INFER_DISPLAY_DECIMALS),
    )


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


def _anova_grouping_columns(df: pd.DataFrame, exclude: str) -> list[str]:
    """Categorical factors with at least three levels (typical one-way ANOVA)."""
    out: list[str] = []
    for c in df.columns:
        if c == exclude:
            continue
        s = df[c]
        if is_categorical(s) and s.dropna().nunique() >= 3:
            out.append(c)
    return sorted(out, key=str.lower)


def render_inference_tab(df: pd.DataFrame, drop_na_rows: bool) -> None:
    mode_help = (
        "- **One proportion**: Test the proportion of a single categorical column against a null hypothesis.\n"
        "- **Two proportions**: Test the proportion of two categorical columns against a null hypothesis.\n"
        "- **Chisq for indenpedence**: Test the independence of two categorical columns.\n"
        "- **Chisq for goodness of fit**: Test the goodness of fit of a categorical column against a null hypothesis.\n"
        "- **One mean**: One-sample t-test for a numeric column vs a null mean μ₀.\n"
        "- **Two means**: Independent two-sample t-test (numeric response, two-level factor).\n"
        "- **Paired means**: Paired t-test on two numeric columns (same rows).\n"
        "- **ANOVA**: One-way ANOVA (numeric response, factor with three or more levels).\n"
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
            # 3×2 grid: row1 categorical | success; row2 H₀ | alternative; row3 conf | (empty)
            g11, g12 = st.columns(2)
            with g11:
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
            with g12:
                success = st.selectbox(
                    "Success value",
                    uniques,
                    key="infer_one_prop_success",
                    help="Category counted as a success for the proportion.",
                )

            g21, g22 = st.columns(2)
            with g21:
                null_p = st.number_input(
                    "Null proportion (H₀)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Hypothesized population proportion under the null (point null).",
                    key="infer_one_prop_null_p",
                )
            with g22:
                direction = st.selectbox(
                    "Alternative hypothesis",
                    ["two-sided", "less", "greater"],
                    index=0,
                    key="infer_one_prop_direction",
                )

            g31, _g32 = st.columns(2)
            with g31:
                conf_level_raw = st.text_input(
                    "Confidence level",
                    value="0.95",
                    key="infer_one_prop_conf_level",
                    help="Decimal between 0.80 and 0.999 (e.g. 0.95). Used for the proportion CI (Wald/normal).",
                )
            conf_level = _parse_confidence_level(conf_level_raw)
            if conf_level is None:
                st.warning("Confidence level must be a number between 0.80 and 0.999 (e.g. 0.95).")
                return

            if (y == success).sum() == 0:
                st.warning("No rows equal the chosen success value; the sample proportion is 0.")
            if (y == success).sum() == len(y):
                st.warning("Every row equals the chosen success value; the sample proportion is 1.")

        with right:
            test = (
                Infer(work)
                .specify(response=col, success=success, test_type="one prop")
                .hypothesize(p=null_p)
                .calculate(direction=direction, conf_level=conf_level)
            )
            st.dataframe(_infer_display_df(test.results), use_container_width=True, hide_index=True)
            if test.ci is not None and len(test.ci) >= 2:
                lo, hi = _infer_display_ci(float(test.ci[0]), float(test.ci[1]))
                ci_pct = int(round(conf_level * 100))
                ci_table = pd.DataFrame(
                    {
                        f"Lower bound ({ci_pct}% CI)": [lo],
                        f"Upper bound ({ci_pct}% CI)": [hi],
                    }
                )
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
            # 3×2 grid: row1 response | explanatory; row2 success | alternative; row3 conf | (empty)
            g11, g12 = st.columns(2)
            with g11:
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
            with g12:
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
            g21, g22 = st.columns(2)
            with g21:
                success = st.selectbox(
                    "Success value (in response)",
                    uniques,
                    key="infer_two_prop_success",
                    help="Category counted as a success when computing proportions within each group.",
                )
            with g22:
                direction = st.selectbox(
                    "Alternative hypothesis",
                    ["two-sided", "less", "greater"],
                    index=0,
                    key="infer_two_prop_direction",
                )

            g31, _g32 = st.columns(2)
            with g31:
                conf_level_raw = st.text_input(
                    "Confidence level",
                    value="0.95",
                    key="infer_two_prop_conf_level",
                    help="Decimal between 0.80 and 0.999 (e.g. 0.95). Used for the CI for p₁ − p₂ (Wald).",
                )
            conf_level = _parse_confidence_level(conf_level_raw)
            if conf_level is None:
                st.warning("Confidence level must be a number between 0.80 and 0.999 (e.g. 0.95).")
                return

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
                .calculate(direction=direction, conf_level=conf_level)
            )
            st.dataframe(_infer_display_df(test.results), use_container_width=True, hide_index=True)
            if test.ci is not None and len(test.ci) >= 2:
                lo, hi = _infer_display_ci(float(test.ci[0]), float(test.ci[1]))
                ci_pct = int(round(conf_level * 100))
                ci_table = pd.DataFrame(
                    {
                        f"Lower bound ({ci_pct}% CI)": [lo],
                        f"Upper bound ({ci_pct}% CI)": [hi],
                    }
                )
                st.dataframe(ci_table, use_container_width=True, hide_index=True)
            fig = test.visualize()
            st.pyplot(fig)
            plt.close(fig)

    elif mode == "Chisq for goodness of fit":
        cat_cols = _categorical_columns_for_inference(df)
        if not cat_cols:
            st.warning(
                "No categorical columns found. Use columns inferred as categorical in Data Preview, or upload different data."
            )
            return

        left, right = st.columns(2, gap="large")

        with left:
            col = st.selectbox(
                "Categorical variable",
                cat_cols,
                key="infer_gof_col",
                help="Counts of these categories are compared to the null probabilities you specify.",
            )
            work = df[[col]].copy()
            if drop_na_rows:
                work = work.dropna(subset=[col])
            if work.empty:
                st.error("No rows left after dropping missing values in the selected column.")
                return

            obs_counts = work[col].value_counts().sort_index()
            cats = list(obs_counts.index)
            if len(cats) < 2:
                st.error("Goodness-of-fit χ² needs at least two categories in the filtered data.")
                return

            st.markdown(
                "**H₀:** Category probabilities match the table below. "
                "Rows follow **sorted category order** (same order used in the test). "
                "Probabilities must be **non-negative** and **sum to 1**."
            )
            if st.button("Reset to uniform **1/k**", key="infer_gof_uniform"):
                st.session_state["infer_gof_uniform_tick"] = st.session_state.get("infer_gof_uniform_tick", 0) + 1

            tick = st.session_state.get("infer_gof_uniform_tick", 0)
            cats_key = "|".join(repr(c) for c in cats)
            cats_digest = hashlib.sha256(cats_key.encode("utf-8")).hexdigest()[:16]
            editor_key = f"infer_gof_ed_{col}_{cats_digest}_{tick}"

            default_p = 1.0 / len(cats)
            edit_df = pd.DataFrame(
                {"category": cats, "probability": [default_p] * len(cats)},
            )
            edited = st.data_editor(
                edit_df,
                column_config={
                    "category": st.column_config.Column("Category", disabled=True, required=True),
                    "probability": st.column_config.NumberColumn(
                        "P(category under H₀)",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.6f",
                        required=True,
                    ),
                },
                hide_index=True,
                num_rows="fixed",
                key=editor_key,
            )

            probs = edited["probability"].astype(float)
            prob_sum = float(probs.sum())
            st.caption(f"**Sum of probabilities:** {prob_sum:.6f}")
            if (probs < 0).any():
                st.warning("All probabilities must be non-negative.")
                return
            if not np.isclose(prob_sum, 1.0, rtol=0.0, atol=1e-5):
                st.warning("Probabilities must sum to 1 (within 0.00001). Adjust the table or use **1/k**.")
                return

            null_probs = probs.to_numpy()

        with right:
            test = (
                Infer(work)
                .specify(response=col, test_type="chisq_gof")
                .hypothesize(p=null_probs)
                .calculate(conf_level=0.95)
            )
            st.dataframe(_infer_display_df(test.results), use_container_width=True, hide_index=True)
            exp_ct = null_probs * float(obs_counts.sum())
            fit_table = pd.DataFrame(
                {
                    "observed": obs_counts.values,
                    "expected_under_H0": exp_ct,
                },
                index=obs_counts.index.rename("category"),
            )
            st.caption("Counts used in the test (sorted categories).")
            st.dataframe(fit_table, use_container_width=True)
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
                .calculate(conf_level=0.95)
            )
            st.dataframe(_infer_display_df(test.results), use_container_width=True, hide_index=True)
            fig = test.visualize()
            st.pyplot(fig)
            plt.close(fig)

    elif mode == "One mean":
        num_cols = _numeric_columns_for_inference(df)
        if not num_cols:
            st.warning(
                "No numeric quantitative columns found. Use columns inferred as numeric (not categorical) in Data Preview, or upload different data."
            )
            return

        left, right = st.columns(2, gap="large")

        with left:
            # 2×2 grid: row1 response | μ₀; row2 direction | confidence level
            g11, g12 = st.columns(2)
            with g11:
                response = st.selectbox(
                    "Response variable",
                    num_cols,
                    key="infer_one_mean_response",
                    help="Numeric outcome; one-sample t-test vs μ₀.",
                )
            with g12:
                mu0 = st.number_input(
                    "μ₀ (null mean)",
                    value=0.0,
                    format="%.6g",
                    key="infer_one_mean_mu0",
                    help="Hypothesized population mean under H₀.",
                )

            work = df[[response]].copy()
            if drop_na_rows:
                work = work.dropna(subset=[response])
            if work.empty:
                st.error("No rows left after dropping missing values in the selected column.")
                return
            if work[response].shape[0] < 2:
                st.error("Need at least two non-missing values for a one-sample t-test.")
                return

            g21, g22 = st.columns(2)
            with g21:
                direction = st.selectbox(
                    "Alternative hypothesis",
                    ["two-sided", "less", "greater"],
                    index=0,
                    key="infer_one_mean_direction",
                )
            with g22:
                conf_level_raw = st.text_input(
                    "Confidence level",
                    value="0.95",
                    key="infer_one_mean_conf_level",
                    help="Decimal between 0.80 and 0.999 (e.g. 0.95). Used for the mean CI.",
                )
            conf_level = _parse_confidence_level(conf_level_raw)
            if conf_level is None:
                st.warning("Confidence level must be a number between 0.80 and 0.999 (e.g. 0.95).")
                return

        with right:
            test = (
                Infer(work)
                .specify(response=response, test_type="one mean")
                .hypothesize(p=mu0)
                .calculate(direction=direction, conf_level=conf_level)
            )
            st.dataframe(_infer_display_df(test.results), use_container_width=True, hide_index=True)
            if test.ci is not None and len(test.ci) >= 2:
                lo, hi = _infer_display_ci(float(test.ci[0]), float(test.ci[1]))
                ci_pct = int(round(conf_level * 100))
                ci_table = pd.DataFrame(
                    {
                        f"Lower bound ({ci_pct}% CI)": [lo],
                        f"Upper bound ({ci_pct}% CI)": [hi],
                    }
                )
                st.dataframe(ci_table, use_container_width=True, hide_index=True)
            fig = test.visualize()
            st.pyplot(fig)
            plt.close(fig)

    elif mode == "Two means":
        num_cols = _numeric_columns_for_inference(df)
        if not num_cols:
            st.warning(
                "No numeric quantitative columns found. Use columns inferred as numeric in Data Preview, or upload different data."
            )
            return

        left, right = st.columns(2, gap="large")

        with left:
            # 2×2 grid: row1 response | explanatory; row2 direction | confidence level
            g11, g12 = st.columns(2)
            with g11:
                response = st.selectbox(
                    "Response variable",
                    num_cols,
                    key="infer_two_mean_response",
                    help="Numeric outcome compared across two independent groups.",
                )
            expl_candidates = _two_level_categorical_columns(df, exclude=response)
            if not expl_candidates:
                st.error(
                    "No categorical column with exactly two levels (grouping variable). Add a binary factor or pick another response."
                )
                return
            with g12:
                explanatory = st.selectbox(
                    "Explanatory variable (two groups)",
                    expl_candidates,
                    key="infer_two_mean_explanatory",
                    help="Exactly two categories (e.g. treatment vs control).",
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
            if not is_numeric(work[response]) or is_categorical(work[response]):
                st.error("Response must be numeric (quantitative).")
                return
            if not is_categorical(work[explanatory]):
                st.error("Explanatory must be categorical.")
                return
            vc = work.groupby(explanatory)[response].count()
            if (vc < 2).any():
                st.warning("At least one group has fewer than two observations; the t-test may be unreliable.")

            g21, g22 = st.columns(2)
            with g21:
                direction = st.selectbox(
                    "Alternative hypothesis",
                    ["two-sided", "less", "greater"],
                    index=0,
                    key="infer_two_mean_direction",
                )
            with g22:
                conf_level_raw = st.text_input(
                    "Confidence level",
                    value="0.95",
                    key="infer_two_mean_conf_level",
                    help="Decimal between 0.80 and 0.999 (e.g. 0.95). Used for the difference-in-means CI.",
                )
            conf_level = _parse_confidence_level(conf_level_raw)
            if conf_level is None:
                st.warning("Confidence level must be a number between 0.80 and 0.999 (e.g. 0.95).")
                return

        with right:
            test = (
                Infer(work)
                .specify(response=response, explanatory=explanatory, test_type="two means")
                .hypothesize(p=0)
                .calculate(direction=direction, conf_level=conf_level, paired=False)
            )
            st.dataframe(_infer_display_df(test.results), use_container_width=True, hide_index=True)
            if test.ci is not None and len(test.ci) >= 2:
                lo, hi = _infer_display_ci(float(test.ci[0]), float(test.ci[1]))
                ci_pct = int(round(conf_level * 100))
                ci_table = pd.DataFrame(
                    {
                        f"Lower bound ({ci_pct}% CI)": [lo],
                        f"Upper bound ({ci_pct}% CI)": [hi],
                    }
                )
                st.dataframe(ci_table, use_container_width=True, hide_index=True)
            fig = test.visualize()
            st.pyplot(fig)
            plt.close(fig)

    elif mode == "Paired means":
        num_cols = _numeric_columns_for_inference(df)
        if not num_cols:
            st.warning(
                "No numeric quantitative columns found. Use columns inferred as numeric in Data Preview, or upload different data."
            )
            return

        left, right = st.columns(2, gap="large")

        with left:
            # 2×2 grid: row1 column 1 | column 2; row2 direction | confidence level
            g11, g12 = st.columns(2)
            with g11:
                col1 = st.selectbox(
                    "Column 1",
                    num_cols,
                    key="infer_paired_col1",
                    help="First numeric measurement (paired with column 2).",
                )
            col2_candidates = _numeric_columns_excluding(df, exclude=col1)
            if not col2_candidates:
                st.error("Pick a second numeric column different from column 1.")
                return
            with g12:
                col2 = st.selectbox(
                    "Column 2",
                    col2_candidates,
                    key="infer_paired_col2",
                    help="Second numeric measurement on the same rows as column 1.",
                )

            cols_needed = [col1, col2]
            work = df[cols_needed].copy()
            if drop_na_rows:
                work = work.dropna(subset=cols_needed)
            if work.shape[0] < 2:
                st.error("Need at least two complete pairs (non-missing in both columns) for a paired t-test.")
                return

            g21, g22 = st.columns(2)
            with g21:
                direction = st.selectbox(
                    "Alternative hypothesis",
                    ["two-sided", "less", "greater"],
                    index=0,
                    key="infer_paired_direction",
                )
            with g22:
                conf_level_raw = st.text_input(
                    "Confidence level",
                    value="0.95",
                    key="infer_paired_conf_level",
                    help="Decimal between 0.80 and 0.999 (e.g. 0.95). Used for the paired mean-difference CI.",
                )
            conf_level = _parse_confidence_level(conf_level_raw)
            if conf_level is None:
                st.warning("Confidence level must be a number between 0.80 and 0.999 (e.g. 0.95).")
                return

        with right:
            test = (
                Infer(work)
                .specify(response=col1, explanatory=col2, test_type="two means")
                .hypothesize(p=0)
                .calculate(direction=direction, conf_level=conf_level, paired=True)
            )
            st.dataframe(_infer_display_df(test.results), use_container_width=True, hide_index=True)
            if test.ci is not None and len(test.ci) >= 2:
                lo, hi = _infer_display_ci(float(test.ci[0]), float(test.ci[1]))
                ci_pct = int(round(conf_level * 100))
                ci_table = pd.DataFrame(
                    {
                        f"Lower bound ({ci_pct}% CI)": [lo],
                        f"Upper bound ({ci_pct}% CI)": [hi],
                    }
                )
                st.dataframe(ci_table, use_container_width=True, hide_index=True)
            fig = test.visualize()
            st.pyplot(fig)
            plt.close(fig)

    elif mode == "ANOVA":
        num_cols = _numeric_columns_for_inference(df)
        if not num_cols:
            st.warning(
                "No numeric quantitative columns found. Use columns inferred as numeric in Data Preview, or upload different data."
            )
            return

        left, right = st.columns(2, gap="large")

        with left:
            g11, g12 = st.columns(2)
            with g11:
                response = st.selectbox(
                    "Response variable",
                    num_cols,
                    key="infer_anova_response",
                    help="Numeric outcome (dependent variable).",
                )
            expl_candidates = _anova_grouping_columns(df, exclude=response)
            if not expl_candidates:
                st.error(
                    "No categorical column with at least three levels for grouping. One-way ANOVA needs a factor with three or more categories."
                )
                return
            with g12:
                explanatory = st.selectbox(
                    "Explanatory variable (factor)",
                    expl_candidates,
                    key="infer_anova_explanatory",
                    help="Categorical factor with three or more groups.",
                )

            cols_needed = [response, explanatory]
            work = df[cols_needed].copy()
            if drop_na_rows:
                work = work.dropna(subset=cols_needed)
            if work.empty:
                st.error("No rows left after dropping missing values in the selected columns.")
                return
            if not is_numeric(work[response]) or is_categorical(work[response]):
                st.error("Response must be numeric (quantitative).")
                return
            if not is_categorical(work[explanatory]):
                st.error("Explanatory must be categorical.")
                return
            n_groups = work[explanatory].nunique()
            if n_groups < 3:
                st.error("After filtering, the factor needs at least three groups for this ANOVA setup.")
                return

        with right:
            test = (
                Infer(work)
                .specify(response=response, explanatory=explanatory, test_type="anova")
                .hypothesize(p=0)
                .calculate(conf_level=0.95)
            )
            st.dataframe(_infer_display_df(test.results), use_container_width=True, hide_index=True)
            fig = test.visualize()
            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info("Choose a hypothesis test from the options above.")