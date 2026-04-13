"""
Distribution calculators: p* (CDF / tail probability) and q* (quantiles), with plots.

Each distribution is a small class wrapping scipy.stats. The Streamlit UI uses tabs
at the top (Normal, Student's t, Chi-square, F) mirroring other app sections.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st

# Keys for number_inputs cleared when leaving Distribution Tools (see StatSketch.py).
DISTRIBUTION_TOOLS_WIDGET_KEYS_TO_CLEAR: tuple[str, ...] = (
    "normal_mu_input",
    "normal_sigma_input",
    "normal_q_input",
    "normal_q1_input",
    "normal_q2_input",
    "normal_p_lower_input",
    "normal_p_upper_input",
    "t_df_input",
    "t_q_input",
    "t_q1_input",
    "t_q2_input",
    "t_p_lower_input",
    "t_p_upper_input",
    "chisq_df_input",
    "chisq_q_input",
    "chisq_q1_input",
    "chisq_q2_input",
    "chisq_p_lower_input",
    "chisq_p_upper_input",
    "f_dfn_input",
    "f_dfd_input",
    "f_q_input",
    "f_q1_input",
    "f_q2_input",
    "f_p_lower_input",
    "f_p_upper_input",
)


class NormalDist:
    """Normal(μ, σ²): R-style pnorm / qnorm."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.dist = stats.norm(loc=self.mu, scale=self.sigma)

    def _draw_base(self, title: str):
        span = 4 * self.sigma
        x = np.linspace(self.mu - span, self.mu + span, 1000)
        y = self.dist.pdf(x)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, y, color="black", lw=2)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        return fig, ax

    def pnorm(self, q: float, q2: float | None = None, *, lower_tail: bool = True):
        q_label = f"{q:.4g}"
        if q2 is not None:
            q2_label = f"{q2:.4g}"
            prob = float(self.dist.cdf(q2) - self.dist.cdf(q))
            label = f"P({q_label} < X < {q2_label}) = {prob:.4f}"
            x_fill = np.linspace(q, q2, 100)
        elif lower_tail:
            prob = float(self.dist.cdf(q))
            label = f"P(X < {q_label}) = {prob:.4f}"
            x_fill = np.linspace(self.mu - 4 * self.sigma, q, 100)
        else:
            prob = float(1.0 - self.dist.cdf(q))
            label = f"P(X > {q_label}) = {prob:.4f}"
            x_fill = np.linspace(q, self.mu + 4 * self.sigma, 100)

        fig, ax = self._draw_base(label)
        ax.fill_between(x_fill, self.dist.pdf(x_fill), color="skyblue", alpha=0.6)
        return prob, fig

    def qnorm(self, p: float, *, lower_tail: bool = True):
        target_p = p if lower_tail else 1.0 - p
        cutoff = float(self.dist.ppf(target_p))
        direction = "below" if lower_tail else "above"
        title = f"{p*100:.1f}% of values are {direction} {cutoff:.4g}"
        fig, ax = self._draw_base(title)
        if lower_tail:
            x_fill = np.linspace(self.mu - 4 * self.sigma, cutoff, 100)
        else:
            x_fill = np.linspace(cutoff, self.mu + 4 * self.sigma, 100)
        ax.fill_between(x_fill, self.dist.pdf(x_fill), color="salmon", alpha=0.4)
        ax.axvline(cutoff, color="red", linestyle="--", label=f"Cutoff: {cutoff:.4g}")
        ax.legend()
        return cutoff, fig


class StudentTDist:
    """Student t(df): R-style pt / qt (standard t; location 0, scale 1)."""

    def __init__(self, df: float):
        self.df = float(df)
        self.dist = stats.t(self.df)

    def _draw_base(self, title: str):
        lo = float(self.dist.ppf(0.0005))
        hi = float(self.dist.ppf(0.9995))
        x = np.linspace(lo, hi, 1000)
        y = self.dist.pdf(x)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, y, color="black", lw=2)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        return fig, ax

    def pt(self, q: float, q2: float | None = None, *, lower_tail: bool = True):
        q_label = f"{q:.4g}"
        if q2 is not None:
            q2_label = f"{q2:.4g}"
            prob = float(self.dist.cdf(q2) - self.dist.cdf(q))
            label = f"P({q_label} < T < {q2_label}) = {prob:.4f}"
            x_fill = np.linspace(q, q2, 100)
        elif lower_tail:
            prob = float(self.dist.cdf(q))
            label = f"P(T < {q_label}) = {prob:.4f}"
            lo = float(self.dist.ppf(0.0005))
            x_fill = np.linspace(lo, q, 100)
        else:
            prob = float(1.0 - self.dist.cdf(q))
            label = f"P(T > {q_label}) = {prob:.4f}"
            hi = float(self.dist.ppf(0.9995))
            x_fill = np.linspace(q, hi, 100)

        fig, ax = self._draw_base(label)
        ax.fill_between(x_fill, self.dist.pdf(x_fill), color="skyblue", alpha=0.6)
        return prob, fig

    def qt(self, p: float, *, lower_tail: bool = True):
        target_p = p if lower_tail else 1.0 - p
        cutoff = float(self.dist.ppf(target_p))
        direction = "below" if lower_tail else "above"
        title = f"{p*100:.1f}% of values are {direction} {cutoff:.4g}"
        fig, ax = self._draw_base(title)
        lo = float(self.dist.ppf(0.0005))
        hi = float(self.dist.ppf(0.9995))
        if lower_tail:
            x_fill = np.linspace(lo, cutoff, 100)
        else:
            x_fill = np.linspace(cutoff, hi, 100)
        ax.fill_between(x_fill, self.dist.pdf(x_fill), color="salmon", alpha=0.4)
        ax.axvline(cutoff, color="red", linestyle="--", label=f"Cutoff: {cutoff:.4g}")
        ax.legend()
        return cutoff, fig


class ChiSqDist:
    """Chi-square(df): R-style pchisq / qchisq (support x ≥ 0)."""

    def __init__(self, df: float):
        self.df = float(df)
        self.dist = stats.chi2(self.df)

    def _upper_x(self) -> float:
        return max(12.0, float(self.dist.ppf(0.9995)) * 1.05)

    def _draw_base(self, title: str):
        hi = self._upper_x()
        x = np.linspace(0.0, hi, 1000)
        y = self.dist.pdf(x)
        y[0] = 0.0 if not np.isfinite(y[0]) else y[0]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, y, color="black", lw=2)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.set_xlim(left=0)
        return fig, ax

    def pchisq(self, q: float, q2: float | None = None, *, lower_tail: bool = True):
        q_label = f"{q:.4g}"
        hi = self._upper_x()
        if q2 is not None:
            q2_label = f"{q2:.4g}"
            prob = float(self.dist.cdf(q2) - self.dist.cdf(q))
            label = f"P({q_label} < X < {q2_label}) = {prob:.4f}"
            x_fill = np.linspace(max(0, q), q2, 100)
        elif lower_tail:
            prob = float(self.dist.cdf(q))
            label = f"P(X < {q_label}) = {prob:.4f}"
            x_fill = np.linspace(0, max(0, q), 100)
        else:
            prob = float(1.0 - self.dist.cdf(q))
            label = f"P(X > {q_label}) = {prob:.4f}"
            x_fill = np.linspace(max(0, q), hi, 100)

        fig, ax = self._draw_base(label)
        ax.fill_between(x_fill, self.dist.pdf(x_fill), color="skyblue", alpha=0.6)
        return prob, fig

    def qchisq(self, p: float, *, lower_tail: bool = True):
        target_p = p if lower_tail else 1.0 - p
        cutoff = float(self.dist.ppf(target_p))
        direction = "below" if lower_tail else "above"
        title = f"{p*100:.1f}% of values are {direction} {cutoff:.4g}"
        fig, ax = self._draw_base(title)
        hi = self._upper_x()
        if lower_tail:
            x_fill = np.linspace(0, cutoff, 100)
        else:
            x_fill = np.linspace(cutoff, hi, 100)
        ax.fill_between(x_fill, self.dist.pdf(x_fill), color="salmon", alpha=0.4)
        ax.axvline(cutoff, color="red", linestyle="--", label=f"Cutoff: {cutoff:.4g}")
        ax.legend()
        return cutoff, fig


class FDist:
    """F(dfn, dfd): R-style pf / qf (support x > 0)."""

    def __init__(self, dfn: float, dfd: float):
        self.dfn = float(dfn)
        self.dfd = float(dfd)
        self.dist = stats.f(self.dfn, self.dfd)

    def _upper_x(self) -> float:
        return max(5.0, float(self.dist.ppf(0.999)) * 1.1)

    def _draw_base(self, title: str):
        hi = self._upper_x()
        lo = 1e-4
        x = np.linspace(lo, hi, 1000)
        y = self.dist.pdf(x)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, y, color="black", lw=2)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.set_xlim(left=0)
        return fig, ax

    def pf(self, q: float, q2: float | None = None, *, lower_tail: bool = True):
        q_label = f"{q:.4g}"
        hi = self._upper_x()
        if q2 is not None:
            q2_label = f"{q2:.4g}"
            prob = float(self.dist.cdf(q2) - self.dist.cdf(q))
            label = f"P({q_label} < F < {q2_label}) = {prob:.4f}"
            x_fill = np.linspace(max(1e-6, q), q2, 100)
        elif lower_tail:
            prob = float(self.dist.cdf(q))
            label = f"P(F < {q_label}) = {prob:.4f}"
            x_fill = np.linspace(1e-6, max(1e-6, q), 100)
        else:
            prob = float(1.0 - self.dist.cdf(q))
            label = f"P(F > {q_label}) = {prob:.4f}"
            x_fill = np.linspace(max(1e-6, q), hi, 100)

        fig, ax = self._draw_base(label)
        ax.fill_between(x_fill, self.dist.pdf(x_fill), color="skyblue", alpha=0.6)
        return prob, fig

    def qf(self, p: float, *, lower_tail: bool = True):
        target_p = p if lower_tail else 1.0 - p
        cutoff = float(self.dist.ppf(target_p))
        direction = "below" if lower_tail else "above"
        title = f"{p*100:.1f}% of values are {direction} {cutoff:.4g}"
        fig, ax = self._draw_base(title)
        hi = self._upper_x()
        if lower_tail:
            x_fill = np.linspace(1e-6, cutoff, 100)
        else:
            x_fill = np.linspace(cutoff, hi, 100)
        ax.fill_between(x_fill, self.dist.pdf(x_fill), color="salmon", alpha=0.4)
        ax.axvline(cutoff, color="red", linestyle="--", label=f"Cutoff: {cutoff:.4g}")
        ax.legend()
        return cutoff, fig


def _sync_widget_from_saved(widget_key: str, saved_key: str, *, force_restore: bool = False) -> None:
    if force_restore or widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[saved_key]


def _render_normal_tab() -> None:
    pnorm_modes = ["Left tail: P(X < q)", "Right tail: P(X > q)", "Between: P(q1 < X < q2)"]
    qnorm_tails = ["Lower tail (left)", "Upper tail (right)"]

    defaults = {
        "normal_calc_kind": "Normal Distribution (pnorm)",
        "normal_mu_saved": 0.0,
        "normal_sigma_saved": 1.0,
        "normal_q_saved": 1.96,
        "normal_q1_saved": -1.96,
        "normal_q2_saved": 1.96,
        "normal_p_lower_saved": 0.95,
        "normal_p_upper_saved": 0.05,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "normal_pnorm_mode" not in st.session_state:
        st.session_state["normal_pnorm_mode"] = pnorm_modes[0]
    if "normal_qnorm_tail" not in st.session_state:
        st.session_state["normal_qnorm_tail"] = qnorm_tails[0]

    c_left, c_right = st.columns([1, 2])
    prob = None
    cutoff = None
    fig = None
    interp_text = ""
    validation_error = ""

    with c_left:
        st.markdown("#### Inputs")
        calc = st.radio(
            "Calculator",
            ["Normal Distribution (pnorm)", "Inverse Normal (qnorm)"],
            horizontal=False,
            key="normal_calc_kind",
        )
        mu_col, sigma_col = st.columns(2, gap="small")
        with mu_col:
            _sync_widget_from_saved("normal_mu_input", "normal_mu_saved")
            mu = st.number_input("Mean (μ)", step=1.0, key="normal_mu_input")
            st.session_state["normal_mu_saved"] = mu
        with sigma_col:
            _sync_widget_from_saved("normal_sigma_input", "normal_sigma_saved")
            sigma = st.number_input("Std dev (σ)", min_value=0.0, step=0.1, key="normal_sigma_input")
            st.session_state["normal_sigma_saved"] = sigma

        if calc == "Normal Distribution (pnorm)":
            mode = st.radio("Calculation type", pnorm_modes, key="normal_pnorm_mode")
            active_signature = "pnorm_between" if mode == "Between: P(q1 < X < q2)" else "pnorm_single"
            restore_active_inputs = st.session_state.get("normal_active_signature") != active_signature
            st.session_state["normal_active_signature"] = active_signature
            if mode == "Between: P(q1 < X < q2)":
                q1_col, q2_col = st.columns(2, gap="small")
                with q1_col:
                    _sync_widget_from_saved("normal_q1_input", "normal_q1_saved", force_restore=restore_active_inputs)
                    q1 = st.number_input("q1", step=0.1, key="normal_q1_input")
                    st.session_state["normal_q1_saved"] = q1
                with q2_col:
                    _sync_widget_from_saved("normal_q2_input", "normal_q2_saved", force_restore=restore_active_inputs)
                    q2 = st.number_input("q2", step=0.1, key="normal_q2_input")
                    st.session_state["normal_q2_saved"] = q2
            else:
                _sync_widget_from_saved("normal_q_input", "normal_q_saved", force_restore=restore_active_inputs)
                q = st.number_input("q", step=0.1, key="normal_q_input")
                st.session_state["normal_q_saved"] = q
        else:
            tail = st.radio("Tail type", qnorm_tails, key="normal_qnorm_tail")
            active_signature = "qnorm_lower" if tail == "Lower tail (left)" else "qnorm_upper"
            restore_active_inputs = st.session_state.get("normal_active_signature") != active_signature
            st.session_state["normal_active_signature"] = active_signature
            p_key = "normal_p_lower_input" if tail == "Lower tail (left)" else "normal_p_upper_input"
            p_saved_key = "normal_p_lower_saved" if tail == "Lower tail (left)" else "normal_p_upper_saved"
            _sync_widget_from_saved(p_key, p_saved_key, force_restore=restore_active_inputs)
            p = st.number_input("p (0 < p < 1)", min_value=0.0, max_value=1.0, step=0.01, key=p_key)
            st.session_state[p_saved_key] = p

    if sigma <= 0:
        validation_error = "Standard deviation σ must be greater than 0."
    else:
        model = NormalDist(mu=mu, sigma=sigma)
        if calc == "Normal Distribution (pnorm)":
            if mode == "Between: P(q1 < X < q2)":
                if q1 >= q2:
                    validation_error = "For a between probability, q1 must be less than q2."
                else:
                    prob, fig = model.pnorm(q1, q2=q2)
                    interp_text = f"About {prob*100:.2f}% of observations fall between {q1:.4g} and {q2:.4g}."
            else:
                lower_tail = mode == "Left tail: P(X < q)"
                prob, fig = model.pnorm(q, lower_tail=lower_tail)
                if lower_tail:
                    interp_text = f"About {prob*100:.2f}% of observations are below {q:.4g}."
                else:
                    interp_text = f"About {prob*100:.2f}% of observations are above {q:.4g}."
        else:
            if p <= 0 or p >= 1:
                validation_error = "p must be between 0 and 1 (exclusive)."
            else:
                lower_tail = tail == "Lower tail (left)"
                cutoff, fig = model.qnorm(p, lower_tail=lower_tail)
                if lower_tail:
                    interp_text = f"Cutoff ≈ {cutoff:.4g}: about {p*100:.1f}% of observations are below this value."
                else:
                    interp_text = f"Cutoff ≈ {cutoff:.4g}: about {p*100:.1f}% of observations are above this value."

    with c_right:
        st.markdown("#### Result")
        if validation_error:
            st.error(validation_error)
            return
        st.write(interp_text)
        st.pyplot(fig, width="stretch")
        plt.close(fig)


def _render_t_tab() -> None:
    p_modes = ["Left tail: P(T < q)", "Right tail: P(T > q)", "Between: P(q1 < T < q2)"]
    q_tails = ["Lower tail (left)", "Upper tail (right)"]
    defaults = {
        "t_calc_kind": "Student's t (pt)",
        "t_df_saved": 10.0,
        "t_q_saved": 0.0,
        "t_q1_saved": -1.0,
        "t_q2_saved": 1.0,
        "t_p_lower_saved": 0.95,
        "t_p_upper_saved": 0.05,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "t_p_mode" not in st.session_state:
        st.session_state["t_p_mode"] = p_modes[0]
    if "t_q_tail" not in st.session_state:
        st.session_state["t_q_tail"] = q_tails[0]

    c_left, c_right = st.columns([1, 2])
    fig = None
    interp_text = ""
    validation_error = ""

    with c_left:
        st.markdown("#### Inputs")
        calc = st.radio(
            "Calculator",
            ["Student's t (pt)", "Inverse t (qt)"],
            horizontal=False,
            key="t_calc_kind",
        )
        _sync_widget_from_saved("t_df_input", "t_df_saved")
        df_t = st.number_input("Degrees of freedom (df)", min_value=0.0, step=1.0, key="t_df_input")
        st.session_state["t_df_saved"] = df_t

        if calc == "Student's t (pt)":
            mode = st.radio("Calculation type", p_modes, key="t_p_mode")
            sig = "t_pbetween" if mode == "Between: P(q1 < T < q2)" else "t_psingle"
            restore = st.session_state.get("t_active_sig") != sig
            st.session_state["t_active_sig"] = sig
            if mode == "Between: P(q1 < T < q2)":
                c1, c2 = st.columns(2, gap="small")
                with c1:
                    _sync_widget_from_saved("t_q1_input", "t_q1_saved", force_restore=restore)
                    q1 = st.number_input("q1", step=0.1, key="t_q1_input")
                    st.session_state["t_q1_saved"] = q1
                with c2:
                    _sync_widget_from_saved("t_q2_input", "t_q2_saved", force_restore=restore)
                    q2 = st.number_input("q2", step=0.1, key="t_q2_input")
                    st.session_state["t_q2_saved"] = q2
            else:
                _sync_widget_from_saved("t_q_input", "t_q_saved", force_restore=restore)
                q = st.number_input("q", step=0.1, key="t_q_input")
                st.session_state["t_q_saved"] = q
        else:
            tail = st.radio("Tail type", q_tails, key="t_q_tail")
            sig = "t_qlower" if tail == "Lower tail (left)" else "t_qupper"
            restore = st.session_state.get("t_active_sig") != sig
            st.session_state["t_active_sig"] = sig
            pk = "t_p_lower_input" if tail == "Lower tail (left)" else "t_p_upper_input"
            sk = "t_p_lower_saved" if tail == "Lower tail (left)" else "t_p_upper_saved"
            _sync_widget_from_saved(pk, sk, force_restore=restore)
            p = st.number_input("p (0 < p < 1)", min_value=0.0, max_value=1.0, step=0.01, key=pk)
            st.session_state[sk] = p

    if df_t <= 0:
        validation_error = "Degrees of freedom must be greater than 0."
    else:
        model = StudentTDist(df_t)
        if calc == "Student's t (pt)":
            if mode == "Between: P(q1 < T < q2)":
                if q1 >= q2:
                    validation_error = "q1 must be less than q2."
                else:
                    prob, fig = model.pt(q1, q2=q2)
                    interp_text = f"P(q1 < T < q2) ≈ {prob:.4f} ({prob*100:.2f}%)."
            else:
                lower = mode == "Left tail: P(T < q)"
                prob, fig = model.pt(q, lower_tail=lower)
                interp_text = f"Tail probability ≈ {prob:.4f} ({prob*100:.2f}%)."
        else:
            if p <= 0 or p >= 1:
                validation_error = "p must be between 0 and 1 (exclusive)."
            else:
                lower = tail == "Lower tail (left)"
                cutoff, fig = model.qt(p, lower_tail=lower)
                interp_text = f"Quantile ≈ {cutoff:.4g} for the chosen tail and p."

    with c_right:
        st.markdown("#### Result")
        if validation_error:
            st.error(validation_error)
            return
        st.write(interp_text)
        st.pyplot(fig, width="stretch")
        plt.close(fig)


def _render_chisq_tab() -> None:
    p_modes = ["Left tail: P(X < q)", "Right tail: P(X > q)", "Between: P(q1 < X < q2)"]
    q_tails = ["Lower tail (left)", "Upper tail (right)"]
    defaults = {
        "chisq_calc_kind": "Chi-square (pchisq)",
        "chisq_df_saved": 5.0,
        "chisq_q_saved": 5.0,
        "chisq_q1_saved": 1.0,
        "chisq_q2_saved": 8.0,
        "chisq_p_lower_saved": 0.95,
        "chisq_p_upper_saved": 0.05,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "chisq_p_mode" not in st.session_state:
        st.session_state["chisq_p_mode"] = p_modes[0]
    if "chisq_q_tail" not in st.session_state:
        st.session_state["chisq_q_tail"] = q_tails[0]

    c_left, c_right = st.columns([1, 2])
    fig = None
    interp_text = ""
    validation_error = ""

    with c_left:
        st.markdown("#### Inputs")
        calc = st.radio(
            "Calculator",
            ["Chi-square (pchisq)", "Inverse χ² (qchisq)"],
            horizontal=False,
            key="chisq_calc_kind",
        )
        _sync_widget_from_saved("chisq_df_input", "chisq_df_saved")
        df_c = st.number_input("Degrees of freedom (df)", min_value=0.0, step=1.0, key="chisq_df_input")
        st.session_state["chisq_df_saved"] = df_c

        if calc == "Chi-square (pchisq)":
            mode = st.radio("Calculation type", p_modes, key="chisq_p_mode")
            sig = "cx_pbetween" if mode == "Between: P(q1 < X < q2)" else "cx_psingle"
            restore = st.session_state.get("chisq_active_sig") != sig
            st.session_state["chisq_active_sig"] = sig
            if mode == "Between: P(q1 < X < q2)":
                c1, c2 = st.columns(2, gap="small")
                with c1:
                    _sync_widget_from_saved("chisq_q1_input", "chisq_q1_saved", force_restore=restore)
                    q1 = st.number_input("q1 (≥ 0)", min_value=0.0, step=0.5, key="chisq_q1_input")
                    st.session_state["chisq_q1_saved"] = q1
                with c2:
                    _sync_widget_from_saved("chisq_q2_input", "chisq_q2_saved", force_restore=restore)
                    q2 = st.number_input("q2 (≥ 0)", min_value=0.0, step=0.5, key="chisq_q2_input")
                    st.session_state["chisq_q2_saved"] = q2
            else:
                _sync_widget_from_saved("chisq_q_input", "chisq_q_saved", force_restore=restore)
                q = st.number_input("q (≥ 0)", min_value=0.0, step=0.5, key="chisq_q_input")
                st.session_state["chisq_q_saved"] = q
        else:
            tail = st.radio("Tail type", q_tails, key="chisq_q_tail")
            sig = "cx_qlower" if tail == "Lower tail (left)" else "cx_qupper"
            restore = st.session_state.get("chisq_active_sig") != sig
            st.session_state["chisq_active_sig"] = sig
            pk = "chisq_p_lower_input" if tail == "Lower tail (left)" else "chisq_p_upper_input"
            sk = "chisq_p_lower_saved" if tail == "Lower tail (left)" else "chisq_p_upper_saved"
            _sync_widget_from_saved(pk, sk, force_restore=restore)
            p = st.number_input("p (0 < p < 1)", min_value=0.0, max_value=1.0, step=0.01, key=pk)
            st.session_state[sk] = p

    if df_c <= 0:
        validation_error = "Degrees of freedom must be greater than 0."
    else:
        model = ChiSqDist(df_c)
        if calc == "Chi-square (pchisq)":
            if mode == "Between: P(q1 < X < q2)":
                if q1 >= q2:
                    validation_error = "q1 must be less than q2."
                else:
                    prob, fig = model.pchisq(q1, q2=q2)
                    interp_text = f"P(q1 < χ² < q2) ≈ {prob:.4f}."
            else:
                lower = mode == "Left tail: P(X < q)"
                prob, fig = model.pchisq(q, lower_tail=lower)
                interp_text = f"Tail probability ≈ {prob:.4f}."
        else:
            if p <= 0 or p >= 1:
                validation_error = "p must be between 0 and 1 (exclusive)."
            else:
                lower = tail == "Lower tail (left)"
                cutoff, fig = model.qchisq(p, lower_tail=lower)
                interp_text = f"Quantile ≈ {cutoff:.4g}."

    with c_right:
        st.markdown("#### Result")
        if validation_error:
            st.error(validation_error)
            return
        st.write(interp_text)
        st.pyplot(fig, width="stretch")
        plt.close(fig)


def _render_f_tab() -> None:
    p_modes = ["Left tail: P(F < q)", "Right tail: P(F > q)", "Between: P(q1 < F < q2)"]
    q_tails = ["Lower tail (left)", "Upper tail (right)"]
    defaults = {
        "f_calc_kind": "F distribution (pf)",
        "f_dfn_saved": 5.0,
        "f_dfd_saved": 20.0,
        "f_q_saved": 2.0,
        "f_q1_saved": 0.5,
        "f_q2_saved": 3.0,
        "f_p_lower_saved": 0.95,
        "f_p_upper_saved": 0.05,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "f_p_mode" not in st.session_state:
        st.session_state["f_p_mode"] = p_modes[0]
    if "f_q_tail" not in st.session_state:
        st.session_state["f_q_tail"] = q_tails[0]

    c_left, c_right = st.columns([1, 2])
    fig = None
    interp_text = ""
    validation_error = ""

    with c_left:
        st.markdown("#### Inputs")
        calc = st.radio(
            "Calculator",
            ["F distribution (pf)", "Inverse F (qf)"],
            horizontal=False,
            key="f_calc_kind",
        )
        c1, c2 = st.columns(2, gap="small")
        with c1:
            _sync_widget_from_saved("f_dfn_input", "f_dfn_saved")
            dfn = st.number_input("df (numerator)", min_value=0.0, step=1.0, key="f_dfn_input")
            st.session_state["f_dfn_saved"] = dfn
        with c2:
            _sync_widget_from_saved("f_dfd_input", "f_dfd_saved")
            dfd = st.number_input("df (denominator)", min_value=0.0, step=1.0, key="f_dfd_input")
            st.session_state["f_dfd_saved"] = dfd

        if calc == "F distribution (pf)":
            mode = st.radio("Calculation type", p_modes, key="f_p_mode")
            sig = "f_pbetween" if mode == "Between: P(q1 < F < q2)" else "f_psingle"
            restore = st.session_state.get("f_active_sig") != sig
            st.session_state["f_active_sig"] = sig
            if mode == "Between: P(q1 < F < q2)":
                a1, a2 = st.columns(2, gap="small")
                with a1:
                    _sync_widget_from_saved("f_q1_input", "f_q1_saved", force_restore=restore)
                    q1 = st.number_input("q1 (≥ 0)", min_value=0.0, step=0.1, key="f_q1_input")
                    st.session_state["f_q1_saved"] = q1
                with a2:
                    _sync_widget_from_saved("f_q2_input", "f_q2_saved", force_restore=restore)
                    q2 = st.number_input("q2 (≥ 0)", min_value=0.0, step=0.1, key="f_q2_input")
                    st.session_state["f_q2_saved"] = q2
            else:
                _sync_widget_from_saved("f_q_input", "f_q_saved", force_restore=restore)
                q = st.number_input("q (≥ 0)", min_value=0.0, step=0.1, key="f_q_input")
                st.session_state["f_q_saved"] = q
        else:
            tail = st.radio("Tail type", q_tails, key="f_q_tail")
            sig = "f_qlower" if tail == "Lower tail (left)" else "f_qupper"
            restore = st.session_state.get("f_active_sig") != sig
            st.session_state["f_active_sig"] = sig
            pk = "f_p_lower_input" if tail == "Lower tail (left)" else "f_p_upper_input"
            sk = "f_p_lower_saved" if tail == "Lower tail (left)" else "f_p_upper_saved"
            _sync_widget_from_saved(pk, sk, force_restore=restore)
            p = st.number_input("p (0 < p < 1)", min_value=0.0, max_value=1.0, step=0.01, key=pk)
            st.session_state[sk] = p

    if dfn <= 0 or dfd <= 0:
        validation_error = "Both numerator and denominator df must be greater than 0."
    else:
        model = FDist(dfn, dfd)
        if calc == "F distribution (pf)":
            if mode == "Between: P(q1 < F < q2)":
                if q1 >= q2:
                    validation_error = "q1 must be less than q2."
                else:
                    prob, fig = model.pf(q1, q2=q2)
                    interp_text = f"P(q1 < F < q2) ≈ {prob:.4f}."
            else:
                lower = mode == "Left tail: P(F < q)"
                prob, fig = model.pf(q, lower_tail=lower)
                interp_text = f"Tail probability ≈ {prob:.4f}."
        else:
            if p <= 0 or p >= 1:
                validation_error = "p must be between 0 and 1 (exclusive)."
            else:
                lower = tail == "Lower tail (left)"
                cutoff, fig = model.qf(p, lower_tail=lower)
                interp_text = f"Quantile ≈ {cutoff:.4g}."

    with c_right:
        st.markdown("#### Result")
        if validation_error:
            st.error(validation_error)
            return
        st.write(interp_text)
        st.pyplot(fig, width="stretch")
        plt.close(fig)


def render_distribution_tools() -> None:
    """Top-level distribution section: tabs for Normal, t, χ², F (default: Normal)."""
    # st.subheader("Distribution tools")
    tab_n, tab_t, tab_c, tab_f = st.tabs(["Normal", "Student's t", "Chi-square", "F"])
    with tab_n:
        _render_normal_tab()
    with tab_t:
        _render_t_tab()
    with tab_c:
        _render_chisq_tab()
    with tab_f:
        _render_f_tab()