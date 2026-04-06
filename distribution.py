import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st

class NormalDist:
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.dist = norm(loc=mu, scale=sigma)
        
    def _draw_base(self, title):
        x = np.linspace(self.mu - 4*self.sigma, self.mu + 4*self.sigma, 1000)
        y = self.dist.pdf(x)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, y, color="black", lw=2)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        return fig, ax

    def pnorm(self, q, q2=None, lower_tail=True):
        """Calculates percentage below, above, or between values."""
        q_label = f"{q:.2f}"
        if q2 is not None:
            q2_label = f"{q2:.2f}"
            prob = self.dist.cdf(q2) - self.dist.cdf(q)
            label = f"P({q_label} < X < {q2_label}) = {prob:.4f}"
            x_fill = np.linspace(q, q2, 100)
        elif lower_tail:
            prob = self.dist.cdf(q)
            label = f"P(X < {q_label}) = {prob:.4f}"
            x_fill = np.linspace(self.mu - 4*self.sigma, q, 100)
        else:
            prob = 1 - self.dist.cdf(q)
            label = f"P(X > {q_label}) = {prob:.4f}"
            x_fill = np.linspace(q, self.mu + 4*self.sigma, 100)

        fig, ax = self._draw_base(label)
        ax.fill_between(x_fill, self.dist.pdf(x_fill), color="skyblue", alpha=0.6)
        return prob, fig

    def qnorm(self, p, lower_tail=True):
        """
        Finds the cutoff x value for a given percentage.
        - lower_tail=True: p is the area to the LEFT (bottom p%)
        - lower_tail=False: p is the area to the RIGHT (top p%)
        """
        # If lower_tail is False, we want the point where (1-p) is to the left
        target_p = p if lower_tail else 1 - p
        cutoff = self.dist.ppf(target_p)
        
        direction = "below" if lower_tail else "above"
        title = f"{p*100:.1f}% of values are {direction} {cutoff:.4f}"
        
        fig, ax = self._draw_base(title)
        
        # Shade the relevant area
        if lower_tail:
            x_fill = np.linspace(self.mu - 4*self.sigma, cutoff, 100)
        else:
            x_fill = np.linspace(cutoff, self.mu + 4*self.sigma, 100)
            
        ax.fill_between(x_fill, self.dist.pdf(x_fill), color="salmon", alpha=0.4)
        ax.axvline(cutoff, color="red", linestyle="--", label=f"Cutoff: {cutoff:.3f}")
        ax.legend()
        return cutoff, fig


def render_normal_calculator() -> None:
    """Render no-data normal distribution tools (pnorm and qnorm).

    Widget values for mu/sigma/q/p are mirrored into *_saved keys each run.
    When this view is hidden, StatSketch.py pops the *input* widget keys so the next
    visit can seed from *_saved (see sync_widget_from_saved).
    """
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

    def sync_widget_from_saved(widget_key: str, saved_key: str, force_restore: bool = False) -> None:
        """Seed a number_input's session key from the persisted *_saved value.

        Streamlit widgets read/write ``widget_key``. We mirror the latest value into
        ``saved_key`` after each widget so it survives reruns. On a *new* visit to
        this page (or after the app pops ``widget_key`` when leaving Distribution
        Tools), ``widget_key`` is absent: we copy ``saved_key`` → ``widget_key`` so
        the input shows the last known good value (e.g. sigma 1.0).

        If ``widget_key`` already exists, we do **not** overwrite it (unless
        ``force_restore``): that value is the widget's current state and must win
        over saved state on the same run. Stale ``widget_key`` entries when the
        calculator was not mounted are avoided by clearing those keys in
        StatSketch.py.

        ``force_restore`` is used when the user switches pnorm mode or qnorm tail:
        the active widget set changes, so we reload the matching *_saved pair
        instead of leaving an unrelated key in session state.
        """
        if force_restore or widget_key not in st.session_state:
            st.session_state[widget_key] = st.session_state[saved_key]

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
            sync_widget_from_saved("normal_mu_input", "normal_mu_saved")
            mu = st.number_input("Mean (mu)", step=1.0, key="normal_mu_input")
            st.session_state["normal_mu_saved"] = mu
        with sigma_col:
            sync_widget_from_saved("normal_sigma_input", "normal_sigma_saved")
            sigma = st.number_input("Std dev (sigma)", min_value=0.0, step=0.1, key="normal_sigma_input")
            st.session_state["normal_sigma_saved"] = sigma

        if calc == "Normal Distribution (pnorm)":
            mode = st.radio("Calculation type", pnorm_modes, key="normal_pnorm_mode")
            active_signature = "pnorm_between" if mode == "Between: P(q1 < X < q2)" else "pnorm_single"
            restore_active_inputs = st.session_state.get("normal_active_signature") != active_signature
            st.session_state["normal_active_signature"] = active_signature
            if mode == "Between: P(q1 < X < q2)":
                q1_col, q2_col = st.columns(2, gap="small")
                with q1_col:
                    sync_widget_from_saved("normal_q1_input", "normal_q1_saved", force_restore=restore_active_inputs)
                    q1 = st.number_input("q1", step=0.1, key="normal_q1_input")
                    st.session_state["normal_q1_saved"] = q1
                with q2_col:
                    sync_widget_from_saved("normal_q2_input", "normal_q2_saved", force_restore=restore_active_inputs)
                    q2 = st.number_input("q2", step=0.1, key="normal_q2_input")
                    st.session_state["normal_q2_saved"] = q2
            else:
                sync_widget_from_saved("normal_q_input", "normal_q_saved", force_restore=restore_active_inputs)
                q = st.number_input("q", step=0.1, key="normal_q_input")
                st.session_state["normal_q_saved"] = q
        else:
            tail = st.radio("Tail type", qnorm_tails, key="normal_qnorm_tail")
            active_signature = "qnorm_lower" if tail == "Lower tail (left)" else "qnorm_upper"
            restore_active_inputs = st.session_state.get("normal_active_signature") != active_signature
            st.session_state["normal_active_signature"] = active_signature
            p_key = "normal_p_lower_input" if tail == "Lower tail (left)" else "normal_p_upper_input"
            p_saved_key = "normal_p_lower_saved" if tail == "Lower tail (left)" else "normal_p_upper_saved"
            sync_widget_from_saved(p_key, p_saved_key, force_restore=restore_active_inputs)
            p = st.number_input("p (must satisfy 0 < p < 1)", min_value=0.0, max_value=1.0, step=0.01, key=p_key)
            st.session_state[p_saved_key] = p

    if sigma <= 0:
        validation_error = "Standard deviation must be greater than 0."
    else:
        model = NormalDist(mu=mu, sigma=sigma)
        if calc == "Normal Distribution (pnorm)":
            if mode == "Between: P(q1 < X < q2)":
                if q1 >= q2:
                    validation_error = "For a between probability, q1 must be less than q2."
                else:
                    prob, fig = model.pnorm(q1, q2=q2)
                    interp_text = f"About {prob*100:.2f}% of observations are expected to fall between {q1:.2f} and {q2:.2f}."
            else:
                lower_tail = mode == "Left tail: P(X < q)"
                prob, fig = model.pnorm(q, lower_tail=lower_tail)
                if lower_tail:
                    interp_text = f"About {prob*100:.2f}% of observations are expected to be below {q:.2f}."
                else:
                    interp_text = f"About {prob*100:.2f}% of observations are expected to be above {q:.2f}."
        else:
            if p <= 0 or p >= 1:
                validation_error = "p must be between 0 and 1 (exclusive)."
            else:
                lower_tail = tail == "Lower tail (left)"
                cutoff, fig = model.qnorm(p, lower_tail=lower_tail)
                if lower_tail:
                    interp_text = f"The cutoff is {cutoff:.3f}: about {p*100:.1f}% of observations should be below this value."
                else:
                    interp_text = f"The cutoff is {cutoff:.3f}: about {p*100:.1f}% of observations should be above this value."

    with c_right:
        st.markdown("#### Result")
        if validation_error:
            st.error(validation_error)
            return
        st.write(interp_text)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# model = NormalDist(mu=0, sigma=1)
# model.pnorm(1.96) # Returns ~0.975 and shades the left tail
# model.pnorm(1.96, lower_tail=False) # Returns ~0.025 and shades the right tail
# # Similar to pnorm(q2) - pnorm(q1) in R
# model.pnorm(-1.96, q2=1.96) # Returns ~0.95 and shades the middle

# # What value marks the 95th percentile?
# cutoff = model.qnorm(0.95) # Returns 1.644 and marks the spot

# model = NormalDist(mu=100, sigma=15) # Example: IQ scale

# # Find the cutoff for the TOP 5%
# top_five_percent = model.qnorm(0.05, lower_tail=False)
# # Result: ~124.67