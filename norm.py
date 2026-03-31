import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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