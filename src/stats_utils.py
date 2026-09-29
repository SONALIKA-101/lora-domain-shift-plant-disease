"""
Statistical tools for comparing two classifiers on the same held-out data.

1. McNemar's test compares two models' predictions on the same held-out
   samples, using one trained model per method. It is paired (same test items,
   two sets of predictions), so it does not need repeated training runs. It
   tests whether the two models' error rates differ on this test set; it does
   not capture variation between training runs (seeds).

2. Bootstrap confidence intervals resample the held-out predictions with
   replacement to estimate a 95% interval for a metric such as accuracy or
   macro-F1. They reflect variation in which test images were drawn, not
   variation between training runs.

With only three leave-one-domain-out rotations, a formal paired significance
test across rotations (for example a paired t-test on the three per-rotation
scores) would be underpowered. Across-rotation results are therefore reported
as descriptive statistics (mean and sample standard deviation, see
scripts/run_all_rotations.py). McNemar's test avoids this problem by testing
within a single rotation's predictions.
"""

import numpy as np
from scipy import stats


def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray):
    """
    McNemar's test for two classifiers' predictions on the same test set.

    Returns a dict with the contingency table and p-value. Uses the exact
    binomial version when the discordant count is small (<25, common in
    smaller held-out sets like PlantDoc), otherwise the chi-square
    approximation with continuity correction.
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # b01: A wrong, B right (B's unique wins). b10: A right, B wrong (A's unique wins).
    b01 = int(np.sum((~correct_a) & correct_b))
    b10 = int(np.sum(correct_a & (~correct_b)))
    n_discordant = b01 + b10

    if n_discordant == 0:
        return {"b01": b01, "b10": b10, "n_discordant": 0, "p_value": 1.0, "test_used": "none (no discordant pairs)"}

    if n_discordant < 25:
        # exact binomial test: under H0, discordant pairs split 50/50
        p_value = 2 * stats.binom.cdf(min(b01, b10), n_discordant, 0.5)
        p_value = min(p_value, 1.0)
        test_used = "exact binomial"
    else:
        statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)  # continuity-corrected
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        test_used = "chi-square (continuity-corrected)"

    return {
        "b01": b01, "b10": b10, "n_discordant": n_discordant,
        "p_value": p_value, "test_used": test_used,
    }


def bootstrap_ci(preds: np.ndarray, labels: np.ndarray, metric_fn, n_bootstrap: int = 1000,
                  ci: float = 0.95, seed: int = 42):
    """
    Bootstrap confidence interval for a metric (e.g. accuracy_score or a
    macro-F1 lambda) computed on held-out predictions, by resampling the
    test set with replacement. Returns (point_estimate, ci_low, ci_high).
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    point_estimate = metric_fn(labels, preds)

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        scores.append(metric_fn(labels[idx], preds[idx]))

    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(scores, 100 * alpha))
    ci_high = float(np.percentile(scores, 100 * (1 - alpha)))
    return point_estimate, ci_low, ci_high


def generalization_gap(val_metric: float, held_out_metric: float) -> float:
    """Validation metric on the training domains minus the same metric on the
    unseen held-out domain (Eq. 6 in the paper uses macro-F1). A smaller gap
    means less performance is lost under domain shift."""
    return val_metric - held_out_metric
