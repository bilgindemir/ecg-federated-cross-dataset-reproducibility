"""Patient-vector inference across matched model seeds."""
import itertools
import math
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

def mean_seed_auc(A, B):

    # A is treated as positive/member group.
    y = np.r_[
        np.ones(len(A), dtype=int),
        np.zeros(len(B), dtype=int),
    ]

    vals = []

    for seed_col in range(A.shape[1]):

        s = np.r_[
            A[:, seed_col],
            B[:, seed_col],
        ]

        vals.append(
            roc_auc_score(y, s)
        )

    return float(np.mean(vals)), np.asarray(vals)

def exact_patient_permutation(A, B):

    n1 = len(A)
    n0 = len(B)

    X = np.vstack([A, B])

    # Mean rank over seeds.
    # Because AUC is equivalent to the Mann-Whitney U
    # statistic, averaging these ranks gives exactly
    # the mean AUC across model seeds.
    ranks = np.column_stack([
        rankdata(X[:, j], method="average")
        for j in range(X.shape[1])
    ])

    mean_rank = ranks.mean(axis=1)

    denom = n1 * n0
    base = n1 * (n1 + 1) / 2

    observed = (
        mean_rank[:n1].sum() - base
    ) / denom

    null = np.empty(
        math.comb(n1 + n0, n1),
        dtype=float,
    )

    for k, idx in enumerate(
        itertools.combinations(
            range(n1 + n0),
            n1,
        )
    ):
        null[k] = (
            mean_rank[list(idx)].sum()
            - base
        ) / denom

    # Membership question: is AUC > 0.5?
    p_greater = np.mean(
        null >= observed - 1e-15
    )

    # Also provide two-sided result.
    p_two_sided = np.mean(
        np.abs(null - 0.5)
        >= abs(observed - 0.5) - 1e-15
    )

    return {
        "auc": float(observed),
        "exact_p_greater": float(p_greater),
        "exact_p_two_sided": float(p_two_sided),
        "permutations": len(null),
    }

def patient_bootstrap_ci(
    A,
    B,
    repetitions=20000,
    seed=20261117,
):

    rng = np.random.default_rng(seed)

    vals = np.empty(repetitions)

    for b in range(repetitions):

        ia = rng.integers(
            0, len(A), len(A)
        )

        ib = rng.integers(
            0, len(B), len(B)
        )

        vals[b] = mean_seed_auc(
            A[ia],
            B[ib],
        )[0]

    return (
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
    )
