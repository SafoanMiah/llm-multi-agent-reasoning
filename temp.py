# %% [markdown]
# ---
# ### 9. Statistical Validation
#
# Are the accuracy differences between topologies statistically significant, or could they be noise from a 100-question sample?
#
# We apply a standard three-step testing pipeline:
# 1. **Cochran's Q test** — omnibus test: *is there any difference at all?* (Raschka, 2018; Fleiss et al., 2003)
# 2. **Pairwise McNemar's tests** — post-hoc: *where exactly are the differences?* (Dietterich, 1998)
# 3. **Bootstrap confidence intervals** — uncertainty quantification: *how stable are these accuracy numbers?* (Efron & Tibshirani, 1993)
#
# All three are appropriate because our data is **paired** (same 100 questions across all topologies) and **binary** (correct/incorrect per question).

# %%
from mlxtend.evaluate import cochrans_q, mcnemar, mcnemar_table
from itertools import combinations
import numpy as np

# Build question-level correctness matrix (one row per question, one column per topology)
TOPO_ORDER = ["independent", "chain", "full", "mediator"]

q_correct = (
    df.groupby(["topology", "question_idx"])["correct"]
    .first()
    .reset_index()
    .pivot(index="question_idx", columns="topology", values="correct")
    .astype(int)[TOPO_ORDER]
)

y_true = q_correct.values  # (100, 4) binary matrix
print(f"Questions: {y_true.shape[0]}, Topologies: {y_true.shape[1]}")
print(f"Accuracies: { {t: f'{q_correct[t].mean():.0%}' for t in TOPO_ORDER} }")

# %% [markdown]
# #### Step 1: Cochran's Q Test (Omnibus)
#
# Tests $H_0$: all four topologies have **equal accuracy**.
# If rejected, at least one topology is significantly different — we then proceed to pairwise tests.

# %%
Q_stat, p_cochran = cochrans_q(
    q_correct["independent"].values,
    q_correct["chain"].values,
    q_correct["full"].values,
    q_correct["mediator"].values,
)

print(f"Cochran's Q = {Q_stat:.2f}, p = {p_cochran:.2e}")
print(f"→ {'Reject H₀' if p_cochran < 0.05 else 'Fail to reject H₀'} (α = 0.05): significant difference exists between topologies.")

# %% [markdown]
# #### Step 2: Pairwise McNemar's Tests (Post-Hoc)
#
# Now that Cochran's Q confirms a difference exists, we test each pair individually.
# McNemar's test examines the **discordant pairs** — questions where one topology got it right and the other got it wrong.
#
# We apply **Bonferroni correction** ($\alpha_{\text{adj}} = 0.05 / 6 = 0.0083$) to control for multiple comparisons across 6 pairs.

# %%
pairs = list(combinations(TOPO_ORDER, 2))
n_pairs = len(pairs)
alpha = 0.05
bonferroni_alpha = alpha / n_pairs

mcnemar_results = []

for topo_a, topo_b in pairs:
    ct = mcnemar_table(q_correct[topo_a].values, q_correct[topo_b].values)
    b, c = ct[0, 1], ct[1, 0]  # discordant cells
    chi2, p_raw = mcnemar(ct, exact=True if (b + c) < 25 else False, corrected=True)
    p_adj = min(p_raw * n_pairs, 1.0)  # Bonferroni correction
    sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"

    mcnemar_results.append({
        "Pair": f"{topo_a} vs {topo_b}",
        "A only ✓": b,
        "B only ✓": c,
        "p (raw)": f"{p_raw:.4e}",
        "p (Bonferroni)": f"{p_adj:.4e}",
        "Sig": sig,
    })

mcnemar_df = pd.DataFrame(mcnemar_results)

fig = go.Figure(go.Table(
    header=dict(
        values=list(mcnemar_df.columns),
        fill_color="#3D4854", font=dict(color="white", size=13), align="center",
    ),
    cells=dict(
        values=[mcnemar_df[col] for col in mcnemar_df.columns],
        font=dict(size=12), align="center", height=28,
    ),
))

fig.update_layout(title="Pairwise McNemar's Tests (Bonferroni corrected)", height=350, margin=dict(b=0))
fig.show()
fig.write_image("figures/mcnemar_table.png", width=1000, scale=3)

# %% [markdown]
# All three collaborative topologies significantly outperform independent ($p < 0.001$). Mediator significantly outperforms chain ($p < 0.001$), but the mediator vs full and chain vs full differences, while present, do not survive Bonferroni correction. This suggests the mediator advantage over full is real in direction but would benefit from a larger sample to confirm statistically.

# %% [markdown]
# #### Step 3: Bootstrap Confidence Intervals
#
# We resample the 100 questions with replacement 10,000 times to estimate the uncertainty around each topology's accuracy.

# %%
np.random.seed(42)
N_BOOT = 10_000
n_questions = len(q_correct)

boot_data = {}
for topo in TOPO_ORDER:
    vals = q_correct[topo].values
    boot_accs = np.array([
        vals[np.random.randint(0, n_questions, n_questions)].mean()
        for _ in range(N_BOOT)
    ])
    ci_lo, ci_hi = np.percentile(boot_accs, [2.5, 97.5])
    boot_data[topo] = {"accs": boot_accs, "ci_lo": ci_lo, "ci_hi": ci_hi, "mean": vals.mean()}
    print(f"{topo:12s}: {vals.mean():.0%}  95% CI = [{ci_lo:.0%}, {ci_hi:.0%}]")

# %%
fig = go.Figure()

for topo in TOPO_ORDER:
    d = boot_data[topo]
    fig.add_trace(go.Violin(
        y=d["accs"], name=topo.capitalize(),
        line_color=TOPO_COLORS[topo],
        fillcolor=TOPO_COLORS[topo],
        opacity=0.6, meanline_visible=True,
        box_visible=True, points=False,
    ))

    fig.add_annotation(
        x=topo.capitalize(), y=d["ci_hi"] + 0.02,
        text=f"{d['mean']:.0%}<br>[{d['ci_lo']:.0%}, {d['ci_hi']:.0%}]",
        showarrow=False, font=dict(size=11),
    )

fig.update_layout(
    title="Bootstrap Accuracy Distributions (10,000 resamples)",
    yaxis_title="Accuracy", yaxis_tickformat=".0%",
    showlegend=False, height=500,
)

fig.show()
fig.write_image("figures/bootstrap_confidence_intervals.png", width=1000, scale=3)

# %% [markdown]
# The bootstrap distributions confirm the ranking: mediator > full > chain >> independent, with non-overlapping CIs between mediator and chain/independent. The mediator and full CIs overlap slightly, consistent with the McNemar result — the difference is directionally clear but not definitive at $n=100$.
#
# **References:**
# - Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.* Neural Computation, 10(7), 1895–1923.
# - Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.
# - Raschka, S. (2018). *Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning.* arXiv:1811.12808.
# - Fleiss, J. L., Levin, B., & Paik, M. C. (2003). *Statistical Methods for Rates and Proportions.* Wiley.