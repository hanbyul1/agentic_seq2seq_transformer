import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Publication settings
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

# ============================================================
# Data
# ============================================================

SWE_AC = {
    "Agentic": 0.055301,
    "MoE": 1.000000,
}

SWE_TRAINABLE = {
    "Agentic": 592768,
    "MoE": 12693508,
}

SWE_CQ = {
    "Agentic": 0.940,
    "MoE": 0.390,
}

SWE_COMPONENTS = {
    "Syntax": [1.00, 0.00],
    "Structure": [0.80, 1.00],
    "Uniqueness": [1.00, 0.90],
}

HE_AC = {
    "Agentic": 0.065792,
    "MoE": 1.000000,
}

HE_TRAINABLE = {
    "Agentic": 1664613,
    "MoE": 23409725,
}

HE_CQ = {
    "Agentic": 0.700,
    "MoE": 0.140,
}

HE_COMPONENTS = {
    "Syntax": [1.00, 0.10],
    "Structure": [0.00, 0.00],
    "Uniqueness": [1.00, 0.80],
}

# ============================================================
# Adaptation Cost Figure
# ============================================================

def plot_adaptation_cost(
    title,
    ac_data,
    trainable_data,
    outfile
):

    fig = plt.figure(figsize=(12, 5))

    # --------------------------------------------------------
    # Left panel: Component metric
    # --------------------------------------------------------

    ax_comp = fig.add_axes([
        0.07,
        0.22,
        0.38,
        0.60
    ])

    labels = list(trainable_data.keys())
    values = np.array(list(trainable_data.values()))

    y = np.arange(len(labels))

    ax_comp.hlines(
        y,
        xmin=0,
        xmax=values,
        linewidth=3
    )

    ax_comp.scatter(
        values,
        y,
        s=150,
        zorder=3
    )

    ax_comp.set_xscale("log")

    ax_comp.set_yticks(y)
    ax_comp.set_yticklabels(labels)

    ax_comp.set_title("Trainable Parameters")

    ax_comp.set_xlabel("Parameter Count (log scale)")

    for yi, val in zip(y, values):

        ax_comp.text(
            val,
            yi + 0.05,
            f"{val/1e6:.2f}M",
            fontsize=9
        )

    # --------------------------------------------------------
    # Right panel: Main metric
    # --------------------------------------------------------

    ax = fig.add_axes([
        0.57,
        0.22,
        0.35,
        0.60
    ])

    labels = list(ac_data.keys())
    values = list(ac_data.values())

    bars = ax.bar(
        labels,
        values,
        width=0.55
    )

    ax.set_ylim(0, 1.10)

    ax.set_title(title)

    ax.set_ylabel("Adaptation Cost (AC)")

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=4,
        fontsize=12
    )

    # --------------------------------------------------------
    # Subfigure labels
    # --------------------------------------------------------

    fig.text(
        0.26,
        0.08,
        "(a) Component Metrics",
        ha="center",
        fontsize=12
    )

    fig.text(
        0.745,
        0.08,
        "(b) Adaptation Cost",
        ha="center",
        fontsize=12
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# Code Quality Figure
# ============================================================

def plot_code_quality(
    title,
    cq_data,
    component_data,
    outfile
):

    fig = plt.figure(figsize=(12, 5))

    # --------------------------------------------------------
    # Left panel: Component metrics
    # --------------------------------------------------------

    ax_comp = fig.add_axes([
        0.07,
        0.22,
        0.38,
        0.60
    ])

    metrics = list(component_data.keys())

    agentic = [component_data[m][0] for m in metrics]
    moe = [component_data[m][1] for m in metrics]

    x = np.arange(len(metrics))

    ax_comp.plot(
        x,
        agentic,
        marker="o",
        linewidth=2,
        label="Agentic"
    )

    ax_comp.plot(
        x,
        moe,
        marker="s",
        linewidth=2,
        label="MoE"
    )

    ax_comp.set_ylim(0, 1.05)

    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(metrics)

    ax_comp.set_ylabel("Score")

    ax_comp.set_title("CQ Components")

    ax_comp.legend()

    # --------------------------------------------------------
    # Right panel: Main metric
    # --------------------------------------------------------

    ax = fig.add_axes([
        0.57,
        0.22,
        0.35,
        0.60
    ])

    labels = list(cq_data.keys())
    values = list(cq_data.values())

    bars = ax.bar(
        labels,
        values,
        width=0.55
    )

    ax.set_ylim(0, 1.05)

    ax.set_title(title)

    ax.set_ylabel("Code Quality (CQ)")

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=4,
        fontsize=12
    )

    # --------------------------------------------------------
    # Subfigure labels
    # --------------------------------------------------------

    fig.text(
        0.26,
        0.08,
        "(a) Component Metrics",
        ha="center",
        fontsize=12
    )

    fig.text(
        0.745,
        0.08,
        "(b) Code Quality",
        ha="center",
        fontsize=12
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# Generate Figures
# ============================================================

plot_adaptation_cost(
    "SWE-bench Adaptation Cost",
    SWE_AC,
    SWE_TRAINABLE,
    "swe_adaptation_cost.png"
)

plot_code_quality(
    "SWE-bench Code Quality",
    SWE_CQ,
    SWE_COMPONENTS,
    "swe_code_quality.png"
)

plot_adaptation_cost(
    "HumanEval Adaptation Cost",
    HE_AC,
    HE_TRAINABLE,
    "humaneval_adaptation_cost.png"
)

plot_code_quality(
    "HumanEval Code Quality",
    HE_CQ,
    HE_COMPONENTS,
    "humaneval_code_quality.png"
)

print()
print("Saved:")
print("  swe_adaptation_cost.png")
print("  swe_code_quality.png")
print("  humaneval_adaptation_cost.png")
print("  humaneval_code_quality.png")