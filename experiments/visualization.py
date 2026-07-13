import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/student_pass.csv")

# ===================================
# Dashboard Layout
# ===================================

fig = plt.figure(figsize=(16, 12))

fig.canvas.manager.set_window_title(
    "Logistic Regression Dashboard"
)

gs = fig.add_gridspec(
    3,
    2,
    height_ratios=[1,1,1.3],
    width_ratios=[1,1],
    hspace=0.60,
    wspace=0.30
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

ax5 = fig.add_subplot(gs[2, :])

# ===================================
# Pass vs Fail Distribution
# ===================================

df["Pass"].value_counts().plot(
    kind="bar",
    color=["red", "green"],
    ax=ax1
)

ax1.set_title(
    "Pass vs Fail Distribution",
    fontsize=14,
    pad=0
)

ax1.set_xlabel(
    "Class",
    labelpad=0
)

ax1.set_ylabel("Count")

# ===================================
# Correlation Matrix
# ===================================

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    ax=ax2
)
plt.setp(
    ax2.get_xticklabels(),
    rotation=15,
    ha="right"
)
ax2.set_title(
    "Correlation Matrix",
    fontsize=14,
    pad=0
)

# ===================================
# Hours Distribution by Pass
# ===================================

sns.boxplot(
    x="Pass",
    y="Hours",
    data=df,
    ax=ax3
)

ax3.set_title(
    "Hours Distribution by Pass",
    fontsize=14,
    pad=10
)

ax3.set_xlabel(
    "Pass",
    labelpad=2
)

ax3.set_ylabel("Hours Studied")

# ===================================
# Pass Rate vs Hours
# ===================================

pass_rate = df.groupby("Hours")["Pass"].mean()

ax4.plot(
    pass_rate.index,
    pass_rate.values,
    marker="o",
    linewidth=2
)

ax4.set_title(
    "Pass Rate vs Hours",
    fontsize=14,
    pad=10
)

ax4.set_xlabel(
    "Hours Studied",
    labelpad=2
)

ax4.set_ylabel("Pass Rate")

ax4.grid(True)

# ===================================
# Training Loss Curve
# ===================================
# Load stored loss values to visualize training performance.
with open(
    "experiments/losses.json"
    
 ) as file:
    losses = json.load(file)

ax5.plot(
    range(1, len(losses) + 1),
    losses,
    linewidth=3,
    color="purple"
)

ax5.set_title(
    "Training Loss Over Epochs",
    fontsize=15,
    pad=5
)

ax5.set_xlabel(
    "Epoch",
    fontsize=12,
    labelpad=2
)

ax5.set_ylabel(
    "Loss",
    fontsize=12
)

ax5.grid(True)

# ===================================
# Main Title
# ===================================

fig.suptitle(
    "Student Performance Analysis Dashboard",
    fontsize=22,
    fontweight="bold",
    y= 1
)

fig.subplots_adjust(
    top=0.90,
    bottom=0.08,
    hspace=0.55,
    wspace=0.20
)

plt.show()