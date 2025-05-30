import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_summary(csv_path):
    return pd.read_csv(csv_path)

def plot_summary(df, output_dir):
    metrics = [
        "total_nodes",
        "total_edges",
        "connected_components",
        "largest_component_pct",
        "dead_end_fraction",
        "three_way_fraction",
        "degree_two_fraction"
    ]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="condition", y=metric, data=df, hue="condition", palette="Set2", legend=False)
        sns.stripplot(x="condition", y=metric, data=df, hue="condition", color=".25", jitter=True, legend=False)
        plt.title(f"{metric.replace('_', ' ').capitalize()} by Condition")
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.xlabel("Condition")
        plt.tight_layout()
        plt.show()

    # Pie charts for node type proportions per condition
    conditions = df["condition"].unique()
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 6))

    if len(conditions) == 1:
        axes = [axes]  # Make iterable if only one condition

    for ax, condition in zip(axes, conditions):
        group = df[df["condition"] == condition]
        mean_dead_end = group["dead_end_fraction"].mean()
        mean_degree_two = group["degree_two_fraction"].mean()
        mean_branch = group["three_way_fraction"].mean()

        wedges, texts, autotexts = ax.pie(
            [mean_dead_end, mean_degree_two, mean_branch],
            labels=["Dead-Ends (Deg 1)", "Linear (Deg 2)", "Branch Points (Deg ≥3)"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.set_title(f"{condition}")

    plt.suptitle("Average Node Degree Distribution per Condition", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    summary_csv = Path("/Users/frederic/Nextcloud/MD_MSC_Project/Jana_Mitograph/Output_summary/summary.csv")
    output_dir = summary_csv.parent

    df = load_summary(summary_csv)
    plot_summary(df, output_dir)