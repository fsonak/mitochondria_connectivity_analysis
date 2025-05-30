"""
# Script: analyse_and_visualise_statistics.py

This script loads a CSV summary file containing mitochondrial network metrics
(e.g., node degrees, connectivity components) and generates:
- Box/strip plots for each metric grouped by experimental condition
- Combined pie charts showing the average node type distribution per condition

To use:
1. Ensure a CSV file exists with fields like 'condition', 'dead_end_fraction', etc.
2. Set the correct path to the summary CSV in the __main__ section
3. Run the script to visualize and inspect group-wise mitochondrial connectivity

"""
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

    # Box and strip plots for individual metrics by condition
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        # Boxplot: shows distribution by condition
        sns.boxplot(x="condition", y=metric, data=df, hue="condition", palette="Set2", legend=False)
        # Stripplot: overlays individual data points
        sns.stripplot(x="condition", y=metric, data=df, hue="condition", color=".25", jitter=True, legend=False)
        plt.title(f"{metric.replace('_', ' ').capitalize()} by Condition")
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.xlabel("Condition")
        plt.tight_layout()
        plt.show()

    # Pie chart summarizing average node-type distribution per condition
    conditions = df["condition"].unique()
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 6))

    if len(conditions) == 1:
        axes = [axes]  # Make iterable if only one condition

    for ax, condition in zip(axes, conditions):
        # Compute mean node fractions for this condition
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