import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_summary(csv_path):
    return pd.read_csv(csv_path)

def plot_summary(df, output_dir):
    metrics = ["nodes", "edges", "components", "largest_component_pct"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="condition", y=metric, data=df, hue="condition", palette="Set2", legend=False)
        sns.stripplot(x="condition", y=metric, data=df, hue="condition", color=".25", jitter=True, legend=False)
        plt.title(f"{metric.replace('_', ' ').capitalize()} by Condition")
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.xlabel("Condition")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    summary_csv = Path("/Users/frederic/Nextcloud/MD_MSC_Project/Jana_Mitograph/Output_summary/summary.csv")
    output_dir = summary_csv.parent

    df = load_summary(summary_csv)
    plot_summary(df, output_dir)