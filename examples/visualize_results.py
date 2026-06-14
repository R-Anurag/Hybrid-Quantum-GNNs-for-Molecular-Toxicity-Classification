"""
Generate visualizations from the experiment results CSV.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_model_comparison(results_df, output_dir="../results"):
    """Create comparison plots for different model variants."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    models = results_df["Model"].unique()
    datasets = results_df["Dataset"].unique()
    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, dataset in enumerate(datasets):
        data = results_df[results_df["Dataset"] == dataset]
        means = [data[data["Model"] == model]["auc_mean"].values[0] for model in models]
        stds = [data[data["Model"] == model]["auc_std"].values[0] for model in models]
        offset = width * (i - 0.5)
        ax.bar([xi + offset for xi in x], means, width, yerr=stds,
               label=dataset, alpha=0.8, capsize=5)

    ax.set_ylabel("ROC-AUC Score", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_auc_comparison.png"), dpi=300)
    print(f"Saved: {output_dir}/roc_auc_comparison.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, dataset in enumerate(datasets):
        data = results_df[results_df["Dataset"] == dataset]
        times = [data[data["Model"] == model]["time_mean"].values[0] for model in models]
        offset = width * (i - 0.5)
        ax.bar([xi + offset for xi in x], times, width, label=dataset, alpha=0.8)

    ax.set_ylabel("Training Time per Epoch (s)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Computational Cost Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_time_comparison.png"), dpi=300)
    print(f"Saved: {output_dir}/training_time_comparison.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset in datasets:
        data = results_df[results_df["Dataset"] == dataset]
        ax.scatter(data["Params"], data["auc_mean"], s=100, alpha=0.7, label=dataset)
        for _, row in data.iterrows():
            ax.annotate(
                row["Model"],
                (row["Params"], row["auc_mean"]),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.set_xlabel("Number of Parameters", fontsize=12)
    ax.set_ylabel("ROC-AUC Score", fontsize=12)
    ax.set_title("Model Efficiency: Performance vs Complexity", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_vs_params.png"), dpi=300)
    print(f"Saved: {output_dir}/performance_vs_params.png")
    plt.close()


def print_summary_table(results_df):
    """Print a compact results table."""
    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)

    for dataset in results_df["Dataset"].unique():
        print(f"\n{dataset.upper()}")
        print("-" * 80)
        data = results_df[results_df["Dataset"] == dataset]
        print(f"{'Model':<25} {'ROC-AUC':<20} {'Time/Epoch':<15}")
        print("-" * 80)

        for _, row in data.iterrows():
            auc_str = f"{row['auc_mean']:.4f} +/- {row['auc_std']:.4f}"
            time_str = f"{row['time_mean']:.2f}s"
            print(f"{row['Model']:<25} {auc_str:<20} {time_str:<15}")

    print("\n" + "=" * 80)


def main():
    results_file = "../results/results.csv"
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        print("Run experiments first: python src/run_experiments.py")
        return

    print("Loading results...")
    results_df = pd.read_csv(results_file)
    print_summary_table(results_df)

    print("\nGenerating visualizations...")
    plot_model_comparison(results_df)
    print("\nDone. Check the results/ directory for plots.")


if __name__ == "__main__":
    main()
