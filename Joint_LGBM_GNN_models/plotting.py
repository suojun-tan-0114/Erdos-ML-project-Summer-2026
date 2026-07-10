"""Plotting helpers for experiment diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt

def plot_history_metric(
    df,
    metric_col,
    ylabel,
    title,
    filename,
    output_dir,
):
    plt.figure(figsize=(10, 6))

    for experiment_name, group in df.groupby(
        "experiment",
        sort=False,
    ):
        group = group.sort_values("epoch")

        plt.plot(
            group["epoch"],
            group[metric_col],
            label=experiment_name,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = Path(output_dir) / filename
    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight",
    )
    print("saved plot:", plot_path)
    plt.show()
