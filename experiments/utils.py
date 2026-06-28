import torch
from torchvision.transforms import v2
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).parents[1]
OUTPUT_DIR = str(ROOT / "images")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def _configure_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.edgecolor": "#243244",
            "axes.linewidth": 0.9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )

def get_transform() -> v2.Compose:
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(48, 48)),
        v2.Grayscale(num_output_channels=1),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return transform

def export_confusion_matrix(y_true, y_pred, class_names, dataset):
    _configure_plot_style()
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    cmap = sns.light_palette("#1f4e79", as_cmap=True)
    heatmap = sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0,
        vmax=1,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        linewidths=0.6,
        linecolor="#edf2f7",
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_title(f"Confusion Matrix - {dataset}", pad=18)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    heatmap.collections[0].colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/confusion_matrix_{dataset}.png", dpi=300)
    plt.close(fig)
    print(f"Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix_{dataset}.png")

def export_classification_report(y_true, y_pred, class_names, dataset):
    _configure_plot_style()
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    report_frame = df_report.iloc[:-1, :-1]

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    cmap = sns.light_palette("#2a9d8f", as_cmap=True)
    heatmap = sns.heatmap(
        report_frame,
        annot=True,
        cmap=cmap,
        vmin=0,
        vmax=1,
        fmt=".2f",
        linewidths=0.6,
        linecolor="#edf2f7",
        cbar=True,
        square=False,
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax.set_title(f"Classification Report - {dataset}", pad=18)
    ax.set_ylabel("Metrics")
    ax.set_xlabel("Classes")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    heatmap.collections[0].colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/classification_report_{dataset}.png", dpi=300)
    plt.close(fig)
    print(f"Classification report saved to {OUTPUT_DIR}/classification_report_{dataset}.png")