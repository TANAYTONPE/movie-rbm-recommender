"""
Visualization utilities for Restricted Boltzmann Machine recommender.
Generates Matplotlib charts for training loss, weight matrix heatmaps,
Gibbs sampling schematics, evaluation metrics, and architectural diagrams.
"""

import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Modern dark theme palette matching Streamlit dashboard
BG_DARK = "#0F172A"
BG_CARD = "#1E293B"
TEXT_COLOR = "#F8FAFC"
ACCENT_BLUE = "#38BDF8"
ACCENT_PURPLE = "#A855F7"
ACCENT_CYAN = "#06B6D4"
ACCENT_PINK = "#EC4899"
GRID_COLOR = "#334155"


def _apply_dark_style(fig, ax):
    """Applies clean dark theme styling to matplotlib axes."""
    fig.patch.set_facecolor(BG_DARK)
    if isinstance(ax, np.ndarray):
        for a in ax.flat:
            _apply_single_axis_dark(a)
    else:
        _apply_single_axis_dark(ax)


def _apply_single_axis_dark(ax):
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle="--", alpha=0.4)


def plot_training_history(history: Dict) -> plt.Figure:
    """
    Plots Train Reconstruction Loss and Test RMSE across training epochs.
    """
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    test_rmse = history.get("test_rmse", [])

    fig, ax1 = plt.subplots(figsize=(8, 4.2), dpi=120)
    _apply_dark_style(fig, ax1)

    line1 = ax1.plot(
        epochs, train_loss, color=ACCENT_CYAN, marker="o",
        linewidth=2.2, label="Train Loss (Reconstruction MSE)"
    )
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Train Loss (MSE)", color=ACCENT_CYAN, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=ACCENT_CYAN)

    # Second axis for Test RMSE
    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    line2 = ax2.plot(
        epochs, test_rmse, color=ACCENT_PURPLE, marker="s",
        linewidth=2.2, linestyle="--", label="Test RMSE"
    )
    ax2.set_ylabel("Held-Out Test RMSE", color=ACCENT_PURPLE, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=ACCENT_PURPLE)
    for spine in ax2.spines.values():
        spine.set_color(GRID_COLOR)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", facecolor=BG_DARK, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax1.set_title("RBM Training Convergence: Loss & Generalization Error", fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    return fig


def plot_weight_matrix_heatmap(
    W_tensor: torch.Tensor,
    df_movies: pd.DataFrame,
    num_visible_show: int = 20,
    num_hidden_show: int = 20
) -> plt.Figure:
    """
    Plots a 2D heatmap showing learned connection weights between movies and latent preference neurons.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    _apply_dark_style(fig, ax)

    # Extract weights for top popular movies
    W_sub = W_tensor[:num_visible_show, :num_hidden_show].detach().cpu().numpy()

    # Movie labels
    movie_labels = [
        df_movies.iloc[i]["clean_title"][:22]
        if i < len(df_movies) else f"Movie {i+1}"
        for i in range(num_visible_show)
    ]
    hidden_labels = [f"h_{j+1}" for j in range(num_hidden_show)]

    im = ax.imshow(W_sub, cmap="magma", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Connection Weight (W_ij)", color=TEXT_COLOR, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_COLOR)

    ax.set_xticks(range(num_hidden_show))
    ax.set_xticklabels(hidden_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(num_visible_show))
    ax.set_yticklabels(movie_labels, fontsize=8)

    ax.set_xlabel("Latent Hidden Features (Hidden Units)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Visible Movie Units", fontsize=10, fontweight="bold")
    ax.set_title("RBM Learned Weight Matrix Subsample (W)", fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    return fig


def plot_evaluation_metrics_bar(ranking_dict: Dict[str, Dict[str, float]]) -> plt.Figure:
    """
    Plots Top-K Ranking performance across K=5, 10, 15, 20.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=120)
    _apply_dark_style(fig, ax)

    k_labels = list(ranking_dict.keys())
    precisions = [ranking_dict[k]["Precision@K"] * 100 for k in k_labels]
    recalls = [ranking_dict[k]["Recall@K"] * 100 for k in k_labels]
    hit_rates = [ranking_dict[k]["Hit_Rate@K"] * 100 for k in k_labels]

    x = np.arange(len(k_labels))
    width = 0.25

    rects1 = ax.bar(x - width, precisions, width, label="Precision@K (%)", color=ACCENT_CYAN, alpha=0.9)
    rects2 = ax.bar(x, recalls, width, label="Recall@K (%)", color=ACCENT_PURPLE, alpha=0.9)
    rects3 = ax.bar(x + width, hit_rates, width, label="Hit Rate@K (%)", color=ACCENT_PINK, alpha=0.9)

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Test Set Recommendation Ranking Performance Across K", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(k_labels, fontsize=10, fontweight="bold")
    ax.legend(facecolor=BG_DARK, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Attach values above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, color=TEXT_COLOR, fontweight="bold"
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    return fig


def generate_architecture_diagram(save_path: Optional[str] = None) -> plt.Figure:
    """
    Generates a high-quality academic architecture diagram of the Restricted Boltzmann Machine.
    Saves to save_path (e.g. assets/architecture.png) if specified.
    """
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=150)
    _apply_dark_style(fig, ax)
    ax.axis("off")

    # Positions
    visible_y = 1.0
    hidden_y = 4.0

    v_coords = [(1.5 + i * 1.0, visible_y) for i in range(8)]
    h_coords = [(2.0 + j * 1.25, hidden_y) for j in range(6)]

    # Draw bipartite connections
    for vx, vy in v_coords:
        for hx, hy in h_coords:
            ax.plot([vx, hx], [vy, hy], color="#475569", alpha=0.25, linewidth=0.9, zorder=1)

    # Highlight a few connections
    for i in [1, 3, 5]:
        for j in [1, 3]:
            ax.plot(
                [v_coords[i][0], h_coords[j][0]],
                [v_coords[i][1], h_coords[j][1]],
                color=ACCENT_CYAN, alpha=0.6, linewidth=1.4, zorder=2
            )

    # Draw Hidden units
    for idx, (hx, hy) in enumerate(h_coords):
        circle = plt.Circle((hx, hy), 0.32, color=ACCENT_PURPLE, ec="#C084FC", linewidth=2.0, zorder=4)
        ax.add_patch(circle)
        label = f"$h_{{{idx+1}}}$" if idx < 5 else "$h_{M}$"
        ax.text(hx, hy, label, ha="center", va="center", color="#FFFFFF", fontsize=10, fontweight="bold", zorder=5)

    # Draw Visible units
    for idx, (vx, vy) in enumerate(v_coords):
        circle = plt.Circle((vx, vy), 0.32, color=ACCENT_BLUE, ec="#7DD3FC", linewidth=2.0, zorder=4)
        ax.add_patch(circle)
        label = f"$v_{{{idx+1}}}$" if idx < 7 else "$v_{N}$"
        ax.text(vx, vy, label, ha="center", va="center", color="#FFFFFF", fontsize=10, fontweight="bold", zorder=5)

    # Labels and annotations
    ax.text(
        5.0, hidden_y + 0.8,
        "HIDDEN LAYER: Latent User Preferences (M = 100 Neurons)\nStochastic Binary: P(h_j = 1 | v) = σ(b_j + Σ v_i W_ij)",
        ha="center", va="center", color=ACCENT_PURPLE, fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_DARK, edgecolor=ACCENT_PURPLE, alpha=0.9)
    )

    ax.text(
        5.0, visible_y - 0.8,
        "VISIBLE LAYER: Movie Ratings (N = 1,682 Movies)\nReconstruction: P(v_i = 1 | h) = σ(a_i + Σ h_j W_ij)",
        ha="center", va="center", color=ACCENT_BLUE, fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_DARK, edgecolor=ACCENT_BLUE, alpha=0.9)
    )

    ax.text(
        9.2, 2.5,
        "Bipartite Graph\n(NO Intra-layer\nconnections!)\n\nWeights: W ∈ ℝ^(N×M)\nBiases: a ∈ ℝ^N, b ∈ ℝ^M",
        ha="left", va="center", color=TEXT_COLOR, fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_DARK, edgecolor=GRID_COLOR, alpha=0.9)
    )

    # Gibbs arrows
    ax.annotate(
        "Positive Phase:\nP(h|v)", xy=(1.1, 3.2), xytext=(1.1, 1.8),
        arrowprops=dict(arrowstyle="->", color=ACCENT_CYAN, lw=2.0),
        color=ACCENT_CYAN, fontsize=8.5, fontweight="bold", ha="center"
    )
    ax.annotate(
        "Negative Phase:\nP(v|h) [Gibbs]", xy=(0.3, 1.8), xytext=(0.3, 3.2),
        arrowprops=dict(arrowstyle="->", color=ACCENT_PINK, lw=2.0),
        color=ACCENT_PINK, fontsize=8.5, fontweight="bold", ha="center"
    )

    ax.set_xlim(-0.2, 11.5)
    ax.set_ylim(-0.3, 5.3)
    ax.set_title(
        "Restricted Boltzmann Machine (RBM) Architecture for Movie Recommendations",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=16
    )

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, facecolor=BG_DARK, edgecolor="none", bbox_inches="tight")
        print(f"Architecture diagram saved to: {save_path}")

    return fig

