#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:53:23 2025

@author: user
"""

# visualization.py

import matplotlib.pyplot as plt
import numpy as np

def plot_aligned_signals(df_all, cell_ids, best_shifts, global_time, time_points, features, mean_trace=None, title_prefix="Aligned"):
    """
    Plots each feature over global time for aligned cells.
    Optionally includes mean trace.
    """
    n_features = len(features)
    fig, axs = plt.subplots(nrows=n_features, figsize=(10, 2.5 * n_features), sharex=True)

    for i, feature in enumerate(features):
        ax = axs[i] if n_features > 1 else axs
        for cid in cell_ids:
            df_cell = df_all[df_all["cell_id"] == cid].sort_values("time_point")
            values = df_cell.set_index("time_point").reindex(time_points)[feature].values
            aligned = np.full_like(global_time, np.nan, dtype=np.float64)
            shift = best_shifts[cid]
            aligned[shift:shift+len(time_points)] = values
            ax.plot(global_time, aligned, alpha=0.3)

        if mean_trace is not None:
            ax.plot(global_time, mean_trace[i], color='black', linewidth=2, label='Mean')

        ax.set_title(f"{title_prefix}: {feature}")
        ax.set_ylabel("Value")
        ax.grid(True)

    axs[-1].set_xlabel("Global Master Timeline")
    plt.tight_layout()
    plt.show()
    
import seaborn as sns

def plot_aligned_heatmaps(df_all, cell_ids, best_shifts, global_time, time_points, features, cmap_list=None, title="Aligned Heatmaps"):
    """
    Create a row of heatmaps, one for each feature, aligned over global time.
    """
    fig, axs = plt.subplots(len(features), 1, figsize=(12, len(features)*2.5), sharex=True)

    ordered_cells = sorted(cell_ids, key=lambda cid: best_shifts[cid])

    for i, feature in enumerate(features):
        heatmap_data = []
        for cid in ordered_cells:
            df_cell = df_all[df_all["cell_id"] == cid].sort_values("time_point")
            values = df_cell.set_index("time_point").reindex(time_points)[feature].values
            aligned = np.full_like(global_time, np.nan, dtype=np.float64)
            shift = best_shifts[cid]
            aligned[shift:shift+len(time_points)] = values
            heatmap_data.append(aligned)

        heatmap_array = np.array(heatmap_data)
        cmap = cmap_list[i] if cmap_list else "viridis"

        sns.heatmap(
            heatmap_array,
            ax=axs[i],
            cmap=cmap,
            xticklabels=False,
            yticklabels=False,
            cbar=True
        )
        axs[i].set_title(f"{feature}")
        axs[i].set_ylabel("Cells")

    axs[-1].set_xlabel("Global Master Timeline")
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
