#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:05:32 2025
main_spectral_only.py - Runs visualization and spectral analysis using preprocessed data
Created on Fri Jun 21 2025
@author: user
"""
import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
import os

import numpy as np
import pandas as pd

from SingleCellDataAnalysis.config import WORKING_DIR, FRAME_NUMBER
from SingleCellDataAnalysis.visualization import plot_aligned_signals, plot_aligned_heatmaps
from SingleCellDataAnalysis.spectral_analysis import (
    fft_dominant_frequencies, plot_fft_dominant_frequencies,
    gp_infer_periods, plot_gp_periods, plot_multi_periodic_gp_grid
)
from SingleCellDataAnalysis.load_data import load_preprocessed_data
from SingleCellDataAnalysis.spectral_analysis import quantify_gp_features
from SingleCellDataAnalysis.simple_shape_analysis import (
    plot_simple_model_grid, simulate_stepwise_cells, quantify_all_cells,
    summarize_model_distribution, build_model_category_heatmap_df, 
    pivot_heatmap_matrix,plot_model_heatmap,extract_oscillation_data,
    plot_oscillation_scatter,get_median_global_times,extract_frequency_with_time,
    plot_frequency_timeline,plot_amplitude_timeline, extract_slope_with_time, plot_slope_timeline,filter_slope_data,
    fit_slope_vs_time,plot_slope_with_regression, prepare_pol1_pol2_slope_with_time,plot_pol1_vs_pol2_with_lines,
    plot_first_time_distribution,compute_correlation_corrected,plot_correlation_timeline,
    cluster_cells_from_model_params,plot_amplitude_distributions,cluster_cells_by_amplitude_and_delay
    
)
df_all, time_points, global_time, cell_ids, best_shifts = load_preprocessed_data(WORKING_DIR)
#%%

# ---- Step 3: Visualization ----
# print("🧯 Creating heatmaps for polar features...")
# heatmap_features = ['pol1_int_corr', 'pol2_int_corr', 'pol1_minus_pol2']
# plot_aligned_heatmaps(df_all, cell_ids, best_shifts, global_time, time_points, heatmap_features)

# # ---- Step 4: Spectral Analysis ----
# print("🔬 Running FFT analysis...")
# fft_results = fft_dominant_frequencies(df_all, cell_ids, best_shifts, global_time, time_points, features=['pol1_int_corr', 'pol2_int_corr'])
# plot_fft_dominant_frequencies(fft_results)

# print("🌊 Running GP period inference...")
# gp_results = gp_infer_periods(df_all, cell_ids, best_shifts, global_time, time_points, features=['pol1_int_corr', 'pol2_int_corr'])
# plot_gp_periods(gp_results)

# ---- Step 5: GP Multi-Period Grid Plot ----
#print("📈 Plotting multi-periodic GP fit for 25 cells...")
#plot_multi_periodic_gp_grid(df_all, cell_ids, time_points)



# result = quantify_gp_features(df_all, cell_ids, time_points)
# result.head()


# result.to_csv(os.path.join(WORKING_DIR, "gp_summary_features.csv"), index=False)

# ---- Try simple shapes for single cell trejectories ----

#plot_simple_model_grid(df_all, cell_ids, time_points, model_type='step', start_idx=26)
plot_simple_model_grid(df_all, cell_ids, time_points, model_type='linear', start_idx=0)
#plot_simple_model_grid(df_all, cell_ids, time_points, model_type='constant', start_idx=26)
#plot_simple_model_grid(df_all, cell_ids, time_points, model_type='step', start_idx=51)
#plot_simple_model_grid(df_all, cell_ids, time_points, model_type='step', start_idx=76)


# df_sim = simulate_stepwise_cells()
# cell_ids = df_sim['cell_id'].unique()

# plot_simple_model_grid(df_sim, cell_ids=cell_ids, time_points=np.arange(51), model_type='step')
#%%
df_results = quantify_all_cells(df_all, cell_ids,delta_threshold=10)
summary_table = summarize_model_distribution(df_results)
print(summary_table)
#%%
#df_results = pd.read_csv("model_fits_by_cell.csv")
clustered_df = cluster_cells_from_model_params(df_results, n_harmonics=10)
plot_amplitude_distributions(clustered_df)#, n_harmonics=10)
#%%
df_norm,ordered_cell_ids, row_linkage = cluster_cells_by_amplitude_and_delay(df_results)
#%%
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=0)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=25)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=50)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=75)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=100)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=125)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=150)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=175)
#%%
# Make sure cell_id is the index of df_norm
aligned_times = df_all.groupby('cell_id')['aligned_time'].min()
aligned_times = aligned_times[df_norm.index]  # Filter to clustered cells only
from scipy.cluster.hierarchy import to_tree
from scipy.cluster.hierarchy import dendrogram

tree_root, nodes = to_tree(row_linkage, rd=True)

def annotate_tree_with_aligned_time(node, aligned_time_series):
    if node.is_leaf():
        cell_id = df_norm.index[node.id]
        value = aligned_time_series[cell_id]
        node.aligned_times = [value]
        node.label = f"{value:.2f}"#f"{cell_id}:{value:.2f}"
    else:
        annotate_tree_with_aligned_time(node.left, aligned_time_series)
        annotate_tree_with_aligned_time(node.right, aligned_time_series)
        node.aligned_times = node.left.aligned_times + node.right.aligned_times
        mu = np.mean(node.aligned_times)
        sigma = np.std(node.aligned_times)
        node.label = f"{mu:.2f} ± {1.96*sigma:.2f}"

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


import matplotlib.cm as cm
from matplotlib.colors import Normalize

def plot_tree_with_annotations(tree_root, y_offset=0, x_offset=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 60))

    # Flatten the tree to get all values for color normalization
    all_means = []

    def collect_means(node):
        if hasattr(node, 'aligned_times') and node.aligned_times:
            mu = np.mean(node.aligned_times)
            all_means.append(mu)
        if not node.is_leaf():
            collect_means(node.left)
            collect_means(node.right)

    collect_means(tree_root)

    # Normalize for colormap
    norm = Normalize(vmin=min(all_means), vmax=max(all_means))
    cmap = cm.get_cmap('viridis')  # You can change to 'plasma', 'coolwarm', etc.

    def draw_node(node, x):
        nonlocal y_offset

        if node.is_leaf():
            y = y_offset
            y_offset += 1
            value = np.mean(node.aligned_times) if hasattr(node, 'aligned_times') else 0
            color = cmap(norm(value))
            ax.text(x, y, node.label, va='center', ha='left', fontsize=20, color=color)
        else:
            y_left = draw_node(node.left, x + node.dist)
            y_right = draw_node(node.right, x + node.dist)
            y = (y_left + y_right) / 2
            ax.plot([x + node.dist, x + node.dist], [y_left, y_right], 'k-')
            ax.plot([x, x + node.dist], [y, y], 'k-')
            value = np.mean(node.aligned_times) if hasattr(node, 'aligned_times') else 0
            color = cmap(norm(value))
            ax.text(x, y, node.label, va='center', ha='right', fontsize=20, color=color)
        return y

    draw_node(tree_root, x_offset)
    ax.axis('off')
    plt.title("Clustering Tree Annotated with Aligned Time Stats (Colored by Mean)", fontsize=18)
    plt.tight_layout()
    plt.show()


annotate_tree_with_aligned_time(tree_root, aligned_times)
plot_tree_with_annotations(tree_root)

#%% plot heatmap
# Reuse helper
heatmap_df = build_model_category_heatmap_df(df_all, df_results, best_shifts, time_bin_width=20)
heatmap_matrix = pivot_heatmap_matrix(heatmap_df)
plot_model_heatmap(heatmap_matrix)


#%% plot osciltion par

df_osc = extract_oscillation_data(df_results)
plot_oscillation_scatter(df_osc)

# Step 1: build global time reference per cell
median_global_time_dict = get_median_global_times(df_all, best_shifts)

# Step 2: extract freq + time per pol
df_freq_time = extract_frequency_with_time(df_results, median_global_time_dict)

# Step 3: plot
plot_frequency_timeline(df_freq_time)

plot_amplitude_timeline(df_freq_time)
#%% plot slop

df_slope_time = extract_slope_with_time(df_results, df_all, best_shifts)
plot_slope_timeline(df_slope_time)



df_filtered = filter_slope_data(df_slope_time, time_min=50, time_max=225)
reg = fit_slope_vs_time(df_filtered)
print(f"Regression Coef: {reg.coef_[0]:.4f}, Intercept: {reg.intercept_:.4f}, R^2: {reg.score(df_filtered['global_time'].values.reshape(-1, 1), df_filtered['slope'].values):.4f}")
plot_slope_with_regression(df_filtered, reg)


plot_first_time_distribution(df_slope_time, pol1="pol1", pol2="pol2")

#%%
df_pivoted = prepare_pol1_pol2_slope_with_time(df_slope_time, df_all, best_shifts)
plot_pol1_vs_pol2_with_lines(df_pivoted)


#%% Covariance

df_corr = compute_correlation_corrected(df_all)
plot_correlation_timeline(df_corr)

print("✅ Spectral analysis complete.")
