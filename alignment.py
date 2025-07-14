#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:51:58 2025

@author: user
"""

# alignment.py

import numpy as np
import random
import pandas as pd

def prepare_signals(df_all, features, time_points):
    """
    Returns a dictionary {cell_id: signal matrix} with features × time for each cell.
    """
    cell_ids = df_all['cell_id'].unique()
    signal_dict = {}

    for cid in cell_ids:
        df_cell = df_all[df_all["cell_id"] == cid].sort_values("time_point")
        matrix = np.array([
            df_cell.set_index("time_point").reindex(time_points)[f].values
            for f in features
        ])
        signal_dict[cid] = matrix

    return signal_dict, cell_ids


def compute_mse(shifts, cell_signals, global_time, T, lambda_reg=0.0):
    """
    Compute total MSE after aligning all cells according to 'shifts'.
    Returns total cost and the global average signal.
    """
    n_features = next(iter(cell_signals.values())).shape[0]
    acc = np.zeros((n_features, len(global_time)))
    weights = np.zeros_like(acc)

    for cid, signal in cell_signals.items():
        shift = shifts[cid]
        valid = ~np.isnan(signal)
        acc[:, shift:shift+T] += np.where(valid, signal, 0)
        weights[:, shift:shift+T] += valid

    avg = np.divide(acc, weights, where=weights != 0)

    total_mse = 0
    for cid, signal in cell_signals.items():
        shift = shifts[cid]
        valid = ~np.isnan(signal) & (weights[:, shift:shift+T] > 0)
        diff = signal - avg[:, shift:shift+T]
        mse = np.mean((diff[valid]) ** 2)
        total_mse += mse

    return total_mse, avg


def initialize_shifts(cell_signals, time_points, global_time):
    """
    Assign even-spaced initial shifts based on sorted feature means.
    """
    T = len(time_points)
    shift_range = len(global_time) - T

    # Rank cells by average size
    means = {
        cid: np.nanmean(signal[1])  # feature 1 (e.g. weighted_area)
        for cid, signal in cell_signals.items()
    }

    sorted_cells = sorted(means, key=means.get)
    shift_values = np.linspace(0, shift_range, len(sorted_cells)).astype(int)

    shifts = {cid: shift for cid, shift in zip(sorted_cells, shift_values)}
    return shifts, shift_range


def run_mcmc(cell_signals, global_time, time_points, lambda_reg=0.0,
             n_iter=10000, initial_temp=1.0):
    """
    Run MCMC optimization to minimize alignment MSE.
    """
    T = len(time_points)
    shifts, shift_range = initialize_shifts(cell_signals, time_points, global_time)
    best_shifts = shifts.copy()
    best_score, best_mean = compute_mse(best_shifts, cell_signals, global_time, T, lambda_reg)

    mse_trace = [best_score]

    for i in range(n_iter):
        temperature = initial_temp * (0.99 ** (i / 1))

        proposal = best_shifts.copy()
        cid = random.choice(list(proposal.keys()))
        delta_shift = random.choice([-10, -1, 1, 10])
        new_shift = np.clip(proposal[cid] + delta_shift, 0, shift_range)
        proposal[cid] = new_shift

        new_score, new_mean = compute_mse(proposal, cell_signals, global_time, T, lambda_reg)
        delta = new_score - best_score

        if delta < 0 or np.exp(-delta / temperature) > np.random.rand():
            best_shifts = proposal
            best_score = new_score
            best_mean = new_mean

        mse_trace.append(best_score)

        if i % 100 == 0:
            print(f"Step {i}: MSE = {best_score:.4f}")

    return best_shifts, best_mean, mse_trace
