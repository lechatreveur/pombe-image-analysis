#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:47:22 2025

@author: user
"""

# preprocessing.py

import pandas as pd

def filter_valid_cells(df_all, frame_number):
    valid_cells = df_all.groupby("cell_id")["time_point"].count()
    valid_cells = valid_cells[valid_cells == frame_number].index
    return df_all[df_all['cell_id'].isin(valid_cells)]

def compute_derivatives(df_all, rolling_window=20):
    df_all = df_all.sort_values(by=['cell_id', 'time_point'])

    df_all['d_cell_length'] = df_all.groupby('cell_id')['cell_length'].diff()
    df_all['d_cell_area'] = df_all.groupby('cell_id')['cell_area'].diff()
    df_all['d_nu_dis'] = df_all.groupby('cell_id')['nu_dis'].diff()

    df_all['d_cell_length_avg'] = (
        df_all.groupby('cell_id')['d_cell_length']
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean().reset_index(level=0, drop=True)
    )
    df_all['d_cell_area_avg'] = (
        df_all.groupby('cell_id')['d_cell_area']
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean().reset_index(level=0, drop=True)
    )
    df_all['d_nu_dis_avg'] = (
        df_all.groupby('cell_id')['d_nu_dis']
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean().reset_index(level=0, drop=True)
    )

    return df_all
