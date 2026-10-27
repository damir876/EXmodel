# -*- coding: utf-8 -*-
"""Привязка пожаров к пространственной сетке."""

import numpy as np
import pandas as pd


def assign_to_grid(df, grid_size_deg=0.01):
    """
    Привязывает каждый пожар к ячейке сетки 1x1 км.

    Используется floor (а не round), чтобы ячейка была ровно
    grid_size_deg x grid_size_deg, без «двойного» размера.
    """
    df = df.copy()
    df['grid_lat'] = np.floor(df['lat'] / grid_size_deg).astype(int)
    df['grid_lon'] = np.floor(df['lon'] / grid_size_deg).astype(int)
    df['grid_id'] = (
        df['grid_lat'].astype(str) + '_' + df['grid_lon'].astype(str)
    )
    return df