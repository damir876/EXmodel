# -*- coding: utf-8 -*-
"""Feature Engineering: создание признаков для модели."""

import pandas as pd
import numpy as np
from config import CATEGORICAL_FEATURES


def _calc_dew_point(temp, humidity):
    """Точка росы по формуле Магнуса. Защита от деления на ноль и NaN."""
    a, b = 17.27, 237.7
    humidity = humidity.clip(lower=1, upper=100).fillna(50)
    temp = temp.fillna(0)
    
    alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    
    return dew_point.replace([np.inf, -np.inf], np.nan)


def _add_calendar_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    return df


def _add_lag_features(df, col, lags):
    for lag in lags:
        df[f'{col}_lag_{lag}'] = (
            df.groupby('grid_id', observed=True)[col]
            .shift(lag).fillna(0)
        )
    return df


def _add_rolling_features(df, col, windows):
    for window in windows:
        df[f'{col}_roll_mean_{window}'] = (
            df.groupby('grid_id', observed=True)[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    return df


def _create_grid_date_pairs(df_fires, negative_ratio=5, seed=42):
    """Отрицательный сэмплинг."""
    rng = np.random.default_rng(seed)

    positives = (
        df_fires.groupby(['grid_id', 'date'], observed=True)
        .size().reset_index(name='fire_count')
    )
    positives['is_fire'] = 1

    all_grids = df_fires['grid_id'].unique()
    min_date = df_fires['date'].min()
    max_date = df_fires['date'].max()
    all_dates = pd.date_range(min_date, max_date, freq='D')

    n_neg = len(positives) * negative_ratio
    neg_grids = rng.choice(all_grids, size=n_neg, replace=True)
    neg_dates = rng.choice(all_dates, size=n_neg, replace=True)

    negatives = pd.DataFrame({
        'grid_id': neg_grids,
        'date': pd.to_datetime(neg_dates),
        'fire_count': 0,
        'is_fire': 0,
    })

    merged = negatives.merge(
        positives[['grid_id', 'date']],
        on=['grid_id', 'date'], how='left', indicator=True
    )
    negatives = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')

    dataset = pd.concat([positives, negatives], ignore_index=True)
    return dataset


def create_training_dataset(df_fires, df_meteo, df_forest, negative_ratio=5):
    """Создаёт датасет (ячейка, дата) с отрицательным сэмплингом."""
    print(f"Ячеек: {df_fires['grid_id'].nunique()}, "
          f"дат: {(df_fires['date'].max() - df_fires['date'].min()).days}")
    print(f"Отрицательный сэмплинг: negative_ratio={negative_ratio}")

    dataset = _create_grid_date_pairs(df_fires, negative_ratio=negative_ratio)
    print(f"Размер после сэмплинга: {dataset.shape}, "
          f"доля пожаров: {dataset['is_fire'].mean():.4f}")

    # Подмешиваем внешние данные
    dataset = dataset.merge(df_meteo, on=['grid_id', 'date'], how='left')
    dataset = dataset.merge(df_forest, on='grid_id', how='left')

    dataset = dataset.sort_values(['grid_id', 'date']).reset_index(drop=True)

    # Feature Engineering
    dataset = _add_calendar_features(dataset)

    dataset['dew_point'] = _calc_dew_point(
        dataset['temperature'], dataset['humidity']
    )
    dataset['kp_daily'] = (
        dataset['temperature'] * (dataset['temperature'] - dataset['dew_point'])
    )

    dataset = _add_lag_features(dataset, 'is_fire', [1, 3, 7, 14])
    dataset = _add_lag_features(dataset, 'kp_daily', [1, 3, 7, 14])
    dataset = _add_rolling_features(dataset, 'is_fire', [3, 7, 14])
    dataset = _add_rolling_features(dataset, 'kp_daily', [3, 7, 14])

    # Категориальные признаки (берем из config.py)
    for col in CATEGORICAL_FEATURES:
        if col in dataset.columns:
            dataset[col] = dataset[col].astype('category')
        else:
            print(f"⚠️ Внимание: категориальный признак '{col}' отсутствует в датасете.")

    return dataset