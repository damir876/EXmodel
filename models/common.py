# -*- coding: utf-8 -*-
"""Общие помощники для обучения моделей (LightGBM, CatBoost)."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, precision_recall_curve,
)


def time_split(dataset, test_size, target='is_fire'):
    """Разделяет датасет по времени: прошлое -> train, будущее -> test."""
    sorted_dates = dataset['date'].sort_values().reset_index(drop=True)
    split_idx = int(len(sorted_dates) * (1 - test_size))
    split_date = sorted_dates.iloc[split_idx]

    train = dataset[dataset['date'] <= split_date].copy()
    test = dataset[dataset['date'] > split_date].copy()
    return train, test, split_date


def pick_threshold(y_true, preds_proba, default=0.5):
    """Подбирает порог бинаризации по максимуму F1."""
    prec, rec, thr = precision_recall_curve(y_true, preds_proba)
    f1_curve = 2 * prec * rec / (prec + rec + 1e-9)
    if len(thr) > 0:
        return float(thr[np.argmax(f1_curve[:-1])])
    return default


def print_metrics(y_true, preds_proba, preds_binary):
    print("\n" + "=" * 50)
    print("МЕТРИКИ КАЧЕСТВА МОДЕЛИ")
    print("=" * 50)
    print(f"ROC-AUC:   {roc_auc_score(y_true, preds_proba):.4f}")
    print(f"Precision: {precision_score(y_true, preds_binary, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, preds_binary, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, preds_binary, zero_division=0):.4f}")
    print("\nПолный отчёт:")
    print(classification_report(y_true, preds_binary, zero_division=0))


def print_feature_importance(features, importances, top=15):
    importance = pd.DataFrame({
        'feature': features,
        'importance': importances,
    }).sort_values('importance', ascending=False)

    print(f"\nТоп-{top} важных признаков:")
    print(importance.head(top).to_string(index=False))


def compute_metrics(y_true, preds_proba, preds_binary):
    """Возвращает словарь метрик для сравнения моделей."""
    return {
        'roc_auc': roc_auc_score(y_true, preds_proba),
        'precision': precision_score(y_true, preds_binary, zero_division=0),
        'recall': recall_score(y_true, preds_binary, zero_division=0),
        'f1': f1_score(y_true, preds_binary, zero_division=0),
    }
