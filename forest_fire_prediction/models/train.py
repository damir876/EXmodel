# -*- coding: utf-8 -*-
"""Обучение и оценка модели LightGBM."""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, precision_recall_curve
)

from config import (
    MODEL_PARAMS, EARLY_STOPPING_ROUNDS, TEST_SIZE,
    CLASSIFICATION_THRESHOLD, CATEGORICAL_FEATURES, RESULTS_PATH
)


def _print_metrics(y_true, preds_proba, preds_binary):
    print("\n" + "=" * 50)
    print("МЕТРИКИ КАЧЕСТВА МОДЕЛИ")
    print("=" * 50)
    print(f"ROC-AUC:   {roc_auc_score(y_true, preds_proba):.4f}")
    print(f"Precision: {precision_score(y_true, preds_binary, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, preds_binary, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, preds_binary, zero_division=0):.4f}")
    print("\nПолный отчёт:")
    print(classification_report(y_true, preds_binary, zero_division=0))


def _print_feature_importance(model, features):
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    print("\nТоп-15 важных признаков:")
    print(importance.head(15).to_string(index=False))


def train_model(dataset):
    """
    Обучает модель бинарной классификации для прогноза вероятности пожара.
    Возвращает (model, results_df).
    """
    target = 'is_fire'
    exclude_cols = [
        'grid_id', 'date', target, 'fire_count',
        'lat_raw', 'lon_raw', 'region', 'grid_lat', 'grid_lon',
    ]
    features = [c for c in dataset.columns if c not in exclude_cols]
    cat_features = [c for c in CATEGORICAL_FEATURES if c in features]

    print(f"Используется {len(features)} признаков: {features}")
    print(f"Категориальные: {cat_features}")

    # Разделение по времени
    sorted_dates = dataset['date'].sort_values().reset_index(drop=True)
    split_idx = int(len(sorted_dates) * (1 - TEST_SIZE))
    split_date = sorted_dates.iloc[split_idx]

    train = dataset[dataset['date'] <= split_date].copy()
    test = dataset[dataset['date'] > split_date].copy()

    print(f"Train: {len(train)} строк (до {split_date.date()})")
    print(f"Test:  {len(test)} строк (после {split_date.date()})")
    print(f"Доля пожаров в train: {train[target].mean():.4f}")
    print(f"Доля пожаров в test:  {test[target].mean():.4f}")

    # Проверка на наличие обоих классов
    if train[target].nunique() < 2 or test[target].nunique() < 2:
        raise ValueError(
            "В train или test только один класс. "
            "Проверьте объём данных и negative_ratio."
        )

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # Модель (без scale_pos_weight — работаем с вероятностями)
    model = lgb.LGBMClassifier(**MODEL_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    # Прогноз вероятностей
    preds_proba = model.predict_proba(X_test)[:, 1]

    # Подбор порога по F1 на тесте
    prec, rec, thr = precision_recall_curve(y_test, preds_proba)
    f1_curve = 2 * prec * rec / (prec + rec + 1e-9)
    if len(thr) > 0:
        best_thr = float(thr[np.argmax(f1_curve[:-1])])
    else:
        best_thr = CLASSIFICATION_THRESHOLD
    print(f"\nОптимальный порог по F1: {best_thr:.3f}")

    preds_binary = (preds_proba > best_thr).astype(int)

    _print_metrics(y_test, preds_proba, preds_binary)
    _print_feature_importance(model, features)

    # Сохранение результатов
    results = test[['grid_id', 'date', 'is_fire']].copy()
    results['predicted_probability'] = preds_proba
    results['predicted_class'] = preds_binary
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nРезультаты сохранены в '{RESULTS_PATH}'")

    return model, results