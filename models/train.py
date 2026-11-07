# -*- coding: utf-8 -*-
"""Обучение и оценка модели LightGBM."""

import lightgbm as lgb

from config import (
    MODEL_PARAMS, EARLY_STOPPING_ROUNDS, TEST_SIZE,
    CLASSIFICATION_THRESHOLD, CATEGORICAL_FEATURES, RESULTS_PATH
)
from .common import (
    time_split, pick_threshold, print_metrics, print_feature_importance,
)


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
    train, test, split_date = time_split(dataset, TEST_SIZE, target=target)

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
    best_thr = pick_threshold(y_test, preds_proba, CLASSIFICATION_THRESHOLD)
    print(f"\nОптимальный порог по F1: {best_thr:.3f}")

    preds_binary = (preds_proba > best_thr).astype(int)

    print_metrics(y_test, preds_proba, preds_binary)
    print_feature_importance(features, model.feature_importances_)

    # Сохранение результатов
    results = test[['grid_id', 'date', 'is_fire']].copy()
    results['predicted_probability'] = preds_proba
    results['predicted_class'] = preds_binary
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nРезультаты сохранены в '{RESULTS_PATH}'")

    return model, results
