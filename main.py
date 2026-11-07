# -*- coding: utf-8 -*-
"""
Главный скрипт запуска пайплайна прогнозирования лесных пожаров.

Использование:
    python main.py
"""

import pandas as pd
import os

from config import (
    FILES_PATTERN, PILOT_REGION, GRID_SIZE_DEG, NEGATIVE_RATIO,
)
from data import load_fire_data, load_meteo_data, load_forest_data
from features import assign_to_grid, create_training_dataset
from models import train_model, train_model_catboost, compute_metrics

def main():
    # --- Шаг 1: Загрузка данных о пожарах ---
    print("=" * 50)
    print("ШАГ 1: Загрузка данных о пожарах")
    print("=" * 50)
    df_fires = load_fire_data(FILES_PATTERN)

    if len(df_fires) == 0:
        raise RuntimeError(
            "Данные о пожарах не загружены. Проверьте FILES_PATTERN."
        )

    print("\nПример данных:")
    cols_to_show = ['region', 'date', 'lat', 'lon', 'forest_type', 'forest_cover']
    existing_cols = [c for c in cols_to_show if c in df_fires.columns]
    print(df_fires[existing_cols].head())
    
    print("\nСтатистика координат:")
    print(df_fires[['lat', 'lon']].describe())

    # --- Шаг 2: Привязка к сетке ---
    print("\n" + "=" * 50)
    print("ШАГ 2: Привязка пожаров к сетке")
    print("=" * 50)
    df_fires = assign_to_grid(df_fires, grid_size_deg=GRID_SIZE_DEG)
    print(f"Уникальных ячеек сетки: {df_fires['grid_id'].nunique()}")

    # Фильтр по пилотному региону
    if PILOT_REGION:
        df_fires_pilot = df_fires[df_fires['region'] == PILOT_REGION].copy()
        if len(df_fires_pilot) == 0:
            print(f"⚠️ Данных по '{PILOT_REGION}' не найдено. "
                  f"Используются все данные.")
            df_fires_pilot = df_fires
        else:
            print(f"Отфильтровано по региону '{PILOT_REGION}': "
                  f"{len(df_fires_pilot)} записей")
    else:
        df_fires_pilot = df_fires

    if len(df_fires_pilot) < 100:
        print(f"⚠️ Внимание: всего {len(df_fires_pilot)} записей. "
              f"Модель может быть нестабильной.")

    # --- Шаг 3: Загрузка внешних данных ---
    print("\n" + "=" * 50)
    print("ШАГ 3: Загрузка внешних данных")
    print("=" * 50)
    grid_ids = df_fires_pilot['grid_id'].unique()
    dates = pd.date_range(
        df_fires_pilot['date'].min(),
        df_fires_pilot['date'].max(),
        freq='D',
    )
    
    # Загружаем метеоданные (NASA POWER)
    df_meteo = load_meteo_data(grid_ids, dates)
    
    # Загружаем лесохозяйственные данные ИЗ САМОГО РЕЕСТРА
    df_forest = load_forest_data(df_fires_pilot, grid_ids) 

    # --- Шаг 4: Создание датасета ---
    print("\n" + "=" * 50)
    print("ШАГ 4: Создание обучающего датасета")
    print("=" * 50)
    dataset = create_training_dataset(
        df_fires_pilot, df_meteo, df_forest,
        negative_ratio=NEGATIVE_RATIO,
    )
    print(f"Размер датасета: {dataset.shape}")
    print(f"Пропуски по колонкам (топ-10):")
    print(dataset.isna().sum().sort_values(ascending=False).head(10))

    # --- Шаг 5: Обучение моделей ---
    print("\n" + "=" * 50)
    print("ШАГ 5: Обучение моделей (LightGBM и CatBoost)")
    print("=" * 50)
    model_lgb, results_lgb = train_model(dataset)
    model_cb, results_cb = train_model_catboost(dataset)

    # Сравнение моделей по метрикам на тесте
    print("\n" + "=" * 50)
    print("СРАВНЕНИЕ МОДЕЛЕЙ (тест)")
    print("=" * 50)
    metrics_lgb = compute_metrics(
        results_lgb['is_fire'],
        results_lgb['predicted_probability'],
        results_lgb['predicted_class'],
    )
    metrics_cb = compute_metrics(
        results_cb['is_fire'],
        results_cb['predicted_probability'],
        results_cb['predicted_class'],
    )
    print(f"{'Модель':<12}{'ROC-AUC':>10}{'F1':>10}{'Precision':>12}{'Recall':>10}")
    for name, m in [('LightGBM', metrics_lgb), ('CatBoost', metrics_cb)]:
        print(f"{name:<12}{m['roc_auc']:>10.4f}{m['f1']:>10.4f}"
              f"{m['precision']:>12.4f}{m['recall']:>10.4f}")

    if metrics_cb['roc_auc'] >= metrics_lgb['roc_auc']:
        best_name, best_model = 'CatBoost', model_cb
    else:
        best_name, best_model = 'LightGBM', model_lgb
    print(f"\nДля прогноза выбрана модель с лучшим ROC-AUC: {best_name}")

    # --- Шаг 6: Прогноз на будущее (Краткосрочный) ---
    print("\n" + "=" * 50)
    print("ШАГ 6: Прогнозирование на будущее (3 дня)")
    print("=" * 50)
    
    if os.path.exists('forecast.py'):
        from forecast import predict_future
        
        # Берем 50 случайных ячеек из пилотного региона для теста прогноза
        test_grids = df_fires_pilot['grid_id'].unique()[:50]
        
        df_future = predict_future(best_model, df_fires_pilot, test_grids, days_ahead=3)

        if df_future.empty:
            print("⚠️ Прогноз не получен (нет данных о погоде). Пропуск сохранения.")
        else:
            print("\nПример прогноза на будущее:")
            print(df_future[['grid_id', 'date', 'predicted_probability']].head())

            df_future.to_csv('future_forecast.csv', index=False)
            print("✅ Результаты прогноза сохранены в 'future_forecast.csv'")
    else:
        print("⚠️ Модуль forecast.py не найден. Шаг прогноза пропущен.")

    print("\n" + "=" * 50)
    print("ГОТОВО! ВСЕ ЭТАПЫ УСПЕШНО ЗАВЕРШЕНЫ.")
    print("=" * 50)


if __name__ == "__main__":
    main()