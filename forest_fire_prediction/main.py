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
from models import train_model

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

    # --- Шаг 5: Обучение модели ---
    print("\n" + "=" * 50)
    print("ШАГ 5: Обучение модели")
    print("=" * 50)
    model, results = train_model(dataset)

    # --- Шаг 6: Прогноз на будущее (Краткосрочный) ---
    print("\n" + "=" * 50)
    print("ШАГ 6: Прогнозирование на будущее (3 дня)")
    print("=" * 50)
    
    if os.path.exists('forecast.py'):
        from forecast import predict_future
        
        # Берем 50 случайных ячеек из пилотного региона для теста прогноза
        test_grids = df_fires_pilot['grid_id'].unique()[:50]
        
        df_future = predict_future(model, df_fires_pilot, test_grids, days_ahead=3)
        
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