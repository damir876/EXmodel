# -*- coding: utf-8 -*-
"""Модуль краткосрочного прогнозирования на основе прогноза погоды."""

import pandas as pd
import numpy as np
import requests
from config import GRID_SIZE_DEG

# --- Вспомогательные функции (скопированы из features/engineering.py, чтобы избежать ошибок импорта) ---
def _add_calendar_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    return df

def get_weather_forecast(lat, lon, days=3):
    """Получает прогноз погоды с Open-Meteo (бесплатно, без ключа)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_mean", "relative_humidity_2m_mean", "precipitation_sum", "wind_speed_10m_max"],
        "timezone": "auto",
        "forecast_days": days
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        df = pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            'temperature': data['daily']['temperature_2m_mean'],
            'humidity': data['daily']['relative_humidity_2m_mean'],
            'precipitation': data['daily']['precipitation_sum'],
            'wind_speed': data['daily']['wind_speed_10m_max']
        })
        return df
    except Exception as e:
        print(f"Ошибка получения прогноза для {lat}, {lon}: {e}")
        return None

def predict_future(model, df_fires_history, grid_ids, days_ahead=3):
    """
    Строит прогноз на будущее.
    df_fires_history: исторические данные (для расчета лагов)
    grid_ids: список ячеек для прогноза
    """
    print(f"\nЗапуск прогноза на {days_ahead} дней вперед...")
    
    # 1. Получаем прогноз погоды для каждой ячейки
    forecast_records = []
    for gid in grid_ids:
        lat = (int(gid.split('_')[0]) + 0.5) * GRID_SIZE_DEG
        lon = (int(gid.split('_')[1]) + 0.5) * GRID_SIZE_DEG
        
        df_w = get_weather_forecast(lat, lon, days=days_ahead)
        if df_w is not None:
            df_w['grid_id'] = gid
            forecast_records.append(df_w)
            
    if not forecast_records:
        print("⚠️ Не удалось получить прогноз погоды.")
        return pd.DataFrame()

    df_forecast = pd.concat(forecast_records, ignore_index=True)
    
    # 2. Подмешиваем лесохозяйственные данные (берем последние известные)
    forest_info = df_fires_history[['grid_id', 'forest_type', 'forest_cover']].drop_duplicates('grid_id')
    df_forecast = df_forecast.merge(forest_info, on='grid_id', how='left')
    df_forecast['forest_type'] = df_forecast['forest_type'].fillna('Unknown')
    df_forecast['forest_cover'] = df_forecast['forest_cover'].fillna('Unknown')
    df_forecast['elevation'] = 0.0

    # 3. Добавляем календарные признаки
    df_forecast = _add_calendar_features(df_forecast)
    
    # Считаем производные погодные признаки (точка росы и kp_daily)
    a, b = 17.27, 237.7
    humidity = df_forecast['humidity'].clip(1, 100).fillna(50)
    temp = df_forecast['temperature'].fillna(0)
    alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
    df_forecast['dew_point'] = (b * alpha) / (a - alpha)
    df_forecast['kp_daily'] = df_forecast['temperature'] * (df_forecast['temperature'] - df_forecast['dew_point'])
    
    # 4. Имитация лагов (для прототипа заполняем нулями, так как у нас нет истории на будущее)
    for col in ['is_fire', 'kp_daily']:
        for lag in [1, 3, 7, 14]:
            df_forecast[f'{col}_lag_{lag}'] = 0
        for window in [3, 7, 14]:
            df_forecast[f'{col}_roll_mean_{window}'] = 0
            
    # 5. Прогноз!
    feature_cols = model.feature_name_
    
    # Проверка на наличие всех нужных колонок
    missing_cols = [c for c in feature_cols if c not in df_forecast.columns]
    if missing_cols:
        print(f"⚠️ Внимание: в прогнозе не хватает колонок: {missing_cols}. Заполняем нулями.")
        for c in missing_cols:
            df_forecast[c] = 0

    X_future = df_forecast[feature_cols]
    
    # Важно: LightGBM ожидает категориальные типы
    for col in ['forest_type', 'forest_cover']:
        if col in X_future.columns:
            X_future[col] = X_future[col].astype('category')
            
    df_forecast['predicted_probability'] = model.predict_proba(X_future)[:, 1]
    
    print("✅ Прогноз готов!")
    return df_forecast