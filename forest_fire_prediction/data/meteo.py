# data/meteo.py
import pandas as pd
import numpy as np
import requests
import time

def load_meteo_data(grid_ids, dates):
    """
    Загрузка метеоданных через NASA POWER.
    Возвращает DataFrame с колонками:
        grid_id, date, temperature, humidity, wind_speed, precipitation
    """
    print("Загрузка метеоданных NASA POWER...")
    
    records = []
    # Преобразуем grid_id обратно в координаты
    # Предполагая, что grid_id имеет вид "6434_3021" (grid_lat_grid_lon)
    for gid in grid_ids:
        try:
            lat_int, lon_int = gid.split('_')
            # Восстанавливаем реальные координаты (умножаем на GRID_SIZE_DEG)
            # Внимание: из-за floor координаты — это нижний левый угол ячейки
            # Для запроса к NASA POWER нужна центральная точка, добавляем половину шага
            from config import GRID_SIZE_DEG
            lat = (int(lat_int) + 0.5) * GRID_SIZE_DEG
            lon = (int(lon_int) + 0.5) * GRID_SIZE_DEG
            
            # Формируем запрос
            url = "https://power.larc.nasa.gov/api/temporal/daily/point"
            params = {
                "parameters": "T2M,RH2M,WS2M,PRECTOTCORR",
                "community": "AG",  # Сельскохозяйственный архив, ближе всего к вашим нуждам
                "longitude": lon,
                "latitude": lat,
                "start": dates.min().strftime("%Y%m%d"),
                "end": dates.max().strftime("%Y%m%d"),
                "format": "JSON"
            }
            
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            # Извлекаем данные: структура ответа — properties.parameter.{PARAM_NAME}.{YYYYMMDD}
            params_data = data["properties"]["parameter"]
            
            # Для каждой даты в диапазоне создаём запись
            for d in dates:
                date_str = d.strftime("%Y%m%d")
                records.append({
                    "grid_id": gid,
                    "date": d,
                    "temperature": params_data.get("T2M", {}).get(date_str, np.nan),
                    "humidity": params_data.get("RH2M", {}).get(date_str, np.nan),
                    "wind_speed": params_data.get("WS2M", {}).get(date_str, np.nan),
                    "precipitation": params_data.get("PRECTOTCORR", {}).get(date_str, np.nan),
                })
            
            # Вежливая пауза, чтобы не перегружать сервер
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Ошибка для {gid}: {e}")
            # При ошибке заполняем NaN для этой ячейки
            for d in dates:
                records.append({
                    "grid_id": gid,
                    "date": d,
                    "temperature": np.nan,
                    "humidity": np.nan,
                    "wind_speed": np.nan,
                    "precipitation": np.nan,
                })
    
    df = pd.DataFrame(records)
    print(f"Загружено {len(df)} записей, пропусков: {df.isna().sum().sum()}")
    return df
