# -*- coding: utf-8 -*-
"""Загрузка и очистка данных о лесных пожарах."""

import glob
import pandas as pd

from config import INVALID_REGIONS
from utils.coords import parse_coordinates


def _find_header_row(raw_df, max_scan=10):
    """Ищет индекс строки заголовка, содержащей 'Субъект'."""
    for idx in range(min(max_scan, len(raw_df))):
        if 'Субъект' in str(raw_df.iloc[idx].values):
            return idx
    return None


def _find_column(df, keyword, check_values=None):
    """
    Возвращает первую колонку, содержащую keyword в названии.
    Если не находит по названию, проверяет значения в колонках (check_values).
    """
    # 1. Ищем по названию колонки
    matches = [c for c in df.columns if keyword in str(c).lower()]
    if matches:
        return matches[0]
    
    # 2. Если не нашли, ищем по содержимому (например, 'Лишайн', 'Зеленомош')
    if check_values:
        for col in df.columns:
            # Берем первые 100 непустых строк, чтобы быстро проверить
            sample = df[col].dropna().astype(str).head(100).str.lower()
            for val in check_values:
                if sample.str.contains(val.lower()).any():
                    return col

    raise KeyError(
        f"Не найдена колонка с ключевым словом '{keyword}'. "
        f"Доступные колонки: {df.columns.tolist()}"
    )


def load_fire_data(files_pattern):
    """
    Загружает и очищает данные о пожарах из всех файлов.
    
    Возвращает DataFrame с колонками:
        region, date, lat, lon, lat_raw, lon_raw, forest_type, forest_cover
    """
    files = sorted(glob.glob(files_pattern))
    if not files:
        raise FileNotFoundError(
            f"Файлы по паттерну не найдены: {files_pattern}"
        )

    print(f"Найдено файлов: {len(files)}")

    dfs = []
    for file in files:
        try:
            raw = pd.read_excel(file, header=None)
            header_idx = _find_header_row(raw)
            if header_idx is None:
                print(f"⚠️ Пропущен файл (нет строки 'Субъект'): {file}")
                continue
            dfs.append(pd.read_excel(file, header=header_idx))
        except Exception as e:
            print(f"⚠️ Ошибка при чтении файла {file}: {e}")
            continue

    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла.")

    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"Всего строк в объединённом датасете: {len(df_raw)}")

    # Поиск обязательных колонок
    sub_col = _find_column(df_raw, 'субъект')
    date_col = _find_column(df_raw, 'обнаруж')
    lat_col = _find_column(df_raw, 'широта')
    lon_col = _find_column(df_raw, 'долгота')

    # Поиск лесохозяйственных колонок
    try:
        forest_type_col = _find_column(df_raw, 'порода')
    except KeyError:
        forest_type_col = None
        print("⚠️ Колонка 'Порода' не найдена в реестре")
        
    try:
        # Для покрова ищем по слову 'покров' или по типичным значениям
        forest_cover_col = _find_column(
            df_raw, 
            'покров', 
            check_values=['лишайн', 'зеленомош', 'сфагнум', 'мох', 'трава', 'брусник']
        )
    except KeyError:
        forest_cover_col = None
        print("⚠️ Колонка 'Покров' не найдена в реестре")

    # Собираем нужные колонки
    cols_to_keep = [sub_col, date_col, lat_col, lon_col]
    new_col_names = ['region', 'date', 'lat_raw', 'lon_raw']
    
    if forest_type_col:
        cols_to_keep.append(forest_type_col)
        new_col_names.append('forest_type')
    if forest_cover_col:
        cols_to_keep.append(forest_cover_col)
        new_col_names.append('forest_cover')

    df_clean = df_raw[cols_to_keep].copy()
    df_clean.columns = new_col_names

    # Если колонок не было, создаем пустые
    if 'forest_type' not in df_clean.columns:
        df_clean['forest_type'] = 'Unknown'
    if 'forest_cover' not in df_clean.columns:
        df_clean['forest_cover'] = 'Unknown'

    # Очистка региона
    df_clean['region'] = df_clean['region'].astype(str).str.strip()
    df_clean = df_clean[~df_clean['region'].str.lower().isin(INVALID_REGIONS)]
    df_clean = df_clean[df_clean['region'].notna()]

    # Парсинг даты
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['date'])

    # Парсинг координат
    df_clean['lat'] = df_clean['lat_raw'].apply(parse_coordinates)
    df_clean['lon'] = df_clean['lon_raw'].apply(parse_coordinates)

    n_before = len(df_clean)
    df_clean = df_clean.dropna(subset=['lat', 'lon'])
    n_after = len(df_clean)
    print(f"Удалено строк с невалидными координатами: {n_before - n_after}")

    print(f"Итоговый размер: {df_clean.shape}")
    print(f"Уникальных регионов: {df_clean['region'].nunique()}")

    return df_clean