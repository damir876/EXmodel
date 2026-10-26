# -*- coding: utf-8 -*-
"""Загрузка лесохозяйственных данных из реестра пожаров."""

import pandas as pd
import numpy as np


def load_forest_data(df_fires, grid_ids):
    """
    Извлекает лесохозяйственные данные (породу и покров) 
    из уже загруженного датафрейма пожаров.
    
    Аргументы:
        df_fires: DataFrame, полученный из load_fire_data
        grid_ids: массив уникальных grid_id
    """
    print("Загрузка лесохозяйственных данных из реестра...")
    
    # Берем только нужные колонки из пожаров
    # drop_duplicates нужен, так как в одном grid_id могло быть несколько пожаров
    df_forest = df_fires[['grid_id', 'forest_type', 'forest_cover']].drop_duplicates(subset=['grid_id'])
    
    # Убеждаемся, что для всех grid_ids есть запись (даже если пожара не было)
    df_forest = df_forest.set_index('grid_id').reindex(grid_ids).reset_index()
    
    # Заполняем пропуски, если для какой-то сетки не нашлось данных
    df_forest['forest_type'] = df_forest['forest_type'].fillna('Unknown')
    df_forest['forest_cover'] = df_forest['forest_cover'].fillna('Unknown')
    
    # Добавляем фиктивную высоту (elevation), так как в реестре ее нет.
    # Если модель ее не использует, она не повлияет. 
    # Можно поставить 0.0.
    df_forest['elevation'] = 0.0
    
    # На всякий случай убираем дубликаты (если в реестре для одной ячейки были разные породы)
    df_forest = df_forest.drop_duplicates(subset=['grid_id'])
    
    print(f"Лесохозяйственные данные загружены для {len(df_forest)} ячеек.")
    return df_forest
