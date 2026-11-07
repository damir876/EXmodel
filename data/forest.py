# -*- coding: utf-8 -*-
"""Загрузка лесохозяйственных данных из реестра пожаров."""

import pandas as pd
import numpy as np


def load_forest_data(df_fires, grid_ids):
    """
    Извлекает лесохозяйственные данные (породу и покров) 
    из уже загруженного датафрейма пожаров.
    """
    print("Загрузка лесохозяйственных данных из реестра...")
    
    # Берем только нужные колонки из пожаров
    df_forest = df_fires[['grid_id', 'forest_type', 'forest_cover']].drop_duplicates(subset=['grid_id'])
    
    # Убеждаемся, что для всех grid_ids есть запись
    df_forest = df_forest.set_index('grid_id').reindex(grid_ids).reset_index()
    
    # Заполняем пропуски
    df_forest['forest_type'] = df_forest['forest_type'].fillna('Unknown')
    df_forest['forest_cover'] = df_forest['forest_cover'].fillna('Unknown')
    
    # Добавляем фиктивную высоту (elevation)
    df_forest['elevation'] = 0.0
    
    df_forest = df_forest.drop_duplicates(subset=['grid_id'])
    
    print(f"Лесохозяйственные данные загружены для {len(df_forest)} ячеек.")
    return df_forest