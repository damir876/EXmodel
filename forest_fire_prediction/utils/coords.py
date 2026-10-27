# -*- coding: utf-8 -*-
"""Парсинг географических координат."""

import pandas as pd
import numpy as np


def parse_coordinates(coord_str):
    """
    Преобразует строку координат в десятичные градусы.

    Поддерживаемые форматы:
      - "64°20'24,0"N"
      - "64°20'24.0"N"
      - "64 20 24 N"
      - "64°20′24″N"  (типографские кавычки)
      - "64.34"       (десятичные градусы)

    Возвращает float или np.nan при ошибке.
    """
    if pd.isna(coord_str):
        return np.nan

    try:
        s = str(coord_str).strip().upper()
        if not s:
            return np.nan

        # Нормализация символов
        s = (s.replace('°', ' ')
               .replace('′', ' ')
               .replace('″', ' ')
               .replace("'", ' ')
               .replace('"', ' ')
               .replace(',', '.'))

        direction = ''
        if s and s[-1] in 'NSEW':
            direction = s[-1]
            s = s[:-1].strip()

        parts = s.split()
        if not parts:
            return np.nan

        if len(parts) == 1:
            decimal = float(parts[0])
        else:
            degrees = float(parts[0])
            minutes = float(parts[1]) if len(parts) > 1 else 0.0
            seconds = float(parts[2]) if len(parts) > 2 else 0.0
            decimal = degrees + minutes / 60.0 + seconds / 3600.0

        if direction in ('S', 'W'):
            decimal = -decimal

        return decimal

    except Exception:
        return np.nan