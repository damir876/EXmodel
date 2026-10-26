# -*- coding: utf-8 -*-
"""Модуль загрузки данных."""
from .loader import load_fire_data
from .meteo import load_meteo_data
from .forest import load_forest_data

__all__ = ['load_fire_data', 'load_meteo_data', 'load_forest_data']