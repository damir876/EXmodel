# -*- coding: utf-8 -*-
"""Все настройки проекта в одном месте."""

# Путь к файлам с реестрами пожаров
# config.py
import os

# Путь к файлам с реестрами пожаров.
# Можно переопределить через переменную окружения FIRE_FILES_PATTERN.
FILES_PATTERN = os.environ.get(
    "FIRE_FILES_PATTERN",
    "/content/drive/MyDrive/Roslesinforg/Прогнозирование возникновения лесных пожаров/Реестры лесных пожаров за 10 лет/*.xls*",
)

# Пилотный регион (None - использовать все данные)
PILOT_REGION = "Республика Карелия"

# Размер ячейки сетки в градусах (0.01 ~ 1 км)
GRID_SIZE_DEG = 0.01

# Доля данных для теста (по времени)
TEST_SIZE = 0.2

# Отрицательный сэмплинг: сколько «не-пожаров» на 1 пожар
NEGATIVE_RATIO = 5

# Параметры модели LightGBM
MODEL_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.02,
    'max_depth': 8,
    'num_leaves': 63,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

# Параметры модели CatBoost
CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.02,
    'depth': 8,
    'random_seed': 42,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'thread_count': -1,
    'verbose': False,
}

# Ранняя остановка
EARLY_STOPPING_ROUNDS = 100

# Порог для бинарной классификации (если не удалось подобрать по F1)
CLASSIFICATION_THRESHOLD = 0.5
# Категориальные признаки
CATEGORICAL_FEATURES = ['forest_type', 'forest_cover']
# Путь для сохранения результатов
RESULTS_PATH = 'fire_predictions_grid.csv'
CATBOOST_RESULTS_PATH = 'fire_predictions_grid_catboost.csv'

# Невалидные значения регионов
INVALID_REGIONS = ['nan', 'none', '', 'субъект', 'российская федерация', 'а1']
