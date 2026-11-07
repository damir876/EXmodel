# -*- coding: utf-8 -*-
"""Модели машинного обучения."""
from .train import train_model
from .train_catboost import train_model_catboost
from .common import compute_metrics

__all__ = ['train_model', 'train_model_catboost', 'compute_metrics']
