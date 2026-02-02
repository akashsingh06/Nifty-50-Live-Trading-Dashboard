"""
Machine Learning Models for Quantitative Trading
=================================================

Inspired by Renaissance Technologies and Jim Simons' approach:
- Statistical pattern recognition
- Hidden Markov Models for regime detection
- Ensemble methods for prediction
- Feature importance analysis
- Mean reversion with ML
"""

from .ml_predictor import (
    MLPredictor,
    PredictionResult,
    MarketRegime,
    PatternRecognizer,
    StatisticalArbitrage,
    EnsemblePredictor
)

__all__ = [
    'MLPredictor',
    'PredictionResult',
    'MarketRegime',
    'PatternRecognizer',
    'StatisticalArbitrage',
    'EnsemblePredictor'
]