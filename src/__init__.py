"""
Coffee Cup Background Resolution Benchmark

A comprehensive framework for evaluating AI image generation models
on their ability to coherently resolve background coffee cups.
"""

__version__ = "1.0.0"
__author__ = "Ari Leavesley"

from src.detector import CoffeeCupDetector
from src.evaluator import CupQualityEvaluator
from src.benchmark import CoffeeCupBenchmark

__all__ = [
    "CoffeeCupDetector",
    "CupQualityEvaluator",
    "CoffeeCupBenchmark",
]
