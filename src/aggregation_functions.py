from typing import List
from abc import ABC, abstractmethod
import math
from functools import reduce
from operator import mul


class BaseAggregationFunction(ABC):
    @staticmethod
    @abstractmethod
    def calc(x: List[float]) -> float:
        pass

    @staticmethod
    @abstractmethod
    def derivative(x: List[float]) -> List[float]:
        pass


class SumAggregationFunction(BaseAggregationFunction):
    @staticmethod
    def calc(x: List[float]) -> float:
        return sum(x)

    @staticmethod
    def derivative(x: List[float]) -> List[float]:
        return [1.] * len(x)


class MeanAggregationFunction(BaseAggregationFunction):
    @staticmethod
    def calc(x: List[float]) -> float:
        return sum(x) / len(x)

    @staticmethod
    def derivative(x: List[float]) -> List[float]:
        return [1. / len(x)] * len(x)


class ProductAggregationFunction(BaseAggregationFunction):
    @staticmethod
    def calc(x: List[float]) -> float:
        return reduce(mul, x, 1.)

    @staticmethod
    def derivative(x: List[float]) -> List[float]:
        y = [0.] * len(x)
        for i, val in enumerate(x):
            x_ = [v_ for i_, v_ in enumerate(x) if i_ != i]
            y[i] = reduce(mul, x_, 1.)
        return y
