from abc import ABC, abstractmethod
import math


class BaseActivationFunction(ABC):
    @staticmethod
    @abstractmethod
    def calc(x: float) -> float:
        pass

    @staticmethod
    @abstractmethod
    def derivative(x: float) -> float:
        pass


class SigmoidActivationFunction(BaseActivationFunction):
    @staticmethod
    def calc(x: float) -> float:
        return 1. / (1. + math.exp(-x))

    @staticmethod
    def derivative(x: float) -> float:
        y = 1. / (1. + math.exp(-x))
        return y * (1. - y)


class TanhActivationFunction(BaseActivationFunction):
    @staticmethod
    def calc(x: float) -> float:
        return math.tanh(x)

    @staticmethod
    def derivative(x: float) -> float:
        y = math.tanh(x)
        return 1. - y**2


class ReluActivationFunction(BaseActivationFunction):
    @staticmethod
    def calc(x: float) -> float:
        return 0. if x <= 0 else x

    @staticmethod
    def derivative(x: float) -> float:
        return 0. if x <= 0 else 1.


class GaussianActivationFunction(BaseActivationFunction):
    @staticmethod
    def calc(x: float) -> float:
        return math.exp(-x**2)

    @staticmethod
    def derivative(x: float) -> float:
        return -2.*x*math.exp(-x**2)
