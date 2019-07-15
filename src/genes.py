import random
from typing import Any, Tuple, List
from abc import ABC, abstractmethod
from collections import deque
from attributes import FloatAttr, BoolAttr, StringAttr


class BaseGene(ABC):
    @property
    @abstractmethod
    def _gene_attributes(self):
        pass

    def __init__(self, key: Any):
        self.key = key  # also identifier of this gene in genome

    def init_attributes(self, config: object) -> None:
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config))

    def mutate(self, config: object) -> None:
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def copy(self) -> Any:
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))
        return new_gene

    def crossover(self, other) -> Any:
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random.random() < 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(other, a.name))
        return new_gene

    @abstractmethod
    def distance(self, other) -> float:
        pass


class BaseNeuron(ABC):
    @abstractmethod
    def forward(self, config: object, inputs: List[float]) -> float:
        pass

    @abstractmethod
    def backward(self, config: object, grad: float) -> List[float]:
        pass


class DefaultNodeGene(BaseGene):
    _gene_attributes = [
        FloatAttr('response'),
        FloatAttr('bias'),
        StringAttr('activation', default_value='sigmoid'),
        StringAttr('aggregation', default_value='sum')
    ]

    def __init__(self, key: int):
        assert isinstance(key, int), '{0} key must be {1}'.format(self.__class__, int)
        super(DefaultNodeGene, self).__init__(key)

    def distance(self, other) -> float:
        d = 0.
        d += abs(getattr(self, 'response') - getattr(other, 'response'))
        d += abs(getattr(self, 'bias') - getattr(other, 'bias'))
        if getattr(self, 'activation') != getattr(other, 'activation'):
            d += 1.
        if getattr(self, 'aggregation') != getattr(other, 'aggregation'):
            d += 1.
        return d


class DefaultConnectionGene(BaseGene):
    _gene_attributes = [
        FloatAttr('weight'),
        BoolAttr('enabled')
    ]

    def __init__(self, key: Tuple[int, int]):
        assert isinstance(key, tuple), '{0} key must be {1}'.format(self.__class__, tuple)
        super(DefaultConnectionGene, self).__init__(key)

    def distance(self, other) -> float:
        d = 0.
        d += abs(getattr(self, 'weight') - getattr(other, 'weight'))
        if getattr(self, 'enabled') != getattr(other, 'enabled'):
            d += 1.
        return d


class NeuralNodeGene(DefaultNodeGene, BaseNeuron):
    _grad_items_history_len = 4
    _grad_items_history = deque(maxlen=_grad_items_history_len)

    def forward(self, config: object, inputs: List[float]) -> float:
        assert isinstance(inputs, list), 'input must be {0}, not {1}'.format(list, type(inputs))
        assert len(inputs) > 0, 'input length must not be 0'
        _grad_items = {
            'inputs': inputs,
            'gradient': {}
        }
        activation_function_def = getattr(config, 'activation_function_def')
        aggregation_function_def = getattr(config, 'aggregation_function_def')
        activation_f = activation_function_def[self.activation]
        aggregation_f = aggregation_function_def[self.aggregation]
        y = activation_f.calc(self.response * aggregation_f.calc(inputs) + self.bias)
        return y

    def backward(self, config: object, grad: float) -> List[float]:
        # y = activation(response * aggregation(inputs) + self.bias)
        # dy/d_bias = activation.derivative(response * aggregation(inputs) + self.bias) * 1
        # dy/d_response = activation.derivative(response * aggregation(inputs) + self.bias) * aggregation(inputs)
        # dy/d_inputs =  activation.derivative(response * aggregation(inputs) + self.bias) * response * aggregation.derivative(inputs) * 1
        _grad_items = self._grad_items_history[-1]
        inputs = _grad_items['inputs']
        assert inputs is not None and len(inputs) > 0
        _grad_items['gradient']['output'] = grad
        activation_function_def = getattr(config, 'activation_function_def')
        aggregation_function_def = getattr(config, 'aggregation_function_def')
        activation_f = activation_function_def[self.activation]
        aggregation_f = aggregation_function_def[self.aggregation]
        x = activation_f.derivative(self.response * aggregation_f.calc(inputs) + self.bias)
        _grad_items['gradient']['bias'] = x * grad
        _grad_items['gradient']['response'] = x * aggregation_f.calc(inputs) * grad
        _grad_items['gradient']['inputs'] = x * self.response * aggregation_f.derivative(inputs) * grad
        return _grad_items['gradient']['inputs']


class NeuralConnectionGene(DefaultConnectionGene, BaseNeuron):
    _grad_items_history_len = 4
    _grad_items_history = deque(maxlen=_grad_items_history_len)

    def forward(self, inputs: List[float]) -> float:
        # connection gene is 1-1 connection
        assert isinstance(inputs, list), 'input must be {0}, not {1}'.format(list, type(inputs))
        assert len(inputs) > 0, 'input length must not be 0'
        _grad_items = {
            'inputs': inputs[0],
            'gradient': {}
        }
        y = self.weight * inputs[0]
        return y

    def backward(self, grad: float) -> List[float]:
        # y = w * input
        # dy/d_input = w
        _grad_items = self._grad_items_history[-1]
        _grad_items['gradient']['output'] = grad
        _grad_items['gradient']['inputs'] = self.weight * grad
        return _grad_items['gradient']['inputs']
