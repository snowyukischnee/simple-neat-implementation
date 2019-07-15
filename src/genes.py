import random
from typing import Any, Tuple, List
from abc import ABC, abstractmethod
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
    # 'inputs' contain the forward inputs
    # 'gradient' contain the gradient wrt. all the attributes of this neuron
    _grad_items = {
        'inputs': [list, None],
        'gradient': [dict, {}]
    }

    @abstractmethod
    def forward(self, inputs: List[float]) -> float:
        pass

    @abstractmethod
    def backward(self, grad: float) -> List[float]:
        pass

    def clear_grads(self) -> None:
        self._grad_items['inputs'][1] = None
        self._grad_items['gradient'][1] = {}


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
    def forward(self, inputs: List[float]) -> float:
        assert isinstance(inputs, self._grad_items['inputs'][0]), 'input must be {0}, not {1}'.format(self._grad_items['inputs'][0], type(inputs))
        assert len(inputs) > 0, 'input length must not be 0'
        self._grad_items['inputs'][1] = inputs
        y = self.activation(self.response * self.aggregation(inputs) + self.bias)
        return y

    def backward(self, grad: float) -> List[float]:
        # y = activation(response * aggregation(inputs) + self.bias)
        # dy/d_bias = activation.derivative(response * aggregation(inputs) + self.bias) * 1
        # dy/d_response = activation.derivative(response * aggregation(inputs) + self.bias) * aggregation(inputs)
        # dy/d_inputs =  activation.derivative(response * aggregation(inputs) + self.bias) * response * aggregation.derivative(inputs) * 1
        inputs = self._grad_items['inputs'][1]
        assert inputs is not None and len(inputs) > 0
        x = self.activation.derivative(self.response * self.aggregation(inputs) + self.bias)
        self._grad_items['gradient'][1]['bias'] = x * grad
        self._grad_items['gradient'][1]['response'] = x * self.aggregation(inputs) * grad
        self._grad_items['gradient'][1]['inputs'] = x * self.response * self.aggregation.derivative(inputs) * grad
        return self._grad_items['gradient'][1]['inputs']


class NeuralConnectionGene(DefaultConnectionGene, BaseNeuron):
    def forward(self, inputs: List[float]) -> float:
        # connection gene is 1-1 connection
        assert isinstance(inputs, self._grad_items['inputs'][0]), 'input must be {0}, not {1}'.format(self._grad_items['inputs'][0], type(inputs))
        assert len(inputs) > 0, 'input length must not be 0'
        self._grad_items['inputs'][1] = inputs[0]
        y = self.weight * inputs[0]
        return y

    def backward(self, grad: float) -> List[float]:
        # y = w * input
        # dy/d_input = w
        self._grad_items['gradient'][1]['inputs'] = self.weight * grad
        return self._grad_items['gradient'][1]['inputs']
