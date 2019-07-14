import random
from typing import Any, Tuple
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


if __name__ == '__main__':
    x = DefaultNodeGene(111213)
    print(x.key)
    y = DefaultConnectionGene((11, 32))
    print(y.key)