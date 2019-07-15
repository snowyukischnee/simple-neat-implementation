import random
from typing import Any
from abc import ABC, abstractmethod
from utils import clamp, without_keys
import copy

class BaseAttr(ABC):
    @property
    @abstractmethod
    def _config_items(self):
        raise NotImplementedError

    def __init__(self, name: str, **default_dict):
        self.name = name
        self._config_items = copy.copy(self.__class__._config_items)
        for attr_name, attr_default_val in default_dict.items():
            if attr_name in self._config_items:
                self._config_items[attr_name] = [self._config_items[attr_name][0], attr_default_val]

    def get_config_attr(self, config: object, attr_name: str, nullable: bool = False):
        config_attr = self._config_items.get(attr_name)
        if config_attr is None:
            raise RuntimeError('{0}: "{1}_{2}" not exist in _config_items'.format(self.__class__, self.name, attr_name))
        config_attr_type, config_attr_default_val = config_attr
        value = getattr(config, '{0}_{1}'.format(self.name, attr_name), config_attr_default_val)
        if nullable:
            if (value is None) or (isinstance(value, config_attr_type)):
                return value
            else:
                raise RuntimeError('{0}: "{1}_{2}" has invalid type: expected {3}, got {4}'.format(self.__class__, self.name, attr_name, config_attr_type, type(value)))
        else:
            if isinstance(value, config_attr_type):
                return value
            elif value is None:
                raise RuntimeError('{0}: "{1}_{2}" not exist in config'.format(self.__class__, self.name, attr_name))
            else:
                raise RuntimeError('{0}: "{1}_{2}" has invalid type: expected {3}, got {4}'.format(self.__class__, self.name, attr_name, config_attr_type, type(value)))

    @abstractmethod
    def init_value(self, config: object) -> Any:
        pass

    @abstractmethod
    def mutate_value(self, value: Any, config: object) -> Any:
        pass


class FloatAttr(BaseAttr):
    _config_items = {
        'init_type': [str, None],
        'default_value': [float, None],
        'min_value': [float, None],
        'max_value': [float, None],
        'mean': [float, None],
        'stdev': [float, None],
        'replace_rate': [float, None],
        'mutation_rate': [float, None],
        'mutation_power': [float, None],
    }

    def init_value(self, config: object) -> float:
        init_type = self.get_config_attr(config, 'init_type', nullable=True)
        if init_type is None:
            default_value = self.get_config_attr(config, 'default_value')
            return default_value
        elif any(it in init_type for it in ['normal', 'gauss']):
            mean = self.get_config_attr(config, 'mean')
            stdev = self.get_config_attr(config, 'stdev')
            min_value = self.get_config_attr(config, 'min_value')
            max_value = self.get_config_attr(config, 'max_value')
            return clamp(random.gauss(mean, stdev), min_value, max_value)
        elif any(it in init_type for it in ['uniform', 'random']):
            min_value = self.get_config_attr(config, 'min_value')
            max_value = self.get_config_attr(config, 'max_value')
            return random.uniform(min_value, max_value)
        raise RuntimeError('{0}: init_type {1} not recognized'.format(self.__class__, init_type))

    def mutate_value(self, value: float, config: object) -> float:
        mutation_rate = self.get_config_attr(config, 'mutation_rate')
        if random.random() < mutation_rate:
            mutation_power = self.get_config_attr(config, 'mutation_power')
            min_value = self.get_config_attr(config, 'min_value')
            max_value = self.get_config_attr(config, 'max_value')
            return clamp(random.gauss(value, mutation_power), min_value, max_value)
        replace_rate = self.get_config_attr(config, 'replace_rate')
        if random.random() < replace_rate:
            return self.init_value(config)
        return value


class BoolAttr(BaseAttr):
    _config_items = {
        'init_type': [str, None],
        'default_value': [bool, True],
        'mutation_type': [str, None],
        'mutation_rate': [float, None],
        'value_mutation_rate': [dict, {False: 0.5, True: 0.5}]  # the mutation rate has to be in order
    }

    def init_value(self, config: object) -> bool:
        init_type = self.get_config_attr(config, 'init_type', nullable=True)
        if init_type is None:
            default_value = self.get_config_attr(config, 'default_value')
            return default_value
        elif init_type == 'random':
            return random.random() < 0.5
        raise RuntimeError('{0}: init_type {1} not recognized'.format(self.__class__, init_type))

    def mutate_value(self, value: bool, config: object) -> bool:
        value_mutation_rate = self.get_config_attr(config, 'value_mutation_rate')
        mutation_rate = self.get_config_attr(config, 'mutation_rate')
        if random.random() < mutation_rate:
            mutation_type = self.get_config_attr(config, 'mutation_type', nullable=True)
            if mutation_type is None:
                probs_dict = without_keys(value_mutation_rate, [value])
                for val, prob in probs_dict.items():
                    if random.random() < prob:
                        return val
                return value
            elif any(mt in mutation_type for mt in ['uniform', 'random']):
                return random.choice(list(value_mutation_rate.keys()))
            else:
                RuntimeError('{0}: mutation_type {1} not recognized'.format(self.__class__, mutation_type))
        return value


class StringAttr(BaseAttr):
    _config_items = {
        'init_type': [str, None],
        'default_value': [str, None],
        'mutation_type': [str, None],
        'mutation_rate': [float, None],
        'value_mutation_rate': [dict, None]
    }

    def init_value(self, config: object) -> str:
        init_type = self.get_config_attr(config, 'init_type', nullable=True)
        if init_type is None:
            default_value = self.get_config_attr(config, 'default_value')
            return default_value
        elif init_type == 'random':
            value_mutation_rate = self.get_config_attr(config, 'value_mutation_rate')
            return random.choice(list(value_mutation_rate.keys()))
        raise RuntimeError('{0}: init_type {1} not recognized'.format(self.__class__, init_type))

    def mutate_value(self, value: str, config: object) -> str:
        value_mutation_rate = self.get_config_attr(config, 'value_mutation_rate')
        mutation_rate = self.get_config_attr(config, 'mutation_rate')
        if random.random() < mutation_rate:
            mutation_type = self.get_config_attr(config, 'mutation_type', nullable=True)
            if mutation_type is None:
                probs_dict = without_keys(value_mutation_rate, [value])
                for val, prob in probs_dict.items():
                    if random.random() < prob:
                        return val
                return value
            elif any(mt in mutation_type for mt in ['uniform', 'random']):
                return random.choice(list(value_mutation_rate.keys()))
            else:
                RuntimeError('{0}: mutation_type {1} not recognized'.format(self.__class__, mutation_type))
        return value


if __name__ == '__main__':
    x = FloatAttr('weight', default_value=43.435)
    y = {
        'weight_init_type': 'normal',
        'weight_default_value': 1.,
        'weight_mean': 0.,
        'weight_stdev': 1.,
        'weight_max_value': 2.,
        'weight_min_value': -1.,
        'weight_mutation_power': 1.,
        'weight_mutation_rate': 0.6,
        'weight_replace_rate': 0.2
    }
    from collections import namedtuple
    yp = namedtuple('config', y.keys())(*y.values())
    print(x.init_value(yp), x.mutate_value(1, yp))
    print(x._config_items['default_value'])
    x1 = FloatAttr('weight_1111')
    print(x1._config_items['default_value'])

    # x = BoolAttr('conn', init_mean=43.435)
    # y = {
    #     'conn_init_type': None,
    #     'conn_default_value': False,
    #     'conn_mutation_rate': 0.6,
    #     'conn_mutation_type': 'random',
    #     'conn_value_mutation_rate': {False: 0.1, True: 0.5}
    # }
    # from collections import namedtuple
    # yp = namedtuple('config', y.keys())(*y.values())
    # print(x.init_value(yp), x.mutate_value(True, yp))
    #
    # x = StringAttr('agg', init_mean=43.435)
    # y = {
    #     'agg_init_type': 'random',
    #     'agg_default_value': 'dd',
    #     'agg_mutation_rate': 1.,
    #     'agg_mutation_type': None,
    #     'agg_value_mutation_rate': {
    #         'test0': 0.1,
    #         'test1': 0.5,
    #         'test4': 0.5
    #     }
    # }
    # from collections import namedtuple
    # yp = namedtuple('config', y.keys())(*y.values())
    # print(x.init_value(yp), x.mutate_value('dd', yp))
