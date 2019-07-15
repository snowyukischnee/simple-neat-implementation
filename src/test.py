import copy
class BaseAttribute(object):
    def __init__(self, name, **default_dict):
        self.name = name
        self._config_items = copy.copy(self.__class__._config_items)
        for n, default in default_dict.items():
            print(n, default)
            self._config_items[n] = [self._config_items[n][0], default]

class FloatAttribute(BaseAttribute):
    _config_items = {"init_mean": [float, None],
                     "init_stdev": [float, None],
                     "init_type": [str, 'gaussian'],
                     "replace_rate": [float, None],
                     "mutate_rate": [float, None],
                     "mutate_power": [float, None],
                     "max_value": [float, None],
                     "min_value": [float, None]}


x = FloatAttribute('w', init_mean=1.4)
y = FloatAttribute('w1')
print(x._config_items['init_mean'][1], y._config_items['init_mean'][1])
x._config_items['init_mean'][1] = 1.55
print(x._config_items['init_mean'][1], y._config_items['init_mean'][1], FloatAttribute._config_items['init_mean'][1])
