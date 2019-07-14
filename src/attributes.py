import random
from base_classes import BaseAttr
from utils import clamp


class FloatAttr(BaseAttr):
    _config_items = {
        'init_type': [str, None],
        'default_value': [float, None],
        'min_value': [float, None],
        'max_value': [float, None],
        'mean': [float, None],
        'stdev': [float, None],
        'replace_rate': [float, None],
        'mutate_rate': [float, None],
        'mutate_power': [float, None],
    }

    def init_value(self, config: object) -> None:
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
        elif init_type == 'uniform':
            min_value = self.get_config_attr(config, 'min_value')
            max_value = self.get_config_attr(config, 'max_value')
            return random.uniform(min_value, max_value)
        raise RuntimeError('{0}: init_type {1} not recognized'.format(self.__class__, init_type))


x = FloatAttr('weight', init_mean=43.435)
y = {
    'weight_init_type': 'normal',
    'weight_mean': 0.,
    'weight_stdev': 1.,
    'weight_max_value': 2.,
    'weight_min_value': -1.
}
from collections import namedtuple
yp = namedtuple('config', y.keys())(*y.values())
print(x.init_value(yp))
# print(vars(x))
