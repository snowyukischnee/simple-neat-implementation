from abc import ABC, abstractmethod


class BaseAttr(ABC):
    _config_items = {}

    def __init__(self, name: str, **default_dict):
        self.name = name
        for attr_name, attr_default_val in default_dict.items():
            if attr_name in self._config_items:
                self._config_items[attr_name] = [self._config_items[attr_name][0], attr_default_val]

    def get_config_attr(self, config: object, attr_name: str, nullable: bool = False):
        config_attr = self._config_items.get(attr_name)
        if config_attr is None:
            raise RuntimeError('{0}: "{1}" not exist in _config_items'.format(self.__class__, attr_name))
        config_attr_type, config_attr_default_val = config_attr
        value = getattr(config, '{0}_{1}'.format(self.name, attr_name), config_attr_default_val)
        if nullable:
            if (value is None) or (isinstance(value, config_attr_type)):
                return value
            else:
                raise RuntimeError('{0}: "{1}" has invalid type: expected {2}, got {3}'.format(self.__class__, attr_name, config_attr_type, type(value)))
        else:
            if isinstance(value, config_attr_type):
                return value
            elif value is None:
                raise RuntimeError('{0}: "{1}" not exist in config'.format(self.__class__, attr_name))
            else:
                raise RuntimeError('{0}: "{1}" has invalid type: expected {2}, got {3}'.format(self.__class__, attr_name, config_attr_type, type(value)))
