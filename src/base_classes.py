from abc import ABC, abstractmethod


class BaseAttr(ABC):
    _config_items = {}

    def __init__(self, name: str, **default_dict):
        self.name = name
        for config_name, config_default_val in default_dict.items():
            if config_name in self._config_items:
                self._config_items[config_name] = [self._config_items[config_name][0], config_default_val]
        for config_name in self._config_items.keys():
            setattr(self, '{0}_{1}'.format(config_name, 'name'), '{0}_{1}'.format(self.name, config_name))

