import dataclasses
import types
from collections.abc import Mapping
from copy import deepcopy

allowed_types = (int, float, str, bool, type(None))
allowed_iterables = (list, )

class InvalidStructureError(Exception):
    pass

class BaseConfigs(Mapping):
    def __getitem__(self, key):
        return self._storage[key]
    def __iter__(self):
        return iter(self._storage)
    def __len__(self):
        return len(self._storage)
    def __contains__(self, key):
        return key in self._storage

def check_structure(mapping: Mapping):
    seen = set()
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise InvalidStructureError('Keys must be strings')
        if key in seen:
            raise InvalidStructureError('Duplicate keys are not allowed')
        seen.add(key)
        if isinstance(value, allowed_types):
            continue
        if isinstance(value, allowed_iterables):
            seen_types = set()
            for item in value:
                if not isinstance(item, allowed_types):
                    raise InvalidStructureError('Element types must be one of: {}'.format(allowed_types))
                seen_types.add(type(item))
            if len(seen_types) > 1:
                raise InvalidStructureError('Lists must be homogenous')
            continue
        raise InvalidStructureError('Element types must be one of: {}'.format(allowed_types))  

def make_base_config_class(storage: dict, flax_dataclass: bool = False):
    check_structure(storage)
    defaults = {}
    annotations = {}
    for key, value in storage.items():
        annotations[key] = type(value)
    annotations['_storage'] = dict
    def exec_body_callback(ns):
        ns.update(defaults)
        ns['__annotations__'] = annotations
    cls = types.new_class('Configs', (BaseConfigs,), {}, exec_body_callback)
    if flax_dataclass:
        try:
            from flax import struct
            cls = struct.dataclass(cls)
        except ImportError:
            print('Warning: flax is not installed, falling back to dataclasses')
    else:
        cls = dataclasses.dataclass(cls, frozen=True)
    storage['_storage'] = deepcopy(storage)
    return cls(**storage)
