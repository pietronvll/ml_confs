import json
import dataclasses
import types
from collections import UserDict
from typing import Mapping

allowed_types = (int, float, str, bool, type(None))
allowed_iterables = (list, )

def check_structure(mapping: Mapping) -> bool:
    seen = set()
    for key, value in mapping.items():
        if not isinstance(key, str):
            return False
        if key in seen:
            return False
        seen.add(key)
        if isinstance(value, allowed_types):
            continue
        if isinstance(value, allowed_iterables):
            for item in value:
                if not isinstance(item, allowed_types):
                    return False
            continue
        return False
    return True

class InvalidStructureError(Exception):
    pass

class ConfStorage(UserDict):
    def __init__(self, *args, **kwargs):
        UserDict.__init__(self, *args, **kwargs)
        if not check_structure(self.data):
            error_message = """
                Invalid configuration:
                - keys must be strings
                - values must be one of the following types: int, float, str, bool, None
                - no nested structures are allowed except for lists of the above types
                - no duplicate keys are allowed
            """
            raise InvalidStructureError(error_message)

def make_base_config_class(storage: ConfStorage):
    defaults = {}
    annotations = {}
    for key, value in storage.data.items():
        defaults[key] = dataclasses.field(default=value, metadata={'pytree_node': False})
        annotations[key] = type(value)
    defaults['_storage'] = dataclasses.field(default = storage, metadata={'pytree_node': False})
    annotations['_storage'] = ConfStorage
    def exec_body_callback(ns):
        ns.update(defaults)
        ns['__annotations__'] = annotations
    cls = types.new_class('BaseConfig', (), {}, exec_body_callback)
    return cls

def init_base_config_class(cls, storage: ConfStorage):
    cls._storage = storage
    return cls        