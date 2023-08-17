import dataclasses
import types
from collections.abc import Mapping
from copy import deepcopy

allowed_types = (int, float, str, bool, type(None))
allowed_iterables = (list, )

class InvalidStructureError(Exception):
    pass

class Configs(Mapping):
    def __getitem__(self, key):
        return self._storage[key]
    def __iter__(self):
        return iter(self._storage)
    def __len__(self):
        return len(self._storage)
    def __contains__(self, key):
        return key in self._storage
    def __eq__(self, other):
        if not isinstance(other, Configs):            
            return False
        return (self._storage == other._storage) and (self._is_jax_pytree == other._is_jax_pytree)
    
    #JAX pytree compatibility
    def tree_flatten(self):
        return tuple(self._storage.values()), tuple(self._storage.keys()) + (self._is_jax_pytree,)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        #Pop last element from aux_data
        _is_jax_pytree = aux_data[-1]
        aux_data = aux_data[:-1]
        storage = dict(zip(aux_data, children))
        return make_base_config_class(storage, _is_jax_pytree)

def check_structure(mapping: Mapping, _jax_tracer: bool = False):
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

        if _jax_tracer:
            try:
                from jax.core import Tracer
            except ImportError:
                raise ImportError('The argument `_jax_tracer` is not supported without JAX installed')
            if isinstance(value, Tracer):
                continue
        error_str = f"The element {key} is of type {type(value)} while it must be one of:\n"
        for t in allowed_types:
            error_str += f"\t{t.__name__}\n"
        raise InvalidStructureError(error_str)  

def make_base_config_class(storage: dict, register_jax_pytree: bool = False):
    check_structure(storage, register_jax_pytree)
    defaults = {}
    annotations = {}
    for key, value in storage.items():
        annotations[key] = type(value)
    annotations['_storage'] = dict
    annotations['_is_jax_pytree'] = bool
    def exec_body_callback(ns):
        ns.update(defaults)
        ns['__annotations__'] = annotations
    cls = types.new_class('LoadedConfigs', (Configs,), {}, exec_body_callback)
    cls = dataclasses.dataclass(cls, frozen=True, eq=False)
    storage['_storage'] = deepcopy(storage)
    if register_jax_pytree:
        try:
            from jax.tree_util import register_pytree_node_class
            cls = register_pytree_node_class(cls)
            storage['_is_jax_pytree'] = True
        except ImportError:
            print('Warning: unable to import JAX. The argument `register_jax_pytree` will be ignored.')
    else:
        storage['_is_jax_pytree'] = False
    return cls(**storage)
