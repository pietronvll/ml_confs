import os
import json
import yaml
from rich.console import Console
from rich.table import Table
from pathlib import Path
from copy import deepcopy
import re

from ml_confs.config_containers import Configs, make_base_config_class

#Fix for yaml scientific notation https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def create_base_dir(path: os.PathLike):
    path = Path(path)
    base_path = path.parent
    if not base_path.exists():
        base_path.mkdir(parents=True)

def from_msgpack(path: os.PathLike, register_jax_pytree: bool = False):
    raise NotImplementedError

def from_json(path: os.PathLike, register_jax_pytree: bool = False):
    """Load configurations from a JSON file.

    Args:
        path (os.PathLike): Configuration file path.
        register_jax_pytree (bool, optional): Register the configuration as a `JAX` pytree. This allows the configurations to be safely used in `JAX`'s transformations.. Defaults to False.

    Returns:
        Configs: Instance of the loaded configurations.
    """
    with open(path, 'r') as f:
        storage = json.load(f)
    return make_base_config_class(storage, register_jax_pytree)

def from_yaml(path: os.PathLike, register_jax_pytree: bool = False):
    """Load configurations from a YAML file.

    Args:
        path (os.PathLike): Configuration file path.
        register_jax_pytree (bool, optional): Register the configuration as a `JAX` pytree. This allows the configurations to be safely used in `JAX`'s transformations.. Defaults to False.

    Returns:
        Configs: Instance of the loaded configurations.
    """    
    with open(path, 'r') as f:
        storage = yaml.load(f, Loader=loader)
    return make_base_config_class(storage, register_jax_pytree)

def from_dict(storage: dict, register_jax_pytree: bool = False):
    """Load configurations from a python dictionary.

    Args:
        storage (dict): Configuration dictionary.
        register_jax_pytree (bool, optional): Register the configuration as a `JAX` pytree. This allows the configurations to be safely used in `JAX`'s transformations.. Defaults to False.

    Returns:
        Configs: Instance of the loaded configurations.
    """
    storage = deepcopy(storage)
    return make_base_config_class(storage, register_jax_pytree)

def from_file(path: os.PathLike, register_jax_pytree: bool = False):
    """Load configurations from a YAML/JSON file.

    Args:
        path (os.PathLike): Configuration file path.
        register_jax_pytree (bool, optional): Register the configuration as a `JAX` pytree. This allows the configurations to be safely used in `JAX`'s transformations.. Defaults to False.

    Returns:
        Configs: Instance of the loaded configurations.
    """
    path = str(path)
    if path.endswith('.json'):
        return from_json(path, register_jax_pytree)
    elif path.endswith('.yaml') or path.endswith('.yml'):
        return from_yaml(path, register_jax_pytree)
    else:
        raise ValueError('File extension must be one of: .json, .yaml, .yml')

def to_json(path: os.PathLike, configs: Configs):
    """Save configurations to a JSON file.

    Args:
        path (os.PathLike): File path to save the configurations.
        configs (Configs): Instance of the configurations.
    """    
    path = str(path)
    assert path.endswith('.json'), 'File extension must be .json'
    create_base_dir(path)
    with open(path, 'w') as f:
        json.dump(configs._storage, f, indent=4)

def to_yaml(path: os.PathLike, configs: Configs):
    """Save configurations to a YAML file.

    Args:
        path (os.PathLike): File path to save the configurations.
        configs (Configs): Instance of the configurations.
    """
    path = str(path)
    assert path.endswith('.yaml') or path.endswith('.yml'), 'File extension must be .yaml or .yml'
    create_base_dir(path)
    with open(path, 'w') as f:
        yaml.safe_dump(configs._storage, f)

def to_file(path: os.PathLike, configs: Configs):
    """Save configurations to a YAML/JSON file.

    Args:
        path (os.PathLike): File path to save the configurations.
        configs (Configs): Instance of the configurations.
    """
    path = str(path)
    if path.endswith('.json'):
        to_json(path, configs)
    elif path.endswith('.yaml') or path.endswith('.yml'):
        to_yaml(path, configs)
    else:
        raise ValueError('File extension must be one of: .json, .yaml, .yml')

def to_dict(configs: Configs) -> dict:
    """Export configurations to a python dictionary.

    Args:
        configs (Configs): Instance of the configurations.

    Returns:
        dict: A standard python dictionary containing the configurations.
    """    
    return configs._storage

def to_msgpack(path: os.PathLike, configs: Configs):
    raise NotImplementedError

def pprint(configs: Configs):
    """Pretty print configurations.

    Args:
        configs (Configs): An instance of the configurations.
    """    
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Key")
    table.add_column("Value")
    table.add_column("Type")

    for key, value in configs._storage.items():
        if isinstance(value, list):
            value_str = '['
            value_str += ', '.join(str(item) for item in value)
            value_str += ']'
            value_type = r"list\[" + f"{type(value[0]).__name__}]"            
        else:
            value_str = str(value)
            value_type = type(value).__name__
        
        table.add_row(str(key), value_str, value_type)
    console.print(table)