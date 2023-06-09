import os
import json
import yaml
from rich.console import Console
from rich.table import Table
from pathlib import Path
from copy import deepcopy

from ml_confs.config_containers import BaseConfigs, make_base_config_class

def create_base_dir(path: os.PathLike):
    path = Path(path)
    base_path = path.parent
    if not base_path.exists():
        base_path.mkdir(parents=True)

def from_json(path: os.PathLike, flax_dataclass: bool = False):
    with open(path, 'r') as f:
        storage = json.load(f)
    return make_base_config_class(storage, flax_dataclass)

def from_yaml(path: os.PathLike, flax_dataclass: bool = False):
    with open(path, 'r') as f:
        storage = yaml.safe_load(f)
    return make_base_config_class(storage, flax_dataclass)

def from_dict(storage: dict, flax_dataclass: bool = False):
    storage = deepcopy(storage)
    return make_base_config_class(storage, flax_dataclass)

def from_file(path: os.PathLike, flax_dataclass: bool = False):
    path = str(path)
    if path.endswith('.json'):
        return from_json(path, flax_dataclass)
    elif path.endswith('.yaml') or path.endswith('.yml'):
        return from_yaml(path, flax_dataclass)
    else:
        raise ValueError('File extension must be one of: .json, .yaml, .yml')

def to_json(path: os.PathLike, configs: BaseConfigs):
    path = str(path)
    assert path.endswith('.json'), 'File extension must be .json'
    create_base_dir(path)
    with open(path, 'w') as f:
        json.dump(configs._storage, f, indent=4)

def to_yaml(path: os.PathLike, configs: BaseConfigs):
    path = str(path)
    assert path.endswith('.yaml') or path.endswith('.yml'), 'File extension must be .yaml or .yml'
    create_base_dir(path)
    with open(path, 'w') as f:
        yaml.safe_dump(configs._storage, f)

def to_file(path: os.PathLike, configs: BaseConfigs):
    path = str(path)
    if path.endswith('.json'):
        to_json(path, configs)
    elif path.endswith('.yaml') or path.endswith('.yml'):
        to_yaml(path, configs)
    else:
        raise ValueError('File extension must be one of: .json, .yaml, .yml')

def to_dict(configs: BaseConfigs):
    return configs._storage

def pprint(configs: BaseConfigs):
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