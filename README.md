# ML configurations
A small, highly opinionated `python` tool to handle configurations for machine learning pipelines.
The library is designed to load configurations from both `json` and `yaml` files, as well as from standard python dictionaries.
## Design rules
The configurations, once loaded are frozen. Each configuration file can contain only `int`, `float`, `str`, `bool` and `None` fields, as well as _homogeneous_ lists of one of the same types. That's all. No nested structures are allowed.
## Installation
ML configurations can be installed directly from `git` by running
```
pip install ml-confs
```

## Basic usage
A valid `ml_confs` configuration file `configs.yml` in YAML is:
```yaml
int_field: 1
float_field: 1.0
str_field: 'string'
bool_field: true
none_field: null
list_field: [1, 2, 3]
```
To load it we just use:
```python
import ml_confs as mlcfg

#Loading configs
configs = mlcfg.from_file('configs.yml')

#Accessing configs with dot notation
print(configs.int_field) # >>> 1

#Additionally, one can use the ** notation to unpack the configurations
def foo(**kwargs):
    # Do stuff...
foo(**configs)


#Saving configs to json format
mlcfg.to_file('json_configs_copy.json') #Will create a .json file 
```

One can also pretty print a loaded configuration with `ml_confs.pprint`, which in the previous example would output:
```
┏━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Key         ┃ Value     ┃ Type      ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ int_field   │ 1         │ int       │
│ float_field │ 1.0       │ float     │
│ str_field   │ string    │ str       │
│ bool_field  │ True      │ bool      │
│ none_field  │ None      │ NoneType  │
│ list_field  │ [1, 2, 3] │ list[int] │
└─────────────┴───────────┴───────────┘
```
## API reference
<!-- markdownlint-disable -->
<a href="https://github.com/Pietronvll/ml_confs/tree/main/ml_confs/io_utils.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `from_json`

```python
from_json(path: PathLike, flax_dataclass: bool = False)
```

Load configurations from a JSON file. 



**Args:**
 
 - <b>`path`</b> (os.PathLike):  Configuration file path. 
 - <b>`flax_dataclass`</b> (bool, optional):  Returns a flax compatible object. Uses flax.struct.dataclass. Defaults to False. 



**Returns:**
 
 - <b>`Configs`</b>:  Instance of the loaded configurations. 


---

<a href="https://github.com/Pietronvll/ml_confs/tree/main/ml_confs/io_utils.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `from_yaml`

```python
from_yaml(path: PathLike, flax_dataclass: bool = False)
```

Load configurations from a YAML file. 



**Args:**
 
 - <b>`path`</b> (os.PathLike):  Configuration file path. 
 - <b>`flax_dataclass`</b> (bool, optional):  Returns a flax compatible object. Uses flax.struct.dataclass. Defaults to False. 



**Returns:**
 
 - <b>`Configs`</b>:  Instance of the loaded configurations. 


---

<a href="https://github.com/Pietronvll/ml_confs/tree/main/ml_confs/io_utils.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `from_dict`

```python
from_dict(storage: dict, flax_dataclass: bool = False)
```

Load configurations from a python dictionary. 



**Args:**
 
 - <b>`storage`</b> (dict):  Configuration dictionary. 
 - <b>`flax_dataclass`</b> (bool, optional):  Returns a flax compatible object. Uses flax.struct.dataclass. Defaults to False. 



**Returns:**
 
 - <b>`Configs`</b>:  Instance of the loaded configurations. 


---

<a href="https://github.com/Pietronvll/ml_confs/tree/main/ml_confs/io_utils.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `from_file`

```python
from_file(path: PathLike, flax_dataclass: bool = False)
```

Load configurations from a YAML/JSON file. 



**Args:**
 
 - <b>`path`</b> (os.PathLike):  Configuration file path. 
 - <b>`flax_dataclass`</b> (bool, optional):  Returns a flax compatible object. Uses flax.struct.dataclass. Defaults to False. 



**Returns:**
 
 - <b>`Configs`</b>:  Instance of the loaded configurations. 


---

<a href="https://github.com/Pietronvll/ml_confs/tree/main/ml_confs/io_utils.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_json`

```python
to_json(path: PathLike, configs: BaseConfigs)
```

Save configurations to a JSON file. 



**Args:**
 
 - <b>`path`</b> (os.PathLike):  File path to save the configurations. 
 - <b>`configs`</b> (BaseConfigs):  Instance of the configurations. 


---

<a href="https://github.com/Pietronvll/ml_confs/tree/main/ml_confs/io_utils.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_yaml`

```python
to_yaml(path: PathLike, configs: BaseConfigs)
```

Save configurations to a YAML file. 



**Args:**
 
 - <b>`path`</b> (os.PathLike):  File path to save the configurations. 
 - <b>`configs`</b> (BaseConfigs):  Instance of the configurations. 


---

<a href="https://github.com/Pietronvll/ml_confs/tree/main/ml_confs/io_utils.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_file`

```python
to_file(path: PathLike, configs: BaseConfigs)
```

Save configurations to a YAML/JSON file. 



**Args:**
 
 - <b>`path`</b> (os.PathLike):  File path to save the configurations. 
 - <b>`configs`</b> (BaseConfigs):  Instance of the configurations. 


---

<a href="https://github.com/Pietronvll/ml_confs/tree/main/ml_confs/io_utils.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_dict`

```python
to_dict(configs: BaseConfigs) → dict
```

Export configurations to a python dictionary. 



**Args:**
 
 - <b>`configs`</b> (BaseConfigs):  Instance of the configurations. 



**Returns:**
 
 - <b>`dict`</b>:  A standard python dictionary containing the configurations. 


---

<a href="https://github.com/Pietronvll/ml_confs/tree/main/ml_confs/io_utils.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `pprint`

```python
pprint(configs: BaseConfigs)
```

Pretty print configurations. 



**Args:**
 
 - <b>`configs`</b> (BaseConfigs):  An instance of the configurations. 




---

_The API reference was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
