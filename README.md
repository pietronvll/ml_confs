# ML configurations
A small, highly opinionated library to handle configurations used in machine learning pipelines.
The library can load configurations from both `json`, `yaml` and standard python dictionaries.
## Design rules
The configurations, once loaded are frozen. Each configuration file can contain only `int`, `float`, `str`, `bool` and `None` fields, as well as _homogeneous_ lists of one of the same types. That is it. No nested structures are allowed.
## Installation
ML configurations can be installed directly from `git` by running
```
pip install git+https://github.com/Pietronvll/ml_confs.git
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