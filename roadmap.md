# Roadmap for `ml_confs`
The plan is to add functionality to be `JAX`-compatible.

**[Aug 17]**

The `pytree` registration functionality has been implemented and partially tested. I do not like too much how the using JAX/ not using JAX is handled, make it a better way.

(Later): Added a standalone `PytreeConfigs` subclass to handle the JAX registered configs. Tests still pass.

Aim for 100% code coverage. (How to test against optional imports?)


**[BASE]**

1. Check if `flax`'s dataclasses are enough to support `pytree` registration, and to use configurations inside `jax.jit` and `jax.vmap`.
2. Wrap the `to_dict` and `to_file` functions as instance methods.
3. Wrap or rename `BaseConfigs` to `Configs` for nicer typing suggestions
