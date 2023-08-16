# Roadmap for `ml_confs`
The plan is to add functionality to be `JAX`-compatible.

1. Check if `flax`'s dataclasses are enough to support `pytree` registration, and to use configurations inside `jax.jit` and `jax.vmap`.
2. Wrap the `to_dict` and `to_file` functions as instance methods.
3. Wrap or rename `BaseConfigs` to `Configs` for nicer typing suggestions
