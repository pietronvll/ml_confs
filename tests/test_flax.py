from typing import Any
import pytest
import jax
import jax.numpy as jnp
import flax.linen as nn
import ml_confs as mlc
from pathlib import Path

tests_path = Path(__file__).parent
configs_path = tests_path / 'test_configs/flax_test_conf.yaml'

class PositionalEmbedding(nn.Module):
    sequence_length: int
    min_scale: float = 1.0
    max_scale: float = 10000.0

    @nn.compact
    def __call__(self, x):
        num_features = x.shape[-1]
        pe = jnp.zeros((self.sequence_length, num_features))
        positions = jnp.arange(0, self.sequence_length)[:, None]
        scale_factor = -jnp.log(self.max_scale / self.min_scale) / (num_features // 2 - 1)
        div_term = self.min_scale * jnp.exp(jnp.arange(0, num_features // 2) * scale_factor)
        pe = pe.at[:, :num_features // 2].set(jnp.sin(positions * div_term))
        pe = pe.at[:, num_features // 2: 2 * (num_features // 2)].set(jnp.cos(positions * div_term))
        return pe[None, :, :] + x

class Block(nn.Module):
    num_heads: int
    qkv_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        out_features = inputs.shape[-1]
        x = nn.LayerNorm()(inputs)
        x = nn.SelfAttention(self.num_heads, qkv_features = self.qkv_dim, deterministic=True)(x)
        x = x + inputs
        x = nn.Sequential([
                nn.LayerNorm(),
                nn.Dense(4*out_features),
                nn.gelu,
                nn.Dense(out_features),
                nn.Dropout(self.dropout_rate, deterministic = not train)
            ])(x)
        return x + inputs

class Transformer(nn.Module):
    configs: mlc.BaseConfigs
    @nn.compact
    def __call__(self, x, train: bool = False):
        x = PositionalEmbedding(sequence_length=x.shape[1])(x)
        for _ in range(self.configs.num_blocks):
            x = Block(self.configs.num_heads, self.configs.qkv_dim, self.configs.dropout_rate)(x, train = train)
        x = nn.gelu(x)
        x = x.reshape((-1, x.shape[1]*x.shape[2]))
        x = nn.Dense(self.configs.output_dim)(x)
        return x

def make_mock_data():
    return jnp.ones((1, 10, 10), jnp.float32)

def test_transformer_init():
    configs = mlc.from_file(configs_path)
    model = Transformer(configs)
    x = make_mock_data()
    model.init(jax.random.PRNGKey(0), x)

def test_transformer_call():
    configs = mlc.from_file(configs_path)
    model = Transformer(configs)
    variables = model.init(jax.random.PRNGKey(0), make_mock_data())
    apply_fn = jax.jit(model.apply, static_argnames=('train',))
    apply_fn(variables, make_mock_data(), train = True, rngs={"dropout": jax.random.PRNGKey(0)})
    