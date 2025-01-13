# Heavily inspired by VDM codebase.

from typing import Callable, Optional, Iterable

import chex
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np

import pywt

from pcdm import cascade_utils


######### Config and Output Classes ########
@flax.struct.dataclass
class CDMConfig:
  """VDM configurations."""
  vocab_size: int
  sample_softmax: bool
  antithetic_time_sampling: bool
  with_fourier_features: bool
  with_attention: bool

  # cascaded models
  levels: int
  image_sizes: list
  cascade_map: str

  # configurations of the noise schedule
  gamma_type: str
  gamma_min: float
  gamma_max: float

  # configurations of the score model
  sm_n_timesteps: int
  sm_n_embd: int
  sm_n_layer: int
  sm_pdrop: float
  sm_kernel_init: Callable = jax.nn.initializers.normal(0.02)


######### Latent VDM model #########

@flax.struct.dataclass
class CDMOutput:
  loss_recon: chex.Array  # [B]
  loss_klz: chex.Array  # [B]
  loss_diff: chex.Array  # [B]
  var_0: float
  var_1: float


######### Encoder and decoder #########


class EncDec(nn.Module):
  """Encoder and decoder. """
  config: CDMConfig

  def __call__(self, x, g_0):
    # For initialization purposes
    h = self.encode(x)
    return self.decode(h, g_0)

  def encode(self, x):
    # This transforms x from discrete values (0, 1, ...)
    # to the domain (-1,1).
    # Rounding here just a safeguard to ensure the input is discrete
    # (although typically, x is a discrete variable such as uint8)
    x = x.round()
    return 2 * ((x+.5) / self.config.vocab_size) - 1

  def decode(self, z, g_0):
    config = self.config

    # Logits are exact if there are no dependencies between dimensions of x
    x_vals = jnp.arange(0, config.vocab_size)[:, None]
    x_vals = jnp.repeat(x_vals, 3, 1)
    x_vals = self.encode(x_vals).transpose([1, 0])[None, None, None, :, :]
    inv_stdev = jnp.exp(-0.5 * g_0[..., None])
    logits = -0.5 * jnp.square((z[..., None] - x_vals) * inv_stdev)

    logprobs = jax.nn.log_softmax(logits)
    return logprobs

  def logprob(self, x, z, g_0):
    x = x.round().astype('int32')
    x_onehot = jax.nn.one_hot(x, self.config.vocab_size)
    logprobs = self.decode(z, g_0)
    logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2, 3, 4))
    return logprob


######### Score model #########


class ScoreUNet(nn.Module):
  config: CDMConfig

  @nn.compact
  def __call__(self, z, g_t, conditioning, deterministic=True):
    config = self.config

    # Compute conditioning vector based on 'g_t' and 'conditioning'
    n_embd = self.config.sm_n_embd

    lb = config.gamma_min
    ub = config.gamma_max
    t = (g_t - lb) / (ub - lb)  # ---> [0,1]

    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
    if jnp.isscalar(t):
      t = jnp.ones((z.shape[0],), z.dtype) * t
    elif len(t.shape) == 0:
      t = jnp.tile(t[None], z.shape[0])

    temb = get_timestep_embedding(t, n_embd)
    cond = jnp.concatenate([temb, conditioning[:, None]], axis=1)
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense0')(cond))
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense1')(cond))

    # Concatenate Fourier features to input
    if config.with_fourier_features:
      z_f = Base2FourierFeatures(start=6, stop=8, step=1)(z)
      h = jnp.concatenate([z, z_f], axis=-1)
    else:
      h = z

    # Linear projection of input
    h = nn.Conv(features=n_embd, kernel_size=(
        3, 3), strides=(1, 1), name='conv_in')(h)
    hs = [h]

    # Downsampling
    for i_block in range(self.config.sm_n_layer):
      block = ResnetBlock(config, out_ch=n_embd, name=f'down.block_{i_block}')
      h = block(hs[-1], cond, deterministic)[0]
      if config.with_attention:
        h = AttnBlock(num_heads=1, name=f'down.attn_{i_block}')(h)
      hs.append(h)

    # Middle
    h = hs[-1]
    h = ResnetBlock(config, name='mid.block_1')(h, cond, deterministic)[0]
    h = AttnBlock(num_heads=1, name='mid.attn_1')(h)
    h = ResnetBlock(config, name='mid.block_2')(h, cond, deterministic)[0]

    # Upsampling
    for i_block in range(self.config.sm_n_layer + 1):
      b = ResnetBlock(config, out_ch=n_embd, name=f'up.block_{i_block}')
      h = b(jnp.concatenate([h, hs.pop()], axis=-1), cond, deterministic)[0]
      if config.with_attention:
        h = AttnBlock(num_heads=1, name=f'up.attn_{i_block}')(h)

    assert not hs

    # Predict noise
    normalize = nn.normalization.GroupNorm()
    h = nn.swish(normalize(h))
    eps_pred = nn.Conv(
        features=z.shape[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h)

    # Base measure
    eps_pred += z

    return eps_pred


def get_timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
  """Build sinusoidal embeddings (from Fairseq).

  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".

  Args:
    timesteps: jnp.ndarray: generate embedding vectors at these timesteps
    embedding_dim: int: dimension of the embeddings to generate
    dtype: data type of the generated embeddings

  Returns:
    embedding vectors with shape `(len(timesteps), embedding_dim)`
  """
  assert len(timesteps.shape) == 1
  timesteps *= 1000.

  half_dim = embedding_dim // 2
  emb = np.log(10000) / (half_dim - 1)
  emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
  emb = timesteps.astype(dtype)[:, None] * emb[None, :]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


######### Noise Schedule #########

class NoiseSchedule_Scalar(nn.Module):
  config: CDMConfig

  def setup(self):
    init_bias = self.config.gamma_min
    init_scale = self.config.gamma_max - init_bias
    self.w = self.param('w', constant_init(init_scale), (1,))
    self.b = self.param('b', constant_init(init_bias), (1,))

  @nn.compact
  def __call__(self, t):
    return self.b + abs(self.w) * t


class NoiseSchedule_FixedLinear(nn.Module):
  config: CDMConfig

  @nn.compact
  def __call__(self, t):
    config = self.config
    return config.gamma_min + (config.gamma_max-config.gamma_min) * t


class NoiseSchedule_NNet(nn.Module):
  config: CDMConfig
  n_features: int = 1024
  nonlinear: bool = True

  def setup(self):
    config = self.config

    n_out = 1
    kernel_init = nn.initializers.normal()

    init_bias = self.config.gamma_min
    init_scale = self.config.gamma_max - init_bias

    self.l1 = DenseMonotone(n_out,
                            kernel_init=constant_init(init_scale),
                            bias_init=constant_init(init_bias))
    if self.nonlinear:
      self.l2 = DenseMonotone(self.n_features, kernel_init=kernel_init)
      self.l3 = DenseMonotone(n_out, kernel_init=kernel_init, use_bias=False)

  @nn.compact
  def __call__(self, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((1, 1))
    else:
      t = jnp.reshape(t, (-1, 1))

    h = self.l1(t)
    if self.nonlinear:
      _h = 2. * (t - .5)  # scale input to [-1, +1]
      _h = self.l2(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.l3(_h) / self.n_features
      h += _h

    return jnp.squeeze(h, axis=-1)


def constant_init(value, dtype='float32'):
  def _init(key, shape, dtype=dtype):
    return value * jnp.ones(shape, dtype)
  return _init


class DenseMonotone(nn.Dense):
  """Strictly increasing Dense layer."""

  @nn.compact
  def __call__(self, inputs):
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = self.param('kernel',
                        self.kernel_init,
                        (inputs.shape[-1], self.features))
    kernel = abs(jnp.asarray(kernel, self.dtype))
    y = jax.lax.dot_general(inputs, kernel,
                            (((inputs.ndim - 1,), (0,)), ((), ())),
                            precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y


######### ResNet block #########

class ResnetBlock(nn.Module):
  """Convolutional residual block with two convs."""
  config: CDMConfig
  out_ch: Optional[int] = None

  @nn.compact
  def __call__(self, x, cond, deterministic: bool, enc=None):
    config = self.config

    nonlinearity = nn.swish
    normalize1 = nn.normalization.GroupNorm()
    normalize2 = nn.normalization.GroupNorm()

    if enc is not None:
      x = jnp.concatenate([x, enc], axis=-1)

    B, _, _, C = x.shape  # pylint: disable=invalid-name
    out_ch = C if self.out_ch is None else self.out_ch

    h = x
    h = nonlinearity(normalize1(h))
    h = nn.Conv(
        features=out_ch, kernel_size=(3, 3), strides=(1, 1), name='conv1')(h)

    # add in conditioning
    if cond is not None:
      assert cond.shape[0] == B and len(cond.shape) == 2
      h += nn.Dense(
          features=out_ch, use_bias=False, kernel_init=nn.initializers.zeros,
          name='cond_proj')(cond)[:, None, None, :]

    h = nonlinearity(normalize2(h))
    h = nn.Dropout(rate=config.sm_pdrop)(h, deterministic=deterministic)
    h = nn.Conv(
        features=out_ch,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv2')(h)

    if C != out_ch:
      x = nn.Dense(features=out_ch, name='nin_shortcut')(x)

    assert x.shape == h.shape
    x = x + h
    return x, x


class AttnBlock(nn.Module):
  """Self-attention residual block."""

  num_heads: int

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape  # pylint: disable=invalid-name,unused-variable
    assert C % self.num_heads == 0

    normalize = nn.normalization.GroupNorm()

    h = normalize(x)
    if self.num_heads == 1:
      q = nn.Dense(features=C, name='q')(h)
      k = nn.Dense(features=C, name='k')(h)
      v = nn.Dense(features=C, name='v')(h)
      h = dot_product_attention(
          q[:, :, :, None, :],
          k[:, :, :, None, :],
          v[:, :, :, None, :],
          axis=(1, 2))[:, :, :, 0, :]
      h = nn.Dense(
          features=C, kernel_init=nn.initializers.zeros, name='proj_out')(h)
    else:
      head_dim = C // self.num_heads
      q = nn.DenseGeneral(features=(self.num_heads, head_dim), name='q')(h)
      k = nn.DenseGeneral(features=(self.num_heads, head_dim), name='k')(h)
      v = nn.DenseGeneral(features=(self.num_heads, head_dim), name='v')(h)
      assert q.shape == k.shape == v.shape == (
          B, H, W, self.num_heads, head_dim)
      h = dot_product_attention(q, k, v, axis=(1, 2))
      h = nn.DenseGeneral(
          features=C,
          axis=(-2, -1),
          kernel_init=nn.initializers.zeros,
          name='proj_out')(h)

    assert h.shape == x.shape
    return x + h


def dot_product_attention(query,
                          key,
                          value,
                          dtype=jnp.float32,
                          bias=None,
                          axis=None,
                          # broadcast_dropout=True,
                          # dropout_rng=None,
                          # dropout_rate=0.,
                          # deterministic=False,
                          precision=None):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs.


  Args:
    query: queries for calculating attention with shape of `[batch_size, dim1,
      dim2, ..., dimN, num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
      ..., dimN, num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size, dim1,
      dim2,..., dimN, num_heads, value_channels]`.
    dtype: the dtype of the computation (default: float32)
    bias: bias for the attention weights. This can be used for incorporating
      autoregressive mask, padding mask, proximity bias.
    axis: axises over which the attention is applied.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
  """
  assert key.shape[:-1] == value.shape[:-1]
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1])
  assert query.dtype == key.dtype == value.dtype
  input_dtype = query.dtype

  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
      raise ValueError('Attention axis must be between the batch '
                       'axis and the last-two axes.')
  depth = query.shape[-1]
  n = key.ndim
  # batch_dims is  <bs, <non-attention dims>, num_heads>
  batch_dims = tuple(np.delete(range(n), axis + (n - 1,)))
  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
  qk_perm = batch_dims + axis + (n - 1,)
  key = key.transpose(qk_perm)
  query = query.transpose(qk_perm)
  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  key = key.astype(dtype)
  query = query.astype(dtype) / np.sqrt(depth)
  batch_dims_t = tuple(range(len(batch_dims)))
  attn_weights = jax.lax.dot_general(
      query,
      key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
      precision=precision)

  # apply attention bias: masking, droput, proximity bias, ect.
  if bias is not None:
    attn_weights = attn_weights + bias

  # normalize the attention weights
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims)
  assert attn_weights.dtype == dtype
  attn_weights = attn_weights.astype(input_dtype)

  # compute the new values given the attention weights
  assert attn_weights.dtype == value.dtype
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = jax.lax.dot_general(
      attn_weights,
      value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
      precision=precision)

  # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  assert y.dtype == input_dtype
  return y


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)


class Base2FourierFeatures(nn.Module):
  start: int = 0
  stop: int = 8
  step: int = 1

  @nn.compact
  def __call__(self, inputs):
    freqs = range(self.start, self.stop, self.step)

    # Create Base 2 Fourier features
    w = 2.**(jnp.asarray(freqs, dtype=inputs.dtype)) * 2 * jnp.pi
    w = jnp.tile(w[None, :], (1, inputs.shape[-1]))

    # Compute features
    h = jnp.repeat(inputs, len(freqs), axis=-1)
    h = w * h
    h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)
    return h

######### Wavelet score model #######
class WScoreUNet(nn.Module):
  config: CDMConfig
  level: int

  @nn.compact
  def __call__(self, z, g_t, conditioning, skip, deterministic=True):
    config = self.config

    # Compute conditioning vector based on 'g_t' and 'conditioning'
    n_embd = self.config.sm_n_embd
    n_layer = self.config.sm_n_layer
    if self.level > 0:
        n_embd = n_embd // 2
        n_layer = n_layer // 2

    lb = config.gamma_min
    ub = config.gamma_max
    t = (g_t - lb) / (ub - lb)  # ---> [0,1]

    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
    if jnp.isscalar(t):
      t = jnp.ones((z.shape[0],), z.dtype) * t
    elif len(t.shape) == 0:
      t = jnp.tile(t[None], z.shape[0])

    temb = get_timestep_embedding(t, n_embd)
    cond = jnp.concatenate([temb, conditioning[:, None]], axis=1)
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense0')(cond))
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense1')(cond))

    # Concatenate Fourier features to input
    if config.with_fourier_features:
      z_f = Base2FourierFeatures(start=6, stop=8, step=1)(z)
      h = jnp.concatenate([z, z_f], axis=-1)
    else:
      h = z

    # Linear projection of input
    h = nn.Conv(features=n_embd, kernel_size=(
        3, 3), strides=(1, 1), name='conv_in')(h)
    hs = [h]

    # Downsampling
    for i_block in range(n_layer):
      block = ResnetBlock(config, out_ch=n_embd, name=f'down.block_{i_block}')
      h = block(hs[-1], cond, deterministic)[0]
      if config.with_attention:
        h = AttnBlock(num_heads=1, name=f'down.attn_{i_block}')(h)
      hs.append(h)

    # Middle
    h = hs[-1]
    h = ResnetBlock(config, name='mid.block_1')(h, cond, deterministic)[0]
    h = AttnBlock(num_heads=1, name='mid.attn_1')(h)
    h = ResnetBlock(config, name='mid.block_2')(h, cond, deterministic)[0]

    # Upsampling
    for i_block in range(n_layer + 1):
      b = ResnetBlock(config, out_ch=n_embd, name=f'up.block_{i_block}')
      h = b(jnp.concatenate([h, hs.pop()], axis=-1), cond, deterministic)[0]
      if config.with_attention:
        h = AttnBlock(num_heads=1, name=f'up.attn_{i_block}')(h)

    assert not hs

    # Predict noise
    normalize = nn.normalization.GroupNorm()
    h = nn.swish(normalize(h))
    eps_pred = nn.Conv(
        features=skip.shape[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h)

    # Base measure
    eps_pred += skip

    return eps_pred

######### Wavelet model #######
class Wavelet(nn.Module):
  config: CDMConfig

  def x_to_y_w(self, level, x):
    '''
    Takes image x and level and returns wavelet coefficient y at that level, wavelet conditioning sub-image w,
    and the high frequency remainder hfr
    '''
    Y = self.wavelet_from_image(x)
    skip = x.shape[1] // self.config.image_sizes[level]
    w_level = max(level, 1)
    w = self.wavelet_to_image(Y[:w_level] + [jnp.zeros_like(y) for y in Y[w_level:]])[:,::skip,::skip,:]
    y = Y[level] if level > 0 else None

    assert level == 0 or y.shape[1] == self.config.image_sizes[level], f"skip: {skip}, level: {level}, y.shape[1]: {y.shape[1] if y is not None else 'y is None'}, self.config.image_sizes[level]:{self.config.image_sizes[level]}, x: {x.shape}"
    assert w.shape[1] == self.config.image_sizes[level], f"level: {level}, w: {w.shape}, image_sizes: {self.config.image_sizes}"

    return y, w, Y[level + 1:]

  def x_from_y_w(self, level, y=None, w=None, hfr=None):
    size = np.sum(self.config.image_sizes)
    if w is not None:
        x = jax.image.resize(w, shape=(w.shape[0], size, size, w.shape[-1]), method='nearest')
    else:
        x = jnp.zeros(shape=(y.shape[0], size, size, y.shape[-1] // 3))

    Y = self.wavelet_from_image(x)

    if y is not None:
        Y[level] = y

    if hfr is not None:
        Y[level + 1:] = hfr

    return self.wavelet_to_image(Y)

  def wavelet_from_image(self, x):
    n = len(x)

    # functions for (un-)permuting and (un-)combining channel dim with batch dim
    shaper = lambda x: jnp.transpose(x, [0, 3, 1, 2]).reshape((-1, *x.shape[1:3]))
    unshaper = lambda x: jnp.transpose(x.reshape((n, -1, *x.shape[1:])), [0, 2, 3, 1])

    # wavelet transform
    Y = cascade_utils.wavedec2(shaper(x), pywt.Wavelet("haar"), level=self.config.levels - 1, mode="zero")

    Y[1:] = [jnp.concatenate([unshaper(y) for y in ys], axis=3) for ys in Y[1:]]
    Y[0] = unshaper(Y[0])

    return Y

  def wavelet_to_image(self, Y):
    n = len(Y[0])

    # functions for (un-)permuting and (un-)combining channel dim with batch dim
    shaper = lambda x: jnp.transpose(x, [0, 3, 1, 2]).reshape((-1, *x.shape[1:3]))
    unshaper = lambda x: jnp.transpose(x.reshape((n, -1, *x.shape[1:])), [0, 2, 3, 1])


    # reshape
    c = Y[0].shape[-1]
    Y[1:] = [[shaper(y[...,(i * c):(i + 1) * c]) for i in range(3)] for y in Y[1:]]
    Y[0] = shaper(Y[0])

    # inverse wavelet transform
    x = cascade_utils.waverec2(Y, pywt.Wavelet("haar"))

    return unshaper(x)

# class LaplacianPyramid(nn.Module):
#     config: CDMConfig

#     def d(self, x):
#         x = cascade_utils.lap_conv(x)
#         shape = x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]
#         return jax.image.resize(x, shape, 'nearest')

#     def u(self, x):
#         shape = x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]
#         return jax.image.resize(x, shape, 'nearest')

#     def x_to_y_w(self, level, x):
#         level = self.config.levels - level
#         zs = self.encode(x, level)

#         if level == 0:
#             w, c, hfr = None, zs[0], zs[1:]
#         else:
#             w, c, hfr = zs[1], self.u(zs[0]), zs[2:]

#         return w, c, hfr

#     def x_from_y_w(self, level, y=None, w=None, hfr=None):
#         hfr = [] if hfr is None else hfr
#         if level == 0:
#             assert w is not None
#             zs = [w] + hfr
#         else:
#             if w is not None:
#                 shape = w.shape[0], w.shape[1] // 2, w.shape[2] // 2, w.shape[3]
#                 w = jax.image.resize(w, shape, 'nearest')
#             elif y is not None:
#                 shape = y.shape[0], y.shape[1] // 2, y.shape[2] // 2, y.shape[3]
#                 w = jnp.zeros(shape=shape)
#             else:
#                 assert w is not None and y is not None, f"{w}, {y}"
#                 assert False
#             zs = [w, y] + hfr

#         level = self.config.levels - level
#         for i in range(len(zs), level):
#             n, h, h, ch = zs[-1].shape
#             shape = n, h * 2, h * 2, ch
#             zs.append(jnp.zeros(shape=shape))

#         return self.decode(zs)

#     def encode(self, x, level):
#         z = x
#         zs = []
#         for level in range(level):
#             zs.append(z - self.u(self.d(z)))
#             z = self.d(z)

#         zs.append(z)

#         return [x for x in reversed(zs)]

#     def decode(self, zs):
#         x = zs[0]
#         for z in zs[1:]:
#             x = self.u(x) + z

#         return x

class LaplacianPyramid(nn.Module):
    config: CDMConfig

    def d(self, x):
        x = cascade_utils.lap_conv(x)
        shape = x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]
        return jax.image.resize(x, shape, 'nearest')

    def u(self, x):
        shape = x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]
        return jax.image.resize(x, shape, 'nearest')

    def x_to_y_w(self, level, x):
        # level = self.config.levels - level
        zs = self.encode(x, level)

        if level == 0:
            w, c, hfr = None, zs[0], zs[1:]
        else:
            w, c, hfr = zs[1], self.u(zs[0]), zs[2:]

        return w, c, hfr

    def x_from_y_w(self, level, y=None, w=None, hfr=None):
        hfr = [] if hfr is None else hfr
        if level == 0:
            assert w is not None
            zs = [w] + hfr
        else:
            if w is not None:
                shape = w.shape[0], w.shape[1] // 2, w.shape[2] // 2, w.shape[3]
                w = jax.image.resize(w, shape, 'nearest')
            elif y is not None:
                shape = y.shape[0], y.shape[1] // 2, y.shape[2] // 2, y.shape[3]
                w = jnp.zeros(shape=shape)
            else:
                assert w is not None and y is not None, f"{w}, {y}"
                assert False
            zs = [w, y] + hfr

        # level = self.config.levels - level
        for i in range(len(zs), level):
            n, h, h, ch = zs[-1].shape
            shape = n, h * 2, h * 2, ch
            zs.append(jnp.zeros(shape=shape))

        return self.decode(zs)

    def encode(self, x, level):
        z = x
        zs = []
        for level in range(level):
            zs.append(z - self.u(self.d(z)))
            z = self.d(z)

        zs.append(z)

        return [x for x in reversed(zs)]

    def decode(self, zs):
        x = zs[0]
        for z in zs[1:]:
            x = self.u(x) + z

        return x

class Cascade(nn.Module):
    config: CDMConfig
    def d(self, x):
        shape = x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]
        return jax.image.resize(x, shape, 'nearest')

    def u(self, x):
        shape = x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]
        return jax.image.resize(x, shape, 'nearest')

    def x_to_y_w(self, level, x):
        level = self.config.levels - level
        zs = self.encode(x, level)

        if level == 0:
            w, c, hfr = None, zs[0], zs[1:]
        else:
            w, c, hfr = zs[1], self.u(zs[0]), zs[2:]

        return w, c, hfr

    def x_from_y_w(self, level, y=None, w=None, hfr=None):
        hfr = [] if hfr is None else hfr
        if level == 0:
            assert w is not None
            zs = [w] + hfr
        else:
            if w is not None:
                shape = w.shape[0], w.shape[1] // 2, w.shape[2] // 2, w.shape[3]
                w = jax.image.resize(w, shape, 'nearest')
            elif y is not None:
                shape = y.shape[0], y.shape[1] // 2, y.shape[2] // 2, y.shape[3]
                w = jnp.zeros(shape=shape)
            else:
                assert w is not None and y is not None, f"{w}, {y}"
                assert False
            zs = [w, y] + hfr

        level = self.config.levels - level
        for i in range(len(zs), level):
            n, h, h, ch = zs[-1].shape
            shape = n, h * 2, h * 2, ch
            z = jax.image.resize(zs[-1], shape, 'nearest')
            zs.append(z)

        return self.decode(zs)

    def encode(self, x, level):
        z = x
        zs = []
        for level in range(level):
            zs.append(z)
            z = self.d(z)

        zs.append(z)

        return [x for x in reversed(zs)]

    def decode(self, zs):
        return zs[-1]