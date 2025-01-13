# Heavily inspired by VDM codebase.

from typing import Callable, Optional, Iterable

import chex
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import pcdm.cascade_utils as cascade_utils
import pywt

from pcdm.nn import *


######### Latent PCDM model #########
class PCDM(nn.Module):
  config: CDMConfig

  def setup(self):
    if self.config.cascade_map == 'laplacian_pyramid':
        self.w = LaplacianPyramid(self.config)
        self.score_models = dict(
          [(level, WScoreUNet(self.config, level)) for level in range(self.config.levels)])
    elif self.config.cascade_map == 'super_resolution':
        self.w = Cascade(self.config)
        self.score_models = dict(
          [(level, WScoreUNet(self.config, level)) for level in range(self.config.levels)])
    elif self.config.cascade_map == 'wavelet':
        self.w = Wavelet(self.config)
        self.score_models = dict(
          [(level, WScoreUNet(self.config, level)) for level in range(self.config.levels)])

    self.encdec = EncDec(self.config)
    if self.config.gamma_type == 'learnable_nnet':
      self.gamma = NoiseSchedule_NNet(self.config)
    elif self.config.gamma_type == 'fixed':
      self.gamma = NoiseSchedule_FixedLinear(self.config)
    elif self.config.gamma_type == 'learnable_scalar':
      self.gamma = NoiseSchedule_Scalar(self.config)
    else:
      raise Exception("Unknown self.var_model")

  def __call__(self, images, conditioning, deterministic: bool = True):
    x = images
    n_batch = images.shape[0]

    # encode
    f = self.encdec.encode(x)

    # sample time steps
    rng1 = self.make_rng("sample")
    if self.config.antithetic_time_sampling:
      t0 = jax.random.uniform(rng1)
      t = jnp.mod(t0 + jnp.arange(0., 1., step=1. / n_batch), 1.)
    else:
      t = jax.random.uniform(rng1, shape=(n_batch,))

    g_0, g_1 = self.gamma(0.), self.gamma(1.)
    var_0, var_1 = nn.sigmoid(g_0), nn.sigmoid(g_1)

    # 1. RECONSTRUCTION LOSS
    # add noise and reconstruct
    eps_0 = jax.random.normal(self.make_rng("sample"), shape=f.shape)
    z_0 = jnp.sqrt(1. - var_0) * f + jnp.sqrt(var_0) * eps_0
    z_0_rescaled = f + jnp.exp(0.5 * g_0) * eps_0  # = z_0/sqrt(1-var)
    loss_recon = - self.encdec.logprob(x, z_0_rescaled, g_0)

    # 2. LATENT LOSS
    # KL z1 with N(0,1) prior
    mean1_sqr = (1. - var_1) * jnp.square(f)
    loss_klz = 0.5 * jnp.sum(mean1_sqr + var_1 - jnp.log(var_1) - 1., axis=(1, 2, 3))

    loss_diff = jnp.zeros_like(t)
    for level in range(self.config.levels):
      loss_diff += self.diff_loss_at_level(level, f, conditioning, t, deterministic)

    return CDMOutput(
        loss_recon=loss_recon,
        loss_klz=loss_klz,
        loss_diff=loss_diff,
        var_0=var_0,
        var_1=var_1,
    )

  def diff_loss_at_level(self, level, f, conditioning, t, deterministic):
    # 3. DIFFUSION LOSS
    # discretize time steps if we're working with discrete time
    T = self.config.sm_n_timesteps
    if T > 0:
      t = jnp.ceil(t * T) / T

    # sample z_t
    g_t = self.gamma(t)
    var_t = nn.sigmoid(g_t)[:, None, None, None]
    eps = jax.random.normal(self.make_rng("sample"), shape=f.shape, dtype=jnp.float32)
    z_t = jnp.sqrt(1. - var_t) * f + jnp.sqrt(var_t) * eps
    assert z_t.dtype == jnp.float32

    target = self.w.x_to_y_w(level, eps)[1] if level == 0 else self.w.x_to_y_w(level, eps)[0]
    # # compute predicted noise
    z_t, skip, score_model = self.get_score_model_and_inputs(level, z_t, z_clean=f)
    output = score_model(z_t, g_t, conditioning, skip, deterministic)

    # compute MSE of predicted noise
    loss_diff_mse = jnp.sum(jnp.square(target - output), axis=[1, 2, 3])

    if T == 0:
      # loss for infinite depth T, i.e. continuous time
      _, g_t_grad = jax.jvp(self.gamma, (t,), (jnp.ones_like(t),))
      loss_diff = .5 * g_t_grad * loss_diff_mse
    else:
      # loss for finite depth T, i.e. discrete time
      s = t - (1./T)
      g_s = self.gamma(s)
      loss_diff = .5 * T * jnp.expm1(g_t - g_s) * loss_diff_mse

    # End of diffusion loss computation

    return loss_diff

  def step(self, i, T, z_t, conditioning, level, sde_rng=None):
    if sde_rng is not None:
      rng_body = jax.random.fold_in(sde_rng, i)
      eps = jax.random.normal(rng_body, z_t.shape, dtype=jnp.float32)
    else:
      eps = jnp.zeros_like(z_t, dtype=jnp.float32)

    t = (T - i) / T
    s = (T - i - 1) / T

    g_s, g_t = self.gamma(s), self.gamma(t)

    # project the score vector
    z_model, skip, score_model = self.get_score_model_and_inputs(level, z_t)
    eps_hat = score_model(
        z_model,
        g_t * jnp.ones((z_t.shape[0],), g_t.dtype),
        conditioning,
        skip,
        deterministic=True)
    eps_hat = self.w.x_from_y_w(level, w=eps_hat) if level == 0 else self.w.x_from_y_w(level, y=eps_hat)

    # get other variables
    a = nn.sigmoid(-g_s)
    b = nn.sigmoid(-g_t)
    c = - jnp.expm1(g_s - g_t)
    sigma_t = jnp.sqrt(nn.sigmoid(g_t))

    # compute f, g
    f = jnp.sqrt(a / b)
    g2 = -f * sigma_t * c
    g = jnp.sqrt((1. - a) * c)

    g2 = g2 * 0.5 if sde_rng is None else g2

    step = f * z_t + g2 * eps_hat + g * eps - z_t

    # project the step
    step = self.get_score_model_and_inputs(level, step)[1]
    step = self.w.x_from_y_w(level, w=step) if level == 0 else self.w.x_from_y_w(level, y=step)

    return step

  def generate_x(self, z_0):
    g_0 = self.gamma(0.)

    var_0 = nn.sigmoid(g_0)
    z_0_rescaled = z_0 / jnp.sqrt(1. - var_0)

    logits = self.encdec.decode(z_0_rescaled, g_0)

    # get output samples
    if self.config.sample_softmax:
      out_rng = self.make_rng("sample")
      samples = jax.random.categorical(out_rng, logits)
    else:
      samples = jnp.argmax(logits, axis=-1)

    return samples

  def get_score_model_and_inputs(self, level, z, z_clean=None):
    y, w, _ = self.w.x_to_y_w(level, z) # wavelet decomposition of latent var
    if level == 0:
      z, skip = w, w
    else:
      if z_clean is not None:
        wc = self.w.x_to_y_w(level, z_clean)[1]
      else:
        wc = w
      z, skip = jnp.concatenate([y, wc], axis=-1), y

    return z, skip, self.score_models[level]
