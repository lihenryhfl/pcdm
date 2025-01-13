# Heavily inspired by VDM codebase.

import numpy as np
import jax.numpy as jnp
from jax._src.random import PRNGKey
import jax
from typing import Any, Tuple

from pcdm.experiment import Experiment
import pcdm.model_pcdm


class Experiment_PCDM(Experiment):
  """Train and evaluate a PCDM model."""

  def get_model_and_params(self, rng: PRNGKey):
    config = self.config
    config = pcdm.model_pcdm.CDMConfig(**config.model)
    model = pcdm.model_pcdm.PCDM(config)

    inputs = {"images": jnp.zeros((2, 32, 32, 3), "uint8")}
    inputs["conditioning"] = jnp.zeros((2,))
    rng1, rng2 = jax.random.split(rng)
    params = model.init({"params": rng1, "sample": rng2}, **inputs)
    return model, params

  def loss_fn(self, params, inputs, rng, is_train) -> Tuple[float, Any]:
    rng, sample_rng = jax.random.split(rng)
    rngs = {"sample": sample_rng}
    if is_train:
      rng, dropout_rng = jax.random.split(rng)
      rngs["dropout"] = dropout_rng

    # sample time steps, with antithetic sampling
    outputs = self.state.apply_fn(
        variables={'params': params},
        **inputs,
        rngs=rngs,
        deterministic=not is_train,
    )

    rescale_to_bpd = 1./(np.prod(inputs["images"].shape[1:]) * np.log(2.))
    bpd_latent = jnp.mean(outputs.loss_klz) * rescale_to_bpd
    bpd_recon = jnp.mean(outputs.loss_recon) * rescale_to_bpd
    bpd_diff = jnp.mean(outputs.loss_diff) * rescale_to_bpd
    bpd = bpd_recon + bpd_latent + bpd_diff
    scalar_dict = {
        "bpd": bpd,
        "bpd_latent": bpd_latent,
        "bpd_recon": bpd_recon,
        "bpd_diff": bpd_diff,
        "var0": outputs.var_0,
        "var": outputs.var_1,
    }
    img_dict = {"inputs": inputs["images"]}
    metrics = {"scalars": scalar_dict, "images": img_dict}

    return bpd, metrics

  def sample_fn(self, *, dummy_inputs, rng, params):
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    if self.model.config.sm_n_timesteps > 0:
      T = self.model.config.sm_n_timesteps
    else:
      T = 1000

    conditioning = jnp.zeros((dummy_inputs.shape[0],), dtype='uint8')

    # sample z_0 from the diffusion model
    rng, sample_rng = jax.random.split(rng)
    z_0 = jax.random.normal(sample_rng, dummy_inputs.shape)

    for level in range(self.model.config.levels):
      def body_fn(i, z_t):
        step = self.state.apply_fn(
            variables={'params': params},
            i=i,
            T=T,
            z_t=z_t,
            conditioning=conditioning,
            level=level,
            sde_rng=rng,
            method=self.model.step,
        )
        return z_t + step

      z_0 = jax.lax.fori_loop(
          lower=0, upper=T, body_fun=body_fn, init_val=z_0)

    samples = self.state.apply_fn(
        variables={'params': params},
        z_0=z_0,
        method=self.model.generate_x,
    )

    return samples
