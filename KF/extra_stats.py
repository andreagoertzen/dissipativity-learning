from collections.abc import Callable
from typing import Any

import grain.tensorflow as tfgrain
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from swirl_dynamics.data import hdf5_utils
from swirl_dynamics.data import tfgrain_transforms as transforms
from swirl_dynamics.lib.solvers import ode
import tensorflow as tf


Array = jax.Array
PyTree = Any
DynamicsFn = Callable[[Array, Array, PyTree], Array]

def plot_cos_sims(dt: Array, traj_length: int, trajs: Array, pred_trajs: Array):
  """Plot cosine similarities over time."""

  def sum_non_batch_dims(x: Array) -> Array:
    """Helper method to sum array along all dimensions except the 0th."""
    ndim = x.ndim
    return x.sum(axis=tuple(range(1, ndim)))

  def state_cos_sim(x: Array, y: Array) -> Array:
    """Compute cosine similiarity between two batches of states.

      Computes x^Ty / ||x||*||y|| averaged across batch dimension (axis = 0).

    Args:
      x: array of states; shape: batch_size x state_dimension
      y: array of states; shape: batch_size x state_dimension

    Returns:
      cosine similarity averaged along batch dimension.
    """
    x_norm = jnp.expand_dims(
        jnp.sqrt(sum_non_batch_dims((x**2))), axis=tuple(range(1, x.ndim))
    )
    x /= x_norm
    y_norm = jnp.expand_dims(
        jnp.sqrt(sum_non_batch_dims((y**2))), axis=tuple(range(1, y.ndim))
    )
    y /= y_norm
    return sum_non_batch_dims(x * y).mean(axis=0)

  plot_time = jnp.arange(traj_length) * dt
  t_max = plot_time.max()
  fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)

  # Plot 0.9, 0.8 threshold lines.
  ax.plot(
      plot_time,
      jnp.ones(traj_length) * 0.9,
      color="black",
      linestyle="dashed",
      label="0.9 threshold",
  )
  ax.plot(
      plot_time,
      jnp.ones(traj_length) * 0.8,
      color="red",
      linestyle="dashed",
      label="0.8 threshold",
  )

  # Plot correlation lines.
  cosine_sims = jax.vmap(state_cos_sim, in_axes=(1, 1))(
      trajs[:, :traj_length, :], pred_trajs[:, :traj_length, :]
  )
  ax.plot(plot_time, cosine_sims)
  ax.set_xlim([0, t_max])
  ax.set_xlabel(r"$t$")
  ax.set_ylabel("Avg. cosine sim.")
  ax.set_title("Cosine Similiarity over time")
  ax.legend(frameon=False, bbox_to_anchor=(1, 1))
  return {"cosine_sim": fig}


from collections.abc import Callable

import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence


Array = jax.Array
MeasureDistFn = Callable[[Array, Array], Array]

def sinkhorn_div(x: Array, y: Array) -> Array:
  """Sinkhorn Divergence.

  Emprical sinkhorn divergence. The lower the result the more evidence that
  distributions are the same.
  Input arrays are reshaped to dimension: `batch_size x -1`, where `-1`
  indicates that all non-batch dimensions are flattened.

  Args:
    x: first sample, distribution P
    y: second sample, distribution Q

  Returns:
    sd value.
  """
  # Samples x and y are of size `batch_size x state_space_dim`, e.g. for Lorenz
  # system `state_space_dim` is `3`, for KS it is `xspan x 1`, for NS it is
  # `h x w x 1`.
  # These arrays are then reshaped to be order two with shape
  # `batch_size x state_space_dim_flattened`.
  ot = sinkhorn_divergence.sinkhorn_divergence(
      pointcloud.PointCloud,  # geom,
      x.reshape((x.shape[0], -1,)),  # geom.x,
      y.reshape((y.shape[0], -1,)),  # geom.y,
      static_b=False,
  )
  return jnp.array(ot.divergence)
