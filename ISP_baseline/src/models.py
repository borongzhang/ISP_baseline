from collections.abc import Callable, Mapping
import dataclasses
import functools
from typing import Any, ClassVar, Protocol

from clu import metrics as clu_metrics
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.templates import models
from swirl_dynamics.templates import trainers
from swirl_dynamics.templates import train_states
from swirl_dynamics.lib import metrics

Array = jax.Array
CondDict = Mapping[str, Array]
Metrics = clu_metrics.Collection
ShapeDict = Mapping[str, Any]  # may be nested
PyTree = Any
VariableDict = trainers.VariableDict


@dataclasses.dataclass(frozen=True, kw_only=True)
class DeterministicModel(models.BaseModel):

    input_shape: tuple[int, ...]
    core_module: nn.Module

    def initialize(self, rng: Array):
        # TODO: Add a dtype object to ensure consistency of types.
        x = jnp.ones((1,) + self.input_shape)
        return self.core_module.init(rng, x)

    def loss_fn(
        self,
        params: models.PyTree,
        batch: models.BatchType,
        rng: Array,
        mutables: models.PyTree,
    ) -> models.LossAndAux:

        y = self.core_module.apply({"params": params}, batch["scatter"])

        loss = jnp.mean(jnp.square(y - batch["eta"]))
        metric = dict(loss=loss)
        return loss, (metric, mutables)

    def eval_fn(
        self,
        variables: models.PyTree,
        batch: models.BatchType,
        rng: Array,
    ) -> models.ArrayDict:

        x = batch["scatter"]
        core = self.inference_fn(variables, self.core_module)
        y = core(x)
        rrmse = functools.partial(
            metrics.mean_squared_error,
            sum_axes=(-1, -2),
            relative=True,
            squared=False,
        )

        return dict(rrmse=rrmse(pred=y, true=batch["eta"]))

    @staticmethod
    def inference_fn(variables: models.PyTree, core_module: nn.Module):

        def _core(x: Array) -> Array:
            return core_module.apply(variables, x)

        return _core
