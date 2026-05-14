from collections.abc import Callable
import functools
from typing import TypeVar

from clu import metrics as clu_metrics
import flax
import jax
import flax.linen as nn
import optax
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers
from ISP_baseline.src.models import DeterministicModel

Array = jax.Array
ArrayLike = jax.typing.ArrayLike
VariableDict = trainers.VariableDict

TrainState = train_states.BasicTrainState


class DeterministicTrainer(trainers.BasicTrainer[DeterministicModel, TrainState]):

    @flax.struct.dataclass
    class TrainMetrics(clu_metrics.Collection):
        train_loss: clu_metrics.Average.from_output("loss")
        train_loss_std: clu_metrics.Std.from_output("loss")

    @flax.struct.dataclass
    class EvalMetrics(clu_metrics.Collection):
        eval_rrmse_mean: clu_metrics.Average.from_output("rrmse")
        eval_rrmse_std: clu_metrics.Std.from_output("rrmse")

    @staticmethod
    def build_inference_fn(
        state: TrainState, core: nn.Module
    ) -> Callable[[ArrayLike], Array]:
        """Builds an encoder inference function from a train state."""
        return DeterministicModel.inference_fn(state.model_variables, core)
