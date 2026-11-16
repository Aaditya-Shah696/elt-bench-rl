"""turn a Grade into one reward between 0 and 1, plus some metrics to log.

    reward = load_weight * (share of raw tables loaded right)
           + transform_weight * mean over target tables of:
                 1                                  if the table is exact
                 partial_credit * column_fraction   otherwise

exact is worth way more than nearly right (2x by default) so the model is pushed to finish
"""

from dataclasses import dataclass

from elt_rl.grading.grader import Grade


@dataclass(frozen=True)
class RewardWeights:
    load: float = 0.2
    transform: float = 0.8
    partial_credit: float = 0.5


DEFAULT_WEIGHTS = RewardWeights()


def compute_reward(
    grade: Grade, weights: RewardWeights = DEFAULT_WEIGHTS
) -> tuple[float, dict[str, float]]:
    model_scores = [
        1.0 if m.exact else weights.partial_credit * m.column_fraction for m in grade.models
    ]
    transform = sum(model_scores) / len(model_scores) if model_scores else 0.0
    reward = weights.load * grade.load_fraction + weights.transform * transform
    n_models = max(len(grade.models), 1)
    metrics = {
        "reward": reward,
        "load_fraction": grade.load_fraction,
        "table_accuracy": sum(m.exact for m in grade.models) / n_models,
        "column_accuracy": sum(m.column_fraction for m in grade.models) / n_models,
        "task_success": float(grade.success),
    }
    return reward, metrics
