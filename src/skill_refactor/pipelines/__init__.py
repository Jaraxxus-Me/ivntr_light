"""Pipeline builder for different training and evaluation strategies. This module
contains the pipeline builders for different training and evaluation strategies.

Currently, it supports:
- pure TAMP
"""

from typing import Callable, Dict

from skill_refactor.pipelines.eval_pure_tamp import evaluate_pure_tamp
from skill_refactor.pipelines.metric import Metrics

PIPELINE_BUILDER: Dict[str, Callable[..., Metrics]] = {
    "pure_tamp": evaluate_pure_tamp,
}
