"""Unit Tests for the data collection, in Cluttered Table environment."""

import logging
from pathlib import Path
from typing import List

import pytest

from skill_refactor import register_all_environments
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (  # type: ignore[name-defined]
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.approaches.pure_tamp import PureTAMPApproach
from skill_refactor.args import reset_config
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


@pytest.mark.skip(reason="The script requires local data to run")
def test_blocked_stacking_planner_data_collection():
    """Test Data collection in BlockedStacking environment."""
    test_config = {
        "num_envs": 1,
        "debug_env": False,
        "max_rl_steps": 20,
        "rl_static_steps": 3,
        "max_env_steps": 240,
        "num_train_episodes_planner": 1000,
        "obstruction_blocking_grasp_prob": 1.0,
        "obstruction_blocking_stacking_prob": 0.0,
        "log_file": "logs/1010_bstacking_collect1.log",
        "lll_config": "config/lifelong_learning/blocked_stacking_sc1.yaml",
        "rl_config": "config/pure_rl/blocked_stacking_ppoc.yaml",
        "control_mode": "pd_joint_delta_pos",
        "loglevel": logging.INFO,
    }
    register_all_environments()
    reset_config(test_config)
    # Set up logging
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if CFG.log_file:
        handlers.append(logging.FileHandler(CFG.log_file, mode="w"))
    logging.basicConfig(
        level=CFG.loglevel, format="%(message)s", handlers=handlers, force=True
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    if CFG.log_file:
        logging.info(f"Logging to {CFG.log_file}")

    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )
    approach = PureTAMPApproach(tamp_system, seed=CFG.seed)

    fall_back_action = tamp_system.env.single_action_space.sample()
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        tamp_system.env, CFG.control_mode
    )

    # Create planner using environment's components
    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        fallback_action=fall_back_action,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        planner_id="pyperplan",
    )


    envs = tamp_system.env
    train_data = approach.collect_data(envs)
    envs.close()
    train_data.save(Path("training_data/blocked_stacking/Planner_data/scenario_1"))