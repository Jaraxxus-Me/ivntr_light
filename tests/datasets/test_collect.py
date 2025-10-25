"""Unit Tests for the data collection, in Cluttered Table environment."""

import logging
from pathlib import Path
from typing import List

from skill_refactor import register_all_environments
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.approaches.pure_tamp import PureTAMPApproach
from skill_refactor.args import reset_config
from skill_refactor.settings import CFG

def test_blocked_stacking_planner_data_collection():
    """Test Data collection in BlockedStacking environment."""
    test_config = {
        "num_envs": 1,
        "debug_env": False,
        "max_env_steps": 240,
        "num_train_episodes_planner": 1, # Change to 200 for actual training
        "log_file": "bstacking_collect1.log",
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

    envs = tamp_system.env
    train_data = approach.collect_data(envs)
    envs.close()
    train_data.save(Path("training_data/scenario_1"))