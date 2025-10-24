"""Create offline datasets by collecting demonstrations."""

import logging
from pathlib import Path
from typing import Optional

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore

from skill_refactor.approaches.pure_tamp import PureTAMPApproach
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import PlannerDataset


def get_or_collect_planner_data(
    envs: ManiSkillVectorEnv,
    approach: PureTAMPApproach,
    num_trajectories: Optional[int] = -1,
) -> PlannerDataset:
    """Get existing or collect new training data for Planner."""
    # Check if saved data exists
    data_path = (
        Path(CFG.training_data_dir)
        / CFG.env
        / "Planner_data"
        / f"scenario_{approach.curr_learning_phase}"  # type: ignore[attr-defined]
    )
    data_path.mkdir(parents=True, exist_ok=True)
    if not CFG.force_collect and data_path.exists():
        logging.info(f"\nLoading existing training data from {data_path}")
        try:
            train_data = PlannerDataset.load(data_path, num_traj=num_trajectories)
            # Verify config matches
            logging.info(f"Loaded {len(train_data)} training trajectories")
            return train_data

        except Exception as e:
            logging.info(f"Error loading training data: {e}")
            logging.info("Collecting new data instead...")

    # Collect new data
    train_data = approach.collect_planner_data(envs)  # type: ignore[attr-defined]

    # Save the collected data
    logging.info(f"\nSaving training data to {data_path}")
    train_data.save(data_path)

    return train_data
