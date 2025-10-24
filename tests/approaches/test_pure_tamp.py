"""Tests for ClutteredTable environment with (pure) TAMP."""

import time

# import imageio.v2 as iio
import pytest
import torch
from mani_skill.utils.wrappers.record import RecordEpisode  # type: ignore
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore

from skill_refactor import register_all_environments
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import MultiEnvRecordVideo
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


@pytest.mark.parametrize("system_cls", [BlockedStackingRLTAMPSystem])
def test_blocked_stacking_far_obstructions(system_cls):
    """Test BlockedStacking environment with a pure TAMP planner."""

    test_config = {
        "num_envs": 8,
        "device": "cuda:0",
        "training_scenario": 1,
        "control_mode": "pd_joint_delta_pos",
        "normalize_action": True,
    }
    reset_config(test_config)
    register_all_environments()

    # Create TAMP system
    tamp_system = system_cls.create_default(render_mode="rgb_array", seed=42)
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

    envs = MultiEnvRecordVideo(tamp_system.env, "videos/stacking-test-tamp-far-easy")
    # envs = tamp_system.env

    obs, info = envs.reset(seed=0)
    s = time.time()
    planner.reset(obs, info)
    print("Planner reset time:", time.time() - s)

    total_reward = 0
    for step in range(150):
        action, _ = planner.step(obs)
        s = time.time()
        obs, reward, terminated, _, _ = envs.step(action)
        # print("Step time:", time.time() - s)
        # iio.imwrite(f"videos/stacking-test-tamp-far/step-{step:04d}.png", envs.render())
        total_reward += reward
        # print(f"Step {step + 1}: Reward: {reward}")
        if torch.any(terminated):
            print("Some episodes finished successfully at step", step)
            break
        if torch.all(terminated):
            print("Episode finished successfully")
            break
    else:
        print("Episode didn't finish within 300 steps")

    envs.close()
