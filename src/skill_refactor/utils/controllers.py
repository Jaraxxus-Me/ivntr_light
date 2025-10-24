"""Utility functions for controllers in the skill refactor project."""

import gymnasium as gym
import torch
from gymnasium import Env
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore
from torch import Tensor

from skill_refactor.settings import CFG


def get_normalize_action_range(
    env: ManiSkillVectorEnv | Env, control_mode: str
) -> tuple[bool, Tensor, Tensor]:
    """Get the action range for normalization in the ManiSkill environment.

    Args:
        env (ManiSkillVectorEnv): The ManiSkill environment instance.

    Returns:
        bool: Whether to normalize actions.
        Tensor: Lower bounds of the action space.
        Tensor: Upper bounds of the action space.
    """
    normalize_action = False
    if control_mode == "pd_joint_delta_pos":
        if isinstance(env, ManiSkillVectorEnv):
            normalize_action = env.agent.controller.configs["arm"].normalize_action
            if normalize_action:
                arm_action_low = env.agent.controller.controllers[
                    "arm"
                ].action_space_low
                arm_action_high = env.agent.controller.controllers[
                    "arm"
                ].action_space_high
            else:
                arm_action_low = torch.tensor([-1.0] * 6, device=env.device)
                arm_action_high = torch.tensor([1.0] * 6, device=env.device)
        else:
            normalize_action = CFG.normalize_action
            assert isinstance(env, Env)
            assert isinstance(env.action_space, gym.spaces.Box)
            if normalize_action:
                assert hasattr(env, "action_low") and hasattr(
                    env, "action_high"
                ), "Environment must have 'action_low' and 'action_high' attributes for normalization."
                arm_action_low = torch.tensor(
                    env.action_low, dtype=torch.float32, device=CFG.device
                )
                arm_action_high = torch.tensor(
                    env.action_high, dtype=torch.float32, device=CFG.device
                )
            else:
                arm_action_low = torch.tensor(
                    env.action_space.low, dtype=torch.float32, device=CFG.device
                )
                arm_action_high = torch.tensor(
                    env.action_space.high, dtype=torch.float32, device=CFG.device
                )
    else:
        raise NotImplementedError(f"Control mode {control_mode} not supported.")
    return normalize_action, arm_action_low, arm_action_high


def get_frozen_action(
    skill_action: Tensor,
    arm_action_low: Tensor,
    arm_action_high: Tensor,
    normalize_action: bool,
    control_mode: str,
) -> Tensor:
    """Get the frozen action for the skill based on the action range.

    Args:
        skill_action (Tensor): The action from the skill.
        arm_action_low (Tensor): Lower bounds of the action space.
        arm_action_high (Tensor): Upper bounds of the action space.
        normalize_action (bool): Whether to normalize actions.
        control_mode (str): The control mode of the environment.

    Returns:
        Tensor: The frozen action.
    """
    frozen_action = skill_action.clone()
    if control_mode == "pd_joint_delta_pos":
        static_actions = torch.zeros_like(skill_action)
        if CFG.delta_finger_control:
            if normalize_action:
                low = arm_action_low.unsqueeze(0).repeat(static_actions.shape[0], 1)
                high = arm_action_high.unsqueeze(0).repeat(static_actions.shape[0], 1)
                delta_qpos_norm = (static_actions - 0.5 * (low + high)) / (
                    0.5 * (high - low)
                )
            else:
                delta_qpos_norm = static_actions.clone()
        else:
            if normalize_action:
                low = arm_action_low.unsqueeze(0).repeat(static_actions.shape[0], 1)
                high = arm_action_high.unsqueeze(0).repeat(static_actions.shape[0], 1)
                delta_arm_qpos = (static_actions[:, :-1] - 0.5 * (low + high)) / (
                    0.5 * (high - low)
                )
            else:
                delta_arm_qpos = static_actions[:, :-1].clone()
            delta_qpos_norm = torch.cat([delta_arm_qpos, skill_action[:, -1:]], dim=-1)

        frozen_action = delta_qpos_norm.clone()

    return frozen_action
