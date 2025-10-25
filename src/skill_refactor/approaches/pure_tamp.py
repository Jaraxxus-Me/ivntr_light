"""Pure TAMP approach without any RL skills."""

import logging
from typing import Any, List

import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # type: ignore
from torch import Tensor

from skill_refactor.approaches.base import ApproachStepResult, BaseApproach
from skill_refactor.benchmarks.base import BaseRLTAMPSystem
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.structs import LowLevelTrajectory, PlannerDataset
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


class PureTAMPApproach(BaseApproach):
    """Pure RL approach that doesn't use TAMP structure."""

    def __init__(
        self,
        system: BaseRLTAMPSystem,
        seed: int,
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed)
        fall_back_action = system.env.single_action_space.sample()  # type: ignore
        normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
            system.env, CFG.control_mode
        )

        # Create planner using environment's components
        self.planner = TaskThenMotionPlanner(
            types=system.types,
            predicates=system.predicates,
            perceiver=system.perceiver,
            operators=system.operators,
            skills=system.skills,
            fallback_action=fall_back_action,
            normalize_action=normalize_action,
            arm_action_low=arm_action_low,
            arm_action_high=arm_action_high,
            planner_id="pyperplan",
        )

    def reset(self, obs: Tensor, info: dict[str, Any]) -> ApproachStepResult:
        """Reset approach with initial observation."""
        self.planner.reset(obs[0:1], info)
        return self.step(obs, 0.0, False, False, info)

    @classmethod
    def get_name(cls) -> str:
        """Get name of the approach."""
        return "pure_tamp"

    def step(
        self,
        obs: Tensor,
        reward: float | Tensor,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult:
        """Step approach with new observation."""
        action, operator = self.planner.step(obs)
        if action is None:
            raise RuntimeError("Task then motion planning failed, no action returned.")
        return ApproachStepResult(_action=action, op=operator)

    def collect_data(
        self,
        envs: ManiSkillVectorEnv,
    ) -> PlannerDataset:
        """Collect Planner training data from the environment"""

        logging.info("Collecting training data with planner\n")

        assert CFG.num_envs == 1, "Only support single env for planner data collection"
        train_task_idx = 0
        trajectories: List[LowLevelTrajectory] = []

        for episode in range(CFG.num_train_episodes_planner):
            logging.info(
                f"Collecting episode {episode + 1}/{CFG.num_train_episodes_planner}"
            )

            # Phase 1: Planning phase with replay bag tracking
            obs, _ = envs.reset(seed=episode)
            init_obs = obs.clone()  # Save initial observation for replay
            res = self.reset(obs, {})

            colliding = False
            # Replay bag to store actions and collision states
            replay_bag = []  # List of (action, operator, colliding_mask) tuples
            step_count = 0

            # Planning phase - track actions in replay bag
            while step_count < CFG.max_env_steps:
                step_count += 1
                action = res.action  # [B, *act_shape]

                # Apply frozen actions for colliding environments
                obs, _, _, _, infos = envs.step(action)

                colliding = infos["is_colliding"].to(torch.bool)[0].item()

                if colliding:
                    logging.info(
                        f"Episode {episode + 1} had collisions at step {step_count}."
                    )
                    break

                # Store in replay bag
                replay_bag.append((action.clone(), res.op, colliding))

                res = self.step(obs, 0.0, False, False, {})

                if infos["success"].sum():
                    logging.info(
                        f"Episode {episode + 1} finished early at step {step_count} with success."
                    )
                    break

            # Phase 2: Data collection phase with replay (and RL execution)
            step_count = 0
            # Reset environment to initial state for clean data collection
            obs, _ = envs.reset(options={"init_state": init_obs})

            # Data collection buffers
            states_steps = [obs.clone()]
            actions_steps = []
            ops_steps = []
            success = False

            for i, (replay_action, replay_op, _) in enumerate(replay_bag):
                step_count += 1
                actions_steps.append(replay_action.clone())
                ops_steps.append(replay_op)
                obs, _, _, _, infos = envs.step(replay_action)
                success = infos["success"].to(torch.bool)[0].item()
                states_steps.append(obs.clone())
                if success:
                    logging.info(
                        f"Episode {episode + 1} finished early at step {step_count} with success."
                    )
                    break

            # Build trajectories from collected data
            if len(states_steps) > 1 and len(actions_steps) > 0:
                states_tensor = torch.stack(states_steps, dim=1)  # [B, S, *obs_shape]
                actions_tensor = torch.stack(actions_steps, dim=1)  # [B, A, *act_shape]

                env_states = states_tensor.unbind(dim=0)
                env_actions = actions_tensor.unbind(dim=0)

                for _, (s, act) in enumerate(zip(env_states, env_actions)):
                    if success:
                        trajectories.append(
                            LowLevelTrajectory(
                                _states=list(s.unbind(0)),
                                _actions=[
                                    ApproachStepResult(_action=a, op=o)
                                    for a, o in zip(act.unbind(0), ops_steps)
                                ],
                                _train_task_idx=train_task_idx,
                            )
                        )
                        train_task_idx += 1

            logging.info(
                f"Collected {len(trajectories)} trajectories for planner learning."
            )

        return PlannerDataset(
            _trajectories=trajectories,
        )