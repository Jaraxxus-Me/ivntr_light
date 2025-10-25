"""Unit Tests for the Lifelong Refactoring Approach, in Cluttered Table environment."""

import os
import pytest
import copy
import logging
import types
from pathlib import Path
from typing import List

import torch
import yaml

from skill_refactor import register_all_environments
from skill_refactor.approaches.pure_tamp import PureTAMPApproach
from skill_refactor.approaches.pred_learner.topdown_learner import (
    TopDownPredicateLearner,
)
from skill_refactor.args import reset_config
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.benchmarks.wrappers import MultiEnvRecordVideo
from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_normalize_action_range
from skill_refactor.utils.structs import (
    LiftedOperator,
    PlannerDataset,
)
from skill_refactor.utils.ttmp import TaskThenMotionPlanner


@pytest.mark.skipif(
    not os.path.exists("top_down_pred_nets"),
    reason="predicate nets not found",
)
def test_loading_learned_predicates_blocked_stacking():
    """Test RL Planning Wrapper with BlockedStacking environment."""
    test_config = {
        "num_envs": 1,
        "debug_env": False,
        "num_eval_episodes": 10,
        "traj_segmenter": "operator_changes",
        "predicate_config": "config/predicates/blocked_stacking_enu.yaml",
        "log_file": "test_learned_predicates.log",
        "pred_net_save_dir": "top_down_pred_nets",
        "middle_state_method": "naive_init",
        "force_skip_pred_learning": True,
        "loglevel": logging.INFO,
        "control_mode": "pd_joint_delta_pos",
        "max_env_steps": 250,
    }
    reset_config(test_config)
    register_all_environments()

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
    fall_back_action = tamp_system.env.single_action_space.sample()
    normalize_action, arm_action_low, arm_action_high = get_normalize_action_range(
        tamp_system.env, CFG.control_mode
    )

    # Create planner with partial predicates
    given_predicate_names = ["On"]
    given_predicate_set = set()
    for pred in list(tamp_system.predicates):
        if pred.name in given_predicate_names:
            given_predicate_set.add(pred)
    partial_perceiver = copy.deepcopy(tamp_system.perceiver)
    all_predicate_interp = copy.deepcopy(partial_perceiver.predicate_interpreters)
    # Delete all predicates that are not given
    for pred in list(all_predicate_interp.keys()):
        if pred not in given_predicate_set:
            partial_perceiver.delete_predicate_interpreter(pred)

    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=given_predicate_set, # use partial predicate set
        perceiver=partial_perceiver, # use partial predicate set
        operators=set(), # will be updated later
        skills=set(), # will be updated later
        fallback_action=fall_back_action,
        normalize_action=normalize_action,
        arm_action_low=arm_action_low,
        arm_action_high=arm_action_high,
        planner_id="pyperplan",
    )

    # Load the learned predicates
    with open(CFG.predicate_config, "rb") as f:
        config_data = yaml.safe_load(f)
    predicate_configures = config_data["predicates"]

    dataset_path = Path(f"training_data/scenario_1")
    planner_dataset = PlannerDataset.load(dataset_path, num_traj=5)

    topdown_learner = TopDownPredicateLearner(
        dataset=planner_dataset,
        tamp_system=tamp_system,
        predicate_configures=predicate_configures,
        given_predicates=given_predicate_set,
        verbose=True,
    )

    invented_pred_interpr, op_set = topdown_learner.invent()

    # Update the approach with the invented predicates
    for pred, interp in invented_pred_interpr.items():
        type_list = pred.types
        planner.perceiver.add_predicate_interpreter(pred.name, type_list, interp)

    skills = copy.deepcopy(tamp_system.skills)
    skill_operator_names = [skill.get_operator_name() for skill in skills]
    for operator in op_set:
        # Note that all operators will be binded to the skills
        # even for "existing" ones.
        assert operator.name in skill_operator_names
        # Case 1, it is from the existing skills
        # initiate a new skill from the existing skill, but with a new operator name
        existing_skill = next(
            skill
            for skill in skills
            if skill.get_operator_name() == operator.name
        )
        new_skill = existing_skill.__class__(
            env=tamp_system.env.unwrapped[0],
            operators=op_set,
        )
        # deepcopy so we don’t accidentally mutate the original
        local_op = copy.deepcopy(operator)

        # capture local_op in the default arg
        def _customget_operator_name(self, op=local_op) -> str:
            del self
            return op.name

        def _custom_get_lifted_operator(self, op=local_op) -> LiftedOperator:
            del self
            return op

        new_skill.get_operator_name = types.MethodType(
            _customget_operator_name, new_skill
        )
        new_skill.get_lifted_operator = types.MethodType(
            _custom_get_lifted_operator, new_skill
        )
        skills.add(new_skill)
    planner.update_domain(op_set, skills)

    # Now test the approach with the latest planner
    # Use Training states if necessary
    approach = PureTAMPApproach(
        tamp_system,
        seed=0,
        planner=planner,
    )
    video_folder = Path(f"videos/planning_learned_predicates")
    envs = MultiEnvRecordVideo(
        tamp_system.env,
        video_folder=video_folder.as_posix(),
        episode_trigger=lambda x: True,
    )
    # envs = tamp_system.env
    success = []
    for epi in range(0, 10):
        traj_init_state = torch.stack(
            [planner_dataset.trajectories[epi].states[0]],
            dim=0,
        ).to(envs.device)

        # Use training initial states
        reset_options = {"init_state": traj_init_state}
        # Use env rnd seed
        # reset_options = {}
        obs, info = envs.reset(options=reset_options)
        step_result = approach.reset(obs, info)
        total_reward = torch.tensor(
            [0.0] * CFG.num_envs, dtype=torch.float32, device=envs.device
        )
        epi_success = False
        for step in range(CFG.max_env_steps + 1):
            # if step < len(planner_dataset.trajectories[epi].actions):
            #     # Store trajectory action for potential use
            #     torch.stack(
            #         [
            #             planner_dataset.trajectories[epi].actions[step]._action  # pylint: disable=protected-access
            #         ],
            #         dim=0,
            #     ).to(envs.device)
            # action_stack = step_result.action
            # if step >= 42:
            #     action_stack = step_result.action
            #     obs, reward, _, _, info = envs.step(action_stack_)
            # else:
            # print(f"Espisode {epi} Step {step}, taking action {step_result.action}")
            obs, _, _, _, info = envs.step(step_result.action)
            if info["success"].all() and (not epi_success):
                epi_success = True
                success.append(1)
                logging.info(f"Episode {epi} succeeded at step {step}.")
            step_result = approach.step(obs, total_reward, False, False, info)
        if not info["success"].all():
            success.append(0)
            logging.info(f"Episode {epi} failed.")
    logging.info(f"Success rate: {sum(success) / len(success)}")
    envs.close()