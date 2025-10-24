"""BlockedStacking environment planning components."""

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Sequence, cast

import gymnasium as gym
import torch
from prbench.envs.dynamic2d.object_types import Dynamic2DType
from relational_structs import (
    GroundAtom,
    Object,
    PDDLDomain,
    Predicate,
    Type,
    Variable,
)
from torch import Tensor

from skill_refactor.benchmarks.base import (
    BaseRLTAMPSystem,
    GraphData,
    TensorPlanningComponents,
)
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking_env import (
    BlockedStacking2DEnv,
    BlockType,
    DynRectangleType,
    KinRectangleType,
    KinRobotType,
    RobotType,
)
from skill_refactor.benchmarks.wrappers import (
    MultiEnvWrapper,
    NormalizeActionMultiEnvWrapper,
)
from skill_refactor.settings import CFG
from skill_refactor.utils.structs import (
    GroundOperator,
    LiftedOperator,
    LiftedOperatorSkill,
    ObjectContainer,
    Perceiver,
    PredicateContainer,
    TypeContainer,
)
from skill_refactor.utils.task_planning import (
    get_object_combinations,
)


def xy_cos_sin_to_matrix(xy_cos_sin: Tensor) -> Tensor:
    """Convert (x, y, cos(theta), sin(theta)) to 3x3 transformation matrix."""
    B = xy_cos_sin.shape[0]
    assert xy_cos_sin.shape[1] == 4
    x = xy_cos_sin[:, 0]
    y = xy_cos_sin[:, 1]
    cos_theta = xy_cos_sin[:, 2]
    sin_theta = xy_cos_sin[:, 3]
    zeros = torch.zeros(B, device=xy_cos_sin.device, dtype=xy_cos_sin.dtype)
    ones = torch.ones(B, device=xy_cos_sin.device, dtype=xy_cos_sin.dtype)
    matrix = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta, x], dim=1),
            torch.stack([sin_theta, cos_theta, y], dim=1),
            torch.stack([zeros, zeros, ones], dim=1),
        ],
        dim=1,
    )  # (B, 3, 3)
    return matrix


def matrix_to_xy_cos_sin(matrix: Tensor) -> Tensor:
    """Convert 3x3 transformation matrix to (x, y, cos(theta), sin(theta))."""
    assert matrix.shape[1] == 3 and matrix.shape[2] == 3
    x = matrix[:, 0, 2]
    y = matrix[:, 1, 2]
    cos_theta = matrix[:, 0, 0]
    sin_theta = matrix[:, 1, 0]
    return torch.stack([x, y, cos_theta, sin_theta], dim=1)  # (B, 4)


class BlockedStackingTypes(TypeContainer):
    """Container for StickButton types."""

    def __init__(self) -> None:
        """Initialize types."""
        self.kin_robot = KinRobotType
        self.robot = RobotType
        self.block = BlockType
        self.kin_rectangle = KinRectangleType
        self.dyn_rectangle = DynRectangleType
        self.dynamic2d = Dynamic2DType

    def as_set(self) -> set[Type]:
        """Convert to set of types."""
        return {
            self.kin_robot,
            self.robot,
            self.block,
            self.kin_rectangle,
            self.dyn_rectangle,
            self.dynamic2d,
        }

    def as_dict(self) -> dict[str, Type]:
        """Convert to dictionary of types."""
        return {
            "robot": self.robot,
            "block": self.block,
            "kin_robot": self.kin_robot,
            "kin_rectangle": self.kin_rectangle,
            "dyn_rectangle": self.dyn_rectangle,
            "dynamic2d": self.dynamic2d,
        }


class BlockedStackingPredicates(PredicateContainer):
    """Container for StickButton predicates."""

    def __init__(self, types: BlockedStackingTypes) -> None:
        """Initialize predicates."""
        readygrasp = Predicate("ReadyGrasp", [types.robot, types.block])
        readyplace = Predicate("ReadyPlace", [types.robot, types.block, types.block])
        holding = Predicate("Holding", [types.robot, types.block])
        on = Predicate("On", [types.block, types.block])
        self.predicates = {
            "ReadyGrasp": readygrasp,
            "ReadyPlace": readyplace,
            "Holding": holding,
            "On": on,
        }


class BlockedStackingObjects(ObjectContainer):
    """Container for ClutteredTable objects."""

    def __init__(self, types: BlockedStackingTypes) -> None:
        """Initialize objects."""
        self.robot = Object("robot", types.robot)
        self.grasp_block = Object("grasp_block", types.block)
        self.base_block = Object("base_block", types.block)

    def as_set(self) -> set[Object]:
        """Convert to set of objects."""
        return {
            self.robot,
            self.grasp_block,
            self.base_block,
        }

    def as_dict(self) -> dict[str, Object]:
        """Convert to dictionary of objects."""
        robot_name = self.robot.name
        grasp_block_name = self.grasp_block.name
        base_block_name = self.base_block.name
        return {
            robot_name: self.robot,
            grasp_block_name: self.grasp_block,
            base_block_name: self.base_block,
        }

    @property
    def object_to_node(self) -> Dict[Object, int]:
        """Get mapping from objects to their node indices."""
        return {
            self.robot: 0,
            self.grasp_block: 1,
            self.base_block: 2,
        }


def extract_robot_pose(obs: Tensor) -> Tensor:
    """Extract robot position + vel from observation."""
    return obs[:, 30:36].clone()  # shape (B, 6)


def extract_grasp_block_pose(obs: Tensor) -> Tensor:
    """Extract grasp_block position + vel from observation."""
    return obs[:, 0:6].clone()


def extract_base_block_pose(obs: Tensor) -> Tensor:
    """Extract base_block position + vel from observation."""
    return obs[:, 15:21].clone()


def extract_robot_joints(obs: Tensor) -> Tensor:
    """Extract robot joint positions from observation."""
    return torch.cat([obs[:, [38, 42]]], dim=1)  # shape (B, 2)


def extract_object_pose(obs: Tensor, obj: Object) -> Tensor:
    """Extract object position from observation."""
    if obj.name == "robot":
        return extract_robot_pose(obs)
    if obj.name == "grasp_block":
        return extract_grasp_block_pose(obs)
    if obj.name == "base_block":
        return extract_base_block_pose(obs)
    raise ValueError(f"Unknown object: {obj.name}")


def extract_object_shape(obs: Tensor, obj: Object) -> Tensor:
    """Extract shape features for the given object from observation tensor."""
    if obj.name == "grasp_block":
        return obs[:, 11:13].clone()
    if obj.name == "base_block":
        return obs[:, 26:28].clone()
    if obj.name == "robot":
        return torch.zeros((obs.shape[0], 2), dtype=obs.dtype, device=obs.device)
    raise ValueError(f"Unknown object: {obj.name}")


class BaseBlockedStackingSkill(LiftedOperatorSkill):
    """Base class for ClutteredTable environment skills."""

    def __init__(
        self, env: BlockedStacking2DEnv, operators: set[LiftedOperator]
    ) -> None:
        """Initialize skill."""
        super().__init__()
        self._all_operators = operators
        self._lifted_operator = self.get_lifted_operator()
        action_space = cast(gym.spaces.Box, env.action_space)
        self.action_dim = action_space.shape[0] if action_space.shape is not None else 0
        self._max_dx = action_space.high[0] * 0.8
        self._max_dy = action_space.high[1] * 0.8
        self._max_dtheta = action_space.high[2] * 0.8
        self._max_darm = action_space.high[3] * 0.8
        self._max_dfinger = action_space.high[4] * 0.8
        self.device = CFG.device  # env has no device in this domain
        self._current_plan: List[Tensor] | None = []
        # NOTE: self.arm_action_low and self.arm_action_high are different
        # from self._max_d*, they are the real min/max of the action space
        # in the environment, used for normalization.
        self.normalize_action = CFG.normalize_action
        self.arm_action_low = torch.tensor(
            [
                action_space.low[0],
                action_space.low[1],
                action_space.low[2],
                action_space.low[3],
                action_space.low[4],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.arm_action_high = torch.tensor(
            [
                action_space.high[0],
                action_space.high[1],
                action_space.high[2],
                action_space.high[3],
                action_space.high[4],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self, ground_operator: GroundOperator) -> None:
        """Reset skill with ground operator."""
        self._current_plan = []
        return super().reset(ground_operator)

    def _obs_to_kinematic_state(self, obs: Tensor) -> dict[str, Tensor]:
        """Convert observation to kinematic state."""
        return {
            "robot_pose": extract_robot_pose(obs),
            "robot_joints": extract_robot_joints(obs),
            "grasp_block_pose": extract_grasp_block_pose(obs),
            "base_block_pose": extract_base_block_pose(obs),
        }

    @abc.abstractmethod
    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], kinematic_state: dict
    ) -> List[Tensor]:
        """Get kinematic plan given objects and observation."""
        raise NotImplementedError

    def get_action_given_objects(
        self, objects: Sequence[Object], obs: Tensor
    ) -> Tensor:
        """Get action given objects and observation."""
        if not self._current_plan:
            kinematic_state = self._obs_to_kinematic_state(obs)
            # get kinematic plan (qpos) given objects and observation
            self._current_plan = self._get_kinematic_plan_given_objects(
                objects, kinematic_state
            )
        delta_action = torch.zeros(
            obs.shape[0], self.action_dim, dtype=obs.dtype, device=obs.device
        )
        curr_robot_base_pos = extract_robot_pose(obs)[:, 0:3]
        curr_robot_joints = extract_robot_joints(obs)
        curr_robot_state = torch.cat([curr_robot_base_pos, curr_robot_joints], dim=1)
        # NOTE: We assume the next action will always take the robot to the next
        # waypoint in the plan, unless it is forced to stop earlier due to reaching
        # desired effects. If the plan is exhausted, we will output frozon action.
        next_tgt_state = self._current_plan.pop(0)
        delta_action = next_tgt_state - curr_robot_state

        # Clip delta action to be within max ranges
        # Create min/max bounds tensor: [dx, dy, dtheta, darm, dfinger]
        max_deltas = torch.tensor(
            [
                self._max_dx,
                self._max_dy,
                self._max_dtheta,
                self._max_darm,
                self._max_dfinger,
            ],
            device=delta_action.device,
            dtype=delta_action.dtype,
        )
        min_deltas = -max_deltas

        # Clip each dimension: (B, 5)
        clipped_delta_action = torch.clamp(delta_action, min_deltas, max_deltas)

        # Normalize if needed
        if self.normalize_action:
            # Normalize the delta_qpos to be within [low, high]
            # self.arm_action_low is -0.2, self.arm_action_high is 0.2
            # NOT the actual delta min/max
            low = self.arm_action_low.unsqueeze(0).repeat(
                clipped_delta_action.shape[0], 1
            )
            high = self.arm_action_high.unsqueeze(0).repeat(
                clipped_delta_action.shape[0], 1
            )
            clipped_delta_action_norm = (clipped_delta_action - 0.5 * (low + high)) / (
                0.5 * (high - low)
            )
            return clipped_delta_action_norm

        return clipped_delta_action

    def get_lifted_operator(self) -> LiftedOperator:
        """Get the operator this skill implements."""
        return next(
            op for op in self._all_operators if op.name == self.get_operator_name()
        )

    def get_operator_name(self) -> str:
        """Get the name of the operator this skill implements."""
        raise NotImplementedError

    def _waypoint_to_traj(self, way_points: List[Tensor]) -> List[Tensor]:
        """Convert sparse waypoints to fine-grained way-points considering max delta."""
        if len(way_points) <= 1:
            return way_points

        # Stack max deltas for vectorized operations: [dx, dy, dtheta, darm, dfinger]
        max_deltas = torch.tensor(
            [
                self._max_dx,
                self._max_dy,
                self._max_dtheta,
                self._max_darm,
                self._max_dfinger,
            ],
            device=self.device,
            dtype=way_points[0].dtype,
        )

        fine_waypoints = [way_points[0]]  # Start with first waypoint

        for i in range(len(way_points) - 1):
            start = way_points[i]  # (B, 5)
            end = way_points[i + 1]  # (B, 5)

            # Compute differences between consecutive waypoints
            diff = end - start  # (B, 5)

            # Compute number of steps needed for each dimension and batch
            steps_needed = torch.abs(diff) / max_deltas.unsqueeze(0)  # (B, 5)

            # Take maximum across dimensions for each batch element
            max_steps_per_batch = torch.ceil(torch.max(steps_needed, dim=1)[0])  # (B,)

            # Find the maximum number of steps across all batches
            max_steps = int(torch.max(max_steps_per_batch).item())

            if max_steps == 0:
                continue

            # Generate all intermediate waypoints at once using vectorized operations
            # Create step indices: [1, 2, ..., max_steps]
            step_indices = torch.arange(
                1, max_steps + 1, device=self.device, dtype=start.dtype
            )  # (max_steps,)

            # Compute interpolation factors for all steps and batches
            # step_indices: (max_steps,), max_steps_per_batch: (B,) -> (max_steps, B)
            alpha = torch.minimum(
                step_indices.unsqueeze(1)
                / max_steps_per_batch.unsqueeze(0),  # (max_steps, B)
                torch.ones_like(max_steps_per_batch).unsqueeze(0),  # (1, B)
            )  # (max_steps, B)

            # Expand dimensions for broadcasting: start/end: (B, 5), diff: (B, 5)
            # alpha: (max_steps, B) -> (max_steps, B, 1)
            alpha = alpha.unsqueeze(-1)  # (max_steps, B, 1)
            start_expanded = start.unsqueeze(0)  # (1, B, 5)
            diff_expanded = diff.unsqueeze(0)  # (1, B, 5)

            # Generate all intermediate waypoints: (max_steps, B, 5)
            all_intermediates = start_expanded + alpha * diff_expanded

            # Convert to list of tensors with shape (B, 5)
            for step_idx in range(max_steps):
                fine_waypoints.append(all_intermediates[step_idx])  # (B, 5)

        return fine_waypoints


class ReachToGraspSkill(BaseBlockedStackingSkill):
    """Skill for Reach an Object for Grasping."""

    def get_operator_name(self) -> str:
        return "ReachToGrasp"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        kinematic_state: dict,
    ) -> List[Tensor]:
        """Get kinematic plan for reaching to grasp."""
        curr_robot_base_pos = kinematic_state["robot_pose"][:, 0:3]
        curr_robot_joints = kinematic_state["robot_joints"]
        way_points = [torch.cat([curr_robot_base_pos, curr_robot_joints], dim=1)]
        target_obj_pose = None
        if objects[1].name == "grasp_block":
            target_obj_pose = kinematic_state["grasp_block_pose"]
        elif objects[1].name == "base_block":
            target_obj_pose = kinematic_state["base_block_pose"]
        else:
            raise ValueError(f"Unknown target object: {objects[1].name}")

        assert target_obj_pose is not None
        # Safe pose above everything
        way_points.append(
            torch.cat(
                [
                    curr_robot_base_pos[:, 0:1],
                    5
                    * CFG.blocked2d_robot_base_radius
                    * torch.ones_like(curr_robot_base_pos[:, 1:2]),
                    curr_robot_base_pos[:, 2:3],
                    CFG.blocked2d_robot_base_radius
                    * torch.ones_like(curr_robot_base_pos[:, 1:2]),
                    CFG.blocked2d_gripper_base_height
                    * torch.ones_like(curr_robot_joints[:, 1:2]),
                ],
                dim=1,
            )
        )
        # Above the target object, facing down
        way_points.append(
            torch.cat(
                [
                    target_obj_pose[:, 0:1],
                    5
                    * CFG.blocked2d_robot_base_radius
                    * torch.ones_like(curr_robot_base_pos[:, 1:2]),
                    (-torch.pi / 2) * torch.ones_like(curr_robot_base_pos[:, 2:3]),
                    CFG.blocked2d_robot_base_radius
                    * torch.ones_like(curr_robot_base_pos[:, 1:2]),
                    CFG.blocked2d_gripper_base_height
                    * torch.ones_like(curr_robot_joints[:, 1:2]),
                ],
                dim=1,
            )
        )
        # Approach the target object as close as possible
        way_points.append(
            torch.cat(
                [
                    target_obj_pose[:, 0:1],
                    target_obj_pose[:, 1:2],
                    (-torch.pi / 2) * torch.ones_like(curr_robot_base_pos[:, 2:3]),
                    CFG.blocked2d_robot_base_radius
                    * torch.ones_like(curr_robot_base_pos[:, 1:2]),
                    CFG.blocked2d_gripper_base_height
                    * torch.ones_like(curr_robot_joints[:, 1:2]),
                ],
                dim=1,
            )
        )
        actions = self._waypoint_to_traj(way_points)
        return actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when close enough to the target object.

        Using ReadyGrasp predicate logic.
        """
        assert len(objects) == 3
        assert objects[0].name == "robot"
        assert objects[1].name in ["grasp_block", "base_block"]
        robot_pos = extract_robot_pose(obs)[:, 0:2]
        target_obj_pos = extract_object_pose(obs, objects[1])[:, 0:2]
        target_obj_shape = extract_object_shape(obs, objects[1])
        distance_x = torch.abs(robot_pos[:, 0] - target_obj_pos[:, 0])
        distance_y = torch.abs(robot_pos[:, 1] - target_obj_pos[:, 1])
        holding = torch.logical_or(
            obs[:, 27].to(torch.bool),
            obs[:, 42].to(torch.bool),
        )  # bool tensor (NUM_ENVS,)
        dy_max = (
            target_obj_shape[:, 1] / 2
            + CFG.blocked2d_robot_base_radius
            + CFG.blocked2d_gripper_finger_width / 2
        )
        distance_ok = (
            (distance_x < CFG.blocked2d_gripper_base_height / 4)
            & (distance_y < dy_max)
            & (~holding)
        )  # shape (B, num_pairs)
        assert self._current_plan is not None
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(distance_ok, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(distance_ok, dtype=torch.bool)
        )
        terminated = distance_ok | not_have_current_plan_tensor
        return terminated


class ReachToPlaceSkill(ReachToGraspSkill):
    """Skill for Reach an Object for Placing."""

    def get_operator_name(self) -> str:
        return "ReachToPlace"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        kinematic_state: dict,
    ) -> List[Tensor]:
        """Get kinematic plan for reaching to place."""
        curr_robot_base_pos = kinematic_state["robot_pose"][:, 0:3]
        curr_robot_joints = kinematic_state["robot_joints"]
        way_points = [torch.cat([curr_robot_base_pos, curr_robot_joints], dim=1)]
        target_obj_pose = None
        if objects[2].name == "grasp_block":
            target_obj_pose = kinematic_state["grasp_block_pose"]
        elif objects[2].name == "base_block":
            target_obj_pose = kinematic_state["base_block_pose"]
        else:
            raise ValueError(f"Unknown target object: {objects[2].name}")

        assert target_obj_pose is not None
        # Safe pose above everything
        way_points.append(
            torch.cat(
                [
                    curr_robot_base_pos[:, 0:1],
                    5
                    * CFG.blocked2d_robot_base_radius
                    * torch.ones_like(curr_robot_base_pos[:, 1:2]),
                    (-torch.pi / 2) * torch.ones_like(curr_robot_base_pos[:, 2:3]),
                    CFG.blocked2d_robot_base_radius
                    * torch.ones_like(curr_robot_base_pos[:, 1:2]),
                    curr_robot_joints[:, -1:],
                ],
                dim=1,
            )
        )
        # Above the target object
        way_points.append(
            torch.cat(
                [
                    target_obj_pose[:, 0:1],
                    5
                    * CFG.blocked2d_robot_base_radius
                    * torch.ones_like(curr_robot_base_pos[:, 1:2]),
                    (-torch.pi / 2) * torch.ones_like(curr_robot_base_pos[:, 2:3]),
                    CFG.blocked2d_robot_base_radius
                    * torch.ones_like(curr_robot_base_pos[:, 1:2]),
                    curr_robot_joints[:, -1:],
                ],
                dim=1,
            )
        )
        # Approach the target object as close as possible
        way_points.append(
            torch.cat(
                [
                    target_obj_pose[:, 0:1],
                    target_obj_pose[:, 1:2],
                    (-torch.pi / 2) * torch.ones_like(curr_robot_base_pos[:, 2:3]),
                    CFG.blocked2d_robot_base_radius
                    * torch.ones_like(curr_robot_base_pos[:, 1:2]),
                    curr_robot_joints[:, -1:],
                ],
                dim=1,
            )
        )
        actions = self._waypoint_to_traj(way_points)
        return actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when close enough to the target object.

        Using ReadyPlace predicate logic.
        """
        assert len(objects) == 3
        assert objects[0].name == "robot"
        assert objects[1].name in ["grasp_block", "base_block"]
        assert objects[2].name in ["grasp_block", "base_block"]
        target_obj1_pos = extract_object_pose(obs, objects[1])[:, 0:2]
        target_obj2_pos = extract_object_pose(obs, objects[2])[:, 0:2]
        target_obj1_shape = extract_object_shape(obs, objects[1])
        target_obj2_shape = extract_object_shape(obs, objects[2])
        distance_x = torch.abs(target_obj1_pos[:, 0] - target_obj2_pos[:, 0])
        distance_y = target_obj1_pos[:, 1] - target_obj2_pos[:, 1]
        holding = (
            obs[:, 27].to(torch.bool)
            if objects[1].name == "grasp_block"
            else obs[:, 42].to(torch.bool)
        )  # bool tensor (NUM_ENVS,)
        dy_max = (
            target_obj2_shape[:, 1] / 2
            + target_obj1_shape[:, 1] / 2
            + CFG.blocked2d_gripper_base_width
        )
        distance_ok = (
            (distance_x < target_obj2_shape[:, 0] / 4)
            & (distance_y > 0)
            & (distance_y < dy_max)
            & holding
        )
        assert self._current_plan is not None
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(distance_ok, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(distance_ok, dtype=torch.bool)
        )
        terminated = distance_ok | not_have_current_plan_tensor
        return terminated


class GraspSkill(BaseBlockedStackingSkill):
    """Skill for grasping the block."""

    def get_operator_name(self) -> str:
        return "Grasp"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        kinematic_state: dict,
    ) -> List[Tensor]:
        """Get kinematic plan for grasping."""
        del objects  # unused
        curr_robot_base_pos = kinematic_state["robot_pose"][:, 0:3]
        curr_robot_joints = kinematic_state["robot_joints"]
        way_points = [torch.cat([curr_robot_base_pos, curr_robot_joints], dim=1)]
        # Close the finger to zero gap
        way_points.append(
            torch.cat(
                [
                    curr_robot_base_pos,
                    CFG.blocked2d_robot_base_radius
                    * torch.ones_like(curr_robot_joints[:, 0:1]),
                    torch.zeros_like(curr_robot_joints[:, -1:]),
                ],
                dim=1,
            )
        )
        actions = self._waypoint_to_traj(way_points)

        return actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when the block is grasped."""
        assert len(objects) == 2
        assert objects[0].name == "robot"
        assert objects[1].name in ["grasp_block", "base_block"]
        holding = (
            obs[:, 27].to(torch.bool)
            if objects[1].name == "grasp_block"
            else obs[:, 42].to(torch.bool)
        )  # bool tensor (NUM_ENVS,)
        assert self._current_plan is not None
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(holding, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(holding, dtype=torch.bool)
        )
        terminated = holding | not_have_current_plan_tensor
        return terminated


class PlaceSkill(BaseBlockedStackingSkill):
    """Skill for placing the hammar."""

    def get_operator_name(self) -> str:
        return "Place"

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        kinematic_state: dict,
    ) -> List[Tensor]:
        """Get kinematic plan for placing."""
        del objects  # unused
        curr_robot_base_pos = kinematic_state["robot_pose"][:, 0:3]
        curr_robot_joints = kinematic_state["robot_joints"]
        way_points = [torch.cat([curr_robot_base_pos, curr_robot_joints], dim=1)]
        # Close the finger to zero gap
        way_points.append(
            torch.cat(
                [
                    curr_robot_base_pos,
                    CFG.blocked2d_robot_base_radius
                    * torch.ones_like(curr_robot_joints[:, 0:1]),
                    CFG.blocked2d_gripper_base_height
                    * torch.ones_like(curr_robot_joints[:, -1:]),
                ],
                dim=1,
            )
        )
        actions = self._waypoint_to_traj(way_points)

        return actions

    def terminate_with_objects(self, objects, obs):
        """Terminate when the placing is done.

        Using On predicate logic.
        """
        assert len(objects) == 3
        assert objects[1].name in ["grasp_block", "base_block"]
        assert objects[2].name in ["grasp_block", "base_block"]
        target_obj1_pos = extract_object_pose(obs, objects[1])[:, 0:2]
        target_obj2_pos = extract_object_pose(obs, objects[2])[:, 0:2]
        target_obj1_shape = extract_object_shape(obs, objects[1])
        target_obj2_shape = extract_object_shape(obs, objects[2])
        distance_x = torch.abs(target_obj1_pos[:, 0] - target_obj2_pos[:, 0])
        distance_y = target_obj1_pos[:, 1] - target_obj2_pos[:, 1]
        dy_max = target_obj2_shape[:, 1] / 2 + target_obj1_shape[:, 1] / 2 + 1e-3
        on = (
            (distance_x < target_obj2_shape[:, 0] / 4)
            & (distance_y > 0)
            & (distance_y < dy_max)
        )
        assert self._current_plan is not None
        not_have_current_plan = len(self._current_plan) == 0
        not_have_current_plan_tensor = (
            torch.ones_like(on, dtype=torch.bool)
            if not_have_current_plan
            else torch.zeros_like(on, dtype=torch.bool)
        )
        terminated = on | not_have_current_plan_tensor
        return terminated


class BlockedStackingPerceiver(Perceiver):
    """Perceiver for BlockedStacking environment."""

    def __init__(
        self, predicates: BlockedStackingPredicates, types: BlockedStackingTypes
    ) -> None:
        """Initialize with required types."""
        self.predicates_container = predicates
        self._types = types
        self.objects = BlockedStackingObjects(types)
        self.predicate_interpreters = {
            predicates["ReadyGrasp"]: self._interpret_readygrasp,
            predicates["ReadyPlace"]: self._interpret_readyplace,
            predicates["Holding"]: self._interpret_holding,
            predicates["On"]: self._interpret_on,
        }

    def reset(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver with observation and info."""
        # self.objects remains the same from any task
        assert (
            obs.shape[0] == 1
        ), "Expected batch size of 1 for observation for perceiver reset."
        # Note that for now we assume all the tasks have the same initial atoms (so the same task plan).
        atoms = self._get_atoms(obs, info)
        # hammer is at the target position
        grasp_block = self.objects.as_dict()["grasp_block"]
        base_block = self.objects.as_dict()["base_block"]
        goal = {self.predicates_container["On"]([grasp_block, base_block])}
        return self.objects.as_set(), atoms[0], goal

    def step(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> List[set[GroundAtom]]:
        """Step perceiver with observation."""
        return self._get_atoms(obs, info)

    def _get_atoms(
        self,
        obs: Tensor,
        info: Optional[Dict] = None,
    ) -> List[set[GroundAtom]]:
        """Convert a batch of observations into a List of GroundAtom-sets, one per batch
        element.

        Args:
            obs: Tensor of shape (B, …)

        Returns:
            List of length B, where each entry is the set of GroundAtoms for that obs.
        """
        B = obs.shape[0]
        # start with an empty set for each batch element
        atoms_List: List[set[GroundAtom]] = [set() for _ in range(B)]

        # each interpreter now returns a List[Set[GroundAtom]] of length B
        desired_predicates = list(self.predicates_container.predicates.keys())
        if info is not None and "desired_predicates" in info:
            # If info is provided, we can filter the predicates to only those desired
            desired_predicates = [
                p
                for p in info["desired_predicates"]
                if p in self.predicates_container.predicates
            ]
        for predicate, interpret_fn in self.predicate_interpreters.items():
            if predicate.name not in desired_predicates:
                # If the predicate is not desired, skip it
                continue
            # First get the arguments for the predicate
            # which will extend the number of observation inputs

            # Now we assume all the batch has the same number of objects
            # meaning that we can use the same arguments for all batches.
            input_args: List[List[Object]] = []
            for args in get_object_combinations(
                self.objects.as_set(),
                predicate.types,
            ):
                input_args.append(args)
            # args: List[List[Object]] of length num_grounding
            # stacked_res: Tensor of shape (B, num_grounding)
            stacked_res = interpret_fn(obs, input_args)  # type: ignore[arg-type]
            assert stacked_res.shape[0] == B
            assert stacked_res.shape[1] == len(input_args)
            for b in range(B):
                for n in range(len(input_args)):
                    if stacked_res[b, n]:
                        if predicate.arity > 0:
                            atoms_List[b].add(predicate(input_args[n]))
                        else:
                            atoms_List[b].add(GroundAtom(predicate, []))
        return atoms_List

    def _interpret_readygrasp(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret ReadyGrasp predicate."""
        # NOTE: We assume the robot is always grasping the object right above it
        # with arm joint = 0.0.
        obj_positions = []
        obj1_shapes = []
        for obj_pair in objects:
            obj0_p = extract_object_pose(obs, obj_pair[0])[:, 0:2]
            obj1_p = extract_object_pose(obs, obj_pair[1])[:, 0:2]
            obj1_shapes.append(extract_object_shape(obs, obj_pair[1]))
            obj_positions.append(torch.stack([obj0_p, obj1_p], dim=1))
        obj_position_stacked = torch.stack(
            obj_positions, dim=1
        )  # shape (B, num_pairs, 2, 2)
        obj1_shapes_stacked = torch.stack(obj1_shapes, dim=1)  # shape (B, num_pairs, 2)
        distance_x = torch.abs(
            obj_position_stacked[:, :, 0, 0] - obj_position_stacked[:, :, 1, 0]
        )  # shape (B, num_pairs)
        distance_y = torch.abs(
            obj_position_stacked[:, :, 0, 1] - obj_position_stacked[:, :, 1, 1]
        )
        holding = torch.logical_or(
            obs[:, 27].to(torch.bool),
            obs[:, 42].to(torch.bool),
        )  # bool tensor (NUM_ENVS,)
        holding = holding.unsqueeze(1).repeat(
            1, obj_position_stacked.shape[1]
        )  # shape (B, num_pairs)
        dy_max = (
            obj1_shapes_stacked[:, :, 1] / 2
            + CFG.blocked2d_robot_base_radius
            + CFG.blocked2d_gripper_finger_width / 2
        )
        distance_ok = (
            (distance_x < CFG.blocked2d_gripper_base_height / 4)
            & (distance_y < dy_max)
            & (~holding)
        )  # shape (B, num_pairs)
        return distance_ok

    def _interpret_readyplace(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret ReadyPlace predicate."""
        obj_positions = []
        obj1_shapes = []
        obj2_shapes = []
        holding_obj1 = []
        for obj_pair in objects:
            obj1_p = extract_object_pose(obs, obj_pair[1])[:, 0:2]
            obj2_p = extract_object_pose(obs, obj_pair[2])[:, 0:2]
            obj1_shape = extract_object_shape(obs, obj_pair[1])
            obj2_shape = extract_object_shape(obs, obj_pair[2])
            obj1_shapes.append(obj1_shape)
            obj2_shapes.append(obj2_shape)
            obj_positions.append(torch.stack([obj1_p, obj2_p], dim=1))
            if obj_pair[1].name == "grasp_block":
                holding_obj1.append(obs[:, 27].to(torch.bool))
            elif obj_pair[1].name == "base_block":
                holding_obj1.append(obs[:, 42].to(torch.bool))
            else:
                holding_obj1.append(
                    torch.zeros((obs.shape[0],), dtype=torch.bool, device=obs.device)
                )
        obj_position_stacked = torch.stack(
            obj_positions, dim=1
        )  # shape (B, num_pairs, 2, 3)
        obj1_shapes_stacked = torch.stack(obj1_shapes, dim=1)  # shape (B, num_pairs, 2)
        obj2_shapes_stacked = torch.stack(obj2_shapes, dim=1)  # shape (B, num_pairs, 2)
        distance_x = torch.abs(
            obj_position_stacked[:, :, 0, 0] - obj_position_stacked[:, :, 1, 0]
        )
        distance_y = obj_position_stacked[:, :, 0, 1] - obj_position_stacked[:, :, 1, 1]
        holding = torch.stack(holding_obj1, dim=1)  # shape (B, num_pairs)
        dy_max = (
            obj2_shapes_stacked[:, :, 1] / 2
            + obj1_shapes_stacked[:, :, 1] / 2
            + CFG.blocked2d_gripper_base_width
        )
        distance_ok = (
            (distance_x < obj2_shapes_stacked[:, :, 0] / 4)
            & (distance_y > 0)
            & (distance_y < dy_max)
            & holding
        )
        return distance_ok

    def _interpret_on(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret On predicate."""
        obj_positions = []
        obj0_shapes = []
        obj1_shapes = []
        holding_obj0 = []
        diff_obj = torch.zeros(
            (obs.shape[0], len(objects)), dtype=torch.bool, device=obs.device
        )
        for obj_pair in objects:
            if (obj_pair[0].name != obj_pair[1].name) and (
                obj_pair[0].name == "grasp_block"
            ):
                # If both objects are the same, we cannot compute distance
                diff_obj[:, objects.index(obj_pair)] = True
            obj0_p = extract_object_pose(obs, obj_pair[0])[:, 0:2]
            obj1_p = extract_object_pose(obs, obj_pair[1])[:, 0:2]
            obj_positions.append(torch.stack([obj0_p, obj1_p], dim=1))
            obj0_shape = extract_object_shape(obs, obj_pair[0])
            obj1_shape = extract_object_shape(obs, obj_pair[1])
            obj0_shapes.append(obj0_shape)
            obj1_shapes.append(obj1_shape)
            if obj_pair[0].name == "grasp_block":
                holding_obj0.append(obs[:, 27].to(torch.bool))
            elif obj_pair[0].name == "base_block":
                holding_obj0.append(obs[:, 42].to(torch.bool))
            else:
                holding_obj0.append(
                    torch.zeros((obs.shape[0],), dtype=torch.bool, device=obs.device)
                )
        obj_position_stacked = torch.stack(
            obj_positions, dim=1
        )  # shape (B, num_pairs, 2, 3)
        obj0_shapes_stacked = torch.stack(obj0_shapes, dim=1)  # shape (B, num_pairs, 2)
        obj1_shapes_stacked = torch.stack(obj1_shapes, dim=1)  # shape (B, num_pairs, 2)
        distance_x = torch.abs(
            obj_position_stacked[:, :, 0, 0] - obj_position_stacked[:, :, 1, 0]
        )
        distance_y = obj_position_stacked[:, :, 0, 1] - obj_position_stacked[:, :, 1, 1]
        holding = torch.stack(holding_obj0, dim=1)
        dy_max = (
            obj1_shapes_stacked[:, :, 1] / 2
            + obj0_shapes_stacked[:, :, 1] / 2
            + CFG.blocked2d_gripper_base_width
        )
        is_on = (
            (distance_x < obj1_shapes_stacked[:, :, 0] / 4)
            & (distance_y > 0)
            & (distance_y < dy_max)
            & ~holding
        )
        return is_on

    def _interpret_holding(
        self,
        obs: Tensor,
        objects: List[Sequence[Object]],
    ) -> Tensor:
        """Interpret Holding predicate."""
        holding_obj1 = []
        for obj_pair in objects:
            if obj_pair[1].name == "grasp_block":
                holding_obj1.append(obs[:, 27].to(torch.bool))
            elif obj_pair[1].name == "base_block":
                holding_obj1.append(obs[:, 42].to(torch.bool))
            else:
                holding_obj1.append(
                    torch.zeros((obs.shape[0],), dtype=torch.bool, device=obs.device)
                )
        return torch.stack(holding_obj1, dim=1)  # shape (B, num_pairs)


class BlockedStackingRLTAMPSystem(BaseRLTAMPSystem):
    """Base TAMP system for  BlockedStacking environment."""

    def __init__(
        self,
        planning_components: TensorPlanningComponents,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize BlockedStacking2D TAMP system."""
        self._render_mode = render_mode
        self.env_kwargs = {
            "render_mode": render_mode,
        }
        self.env_name = "skill_ref/BlockedStacking2D-v0"
        super().__init__(
            planning_components, name=" BlockedStackingTAMPSystem", seed=seed  # type: ignore
        )

    def _create_env(self) -> gym.Env:
        """Create base environment."""

        def make_env():
            return gym.make(
                self.env_name,
                **self.env_kwargs,
            )

        if CFG.normalize_action:
            norm_envs = NormalizeActionMultiEnvWrapper(  # type: ignore
                make_env,
                num_envs=CFG.num_envs,
                auto_reset=False,
                to_tensor=True,
                device=CFG.device,
                max_episode_steps=CFG.max_env_steps,
            )
            return norm_envs  # type: ignore
        envs = MultiEnvWrapper(
            make_env,
            num_envs=CFG.num_envs,
            auto_reset=False,
            to_tensor=True,
            device=CFG.device,
            max_episode_steps=CFG.max_env_steps,
        )
        return envs

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "BlockedStacking2D-domain"

    def get_domain(self) -> PDDLDomain:
        """Get domain."""
        return PDDLDomain(
            self._get_domain_name(),
            self.components.operators,
            self.components.predicate_container.as_set(),
            self.components.type_container.as_set(),
        )

    @classmethod
    def _create_planning_components(cls) -> TensorPlanningComponents:
        """Create planning components for graph-based StickButton system."""
        types_container = BlockedStackingTypes()

        predicates = BlockedStackingPredicates(types_container)
        ReadyGrasp = predicates["ReadyGrasp"]
        ReadyPlace = predicates["ReadyPlace"]
        On = predicates["On"]
        Holding = predicates["Holding"]

        perceiver = BlockedStackingPerceiver(predicates, types_container)

        robot = Variable("?robot", types_container.robot)
        grasp_block = Variable("?grasp_block", types_container.block)
        base_block = Variable("?base_block", types_container.block)

        operators = {
            LiftedOperator(
                "ReachToGrasp",
                [robot, grasp_block],
                preconditions=set(),
                add_effects={
                    ReadyGrasp([robot, grasp_block]),
                },
                delete_effects=set(),
            ),
            LiftedOperator(
                "ReachToPlace",
                [robot, grasp_block, base_block],
                preconditions={
                    Holding([robot, grasp_block]),
                },
                add_effects={
                    ReadyPlace([robot, grasp_block, base_block]),
                },
                delete_effects=set(),
            ),
            LiftedOperator(
                "Grasp",
                [robot, grasp_block],
                preconditions={
                    ReadyGrasp([robot, grasp_block]),
                },
                add_effects={
                    Holding([robot, grasp_block]),
                },
                delete_effects={
                    ReadyGrasp([robot, grasp_block]),
                },
            ),
            LiftedOperator(
                "Place",
                [robot, grasp_block, base_block],
                preconditions={
                    ReadyPlace([robot, grasp_block, base_block]),
                    Holding([robot, grasp_block]),
                },
                add_effects={
                    On([grasp_block, base_block]),
                },
                delete_effects={
                    ReadyPlace([robot, grasp_block, base_block]),
                    Holding([robot, grasp_block]),
                },
            ),
        }

        return TensorPlanningComponents(
            type_container=types_container,
            predicate_container=predicates,
            operators=operators,
            skills=set(),
            perceiver=perceiver,
        )

    @classmethod
    def create_default(
        cls,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> BlockedStackingRLTAMPSystem:
        """Factory method for creating system with default components."""
        planning_components = cls._create_planning_components()
        system = cls(
            planning_components,
            seed=seed,
            render_mode=render_mode,
        )
        assert isinstance(system.env.unwrapped[0], BlockedStacking2DEnv)  # type: ignore
        skills = {
            ReachToGraspSkill(system.env.unwrapped[0], system.components.operators),  # type: ignore
            ReachToPlaceSkill(system.env.unwrapped[0], system.components.operators),  # type: ignore
            GraspSkill(system.env.unwrapped[0], system.components.operators),  # type: ignore
            PlaceSkill(system.env.unwrapped[0], system.components.operators),  # type: ignore
        }
        system.components.skills.update(skills)  # type: ignore
        return system

    def state_to_graph(self, state: Tensor) -> List[GraphData]:
        """Convert Batched BlockedStacking environment state tensor to graph
        representation.

        Args:
            state: State tensor from BlockedStacking environment, batched

        Returns:
            List of GraphData with nodes representing objects and edges representing relationships
        """
        # Get objects from the perceiver's object container
        b = state.shape[0]
        num_nodes = len(self.perceiver.objects.object_to_node.keys())

        # Node features 8: finger (1) + joint (1) + width + height (2) +
        # SE2 (x, y, cos(theta), sin(theta)) (4)
        node_feature_dim = 8
        node_features = torch.zeros(b, num_nodes, node_feature_dim)

        # Extract features for each object using existing helper functions
        robot_node_idx = 0
        for obj, i in self.perceiver.objects.object_to_node.items():
            # Extract object pose (position + rotation)
            obj_pose = extract_object_pose(state, obj)[:, :3].cpu()
            cos_theta = torch.cos(obj_pose[:, 2:3])
            sin_theta = torch.sin(obj_pose[:, 2:3])
            obj_pose_norm = torch.cat([obj_pose[:, 0:2], cos_theta, sin_theta], dim=-1)
            # Fill in node features
            node_features[:, i, 4:] = obj_pose_norm
            if obj.name == "robot":
                robot_node_idx = i
                node_features[:, i, :2] = extract_robot_joints(state).cpu()
            else:
                node_features[:, i, 2:4] = extract_object_shape(state, obj).cpu()

        # Create edges: fully connected graph between all objects
        edge_list = []
        edge_features_list = []
        # Edge features (10): finger (1) + joint (1) +
        # w_1, h_1, w_2, h_2 (4) +
        # SE2 relative pose (x, y, cos(theta), sin(theta)) (4)
        edge_feature_dim = 10

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops
                    edge_list.append([i, j])
                    # Compute proper relative pose between objects using SE(2) transformation
                    pos_mat_i = xy_cos_sin_to_matrix(
                        node_features[:, i, 4:8]
                    )  # [B, 3, 3]
                    pos_mat_j = xy_cos_sin_to_matrix(
                        node_features[:, j, 4:8]
                    )  # [B, 3, 3]
                    rel_pos_mat = torch.bmm(
                        torch.inverse(pos_mat_i), pos_mat_j
                    )  # [B, 3, 3]
                    rel_pos = matrix_to_xy_cos_sin(
                        rel_pos_mat
                    )  # [B, 4] relative pose in SE(2)
                    wh_i = node_features[:, i, 2:4]  # [B, 2]
                    wh_j = node_features[:, j, 2:4]  # [B, 2]

                    if i == robot_node_idx:
                        robo_joints = node_features[:, i, 0:2]  # [B, 2]
                        edge_feat = torch.cat(
                            [robo_joints, wh_i, wh_j, rel_pos], dim=-1
                        )  # 8-dim edge features
                    else:
                        zero_grasping = torch.zeros(
                            (b, 2), dtype=rel_pos.dtype, device=rel_pos.device
                        )
                        edge_feat = torch.cat(
                            [zero_grasping, wh_i, wh_j, rel_pos], dim=-1
                        )
                    edge_features_list.append(edge_feat)

        # Convert edge list to tensor format
        if edge_list:
            edge_indices = torch.tensor(edge_list, dtype=torch.long).T  # [2, num_edges]
            edge_features = torch.stack(
                edge_features_list, dim=1
            )  # [batch_size, num_edges, edge_feature_dim]
        else:
            edge_indices = torch.empty((2, 0), dtype=torch.long)
            edge_features = torch.empty((b, 0, edge_feature_dim))

        # Create GraphData objects for each batch element
        graph_data_list = []
        for batch_idx in range(b):
            graph_data = GraphData(
                node_features=node_features[batch_idx],  # [num_nodes, node_feature_dim]
                edge_features=edge_features[batch_idx],  # [num_edges, edge_feature_dim]
                edge_indices=edge_indices,  # [2, num_edges] - same for all batch elements
                global_features=None,
                object_to_node=self.perceiver.objects.object_to_node.copy(),
            )
            graph_data_list.append(graph_data)

        return graph_data_list
