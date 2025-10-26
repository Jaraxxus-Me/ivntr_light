"""Blocked Stacking Environment based on PRBench Dynamic2D Environment."""

from dataclasses import dataclass

import numpy as np
import pymunk
from gymnasium.utils import seeding
from prbench.core import ConstantObjectPRBenchEnv
from prbench.envs.dynamic2d.base_env import (
    Dynamic2DRobotEnvConfig,
    ObjectCentricDynamic2DRobotEnv,
)
from prbench.envs.dynamic2d.object_types import (
    Dynamic2DRobotEnvTypeFeatures,
    DynRectangleType,
    KinRectangleType,
    KinRobotType,
)
from prbench.envs.dynamic2d.utils import (
    DYNAMIC_COLLISION_TYPE,
    STATIC_COLLISION_TYPE,
    KinRobotActionSpace,
    create_walls_from_world_boundaries,
)
from prbench.envs.geom2d.structs import MultiBody2D, SE2Pose, ZOrder
from prbench.envs.utils import (
    PURPLE,
    rectangle_object_to_geom,
    sample_se2_pose,
    state_2d_has_collision,
)
from relational_structs import Array, Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict
from tomsgeoms2d.structs import Rectangle
from tomsgeoms2d.utils import geom2ds_intersect

from skill_refactor.settings import CFG

# Define custom object types for the obstruction environment
RobotType = Type("robot", parent=KinRobotType)
BlockType = Type("block", parent=DynRectangleType)
Dynamic2DRobotEnvTypeFeatures[BlockType] = list(
    Dynamic2DRobotEnvTypeFeatures[DynRectangleType] + ["grasped"]
)
Dynamic2DRobotEnvTypeFeatures[RobotType] = list(
    Dynamic2DRobotEnvTypeFeatures[KinRobotType] + ["is_colliding"]
)


@dataclass(frozen=True)
class BlockedStackingEnvConfig(Dynamic2DRobotEnvConfig):
    """Scene specification for BlockedStacking2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 0.8 + 0.8 * np.sqrt(5)  # golden ratio :)
    world_min_y: float = 0.0
    world_max_y: float = 2.0

    # Robot parameters
    init_robot_pos: tuple[float, float] = (1.5, 1.5)
    robot_base_radius: float = CFG.blocked2d_robot_base_radius
    robot_arm_length_max: float = CFG.blocked2d_robot_arm_length_max
    gripper_base_width: float = CFG.blocked2d_gripper_base_width
    gripper_base_height: float = CFG.blocked2d_gripper_base_height
    gripper_finger_width: float = CFG.blocked2d_gripper_finger_width
    gripper_finger_height: float = CFG.blocked2d_gripper_finger_height

    # Action space parameters.
    min_dx: float = -5e-2
    max_dx: float = 5e-2
    min_dy: float = -5e-2
    max_dy: float = 5e-2
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -1e-1
    max_darm: float = 1e-1
    min_dgripper: float = -0.02
    max_dgripper: float = 0.02

    # Controller parameters
    kp_pos: float = 50.0
    kv_pos: float = 5.0
    kp_rot: float = 50.0
    kv_rot: float = 5.0

    # Robot hyperparameters.
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(1.5, 1.5, -np.pi / 2),
        SE2Pose(1.5, 1.5, np.pi / 2),
    )

    # Table hyperparameters.
    table_rgb: tuple[float, float, float] = (0.75, 0.75, 0.75)
    table_height: float = 0.1
    table_width: float = world_max_x - world_min_x
    # The table pose is defined at the center
    table_pose: SE2Pose = SE2Pose(
        world_min_x + table_width / 2, world_min_y + table_height / 2, 0.0
    )

    # Grasp block.
    grasp_block_rgb: tuple[float, float, float] = PURPLE
    block_height: float = gripper_base_height - 1.6 * gripper_finger_height
    block_width: float = gripper_base_height - 1.6 * gripper_finger_height
    block_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + robot_base_radius + gripper_base_height,
            table_pose.y + table_height * 0.8 + 1e-6,
            0.0,
        ),
        SE2Pose(
            world_max_x - (robot_base_radius + gripper_base_height),
            table_pose.y + table_height * 0.8 + 1e-6,
            0.0,
        ),
    )
    grasp_block_mass: float = 1.0

    # Base block.
    base_block_rgb: tuple[float, float, float] = (0.75, 0.1, 0.1)
    base_block_mass: float = 1.0

    # Goal parameters.
    on_tol_dy_overlap: float = 1e-2  # distance tolerance for "on" relation
    on_tol_dy_relative: float = 1e-2  # distance tolerance for "on" relation
    on_tol_vel: float = 1e-2  # velocity tolerance for "on" relation

    # For sampling initial states.
    max_initial_state_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 100


class ObjectCentricBlockedStacking2DEnv(
    ObjectCentricDynamic2DRobotEnv[BlockedStackingEnvConfig]
):
    """**Task Description:** A simple task where the objective is to stack a block on
    top of another block with possible obstructions.

    **(Important) Observation:**
    - [0:6]: [x, y, theta, vx, vy, omega] of the grasp block
    - [11:13]: [w, h] of the grasp block
    - [14]: is_grasped for the grasp block (1 if the robot is holding the block, 0 otherwise)
    - [15, 21]: [x, y, theta, vx, vy, omega] of the base block
    - [26, 28]: [w, h] of the base block
    - [29]: is_grasped for the base block (1 if the robot is holding the block, 0 otherwise)
    - [30, 36]: [x, y, theta, vx, vy, omega] of the robot base
    - [38]: robot arm joint position
    - [42]: robot gripper gap
    - [45]: is_colliding (1 if the robot is colliding with the static objects, 0 otherwise)

    **Success Conditions:**
    - the object position is within `goal_thresh` euclidean distance of the goal position
    - the object is not held
    """

    def __init__(
        self,
        num_obstructions: int = 1,
        config: BlockedStackingEnvConfig = BlockedStackingEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._num_obstructions = num_obstructions

        # Store object references for tracking
        self._robot_obj: Object | None = None
        self._grasp_block: Object | None = None
        self._base_block: Object | None = None
        self._obstruction: Object | None = None
        # Provided in info dict
        self.elapsed_steps = 0
        self.robot_is_colliding = False
        self.grasped_obj_name = ""
        self.success = False

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the table.
        table = Object("table", KinRectangleType)
        init_state_dict[table] = {
            "x": self.config.table_pose.x,
            "vx": 0.0,
            "y": self.config.table_pose.y,
            "vy": 0.0,
            "theta": self.config.table_pose.theta,
            "omega": 0.0,
            "width": self.config.table_width,
            "height": self.config.table_height,
            "static": True,
            "color_r": self.config.table_rgb[0],
            "color_g": self.config.table_rgb[1],
            "color_b": self.config.table_rgb[2],
            "z_order": ZOrder.ALL.value,
        }

        # Create room walls.
        assert isinstance(self.action_space, KinRobotActionSpace)
        min_dx, min_dy = self.action_space.low[:2]
        max_dx, max_dy = self.action_space.high[:2]
        wall_state_dict = create_walls_from_world_boundaries(
            self.config.world_min_x,
            self.config.world_max_x,
            self.config.world_min_y,
            self.config.world_max_y,
            min_dx,
            max_dx,
            min_dy,
            max_dy,
        )
        init_state_dict.update(wall_state_dict)

        return init_state_dict

    def _sample_initial_state(self) -> ObjectCentricState:
        """Sample an initial state for the environment."""
        n = self.config.max_initial_state_sampling_attempts
        for _ in range(n):
            # Sample all randomized values.
            robot_pose = sample_se2_pose(
                self.config.robot_init_pose_bounds, self.np_random
            )
            grasp_block_pose = sample_se2_pose(
                self.config.block_init_pose_bounds, self.np_random
            )
            base_block_pose = sample_se2_pose(
                self.config.block_init_pose_bounds, self.np_random
            )
            if (
                abs(grasp_block_pose.x - base_block_pose.x)
                < 3 * self.config.robot_base_radius
            ):
                # Ensure the two blocks are not too close in x axis.
                continue

            grasp_block_shape = (self.config.block_width, self.config.block_height)
            base_block_shape = (self.config.block_width, self.config.block_height)

            state = self._create_initial_state(
                robot_pose,
                grasp_block_pose,
                grasp_block_shape,
                base_block_pose,
                base_block_shape,
            )

            # Check initial state validity: goal not satisfied and no collisions.
            full_state = state.copy()
            full_state.data.update(self.initial_constant_state.data)
            all_objects = set(full_state)
            # We use Geom2D collision checker for now, maybe need to update it.
            if state_2d_has_collision(full_state, all_objects, all_objects, {}):
                continue
            return state

        raise RuntimeError(f"Failed to sample initial state after {n} attempts")

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        grasp_block_pose: SE2Pose,
        grasp_block_shape: tuple[float, float],
        base_block_pose: SE2Pose,
        base_block_shape: tuple[float, float],
    ) -> ObjectCentricState:
        # Shallow copy should be okay because the constant objects should not
        # ever change in this method.
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the robot.
        robot = Object("robot", RobotType)
        self._robot_obj = robot
        init_state_dict[robot] = {
            "x": robot_pose.x,
            "vx": 0.0,
            "y": robot_pose.y,
            "vy": 0.0,
            "theta": robot_pose.theta,
            "omega": 0.0,
            "static": False,
            "base_radius": self.config.robot_base_radius,
            "arm_joint": self.config.robot_base_radius,
            "arm_length": self.config.robot_arm_length_max,
            "gripper_base_width": self.config.gripper_base_width,
            "gripper_base_height": self.config.gripper_base_height,
            "finger_gap": self.config.gripper_base_height,
            "finger_height": self.config.gripper_finger_height,
            "finger_width": self.config.gripper_finger_width,
            "is_colliding": 0.0,
        }

        # Create the block to be grasped.
        grasp_block = Object("grasp_block", BlockType)
        self._grasp_block = grasp_block
        init_state_dict[grasp_block] = {
            "x": grasp_block_pose.x,
            "vx": 0.0,
            "y": grasp_block_pose.y + grasp_block_shape[1] / 2,
            "vy": 0.0,
            "theta": grasp_block_pose.theta,
            "omega": 0.0,
            "width": grasp_block_shape[0],
            "height": grasp_block_shape[1],
            "static": False,
            "mass": self.config.grasp_block_mass,
            "color_r": self.config.grasp_block_rgb[0],
            "color_g": self.config.grasp_block_rgb[1],
            "color_b": self.config.grasp_block_rgb[2],
            "z_order": ZOrder.ALL.value,
            "grasped": 0.0,
        }

        # Create the base block.
        base_block = Object("base_block", BlockType)
        self._base_block = base_block
        init_state_dict[base_block] = {
            "x": base_block_pose.x,
            "vx": 0.0,
            "y": base_block_pose.y + base_block_shape[1] / 2,
            "vy": 0.0,
            "theta": base_block_pose.theta,
            "omega": 0.0,
            "width": base_block_shape[0],
            "height": base_block_shape[1],
            "static": False,
            "mass": self.config.base_block_mass,
            "color_r": self.config.base_block_rgb[0],
            "color_g": self.config.base_block_rgb[1],
            "color_b": self.config.base_block_rgb[2],
            "z_order": ZOrder.ALL.value,
            "grasped": 0.0,
        }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Dynamic2DRobotEnvTypeFeatures)

    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        """Add objects from the state to the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"

        # Add static objects (table, walls)
        for obj in state:
            if obj.is_instance(RobotType):
                self._reset_robot_in_space(obj, state)
            elif obj.is_instance(DynRectangleType) or obj.is_instance(KinRectangleType):
                # Everything else are rectangles in this environment.
                x = state.get(obj, "x")
                y = state.get(obj, "y")
                width = state.get(obj, "width")
                height = state.get(obj, "height")
                theta = state.get(obj, "theta")

                if state.get(obj, "static"):
                    # Static objects
                    # We use Pymunk kinematic bodies for static objects
                    b2 = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                    vs = [
                        (-width / 2, -height / 2),
                        (-width / 2, height / 2),
                        (width / 2, height / 2),
                        (width / 2, -height / 2),
                    ]
                    shape = pymunk.Poly(b2, vs)
                    shape.friction = 1.0
                    shape.density = 1.0
                    shape.mass = 1.0
                    shape.elasticity = 0.99
                    shape.collision_type = STATIC_COLLISION_TYPE
                    self.pymunk_space.add(b2, shape)
                    b2.position = x, y
                    b2.angle = theta
                    self._state_obj_to_pymunk_body[obj] = b2
                else:
                    # Dynamic objects
                    mass = state.get(obj, "mass")
                    moment = pymunk.moment_for_box(mass, (width, height))
                    body = pymunk.Body()
                    vs = [
                        (-width / 2, -height / 2),
                        (-width / 2, height / 2),
                        (width / 2, height / 2),
                        (width / 2, -height / 2),
                    ]
                    shape = pymunk.Poly(body, vs)
                    shape.friction = 1.0
                    shape.density = 1.0
                    shape.collision_type = DYNAMIC_COLLISION_TYPE
                    shape.mass = mass
                    assert shape.body is not None
                    shape.body.moment = moment
                    shape.body.mass = mass
                    self.pymunk_space.add(body, shape)
                    body.angle = theta
                    body.position = x, y
                    self._state_obj_to_pymunk_body[obj] = body

    def _read_state_from_space(self) -> None:
        """Read the current state from the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"
        assert self._current_state is not None, "Current state not initialized"

        state = self._current_state.copy()

        # Update dynamic object positions from PyMunk simulation
        for obj in state:
            if state.get(obj, "static"):
                continue
            if obj.is_instance(RobotType):
                # Update robot state from its body
                assert self.robot is not None, "Robot not initialized"
                robot_obj = state.get_objects(RobotType)[0]
                state.set(robot_obj, "x", self.robot.base_pose.x)
                state.set(robot_obj, "y", self.robot.base_pose.y)
                state.set(robot_obj, "theta", self.robot.base_pose.theta)
                state.set(robot_obj, "vx", self.robot.base_vel[0].x)
                state.set(robot_obj, "vy", self.robot.base_vel[0].y)
                state.set(robot_obj, "omega", self.robot.base_vel[1])
                state.set(robot_obj, "arm_joint", self.robot.curr_arm_length)
                state.set(robot_obj, "finger_gap", self.robot.curr_gripper)
                state.set(robot_obj, "is_colliding", float(self.robot_is_colliding))
            else:
                assert (
                    obj in self._state_obj_to_pymunk_body
                ), f"Object {obj.name} not found in pymunk body cache"
                pymunk_body = self._state_obj_to_pymunk_body[obj]
                # Update object state from body
                state.set(obj, "x", pymunk_body.position.x)
                state.set(obj, "y", pymunk_body.position.y)
                state.set(obj, "theta", pymunk_body.angle)
                state.set(obj, "vx", pymunk_body.velocity.x)
                state.set(obj, "vy", pymunk_body.velocity.y)
                state.set(obj, "omega", pymunk_body.angular_velocity)
                if obj.is_instance(BlockType):
                    if obj.name == self.grasped_obj_name:
                        state.set(obj, "grasped", 1.0)
                    else:
                        state.set(obj, "grasped", 0.0)

        # Update the current state
        self._current_state = state

    def _target_satisfied(
        self,
        state: ObjectCentricState,
        static_object_body_cache: dict[Object, MultiBody2D],
    ) -> bool:
        """Check if the target condition is satisfied.

        This is borrowed from geom2d obstruction env for now.
        """
        # Find grasp and base blocks dynamically
        grasp_block = None
        base_block = None
        for obj in state:
            if obj.name == "grasp_block":
                grasp_block = obj
            elif obj.name == "base_block":
                base_block = obj

        if grasp_block is None or base_block is None:
            return False
        geom_overlap = False
        top_geom = rectangle_object_to_geom(
            state, grasp_block, static_object_body_cache
        )
        bottom_geom = rectangle_object_to_geom(
            state, base_block, static_object_body_cache
        )
        offset_top_geom = Rectangle(
            top_geom.x,
            top_geom.y - self.config.on_tol_dy_overlap,
            top_geom.width,
            top_geom.height,
            top_geom.theta,
        )
        if geom2ds_intersect(offset_top_geom, bottom_geom):
            geom_overlap = True
        still_vel = False
        vels = np.array(
            [
                state.get(grasp_block, "vx"),
                state.get(grasp_block, "vy"),
                state.get(grasp_block, "omega"),
                state.get(base_block, "vx"),
                state.get(base_block, "vy"),
                state.get(base_block, "omega"),
            ]
        )
        # Ensure both blocks are still
        if np.linalg.norm(vels) < self.config.on_tol_vel:
            still_vel = True

        rel_dy = state.get(grasp_block, "y") - state.get(base_block, "y")
        above = abs(
            rel_dy
            - (
                state.get(base_block, "height") / 2
                + state.get(grasp_block, "height") / 2
            )
        )
        above_relation = above < self.config.on_tol_dy_relative
        return (
            geom_overlap
            and still_vel
            and above_relation
            and (self.grasped_obj_name == "")
        )

    def _get_reward_and_done(self):
        """Calculate reward and termination."""
        # Terminate when target object is on the target surface. Give -1 reward
        # at every step until then to encourage fast completion.
        assert self._current_state is not None
        terminated = self._target_satisfied(
            self._current_state,
            self._static_object_body_cache,
        )
        if terminated:
            self.success = True
            return 1.0, terminated
        return 0.0, terminated

    def step(self, action: Array) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        """Step the environment with the given action.

        Here we additional detect grasping and collisions.
        """
        assert self.robot is not None, "Robot not initialized"
        dx, dy, dtheta, darm, _ = action
        # Calculate target positions
        tgt_x = self.robot.base_pose.x + dx
        tgt_y = self.robot.base_pose.y + dy
        tgt_theta = self.robot.base_pose.theta + dtheta
        tgt_arm = max(
            min(self.robot.curr_arm_length + darm, self.robot.arm_length_max),
            self.robot.base_radius,
        )
        obs, reward, terminated, truncated, _ = super().step(action)
        # NOTE: Simply check if the robot reached the target position for collision
        # detection. This is not perfect but should be good enough for now.
        curr_x = self.robot.base_pose.x
        curr_y = self.robot.base_pose.y
        curr_theta = self.robot.base_pose.theta
        curr_arm = self.robot.curr_arm_length
        assert self._robot_obj is not None, "Robot object not initialized"
        if any(
            abs(t - c) > 1e-2
            for t, c in zip(
                (tgt_x, tgt_y, tgt_theta, tgt_arm),
                (curr_x, curr_y, curr_theta, curr_arm),
            )
        ):
            self.robot_is_colliding = True
            obs.set(self._robot_obj, "is_colliding", 1.0)
        else:
            obs.set(self._robot_obj, "is_colliding", 0.0)
            self.robot_is_colliding = False

        # Get the held object info for grasping detection
        assert self._grasp_block is not None and self._base_block is not None
        if len(self.robot.held_objects):
            kin_obj, _, _ = self.robot.held_objects[0]
            held_obj_id = kin_obj[0].id
            for obj in (self._grasp_block, self._base_block):
                if self._state_obj_to_pymunk_body[obj].id == held_obj_id:
                    self.grasped_obj_name = obj.name
                    obs.set(obj, "grasped", 1.0)
                    break
        else:
            self.grasped_obj_name = ""
            for obj in (self._grasp_block, self._base_block):
                obs.set(obj, "grasped", 0.0)
        self.elapsed_steps += 1
        # Force get info to be the end of step
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObjectCentricState, dict]:
        """Reset the environment."""
        self.elapsed_steps = 0
        self.grasped_obj_name = ""
        self.robot_is_colliding = False
        self.success = False

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # Clear existing physics space
        if self.pymunk_space:
            # Remove all bodies and shapes
            for body in list(self.pymunk_space.bodies):
                for shape in list(body.shapes):
                    if body in self.pymunk_space.bodies:
                        self.pymunk_space.remove(body, shape)
            for shape in list(self.pymunk_space.shapes):
                # Some shapes are not attached to bodies (e.g., static lines)
                self.pymunk_space.remove(shape)

        # Set up new physics space
        self._setup_physics_space()
        self._static_object_body_cache = {}
        self._state_obj_to_pymunk_body = {}

        # For testing purposes only, the options may specify an initial scene.
        if options is not None and "init_state" in options:
            self._current_state = options["init_state"].copy()
        # Otherwise, set up the initial scene here.
        else:
            self._current_state = self._sample_initial_state()

        # Add objects to physics space
        self._add_state_to_space(self.full_state)

        # Calculate simulation parameters
        dt = 1.0 / self.config.sim_hz
        # Stepping physics to let things settle
        assert self.pymunk_space is not None, "Space not initialized"
        for _ in range(self.config.sim_hz):
            self.pymunk_space.step(dt)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_info(self) -> dict:
        return {
            "elapsed_steps": self.elapsed_steps,
            "is_colliding": self.robot_is_colliding,
            "is_grasped": self.grasped_obj_name != "",
            "success": self.success,
        }  # no extra info provided right now


class BlockedStacking2DEnv(ConstantObjectPRBenchEnv):
    """Dynamic Obstruction 2D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricBlockedStacking2DEnv:
        return ObjectCentricBlockedStacking2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        del exemplar_state  # unused
        constant_objects = ["grasp_block", "base_block", "robot"]
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create a markdown description of the overall environment."""
        return "PLACEHOLDER: Blocked Stacking 2D Environment"

    def _create_reward_markdown_description(self) -> str:
        """Create a markdown description of the environment rewards."""
        return "PLACEHOLDER: Blocked Stacking 2D Environment"

    def _create_references_markdown_description(self) -> str:
        """Create a markdown description of the reference (e.g. papers) for this env."""
        return "PLACEHOLDER: Blocked Stacking 2D Environment"
