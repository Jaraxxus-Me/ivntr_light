"""Contains global, immutable settings.

Anything that varies between runs should be a command-line arg (args.py).
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict


class GlobalSettings:
    """Unchanging settings."""

    seed: int = 0
    # global parameters
    render: bool = True
    debug_env: bool = False
    num_eval_episodes: int = 10  # num episodes per eval_env
    num_train_episodes_rl: int = 1
    num_train_episodes_planner: int = 1
    training_scenario: int = 1  # 1, 2, or 3 in this work
    max_env_steps: int = 300  # for the entire task
    record_training: bool = True
    training_record_interval: int = 5000
    num_envs: int = 16
    num_eval_envs: int = 16
    log_wandb: bool = False
    control_mode: str = "pd_joint_delta_pos"
    delta_finger_control: bool = True
    arm_joint_delta: float = 0.2  # for pd_joint_delta_pos control mode
    normalize_action: bool = True

    # Output settings
    results_dir: Path = Path("results")
    training_data_dir: Path = Path("training_data")
    rl_policy_save_dir: Path = Path("trained_policies")
    pred_net_save_dir: str = "trained_pred_nets"
    invented_pred_op_json: str = "invented_predicates_operators.json"
    # tb_log_dir = None
    exp_name: str = "debug_experiment"
    device: str = "cuda:0"  # "cpu" or "cuda"

    # planner-learning parameters
    traj_segmenter: str = "operator_changes"
    predicate_config: str = ""
    precondition_threshold: float = 0.8  # minimum fraction for precondition inclusion
    add_effect_threshold: float = 0.8  # minimum fraction for add effect inclusion
    delete_effect_threshold: float = 0.8  # minimum fraction for delete effect inclusion
    super_label = {
        "ignore": -1,  # wrong type match for the predicate
        "change_neg": 0,
        "change_pos": 1,
        "non_change_1": 2,  # the predicate is not in the effect, and the objects (pairs) are not operated
        "non_change_2": 3,  # the predicate is in the effect, but the objects are not operated
        "non_change_3": 4,  # the predicate is not in the effect, but the objects (pairs) are operated
    }
    num_middle_states: int = 10
    num_middle_states_close: int = 9
    middle_state_method: str = (
        "naive_binary"  # "naive_init", "soft_interpolation", "two_endpoint_weighted"
    )
    sesame_task_planning_heuristic: str = "lmcut"
    force_skip_pred_learning: bool = False
    pred_search_num_trajectories: int = 40
    pred_search_expected_nodes_upper_bound = 1e5
    pred_search_max_skeletons_optimized = 5
    pred_search_expected_nodes_optimal_demo_prob = 1 - 1e-5
    pred_search_expected_nodes_backtracking_cost = 1e3
    predicate_search_task_planning_timeout: float = 1.0  # seconds

    # environment parameters
    # ** Blocked Stacking 2D Environment **
    # These can't be accessed directly from the observation,
    # but are used to set up the planner.
    blocked2d_robot_base_radius: float = 0.24
    blocked2d_robot_arm_length_max: float = 0.48
    blocked2d_gripper_base_width: float = 0.06
    blocked2d_gripper_base_height: float = 0.32
    blocked2d_gripper_finger_width: float = 0.2
    blocked2d_gripper_finger_height: float = 0.06


    @classmethod
    def get_arg_specific_settings(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """A workaround for global settings that are derived from the experiment-
        specific args."""
        del args  # unused
        return {}


_attr_to_value = {}
for _attr, _value in GlobalSettings.__dict__.items():
    if _attr.startswith("_"):
        continue
    assert _attr not in _attr_to_value  # duplicate attributes
    _attr_to_value[_attr] = _value
CFG = SimpleNamespace(**_attr_to_value)
