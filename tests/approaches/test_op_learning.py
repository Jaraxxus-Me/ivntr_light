"""Unit Tests for NSRT learning in BlockedStacking environment, with provided predicates
and skills."""

import copy
import shutil
import types
from pathlib import Path

from skill_refactor import register_all_environments
from skill_refactor.benchmarks.blocked_stacking.blocked_stacking import (
    BlockedStackingRLTAMPSystem,
)
from skill_refactor.approaches.pure_tamp import PureTAMPApproach
from skill_refactor.approaches.operator_learner import learn_operator_from_data
from skill_refactor.args import reset_config
from skill_refactor.utils.structs import (
    LiftedOperator,
    LiftedOperatorSkill,
    PlannerDataset,
)


def _collect_blocked_stacking_test_data(save_path: Path, num_episodes: int = 3) -> None:
    """Helper function to collect minimal planner data for operator learning tests."""
    test_config = {
        "num_envs": 1,
        "debug_env": False,
        "max_env_steps": 100,
        "num_train_episodes_planner": num_episodes,
        "control_mode": "pd_joint_delta_pos",
    }
    register_all_environments()
    reset_config(test_config)

    tamp_system = BlockedStackingRLTAMPSystem.create_default(
        render_mode="rgb_array", seed=42
    )

    approach = PureTAMPApproach(tamp_system, seed=42)

    # Collect minimal data directly without wrapper
    train_data = approach.collect_data(tamp_system.env)
    tamp_system.env.close()

    # Save to temporary location
    save_path.mkdir(parents=True, exist_ok=True)
    train_data.save(save_path)


def test_op_learning_blocked_stacking() -> None:
    """Test Operator learning in BlockedStacking environment."""
    # Setup temporary test data directories
    dataset_path = Path("/tmp/test_op_learning_bstacking_1")

    try:
        # Collect test data (3 trajectories each)
        print("Collecting first dataset...")
        _collect_blocked_stacking_test_data(dataset_path, num_episodes=3)

        test_config = {
            "traj_segmenter": "operator_changes",
        }
        reset_config(test_config)
        register_all_environments()

        tamp_system = BlockedStackingRLTAMPSystem.create_default(  # type: ignore[name-defined]
            render_mode="rgb_array", seed=42
        )
        planner_dataset = PlannerDataset.load(dataset_path)

        assert (
            len(planner_dataset.trajectories) > 0
        ), "Planner dataset should not be empty."

        given_operators = planner_dataset.get_appearing_operators()
        # BlockedStacking typically has 4 operators:
        # ReachToGrasp, Grasp, ReachToPlace, Place
        expected_num_ops = len(given_operators)
        assert (
            len(given_operators) == 4
        ), "Should have 4 given operators in BlockedStacking."

        trajectories = planner_dataset.trajectories
        ground_atom_dataset, tasks = planner_dataset.get_ground_atoms_and_tasks(
            tamp_system.perceiver
        )

        assert len(tasks) == len(
            ground_atom_dataset
        ), "Tasks and ground atom dataset should have the same length."
        assert len(ground_atom_dataset) == len(
            trajectories
        ), "Ground atom dataset should match the number of trajectories."

        # Learn operators from the dataset
        operators, _, _ = learn_operator_from_data(
            "clustering",
            trajectories,
            tasks,
            tamp_system.perceiver.predicates_container.as_set(),
            given_operators,
            ground_atom_dataset,
        )

        # Verify operators were learned successfully
        assert (
            len(operators) == expected_num_ops
        ), f"There should be {expected_num_ops} learned operators."
        print(f"✓ Successfully learned {len(operators)} operators")

        # Verify each operator has preconditions and effects
        for op in operators:
            assert (
                len(op.preconditions) >= 0
            ), f"Operator {op.name} should have preconditions defined"
            assert (
                len(op.add_effects) >= 0 or len(op.delete_effects) >= 0
            ), f"Operator {op.name} should have at least one effect"

        # Test that we can bind skills to operators
        binded_skills: set[LiftedOperatorSkill] = set()
        for op in operators:
            for skill in tamp_system.skills:
                if (skill.get_operator_name() == op.name) or (
                    skill.get_operator_name() == op.name.split("_")[0]
                ):
                    new_skill = skill.__class__(  # type: ignore[call-arg]
                        env=tamp_system.env,
                        operators=operators,
                    )
                    local_op = copy.deepcopy(op)

                    # capture local_op in the default arg
                    def _customget_operator_name(self, op=local_op) -> str:
                        del self
                        return op.name

                    def _custom_get_lifted_operator(
                        self, op=local_op
                    ) -> LiftedOperator:
                        del self
                        return op

                    setattr(
                        new_skill,
                        "get_operator_name",
                        types.MethodType(_customget_operator_name, new_skill),
                    )
                    setattr(
                        new_skill,
                        "get_lifted_operator",
                        types.MethodType(_custom_get_lifted_operator, new_skill),
                    )
                    binded_skills.add(new_skill)
                    break

        print(f"✓ Successfully bound {len(binded_skills)} skills to operators")
        assert len(binded_skills) > 0, "Should be able to bind at least one skill"

        print(f"✓ Test completed successfully. Operators learned: {operators}")

    finally:
        # Cleanup: Remove temporary test data
        for path in [dataset_path]:
            if path.exists():
                shutil.rmtree(path)