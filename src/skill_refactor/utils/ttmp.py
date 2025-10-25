"""Planning interface. Borrowed from Tom's task_then_motion_planning. The key difference
here is that the step function here accepts a batched observation and returns a batched
action, allowing for parallel execution of multiple environments.

Right now we assume that the plans for different environments are homogeneous at high-
level, i.e., the single task plan applies to all environments. And skill execution is
also homogeneous, i.e., the same skill is executed in all environments at all times.
"""

import abc
from typing import Any, Tuple, Set
import copy

import torch
from numpy.typing import NDArray
from relational_structs import (
    PDDLDomain,
    PDDLProblem,
    Predicate,
    Type,
)
from torch import Tensor

from skill_refactor.settings import CFG
from skill_refactor.utils.controllers import get_frozen_action
from skill_refactor.utils.task_planning import (
    task_plan_grounding,
    create_task_planning_heuristic,
    task_plan,
)
from skill_refactor.utils.structs import (
    Object,
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    LiftedOperatorSkill,
    Perceiver,
)


class TaskThenMotionPlanningFailure(Exception):
    """Raised when task then motion planning fails."""


class TaskThenMotionPlanner(abc.ABC):
    """Run task then motion planning with greedy execution."""

    def __init__(
        self,
        types: set[Type],
        predicates: set[Predicate],
        perceiver: Perceiver,
        operators: set[LiftedOperator],
        skills: set[LiftedOperatorSkill],
        fallback_action: NDArray,
        normalize_action: bool,
        arm_action_low: Tensor,
        arm_action_high: Tensor,
        planner_id: str = "fd-sat",
        domain_name: str = "ttmp-domain",
    ) -> None:
        self._types = types
        self.predicates = predicates
        self.perceiver = perceiver
        self.operators = operators
        self.skills = skills
        self._planner_id = planner_id
        self._domain_name = domain_name
        self._domain = PDDLDomain(
            self._domain_name, self.operators, self.predicates, self._types
        )
        self._current_problem: PDDLProblem | None = None
        self._current_task_plan: list[GroundOperator] = []
        self._current_operator: GroundOperator | None = None
        self._current_skill: LiftedOperatorSkill | None = None
        self._fallback_action = fallback_action
        self._normalize_action = normalize_action
        self._arm_action_low = arm_action_low
        self._arm_action_high = arm_action_high
        self._last_action: Tensor = torch.tensor(fallback_action)
        self._skill_reached_effects: Tensor = torch.zeros(1, dtype=torch.bool)
        self._skill_exhausted: Tensor = torch.zeros(1, dtype=torch.bool)

    def reset(self, obs: Tensor, info: dict[str, Any]) -> None:
        """Reset on a new task instance."""
        # We commit to the first environment for planning.
        # This is a limitation of the current implementation.
        objects, atoms, goal = self.perceiver.reset(obs[0:1], info)
        self._current_task_plan = self._create_task_plan(objects, atoms, goal)
        self._last_action = (
            torch.tensor(self._fallback_action)
            .unsqueeze(0)
            .repeat(obs.shape[0], 1)
            .to(obs.device)
        )
        self._current_operator = None
        self._current_skill = None
        # If some environments reach desired effects early, freeze them.
        self._skill_reached_effects = torch.zeros(
            obs.shape[0], dtype=torch.bool, device=obs.device
        )
        # If some environments exhaust the skill, freeze them.
        self._skill_exhausted = torch.zeros(
            obs.shape[0], dtype=torch.bool, device=obs.device
        )

    def step(self, obs: Tensor) -> Tuple[Tensor | None, GroundOperator | None]:
        """Get an action to execute."""
        # NOTE: We only step perceiver then the skill terminates.
        if self._current_skill is not None and self._current_skill.terminate(obs).any():
            terminate_mask = self._current_skill.terminate(obs)
            terminate_mask &= ~self._skill_reached_effects
            terminate_obs = obs[terminate_mask]
            terminate_atoms = self.perceiver.step(terminate_obs)

            # new, assuming atoms: List[Set[GroundAtom]]
            assert self._current_operator is not None
            add_ok_list = [
                self._current_operator.add_effects.issubset(atom_set)
                for atom_set in terminate_atoms
            ]
            del_ok_list = [
                not any(self._current_operator.delete_effects & atom_set)
                for atom_set in terminate_atoms
            ]

            # Update skill reached effects
            self._skill_reached_effects[terminate_mask] |= torch.logical_and(
                torch.tensor(add_ok_list), torch.tensor(del_ok_list)
            ).to(obs.device)
            self._skill_exhausted[terminate_mask] |= ~self._skill_reached_effects[
                terminate_mask
            ]

        # Switch conditions:
        # If the current operator is None
        # or terminated/exhausted in all envs.
        # NOTE: A skill has to be terminiated to trigger reach effects or exhaust.
        # So we don't need to check for termination here.
        switch_condition = torch.all(
            self._skill_reached_effects | self._skill_exhausted
        )
        if self._current_skill is None or switch_condition:
            # If there is no more task plan to execute, fail (freeze to last action).
            if not self._current_task_plan:
                actions = get_frozen_action(
                    self._last_action,
                    self._arm_action_low,
                    self._arm_action_high,
                    self._normalize_action,
                    CFG.control_mode,
                )
                return actions, self._current_operator

            self._current_operator = self._current_task_plan.pop(0)
            # Get a skill that can execute this operator.
            self._current_skill = self._get_skill_for_operator(self._current_operator)
            assert self._current_skill is not None, "No skill for operator"
            self._current_skill.reset(self._current_operator)
            self._skill_reached_effects = torch.zeros(
                obs.shape[0], dtype=torch.bool, device=obs.device
            )
            # NOTE: We do not reset exhausted here, because
            # exhausted envs should stay exhausted.

        assert self._current_skill is not None
        skill_action = self._current_skill.get_action(obs)
        # Freeze the skill if it has reached the desired effects.
        # (Or if it is exhausted.)
        frozen_action_reached_eff = get_frozen_action(
            skill_action[self._skill_reached_effects],
            self._arm_action_low,
            self._arm_action_high,
            self._normalize_action,
            CFG.control_mode,
        )
        frozen_action_exhausted = get_frozen_action(
            self._last_action[self._skill_exhausted],
            self._arm_action_low,
            self._arm_action_high,
            self._normalize_action,
            CFG.control_mode,
        )
        skill_action[self._skill_reached_effects] = frozen_action_reached_eff
        skill_action[self._skill_exhausted] = frozen_action_exhausted
        self._last_action = skill_action.clone()
        return skill_action, self._current_operator

    def _get_skill_for_operator(self, operator: GroundOperator) -> LiftedOperatorSkill:
        applicable_skills = [s for s in self.skills if s.can_execute(operator)]
        if not applicable_skills:
            raise TaskThenMotionPlanningFailure("No skill can execute operator")
        assert len(applicable_skills) == 1, "Multiple operators per skill not supported"
        return applicable_skills[0]

    def update_domain(
        self,
        new_operators: Set[LiftedOperator],
        new_skills: Set[LiftedOperatorSkill],
    ) -> None:
        """Update the current domain with new operators and skills."""
        self.operators = copy.deepcopy(new_operators)
        self.skills = copy.deepcopy(new_skills)
        self._domain = PDDLDomain(
            "Learned_Domain",
            self.operators,  # type: ignore[arg-type]
            self.perceiver.predicates_container.as_set(),
            self._types,
        )

    def _create_task_plan(
        self,
        objects: set[Object],
        init_atoms: set[GroundAtom],
        goal: set[GroundAtom],
    ) -> list[GroundOperator]:
        """Create task plan with local search to achieve goal."""
        ground_operators, reachable_atoms = task_plan_grounding(
            init_atoms, objects, list(self.operators), allow_noops=True
        )
        heuristic = create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic,
            init_atoms,
            goal,
            ground_operators,
            self.perceiver.predicates_container.as_set(),
            objects,
        )
        generator = task_plan(
            init_atoms,
            goal,
            ground_operators,
            reachable_atoms,
            heuristic,
            CFG.seed,
            100,
            CFG.pred_search_max_skeletons_optimized,
        )
        # plan_str = run_pddl_planner(
        #     str(self.domain), str(problem), planner=self.planner_id
        # )
        # if plan_str is None:
        #     raise TaskThenMotionPlanningFailure("No plan found")
        result = next(generator, None)
        # NOTE: we only commit to the first plan
        if result is None:
            raise TaskThenMotionPlanningFailure("No plan found")
        first_plan, _, _ = result
        return first_plan  # type: ignore[return-value]