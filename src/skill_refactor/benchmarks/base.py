"""Base environment interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import gymnasium as gym
from relational_structs import Object, PDDLDomain, Predicate, Type
from torch import Tensor

from skill_refactor.utils.structs import (
    LiftedOperator,
    LiftedOperatorSkill,
    Perceiver,
    PredicateContainer,
    TypeContainer,
)


@dataclass
class GraphData:
    """Graph representation of environment state."""

    node_features: Tensor  # [num_nodes, node_dim]
    edge_features: Tensor  # [num_edges, edge_dim]
    edge_indices: Tensor  # [2, num_edges] - [source, target] pairs
    global_features: Optional[Tensor] = None  # [global_dim]
    object_to_node: Optional[Dict[Object, int]] = (
        None  # mapping from objects to node indices
    )

    @property
    def num_nodes(self) -> int:
        """Get the number of nodes in the graph."""
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        """Get the number of edges in the graph."""
        return self.edge_features.shape[0]


@dataclass
class TensorPlanningComponents(ABC):
    """Container for customized planning-related components."""

    type_container: TypeContainer
    predicate_container: PredicateContainer
    skills: set[LiftedOperatorSkill]
    perceiver: Perceiver
    operators: set[LiftedOperator]


class BaseRLTAMPSystem(ABC):
    """Base class for Task-and-Motion Planning (TAMP) systems.

    This class combines:
    1. The actual environment (gym.Env) that represents the physical world
    2. The agent's planning components (types, predicates, operators, etc.)
       that represent the agent's abstract model of the world for planning
    """

    env_kwargs: dict[str, Any] = {}
    env_name: str = ""

    def __init__(
        self,
        planning_components: TensorPlanningComponents,
        name: str = "TAMPSystem",
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize TAMP system.

        Args:
            planning_components: The agent's planning model/components
            seed: Random seed for environment
        """
        self.name = name
        self.components = planning_components
        self.env = self._create_env()
        if seed is not None:
            self.env.reset(seed=seed)
        self._render_mode = render_mode

    @property
    def types(self) -> set[Type]:
        """Get types."""
        return self.components.type_container.as_set()

    @property
    def predicates(self) -> set[Predicate]:
        """Get PDDL predicates."""
        return self.components.predicate_container.as_set()

    @property
    def operators(self) -> set[LiftedOperator]:
        """Get PDDL operators."""
        return self.components.operators

    @property
    def perceiver(self) -> Perceiver:
        """Get state perceiver."""
        return self.components.perceiver

    @property
    def skills(self) -> set[LiftedOperatorSkill]:
        """Get skills."""
        return self.components.skills

    @abstractmethod
    def _create_env(self) -> gym.Env:
        """Create the base environment."""

    @abstractmethod
    def _get_domain_name(self) -> str:
        """Get domain name."""

    @abstractmethod
    def get_domain(self) -> PDDLDomain:
        """Get PDDL domain with or without extra preconditions for skill learning."""

    def reset(self, seed: int | None = None) -> tuple[Tensor, dict[str, Any]]:
        """Reset environment."""
        return self.env.reset(seed=seed)

    @abstractmethod
    def state_to_graph(self, state: Tensor) -> List[GraphData]:
        """Convert environment state tensor to graph representation.

        Args:
            state: State tensor from environment

        Returns:
            GraphData representation of the state
        """

    @classmethod
    def create_default(
        cls,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> BaseRLTAMPSystem:
        """Create a default TAMP system instance."""
        raise NotImplementedError(
            "This method should be implemented in subclasses to create a default instance."
        )
