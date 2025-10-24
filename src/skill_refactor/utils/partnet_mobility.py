"""PartNet Mobility dataset utilities for loading and building articulations."""

from pathlib import Path
from typing import Any, Dict, Optional

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.articulation_builder import ArticulationBuilder
from mani_skill.utils.io_utils import load_json

from skill_refactor import PACKAGE_ASSET_DIR

PARTNET_MOBILITY: Optional[Dict[str, Any]] = None


def _load_partnet_mobility_dataset() -> None:
    """Loads preprocssed partnet mobility metadata."""
    global PARTNET_MOBILITY  # pylint: disable=global-statement
    PARTNET_MOBILITY = {
        "model_data": load_json(
            PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
        ),
    }

    def find_urdf_path(model_id: str) -> Optional[Path]:
        model_dir = PACKAGE_ASSET_DIR / "partnet_mobility/dataset" / str(model_id)
        urdf_names = ["mobility_cvx.urdf", "mobility_fixed.urdf"]
        for urdf_name in urdf_names:
            urdf_path = model_dir / urdf_name
            if urdf_path.exists():
                return urdf_path
        return None

    PARTNET_MOBILITY["model_urdf_paths"] = {}
    for k in PARTNET_MOBILITY["model_data"].keys():
        urdf_path = find_urdf_path(k)
        if urdf_path is not None:
            PARTNET_MOBILITY["model_urdf_paths"][k] = urdf_path

    if len(PARTNET_MOBILITY["model_urdf_paths"]) == 0:
        raise RuntimeError(
            f"Partnet Mobility dataset not found. Download it by running python -m "
            f"mani_skill.utils.download_asset partnet_mobility_cabinet -o "
            f"{PACKAGE_ASSET_DIR}/partnet_mobility"
        )


def get_partnet_mobility_builder(
    scene: ManiSkillScene,
    object_id: str,
    fix_root_link: bool = True,
    urdf_config: dict | None = None,
) -> ArticulationBuilder:
    """Builds an articulation builder for a partnet mobility object."""
    if PARTNET_MOBILITY is None:
        _load_partnet_mobility_dataset()

    assert PARTNET_MOBILITY is not None, "Failed to load PARTNET_MOBILITY dataset"
    metadata = PARTNET_MOBILITY["model_data"][object_id]
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    loader.scale = metadata["scale"]
    loader.load_multiple_collisions_from_file = True
    urdf_path = PARTNET_MOBILITY["model_urdf_paths"][object_id]
    applied_urdf_config = sapien_utils.parse_urdf_config(
        {
            "material": {"static_friction": 1, "dynamic_friction": 1, "restitution": 0},
        }
    )
    if urdf_config is not None:
        applied_urdf_config.update(**urdf_config)
    sapien_utils.apply_urdf_config(loader, applied_urdf_config)
    articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
    builder = articulation_builders[0]
    return builder
