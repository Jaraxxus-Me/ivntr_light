from pathlib import Path

from gymnasium.envs.registration import register

PACKAGE_DIR = Path(__file__).parent.resolve()
PACKAGE_ASSET_DIR = PACKAGE_DIR / "assets"
EXT_LIB_DIR = PACKAGE_DIR / "ext"


def register_all_environments() -> None:
    # BlockedStacking2D env
    register(
        id="skill_ref/BlockedStacking2D-v0",
        entry_point="skill_refactor.benchmarks.blocked_stacking.blocked_stacking_env:BlockedStacking2DEnv",
        order_enforce=False,
        disable_env_checker=True,
    )
