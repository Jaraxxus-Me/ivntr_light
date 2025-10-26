# IVNTR Light

A light weight implementation of IVNTR algorithm: [paper](https://www.arxiv.org/abs/2502.08697), [web](https://jaraxxus-me.github.io/IVNTR/).
This repo includes:

- A simple Pick-Place `gym` environment (based on [PRBench]())
- A (batched) Task-then-motion planner that collects trajectories with operator/option annotations.
- Simple top-down effect-enumeration for predicate discovery on these trajectories.
- Task-then-motion planning with discovered predicates.

### :wrench: Installation
We strongly recommend [uv](https://docs.astral.sh/uv/getting-started/installation/). The steps below assume that you have `uv` installed. If you do not, just remove `uv` from the commands and the installation should still work.
```
# Install this repo
git clone https://github.com/Jaraxxus-Me/ivntr_light.git
cd ivntr_light
git submodule update --init
```
```
# Create venv
uv venv --python=3.11
source .venv/bin/activate
# Install skill_ref
uv pip install -e .[develop]
# Third-party dependencies
uv pip install -e third-party/prpl-mono/prpl-utils
uv pip install -e third-party/prpl-mono/prbench
uv pip install -e third-party/prpl-mono/relational-structs
uv pip install -e third-party/prpl-mono/toms-geoms-2d
```

### :microscope: Check Installation
1. Run `./run_ci_checks.sh`. It should complete with all green  successes.
2. Additionally, see `videos` for a simple dynamic2d execution video with given planner :)
3. Test pre-trained neural predicates:
    ```
    # Download data
    python3 scripts/download_data.py
    # Run test
    pytest -s -v tests/approaches/test_discovered_pred_skills.py::test_loading_learned_predicates_blocked_stacking
    ```

### :mag: Guidelines for Understanding

#### Basics of Symbolic Planning
```py
# Understanding PDDL's Matrix Formulation
pytest -s -v tests/utils/test_op_matrix_converting.py
# Understanding quantifiers on predicates
pytest -s -v tests/utils/test_quantified_predicates.py
# Task planning
pytest -s -v tests/utils/test_task_planning.py
```

#### Environment and Data Collection
```py
# Understand the environment
pytest -s -v tests/benchmarks/test_blockedstacking_env.py
# How a task-then-motion planner works in (batched) gym.Env
pytest -s -v tests/approaches/test_pure_tamp.py
# How to collect trajectoris with the planner in (batched) gym.Env
pytest -s -v tests/datasets/test_collect.py
```

#### Operator and Predicate Learning
```py
# How to learn operators with given predicates
pytest -s -v tests/approaches/test_op_learning.py
# How to learn predicates by enumerating their effects across operators (This takes a long time due to exhaustive enumeration, ~9 hours).
pytest -s -v tests/approaches/test_pred_learning_topdown.py
# How to learn predicates by bilevel learning the effects and classifiers (not tested here, in general should be more efficient).
pytest -s -v tests/approaches/test_pred_learning_bilevel.py
```

#### Planning with Discovered Predicates
```py
# How to load discovered predicates/operators and bind skills to them.
pytest -s -v tests/approaches/test_discovered_pred_skills.py
```