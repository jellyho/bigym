<h1>
  <a href="#"><img alt="BiGym" src="doc/images/bigym.png" width="100%"></a>
</h1>

<p>
  <a href="https://github.com/NeuracoreAI/bigym/actions/workflows/build.yaml?query=branch%3Amaster" alt="GitHub Actions">
    <img src="https://img.shields.io/github/actions/workflow/status/NeuracoreAI/bigym/build.yaml?branch=master">
  </a>
  <a href="#contributing">
    <img src="https://img.shields.io/badge/PRs-welcome-green.svg" alt="PRs" height="20">
  </a>
</p>

[**BiGym: A Demo-Driven Mobile Bi-Manual Manipulation Benchmark**](https://arxiv.org/abs/2407.07788)\
[Nikita Cherniadev*](https://www.linkedin.com/in/nikita-cherniadev-8495417a/), [Nicholas Backshall*](https://www.linkedin.com/in/nicholas-backshall/?originalSubdomain=uk), [Xiao Ma*](https://yusufma03.github.io/), [Yunfan Lu](https://www.linkedin.com/in/yunfan-lu-90170992/?originalSubdomain=sg), [Younggyo Seo](https://younggyo.me/), [Stephen James](https://stepjam.github.io/)

BiGym is a new benchmark and learning environment for mobile bi-manual demo-driven robotic manipulation.
BiGym features 40 diverse tasks set in home environments, ranging from simple target reaching to complex kitchen cleaning. To capture the real-world performance accurately, we provide human-collected demonstrations for each task, reflecting the diverse modalities found in real-world robot trajectories. BiGym supports a variety of observations, including proprioceptive data and visual inputs such as RGB, and depth from 3 camera views.

Check out the project page: [https://chernyadev.github.io/bigym/](https://chernyadev.github.io/bigym/), and follow the repository for the latest updates: [https://github.com/NeuracoreAI/bigym](https://github.com/NeuracoreAI/bigym).

## What this fork changes

This is a fork of [NeuracoreAI/bigym](https://github.com/NeuracoreAI/bigym). Upstream behaviour
is preserved by default except where noted; everything here is opt-in or a bug fix.

### Object state for every task

`BiGymEnv` defines `_get_task_privileged_obs` and its space, and returns `{}` from the base —
so only `MovePlate` and the `ReachTarget` family ever exposed anything, and an agent configured
with `privileged_information=True` still saw proprioception alone on 38 of 40 tasks.

A task now declares `_PRIVILEGED_PROPS`, the attribute names of the props whose state it
exposes, and the base resolves each through `get_state()` for an articulated prop or
`get_pose()` for a free one:

```python
class _DishwasherCupsEnv(BiGymEnv, ABC):
    _PRIVILEGED_PROPS = ("dishwasher", "cabinets", "cups")
```

All 40 tasks are covered. What is exposed is exactly the quantity each task's own `_success`
already reads every step — cabinet and dishwasher joint states, the manipulated object's pose.
The success predicate itself is not exposed, only the state it is computed from.

### `proprioception_mode`: `compact` (default) or `raw`

`ObservationConfig.proprioception_mode` selects how much of the proprioceptive observation is
emitted:

| mode | keys | numbers |
|---|---|---|
| `compact` *(default)* | `proprioception` = `robot.qpos` | 30 (29 with a 3-DOF base) |
| `raw` | qpos+qvel, plus the gripper and floating-base keys | 66 |

`compact` is the default because the other 36 numbers carry nothing the first 30 do not,
measured across all 40 tasks:

- `proprioception_floating_base` (4) is **bit-identical** to qpos's pelvis entries, in 40/40;
- `proprioception_grippers` (2) is `average(driver joint qpos)` rescaled and **rounded to one
  decimal** — a monotone function of qpos entries already present (corr 0.9999), so a *lossier*
  copy rather than new information;
- `qvel` (30) is recoverable from position differences under frame stacking, at corr 0.92–0.99
  for every arm joint and for pelvis x/y.

What `compact` keeps includes the 16 gripper linkage joints, which are **not** redundant: their
effective rank is 3 per gripper, because the fingers deflect differently when something is
held. Use `raw` to reproduce a result published against the old observation.

### 20 Hz is the default control frequency

Upstream defaults `control_frequency` to `CONTROL_FREQUENCY_MAX` — the raw 500 Hz the
demonstrations were recorded at — and leaves the rate to each caller. CQN-AS, for instance,
decimates per task at 10/20/25 while its paper reports uniform hyperparameters. A benchmark
whose control rate is a caller's private choice cannot be compared across papers, so this fork
picks one and states it: **20 Hz, uniform across all 40 tasks** (`CONTROL_FREQUENCY_DEFAULT`).

20 rather than 25 or 50 because a decimation sweep found no consistent cost — the best rate per
task was scattered over 500/50/25/20 within ±0.06, about the standard error at 30–69
demonstrations — so the tie was broken by what the rate buys: an action chunk of a round number
of seconds, and the longest horizon the recordings support.

One consequence worth knowing: `_sub_steps_count` is also the action scale, and
`action_modes.py` widens the floating-base bounds by it even in absolute mode, so the pelvis
range at 20 Hz is 2.5× what it is at 50. A number measured at one rate does not carry to
another.

Pass `control_frequency=500` to get upstream's behaviour back.

### Datasets built with this fork

All 40 tasks, replayed once at 20 Hz with the compact observation and the object state, and
published in two formats. One replay produced both, so the state rows and the video frames are
the same episodes — two replays of the same actions have been measured disagreeing about which
episodes succeed, which is why that matters.

| dataset | what it is |
|---|---|
| [`jellyho/bigym-ogbench-20hz`](https://huggingface.co/datasets/jellyho/bigym-ogbench-20hz) | 40 `.npz` in OGBench's layout, for offline RL |
| `jellyho/bigym-<Task>-20hz` | 40 LeRobot v3.0 datasets, 256×256 video from three cameras |
| [`jellyho/bigym-replay-outcomes`](https://huggingface.co/datasets/jellyho/bigym-replay-outcomes) | 1,868 replay clips labelled success/failure |

**The OGBench layout.** One row per *state*, so a trajectory of T transitions is T+1 rows;
`terminals` marks each trajectory's last row and that row's action is a pad. Keys:
`observations`, `actions`, `terminals`, `rewards`, `episode_success`, `action_low`,
`action_high`.

**The observation** is `robot.qpos` (30 numbers, 29 with a 3-DOF base) followed by the task's
object state, in `sorted(_PRIVILEGED_PROPS)` order. The per-task widths and key order are in
the `.json` beside each file.

**The actions** are rescaled to [-1, 1] with the demonstrations' own per-dimension range, and
that range travels **inside the npz** as `action_low` / `action_high`. Invert before stepping
the environment, whose action space is raw joint limits and which *raises* rather than clips on
an out-of-bounds action:

```python
raw = (a + 1) / 2 * (high - low + 1e-8) + low
```

The bounds are in the file rather than recomputed because recomputing them is how they came
apart once: the caches were rebuilt, a replay put success at a different step, and the training
data and the evaluation environment ended up on two different scalings with nothing to report it.

**Roughly a quarter of the demonstrations do not reach their goal on replay**, and the rate is
per task (0.035 to 1.000; `replay_success_rate` in each sidecar). That is a property of the
build, not of the operator: `bigym 4.0.0`, which the demos were recorded with, was never
published, and its successor's changelog says the floating-base actuator stiffness was adjusted.
The failures are labelled rather than dropped — `episode_success` — because a critic needs the
contrast, and an imitation baseline can filter them out.

**Getting one**, with nothing installed but the standard library:

```python
import urllib.request, numpy as np
url = ('https://huggingface.co/datasets/jellyho/bigym-ogbench-20hz/'
       'resolve/main/DrawersAllOpen_20hz_state-priv.npz')
urllib.request.urlretrieve(url, 'DrawersAllOpen.npz')
z = np.load('DrawersAllOpen.npz')
```

### Fixes

- `DemoStore._create_path` ignored `privileged_information` and the proprioception mode, so a
  replay cached under one configuration was handed back to a caller that had asked for another
  — failing on a missing key, or silently returning a different observation. It now keys on
  both, as it already did on the camera set.
- `MovePlate` declared `plate_pose` with shape `(3,)` while `get_pose()` returns position and
  quaternion, 7.

## Table of Contents

1. [What this fork changes](#what-this-fork-changes)
1. [Datasets built with this fork](#datasets-built-with-this-fork)
1. [Install](#install)
2. [Tasks](#tasks)
3. [Usage](#usage)
4. [Contributing](#contributing)

## Install

```commandline
pip install .
```

## Tasks

| Task                                                            | Description                                                                                                                                                                           | Preview                                                                                        |
|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [ReachTarget](bigym/envs/reach_target.py)                       | Reach the target with either left or right wrist.                                                                                                                                     | <img src="doc/images/tasks/task_preview_reach_target@720x720.png" width=320>                   |
| [ReachTargetSingle](bigym/envs/reach_target.py)                 | Reach the target with specific wrist.                                                                                                                                                 | <img src="doc/images/tasks/task_preview_reach_target_single@720x720.png" width=320>            |
| [ReachTargetDual](bigym/envs/reach_target.py)                   | Reach 2 targets, one with each arm.                                                                                                                                                   | <img src="doc/images/tasks/task_preview_reach_target_dual@720x720.png" width=320>              |
| [StackBlocks](bigym/envs/manipulation.py)                       | Move blocks across the table, and stack them in the target area.                                                                                                                      | <img src="doc/images/tasks/task_preview_stack_blocks@720x720.png" width=320>                   |
| [MovePlate](bigym/envs/move_plates.py)                          | Move the plate between two draining racks.                                                                                                                                            | <img src="doc/images/tasks/task_preview_move_plate@720x720.png" width=320>                     |
| [MoveTwoPlates](bigym/envs/move_plates.py)                      | Move two plates simultaneously from one draining rack to the other.                                                                                                                   | <img src="doc/images/tasks/task_preview_move_two_plates@720x720.png" width=320>                |
| [FlipCup](bigym/envs/manipulation.py)                           | Flip the cup, initially positioned upside down on the table, to an upright position.                                                                                                  | <img src="doc/images/tasks/task_preview_flip_cup@720x720.png" width=320>                       |
| [FlipCutlery](bigym/envs/manipulation.py)                       | Take the cutlery from the static holder, flip it, and place it back into the holder.                                                                                                  | <img src="doc/images/tasks/task_preview_flip_cutlery@720x720.png" width=320>                   |
| [DishwasherOpen](bigym/envs/dishwasher.py)                      | Open the dishwasher door and pull out all trays.                                                                                                                                      | <img src="doc/images/tasks/task_preview_dishwasher_open@720x720.png" width=320>                |
| [DishwasherClose](bigym/envs/dishwasher.py)                     | Push back all trays and close the door of the dishwasher.                                                                                                                             | <img src="doc/images/tasks/task_preview_dishwasher_close@720x720.png" width=320>               |
| [DishwasherOpenTrays](bigym/envs/dishwasher.py)                 | Pull out the dishwasher’s trays with the door initially open.                                                                                                                         | <img src="doc/images/tasks/task_preview_dishwasher_open_trays@720x720.png" width=320>          |
| [DishwasherCloseTrays](bigym/envs/dishwasher.py)                | Push the dishwasher’s trays back with the door initially open.                                                                                                                        | <img src="doc/images/tasks/task_preview_dishwasher_close_trays@720x720.png" width=320>         |
| [DishwasherLoadPlates](bigym/envs/dishwasher_plates.py)         | Move plates from the rack to the lower tray of the dishwasher.                                                                                                                        | <img src="doc/images/tasks/task_preview_dishwasher_load_plates@720x720.png" width=320>         |
| [DishwasherLoadCups](bigym/envs/dishwasher_cups.py)             | Move cups from the table to the upper tray of the dishwasher.                                                                                                                         | <img src="doc/images/tasks/task_preview_dishwasher_load_cups@720x720.png" width=320>           |
| [DishwasherLoadCutlery](bigym/envs/dishwasher_cutlery.py)       | Move cutlery from the table holder to the dishwasher’s cutlery basket.                                                                                                                | <img src="doc/images/tasks/task_preview_dishwasher_load_cutlery@720x720.png" width=320>        |
| [DishwasherUnloadPlates](bigym/envs/dishwasher_plates.py)       | Move plates from the tray of the dishwasher to a table rack.                                                                                                                          | <img src="doc/images/tasks/task_preview_dishwasher_unload_plates@720x720.png" width=320>       |
| [DishwasherUnloadCups](bigym/envs/dishwasher_cups.py)           | Move cups from the upper tray of the dishwasher to the table.                                                                                                                         | <img src="doc/images/tasks/task_preview_dishwasher_unload_cups@720x720.png" width=320>         |
| [DishwasherUnloadCutlery](bigym/envs/dishwasher_cutlery.py)     | Move cutlery from the cutlery basket to a tray on the table.                                                                                                                          | <img src="doc/images/tasks/task_preview_dishwasher_unload_cutlery@720x720.png" width=320>      |
| [DishwasherUnloadPlatesLong](bigym/envs/dishwasher_plates.py)   | A full task of unloading a plate: picking up the plate from dishwasher, placing this plate into the rack located in the closed wall cabinet, and closing the dishwasher and cupboard. | <img src="doc/images/tasks/task_preview_dishwasher_unload_plates_long@720x720.png" width=320>  |
| [DishwasherUnloadCupsLong](bigym/envs/dishwasher_cups.py)       | A full task of unloading a cup: picking up the cup, placing it inside the closed wall cabinet, and closing the dishwasher and cupboard.                                               | <img src="doc/images/tasks/task_preview_dishwasher_unload_cups_long@720x720.png" width=320>    |
| [DishwasherUnloadCutleryLong](bigym/envs/dishwasher_cutlery.py) | A full task of unloading a cutlery: picking up a cutlery, placing it into the cutlery tray inside the closed drawer, and closing the dishwasher and drawer.                           | <img src="doc/images/tasks/task_preview_dishwasher_unload_cutlery_long@720x720.png" width=320> |
| [DrawerTopOpen](bigym/envs/cupboards.py)                        | Open the top drawer of the kitchen cabinet.                                                                                                                                           | <img src="doc/images/tasks/task_preview_drawer_top_open@720x720.png" width=320>                |
| [DrawerTopClose](bigym/envs/cupboards.py)                       | Close the top drawer of the kitchen cabinet.                                                                                                                                          | <img src="doc/images/tasks/task_preview_drawer_top_close@720x720.png" width=320>               |
| [DrawersAllOpen](bigym/envs/cupboards.py)                       | Open all sliding drawers of the kitchen cabinet.                                                                                                                                      | <img src="doc/images/tasks/task_preview_drawers_all_open@720x720.png" width=320>               |
| [DrawersAllClose](bigym/envs/cupboards.py)                      | Close all sliding drawers of the kitchen cabinet.                                                                                                                                     | <img src="doc/images/tasks/task_preview_drawers_all_close@720x720.png" width=320>              |
| [WallCupboardOpen](bigym/envs/cupboards.py)                     | Open doors of the wall cabinet.                                                                                                                                                       | <img src="doc/images/tasks/task_preview_wall_cupboard_open@720x720.png" width=320>             |
| [WallCupboardClose](bigym/envs/cupboards.py)                    | Close doors of the wall cabinet.                                                                                                                                                      | <img src="doc/images/tasks/task_preview_wall_cupboard_close@720x720.png" width=320>            |
| [CupboardsOpenAll](bigym/envs/cupboards.py)                     | Open all drawers and doors of the kitchen set.                                                                                                                                        | <img src="doc/images/tasks/task_preview_cupboards_open_all@720x720.png" width=320>             |
| [CupboardsCloseAll](bigym/envs/cupboards.py)                    | Close all drawers and doors of the kitchen set.                                                                                                                                       | <img src="doc/images/tasks/task_preview_cupboards_close_all@720x720.png" width=320>            |
| [PutCups](bigym/envs/pick_and_place.py)                         | Pick up cups from the table and put them into the closed wall cabinet.                                                                                                                | <img src="doc/images/tasks/task_preview_put_cups@720x720.png" width=320>                       |
| [TakeCups](bigym/envs/pick_and_place.py)                        | Take two cups out from the closed wall cabinet and put them on the table.                                                                                                             | <img src="doc/images/tasks/task_preview_take_cups@720x720.png" width=320>                      |
| [PickBox](bigym/envs/pick_and_place.py)                         | Pick up a large box from the floor and place it on the counter.                                                                                                                       | <img src="doc/images/tasks/task_preview_pick_box@720x720.png" width=320>                       |
| [StoreBox](bigym/envs/pick_and_place.py)                        | Move a large box from the counter to the shelf in the cabinet below.                                                                                                                  | <img src="doc/images/tasks/task_preview_store_box@720x720.png" width=320>                      |
| [SaucepanToHob](bigym/envs/pick_and_place.py)                   | Take the saucepan from the closed cabinet and place it on the hob.                                                                                                                    | <img src="doc/images/tasks/task_preview_saucepan_to_hob@720x720.png" width=320>                |
| [StoreKitchenware](bigym/envs/pick_and_place.py)                | Take all items from the hob and place them in the cabinet below.                                                                                                                      | <img src="doc/images/tasks/task_preview_store_kitchenware@720x720.png" width=320>              |
| [ToastSandwich](bigym/envs/pick_and_place.py)                   | Use the spatula to put the sandwich on the frying pan.                                                                                                                                | <img src="doc/images/tasks/task_preview_toast_sandwich@720x720.png" width=320>                 |
| [FlipSandwich](bigym/envs/pick_and_place.py)                    | Flip the sandwich in the frying pan using the spatula.                                                                                                                                | <img src="doc/images/tasks/task_preview_flip_sandwich@720x720.png" width=320>                  |
| [RemoveSandwich](bigym/envs/pick_and_place.py)                  | Take the sandwich out of the frying pan.                                                                                                                                              | <img src="doc/images/tasks/task_preview_remove_sandwich@720x720.png" width=320>                |
| [GroceriesStoreLower](bigym/envs/groceries.py)                  | Place a random set of groceries in the cabinets below the counter.                                                                                                                    | <img src="doc/images/tasks/task_preview_groceries_store_lower@720x720.png" width=320>          |
| [GroceriesStoreUpper](bigym/envs/groceries.py)                  | Place a random set of groceries in cabinets and shelves on the wall.                                                                                                                  | <img src="doc/images/tasks/task_preview_groceries_store_upper@720x720.png" width=320>          |

## Usage

Directly instantiate the task of interest. Tasks are located in [bigym/envs/](bigym/envs/).

```python
from bigym.action_modes import TorqueActionMode
from bigym.envs.reach_target import ReachTarget
from bigym.utils.observation_config import ObservationConfig, CameraConfig

env = ReachTarget(
    action_mode=TorqueActionMode(floating_base=True),
    observation_config=ObservationConfig(
        cameras=[
            CameraConfig(
                name="head",
                rgb=True,
                depth=False,
                resolution=(128, 128),
            )
        ],
    ),
    render_mode=None,
)
```

Use `ActionModes` to parameterise how you want to control your robot.

## Working with demonstrations

### [Demo Store](demonstrations/demo_store.py)

Please see simple example here: [examples/replay_demo.py](examples/replay_demo.py).

Demonstrations are automatically downloaded from GitHub releases.
When demos are requested by calling `DemoStore.get_demos()`. The current dataset will be cached locally at `$HOME/.bigym/v0.0.0`.
Demonstrations with custom observations and frequency are also cached to `$HOME/.bigym/v0.0.0`.

**⚠️ Warning:** We are working on the dataset. Not all demonstrations will result in successful completion of the task. Please validate before use.

**⚠️ Warning:** The current dataset includes demonstrations for the following action modes only:
- `JointPositionActionMode(floating_base=True, absolute=True)`
- `JointPositionActionMode(floating_base=True, absolute=False)`

### [Demo Player](tools/demo_player/main.py)

Replay existing demos using GUI player.

```bash
python tools/demo_player/main.py
```

<img src="doc/images/demo_player/player_window.png" width=360>
<img src="doc/images/demo_player/player_mujoco.png" width=360>

### [VR Demo Recorder](tools/demo_recorder/main.py)

Record new demos in VR. Follow [VR README](vr/README.md) to configure docker container to run this tool.

```bash
python tools/demo_recorder/main.py
```

<img src="doc/images/demo_recorder/demo_recorder.png" width=360>

## Contributing

On each PR, please ensure to bump the:
- **Major** version if you alter the existing interface in any way.
- **Minor** version if you have added new features (which didn't break the existing interface)
- **Patch** version for bug fixes.

Please ensure that you pass pre-commits before opening a PR: `pre-commit run --all-files` and that you pass all tests: `pytest tests/ --run-slow`.

## Licenses
- [BiGym License (Apache 2.0)](LICENSE) - This repository
- [Mujoco Menagerie (Apache 2.0)](https://github.com/google-deepmind/mujoco_menagerie/blob/main/LICENSE) - Models of robots and grippers
- [3D Assets Attributions (CC0, CC BY 4.0, CC BY NC 4.0)](bigym/envs/xmls/3D_MODELS_ATTRIBUTION.md) - 3D Assets

## Citation
If you find our work helpful, please kindly cite us
```bibtex
@article{chernyadev2024bigym,
  title={BiGym: A Demo-Driven Mobile Bi-Manual Manipulation Benchmark},
  author={Chernyadev, Nikita and Backshall, Nicholas and Ma, Xiao and Lu, Yunfan and Seo, Younggyo and James, Stephen},
  journal={arXiv preprint arXiv:2407.07788},
  year={2024}
}
```
