"""Dishwasher interaction tasks."""
from abc import ABC

import numpy as np
from gymnasium import spaces

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.dishwasher import Dishwasher


class _DishwasherEnv(BiGymEnv, ABC):
    """Base dishwasher environment."""

    RESET_ROBOT_POS = np.array([0, -0.8, 0])

    _PRESET_PATH = PRESETS_PATH / "dishwasher.yaml"
    _TOLERANCE = 0.05

    def _initialize_env(self):
        self.dishwasher = self._preset.get_props(Dishwasher)[0]

    def _get_task_privileged_obs(self):
        """Door, bottom tray and middle tray, normalised -- what `_success` thresholds."""
        return {"dishwasher_state": np.asarray(self.dishwasher.get_state(), np.float32)}

    def _get_task_privileged_obs_space(self):
        return {
            "dishwasher_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        }


class DishwasherOpen(_DishwasherEnv):
    """Open the dishwasher door and pull out all trays."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state(), 1, atol=self._TOLERANCE)

    def _on_reset(self):
        self.dishwasher.set_state(door=0, bottom_tray=0, middle_tray=0)


class DishwasherClose(_DishwasherEnv):
    """Push back all trays and close the door of the dishwasher."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state(), 0, atol=self._TOLERANCE)

    def _on_reset(self):
        self.dishwasher.set_state(door=1, bottom_tray=1, middle_tray=1)


class DishwasherCloseTrays(DishwasherClose):
    """Push the dishwasher’s trays back with the door initially open."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state()[1:], 0, atol=self._TOLERANCE)


class DishwasherOpenTrays(DishwasherClose):
    """Pull out the dishwasher’s trays with the door initially open."""

    def _success(self) -> bool:
        return np.allclose(self.dishwasher.get_state()[1:], 1, atol=self._TOLERANCE)

    def _on_reset(self):
        self.dishwasher.set_state(door=1, bottom_tray=0, middle_tray=0)
