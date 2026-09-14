"""Configuration classes for environment observations."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CameraConfig:
    """Configuration for camera observations."""

    name: str
    rgb: bool = True
    depth: bool = False
    resolution: tuple[int, int] = (128, 128)
    pos: Optional[tuple[float, float, float]] = None
    quat: Optional[tuple[float, float, float, float]] = None

    def __post_init__(self):
        """Validation."""
        assert len(self.resolution) == 2
        if not isinstance(self.resolution, tuple):
            self.resolution = tuple(self.resolution)
        if self.pos is not None:
            assert len(self.pos) == 3
            if not isinstance(self.pos, tuple):
                self.pos = tuple(self.pos)
        if self.quat is not None:
            assert len(self.quat) == 4
            if not isinstance(self.quat, tuple):
                self.quat = tuple(self.quat)

    @classmethod
    def from_safetensors_metadata(cls, metadata: dict):
        """Get metadata from a safetensor metadata dict."""
        camera_config = cls(**metadata)
        camera_config.resolution = tuple(camera_config.resolution)
        return camera_config

    def to_string(self):
        """Get a string representation of the camera configuration."""
        s = self.name
        if self.rgb:
            s += "-rgb"
        if self.depth:
            s += "-depth"
        s += "-" + "x".join(map(str, self.resolution))
        return s


PROPRIOCEPTION_RAW = "raw"
PROPRIOCEPTION_COMPACT = "compact"


@dataclass
class ObservationConfig:
    """Configuration for environment observations."""

    cameras: list[CameraConfig] = field(default_factory=list)
    proprioception: bool = True
    privileged_information: bool = False
    # How much of the proprioceptive observation to emit.
    #
    #   "compact"  joint POSITIONS only -- `robot.qpos`, 30 numbers (29 with a 3-DOF base)
    #   "raw"      what BiGym emitted before this option: qpos and qvel together, plus the
    #              gripper and floating-base keys, 66 numbers
    #
    # `compact` is the default because the other 36 numbers carry nothing the first 30 do not,
    # measured across all 40 tasks:
    #   `proprioception_floating_base` (4) is BIT-IDENTICAL to qpos's pelvis entries, in 40/40;
    #   `proprioception_grippers` (2) is average(driver joint qpos) rescaled and rounded to one
    #      decimal -- a monotone function of qpos entries already present (corr 0.9999), so a
    #      lossier copy rather than new information;
    #   `qvel` (30) is recoverable from position differences under frame stacking, at
    #      corr 0.92-0.99 for every arm joint and for pelvis x/y.
    #
    # What `compact` keeps includes the 16 gripper linkage joints, which are NOT redundant:
    # their effective rank is 3 per gripper, because the fingers deflect differently when
    # something is held. Use "raw" to reproduce a result published against the old observation.
    proprioception_mode: str = PROPRIOCEPTION_COMPACT

    @classmethod
    def from_safetensors_metadata(cls, metadata: dict):
        """Get metadata from a safetensor file."""
        metadata["cameras"] = [
            CameraConfig.from_safetensors_metadata(camera)
            for camera in metadata["cameras"]
        ]
        return cls(**metadata)
