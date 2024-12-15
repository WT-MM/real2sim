"""Module to define the real2sims configurations for different embodiments."""

from dataclasses import dataclass, field


@dataclass
class Cfg:
    @dataclass
    class SimCfg:
        @dataclass
        class Robot:
            # Number of movable joints
            num_joints: int = 22
            kps: list[float] = field(default_factory=lambda: [1.0] * 22)
            kds: list[float] = field(default_factory=lambda: [1.0] * 22)

        dt: float = 0.001
        # Suspend height
        suspend: float | None = None
        lock_orientation: bool = False
        robot: Robot = field(default_factory=lambda: Cfg.SimCfg.Robot())

    @dataclass
    class RealCfg:
        @dataclass
        class Robot:
            left_arm_ids: list[int] = field(default_factory=lambda: [11, 12, 13, 14])
            right_arm_ids: list[int] = field(default_factory=lambda: [21, 22, 23, 24])
            left_leg_ids: list[int] = field(default_factory=lambda: [31, 32, 33, 34, 35])
            right_leg_ids: list[int] = field(default_factory=lambda: [41, 42, 43, 44, 45])

            default_kp: float = 1.0
            default_kd: float = 0.1

        robot: Robot = field(default_factory=lambda: Cfg.RealCfg.Robot())

    sim: SimCfg = field(default_factory=lambda: Cfg.SimCfg())
    real: RealCfg = field(default_factory=lambda: Cfg.RealCfg())


GPR_CONFIG = {
    "path": "real2sim/embodiments/gpr/robot_fixed.xml",
    "config": Cfg(
        sim=Cfg.SimCfg(
            dt=0.001,
            suspend=1.2,
            lock_orientation=False,
            robot=Cfg.SimCfg.Robot(num_joints=10, kps=[100.0] * 10, kds=[10.0] * 10),
        ),
        real=Cfg.RealCfg(
            robot=Cfg.RealCfg.Robot(
                left_arm_ids=[11, 12, 13, 14, 15, 16],
                right_arm_ids=[21, 22, 23, 24, 25, 26],
                left_leg_ids=[31, 32, 33, 34, 35],
                right_leg_ids=[41, 42, 43, 44, 45],
            )
        ),
    ),
}
