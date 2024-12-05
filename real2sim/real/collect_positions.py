import json
import time

import numpy as np
import pykos

from real2sim.configs import Cfg


def collect_positions(
    cfg: Cfg.RealCfg,
    save_path: str,
    loop_time: float = 0.01,
    save_scheme: str = "ALL_LR",
    verbose: bool = True,
) -> None:
    """Collects the positions of the actuators in the robot.

    Args:
        cfg: The configuration of the robot.
        save_path: The path to save the positions to.
        loop_time: The time to wait between loops (secs).
        save_scheme: The scheme to save the positions.
        verbose: Whether to print the feedback.
    """
    all_ids = (
        cfg.robot.left_arm_ids
        + cfg.robot.right_arm_ids
        + cfg.robot.left_leg_ids
        + cfg.robot.right_leg_ids
    )

    recorded_positions: dict[int, list[float]] = {id: [] for id in all_ids}

    kos = pykos.KOS()

    try:
        while True:
            feedback = kos.actuator.get_actuators_state(all_ids)
            feedback_dict = {i.actuator_id: i for i in feedback}

            if verbose:
                print(f"Collected {feedback_dict} feedback")

            for id in all_ids:
                recorded_positions[id].append(feedback_dict[id].position)

            time.sleep(loop_time)
    except KeyboardInterrupt:
        if save_scheme == "LEGS_LR":
            lines = np.array(recorded_positions)[cfg.robot.left_leg_ids]
            + np.array(recorded_positions)[cfg.robot.right_leg_ids]

        elif save_scheme == "LEGS_RL":
            lines = np.array(recorded_positions)[cfg.robot.right_leg_ids]
            + np.array(recorded_positions)[cfg.robot.left_leg_ids]

        elif save_scheme == "ARMS_LR":
            lines = np.array(recorded_positions)[cfg.robot.left_arm_ids]
            + np.array(recorded_positions)[cfg.robot.right_arm_ids]

        elif save_scheme == "ARMS_RL":
            lines = np.array(recorded_positions)[cfg.robot.right_arm_ids]
            + np.array(recorded_positions)[cfg.robot.left_arm_ids]

        elif save_scheme == "ALL_LR":
            lines = np.array(recorded_positions)[cfg.robot.left_arm_ids]
            + np.array(recorded_positions)[cfg.robot.right_arm_ids]
            + np.array(recorded_positions)[cfg.robot.left_leg_ids]
            + np.array(recorded_positions)[cfg.robot.right_leg_ids]

        elif save_scheme == "ALL_RL":
            lines = np.array(recorded_positions)[cfg.robot.right_arm_ids]
            + np.array(recorded_positions)[cfg.robot.left_arm_ids]
            + np.array(recorded_positions)[cfg.robot.right_leg_ids]
            + np.array(recorded_positions)[cfg.robot.left_leg_ids]

        else:
            with open(save_path, "w") as f:
                json.dump(recorded_positions, f)
            raise ValueError(f"Invalid save scheme: {save_scheme}. Dumped to json")

        np.save(save_path, lines)
