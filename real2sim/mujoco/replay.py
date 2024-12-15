"""Replay positions from a given trajectory."""

import math
import time
from collections import deque
from copy import deepcopy
from typing import Tuple

import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from real2sim.configs import GPR_CONFIG, Cfg


def quaternion_to_euler_array(quat: np.ndarray) -> np.ndarray:
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert euler angles (roll, pitch, yaw) to quaternion [x, y, z, w].

    Args:
        euler: Array of euler angles [roll, pitch, yaw] in radians

    Returns:
        Quaternion as [x, y, z, w]
    """
    # Create rotation object from euler angles
    r = R.from_euler("xyz", euler)

    # Get quaternion (scipy returns as [x, y, z, w])
    return r.as_quat()


def get_obs(data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts an observation from the mujoco data structure."""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q: np.ndarray, q: np.ndarray, kps: np.ndarray, dq: np.ndarray, kds: np.ndarray) -> np.ndarray:
    return kps * (target_q - q) - kds * dq


def run_mujoco_orientation(
    model_path: str, cfg: Cfg.SimCfg, orientations: np.ndarray, timescale: float = 1.0, render: bool = True
) -> None:
    """Run the Mujoco simulation using the provided policy and configuration.

    Args:
        model_path: The path to the Mujoco model.
        cfg: The configuration object containing simulation settings.
        orientations: The orientations to replay.
        timescale: Factor to scale the replay timesteps.
        render: Whether to render the simulation.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = cfg.dt
    model.opt.gravity = np.zeros(3)
    data = mujoco.MjData(model)

    try:
        data.qpos = model.keyframe("default").qpos
        default = deepcopy(model.keyframe("default").qpos)[-cfg.robot.num_joints :]
        print("Default position:", default)
    except Exception as _:
        print("No default position found, using zero initialization")
        default = np.zeros(cfg.robot.num_joints)

    mujoco.mj_step(model, data)
    for ii in range(len(data.ctrl) + 1):
        print(data.joint(ii).id, data.joint(ii).name)

    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)

    if render:
        viewer = mujoco_viewer.MujocoViewer(model, data)

    # Add deque to store last 50 diffs
    diff_history: deque[float] = deque(maxlen=50)

    for ori in orientations:
        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.robot.num_joints :]
        dq = dq[-cfg.robot.num_joints :]

        eu_ang = quaternion_to_euler_array(quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        diff = float(np.linalg.norm(ori - eu_ang))
        diff_history.append(diff)
        avg_diff = np.mean(list(diff_history))
        max_diff = max(diff_history)
        print(f"Diff: {diff:.4f}, Running Average (50): {avg_diff:.4f}, Max (50): {max_diff:.4f}")

        quat = euler_to_quaternion(ori)
        data.qpos[3:7] = quat[[3, 0, 1, 2]]  # Reorder from [x,y,z,w] to [w,x,y,z]

        if cfg.suspend:
            data.qpos[2] = cfg.suspend

        mujoco.mj_forward(model, data)

        if render:
            viewer.render()

        time.sleep(cfg.dt / timescale)


def run_mujoco_positions(
    model_path: str, cfg: Cfg.SimCfg, positions: np.ndarray, timescale: float = 1.0, render: bool = True
) -> None:
    """Run the Mujoco simulation using the provided policy and configuration.

    Args:
        model_path: The path to the Mujoco model.
        cfg: The configuration object containing simulation settings.
        positions: The positions to replay.
        timescale: Factor to scale the replay timesteps.
        render: Whether to render the simulation.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = cfg.dt
    model.opt.gravity = np.zeros(3)
    data = mujoco.MjData(model)

    try:
        data.qpos = model.keyframe("default").qpos
        default = deepcopy(model.keyframe("default").qpos)[-cfg.robot.num_joints :]
        print("Default position:", default)
    except Exception as _:
        print("No default position found, using zero initialization")
        default = np.zeros(cfg.robot.num_joints)

    mujoco.mj_step(model, data)
    for ii in range(len(data.ctrl) + 1):
        print(data.joint(ii).id, data.joint(ii).name)

    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)

    if render:
        viewer = mujoco_viewer.MujocoViewer(model, data)

    for pos in positions:
        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.robot.num_joints :]
        dq = dq[-cfg.robot.num_joints :]

        eu_ang = quaternion_to_euler_array(quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        target_q = pos

        # Set joint positions directly
        data.qpos[-cfg.robot.num_joints :] = target_q

        print(f"Target q: {target_q}")

        if cfg.suspend:
            data.qpos[2] = cfg.suspend

        if cfg.lock_orientation:
            data.qpos[3:7] = np.array([1, 0, 0, 0])

        mujoco.mj_forward(model, data)

        if render:
            viewer.render()

        time.sleep(cfg.dt / timescale)


if __name__ == "__main__":
    assert isinstance(GPR_CONFIG["config"], Cfg)
    assert isinstance(GPR_CONFIG["path"], str)

    positions = np.zeros((100000, GPR_CONFIG["config"].sim.robot.num_joints))
    run_mujoco_positions(GPR_CONFIG["path"], GPR_CONFIG["config"].sim, positions, timescale=1)
