"""Script to check GPR joint positions in MuJoCo."""

import argparse
from pathlib import Path

import numpy as np

from retos.configs import GPR_CONFIG, ZBOT_CONFIG, Cfg
from retos.mujoco.replay import run_mujoco_orientation, run_mujoco_positions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", action="store_true", help="Replay positions")
    parser.add_argument("--ori", action="store_true", help="Replay orientations")
    parser.add_argument("--config", type=str, default="GPR_CONFIG", help="Config to use")
    parser.add_argument("--pos_file", type=str, help="Path to positions directory")
    parser.add_argument("--ori_file", type=str, help="Path to orientations directory")
    parser.add_argument("--skip-frames", type=int, default=0, help="Number of initial frames to skip")
    parser.add_argument("--timescale", type=float, default=0.1, help="Playback speed multiplier")
    args = parser.parse_args()

    # Get the config from globals()
    config = globals()[args.config]
    assert isinstance(config["config"], Cfg)
    assert isinstance(config["path"], str)

    print(f"Using config: {args.config}")

    if args.pos:
        positions_file = Path(args.pos_file or "gpr_lr_legs.npy")
        positions = np.load(positions_file)
        if args.skip_frames:
            positions = positions[args.skip_frames:]

        print(f"Loaded positions: {positions}")

        # Real to sim mapping
        signs = np.array([-1, 1, 1, -1, 1, -1, 1, 1, -1, 1])

        sim_positions = np.deg2rad(positions)

        sim_positions = sim_positions * signs[None, :]

        run_mujoco_positions(config["path"], config["config"].sim, sim_positions, timescale=args.timescale)

    if args.ori:
        orientations_file = Path(args.ori_file or "imu_log.npy")
        orientations = np.load(orientations_file)

        orientations = np.array([[0, 0, 0]]*1000)

        orientations = orientations[:, [0, 2, 1]]

        signs = np.array([-1, -1, -1])

        sim_orientations = np.deg2rad(orientations)

        sim_orientations = sim_orientations * signs[None, :]

        run_mujoco_orientation(config["path"], config["config"].sim, sim_orientations, timescale=args.timescale)
