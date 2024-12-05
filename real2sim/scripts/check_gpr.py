import numpy as np

from real2sim.configs import GPR_CONFIG, Cfg
from real2sim.mujoco.replay_positions import run_mujoco

SAVED_POSITIONS_PATH = "gpr_lr_legs.npy"

if __name__ == "__main__":
    assert isinstance(GPR_CONFIG["config"], Cfg)
    assert isinstance(GPR_CONFIG["path"], str)

    positions = np.load(SAVED_POSITIONS_PATH)

    print(f"Loaded positions: {positions}")

    # Real to sim mapping

    sim_positions = np.deg2rad(positions)

    run_mujoco(GPR_CONFIG["path"], GPR_CONFIG["config"].sim, positions, timescale=1)
