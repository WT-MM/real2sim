import numpy as np

from real2sim.configs import GPR_CONFIG, Cfg
from real2sim.mujoco.replay_positions import run_mujoco

SAVED_POSITIONS_PATH = "gpr_lr_legs.npy"

if __name__ == "__main__":
    assert isinstance(GPR_CONFIG["config"], Cfg)
    assert isinstance(GPR_CONFIG["path"], str)

    positions = np.load(SAVED_POSITIONS_PATH)
    positions = positions[400:] # skip the first 100

    print(f"Loaded positions: {positions}")

    # Real to sim mapping
    signs = np.array([-1, 1, 1, -1, 1,
             -1, 1, 1, -1, 1])

    sim_positions = np.deg2rad(positions)

    sim_positions = sim_positions * signs[None, :]

    run_mujoco(GPR_CONFIG["path"], GPR_CONFIG["config"].sim, sim_positions, timescale=0.1)
