from real2sim.configs import GPR_CONFIG, Cfg
from real2sim.real.collect_positions import collect_positions

import pykos

if __name__ == "__main__":
    assert isinstance(GPR_CONFIG["config"], Cfg)
    assert isinstance(GPR_CONFIG["path"], str)

    kos = pykos.KOS()


    robot_config = GPR_CONFIG["config"].real.robot

    ids = (
        robot_config.left_leg_ids
        + robot_config.right_leg_ids
        + robot_config.left_arm_ids
        + robot_config.right_arm_ids
    )

    for id in ids:
        print(f"Configuring actuator {id}")
        kos.actuator.configure_actuator(
            actuator_id=id, kp=10, kd=1, torque_enabled=False, zero_position=True
        )

    collect_positions(
        cfg=GPR_CONFIG["config"].real,
        save_path="gpr_lr_legs.npy",
        save_scheme="LEGS_LR",
    )

    print("Done!")
