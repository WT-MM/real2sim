"""Script to read IMU data from the robot."""

from real2sim.real.collect_imu import collect_imu_data

if __name__ == "__main__":
    collect_imu_data(
        {"port": "can0", "canid": 1, "id": 1},
        "imu_log.npy",
    )
