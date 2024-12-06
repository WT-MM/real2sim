import time

import numpy as np
from imu import HexmoveImuReader

imu_reader = HexmoveImuReader("can0", 1, 1)

num_samples = 50  # 10 Hz for 5 seconds
angles = []
print("Calculating IMU offset...")
for _ in range(num_samples):
    imu_data = imu_reader.get_data()
    angles.append([imu_data.x_angle, imu_data.y_angle, imu_data.z_angle])
    time.sleep(0.1)

angular_offset = np.mean(angles, axis=0)

log_data = []
try:
    while True:
        imu_data = imu_reader.get_data()
        log_data.append([imu_data.x_angle - angular_offset[0], imu_data.y_angle - angular_offset[1], imu_data.z_angle - angular_offset[2]])
        time.sleep(0.01)
except KeyboardInterrupt:
    pass

np.save("imu_log.npy", np.array(log_data))
