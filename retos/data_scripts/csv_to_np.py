import numpy as np
import pandas as pd


def csv_to_euler_np(csv_path: str) -> np.ndarray:
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Extract euler angles (roll, pitch, yaw) and convert to numpy array
    euler_angles = df[['roll (deg)', 'pitch (deg)', 'yaw (deg)']].to_numpy()

    return euler_angles

if __name__ == "__main__":
    # Path to your CSV file
    csv_path = "examples/imu/med1.csv"

    # Convert CSV to numpy array
    euler_data = csv_to_euler_np(csv_path)

    # Optional: Save to .npy file
    np.save('euler_angles.npy', euler_data)

    # Print shape to verify (should be [n_timestamps, 3])
    print(f"Euler angles shape: {euler_data.shape}")
    # Optional: Print first few rows to verify data
    print("\nFirst 3 timestamps (roll, pitch, yaw):")
    print(euler_data[:3])
