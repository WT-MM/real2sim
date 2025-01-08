"""Script to plot and compare corresponding columns from two CSV files."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class TimeseriesData:
    """Container for timeseries data with timestamps and measurements."""
    timestamps: np.ndarray
    measurements: Dict[str, np.ndarray]

def load_csv(filepath: str, timestamp_col: Optional[str] = 'timestamp') -> TimeseriesData:
    """Load timeseries data from CSV file and return structured data."""
    df = pd.read_csv(filepath)

    # Convert timestamp string to seconds if present
    if timestamp_col and timestamp_col in df.columns:
        timestamps = pd.to_datetime(df[timestamp_col])
        timestamps = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    else:
        # For data without timestamps, use index
        timestamps = np.arange(len(df)) * 0.1

    # Extract all measurements except timestamp
    measurements = {col: df[col].values 
                   for col in df.columns 
                   if col != timestamp_col}

    return TimeseriesData(timestamps, measurements)

def find_common_columns(data1: TimeseriesData, data2: TimeseriesData) -> Set[str]:
    """Find column names that exist in both datasets."""
    cols1 = set(data1.measurements.keys())
    cols2 = set(data2.measurements.keys())
    return cols1.intersection(cols2)

def plot_comparison(data1: TimeseriesData, data2: TimeseriesData, 
                   column: str, labels: tuple = ('Data 1', 'Data 2')) -> None:
    """Plot comparison between two datasets for a specified column."""
    plt.plot(data1.timestamps, data1.measurements[column], 
             'b-', label=labels[0], alpha=0.7)
    plt.plot(data2.timestamps, data2.measurements[column], 
             'r--', label=labels[1], alpha=0.7)

    plt.xlabel('Time (seconds)')
    plt.ylabel(column)
    plt.title(f'{column} Comparison')
    plt.grid(True)
    plt.legend()

def plot_csvs(file1: str, file2: str, 
         timestamp_col1: Optional[str] = 'timestamp',
         timestamp_col2: Optional[str] = 'timestamp',
         labels: tuple = ('Data 1', 'Data 2'),
         plots_per_figure: int = 3) -> None:
    """Load and plot comparisons for all common columns between two CSV files."""
    # Load data
    data1 = load_csv(file1, timestamp_col1)
    data2 = load_csv(file2, timestamp_col2)

    # Find common columns
    common_columns = find_common_columns(data1, data2)
    if not common_columns:
        raise ValueError("No common columns found between the two files")

    # Create figures with multiple subplots
    num_columns = len(common_columns)
    num_figures = (num_columns + plots_per_figure - 1) // plots_per_figure

    for fig_num in range(num_figures):
        plt.figure(figsize=(12, 4 * plots_per_figure))

        # Get columns for this figure
        start_idx = fig_num * plots_per_figure
        end_idx = min(start_idx + plots_per_figure, num_columns)
        fig_columns = list(common_columns)[start_idx:end_idx]

        for i, column in enumerate(fig_columns):
            plt.subplot(plots_per_figure, 1, i + 1)
            plot_comparison(data1, data2, column, labels)

        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    # Example usage
    plot_csvs('examples/imu/real_data.csv',
              'examples/imu/sim_data.csv',
              timestamp_col1='timestamp',
              timestamp_col2=None,
              labels=('Real', 'Simulation'),
              plots_per_figure=3)
