"""Script to align and plot two timeseries dataframes."""
import platform
from dataclasses import dataclass
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.widgets import RadioButtons, Slider

plt.ion()

@dataclass
class PlotState:
    """State container for each subplot."""
    current_real_idx: int = 0
    current_sim_idx: int = 0
    real_scale: float = 1.0
    sim_scale: float = 1.0
    line_real: Optional[plt.Line2D] = None
    line_sim: Optional[plt.Line2D] = None

def plot_data_comparison(real_data: pd.DataFrame,
                         sim_data: pd.DataFrame,
                         num_plots: int = 1,
                         real_timestamp_col: Optional[str] = 'timestamp',
                         sim_timestamp_col: Optional[str] = 'timestamp',
                         sim_timestep: float = 0.1) -> None:
    """Interactive plot to compare and align two dataframes with multiple plots.

    Each plot (subplot) is opened in its own window, with radio buttons on left,
    plot and sliders on right.

    Note that that only 1 plot will be interactive due to problems.
    """
    # On macOS, ensure we use a backend that supports interactive widgets
    if platform.system() == 'Darwin':
        matplotlib.use('TkAgg')

    # Prepare columns (excluding the timestamp col)
    real_columns = [col for col in real_data.columns if col != real_timestamp_col]
    sim_columns = [col for col in sim_data.columns if col != sim_timestamp_col]

    # Prepare time data
    if real_timestamp_col:
        real_time = pd.to_datetime(real_data[real_timestamp_col])
        real_time_sec = (real_time - real_time.iloc[0]).dt.total_seconds()
    else:
        real_time_sec = real_data.index

    if sim_timestamp_col:
        sim_time = pd.to_datetime(sim_data[sim_timestamp_col])
        sim_time_sec = (sim_time - sim_time.iloc[0]).dt.total_seconds()
    else:
        sim_time_sec = sim_data.index * sim_timestep

    plot_states = []
    for i in range(num_plots):
        fig = plt.figure(figsize=(15, 6))

        # Top-level GridSpec: 1 row x 2 columns
        #   - Left col (width_ratios=0.3) → radio buttons
        #   - Right col (width_ratios=0.7) → plot + sliders
        gs_main = GridSpec(
            nrows=1,
            ncols=2,
            figure=fig,
            width_ratios=[0.3, 0.7],
            wspace=0.3,
            top=0.90,
            bottom=0.10,
            left=0.05,
            right=0.95
        )

        # Left sub‐GridSpec: 2 rows, each equally tall → Sim radio vs Real radio
        gs_left = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=gs_main[0, 0],
            height_ratios=[1, 1],
            hspace=0.3
        )

        # Right sub‐GridSpec: 3 rows, ratio [5, 0.25, 0.25]
        #   → row 0 (5 units high) = main plot
        #   → row 1 (0.25 units) = real slider
        #   → row 2 (0.25 units) = sim slider
        # This makes the plot bigger than the two sliders combined.
        gs_right = GridSpecFromSubplotSpec(
            3, 1,
            subplot_spec=gs_main[0, 1],
            height_ratios=[5, 0.25, 0.25],
            hspace=0.3
        )

        state = PlotState()
        plot_states.append(state)

        # -----------------------
        # LEFT COLUMN: Radio Buttons
        # -----------------------
        ax_radio_sim = fig.add_subplot(gs_left[0, 0])
        radio_sim = RadioButtons(ax_radio_sim, sim_columns, active=0)
        ax_radio_sim.set_title('Sim Data')

        ax_radio_real = fig.add_subplot(gs_left[1, 0])
        radio_real = RadioButtons(ax_radio_real, real_columns, active=0)
        ax_radio_real.set_title('Real Data')

        # -----------------------
        # RIGHT COLUMN: Plot + Sliders
        # -----------------------
        # Plot uses row=0 in the right sub‐GridSpec
        ax_plot = fig.add_subplot(gs_right[0, 0])
        state.line_real, = ax_plot.plot(
            real_time_sec,
            real_data[real_columns[0]] * state.real_scale,
            label=f'Real {real_columns[0]}'
        )
        state.line_sim, = ax_plot.plot(
            sim_time_sec,
            sim_data[sim_columns[0]] * state.sim_scale,
            label=f'Sim {sim_columns[0]}'
        )
        ax_plot.set_xlabel('Time (seconds)')
        ax_plot.set_ylabel('Value')
        ax_plot.grid(True)
        ax_plot.legend()
        ax_plot.set_title(f'Plot {i+1}')

        # Sliders → rows 1 and 2 in the right sub‐GridSpec
        ax_slider_real = fig.add_subplot(gs_right[1, 0])
        slider_real = Slider(ax_slider_real, 'Real Scale', -10, 10, valinit=1)

        ax_slider_sim = fig.add_subplot(gs_right[2, 0])
        slider_sim = Slider(ax_slider_sim, 'Sim Scale', -10, 10, valinit=1)

        # -----------------------
        # CALLBACKS
        # -----------------------
        def update_plot() -> None:
            assert state.line_real is not None
            assert state.line_sim is not None

            state.line_real.set_ydata(
                real_data[real_columns[state.current_real_idx]] * state.real_scale
            )
            state.line_sim.set_ydata(
                sim_data[sim_columns[state.current_sim_idx]] * state.sim_scale
            )
            state.line_real.set_label(
                f'Real {real_columns[state.current_real_idx]} (*{state.real_scale:.1f})'
            )
            state.line_sim.set_label(
                f'Sim {sim_columns[state.current_sim_idx]} (*{state.sim_scale:.1f})'
            )
            ax_plot.relim()
            ax_plot.autoscale_view()
            ax_plot.legend()
            fig.canvas.draw_idle()

        def real_radio_clicked(label: str) -> None:
            state.current_real_idx = real_columns.index(label)
            update_plot()

        def sim_radio_clicked(label: str) -> None:
            state.current_sim_idx = sim_columns.index(label)
            update_plot()

        def update_real_scale(val: float) -> None:
            state.real_scale = val
            update_plot()

        def update_sim_scale(val: float) -> None:
            state.sim_scale = val
            update_plot()

        # Connect callbacks
        radio_real.on_clicked(real_radio_clicked)
        radio_sim.on_clicked(sim_radio_clicked)
        slider_real.on_changed(update_real_scale)
        slider_sim.on_changed(update_sim_scale)

    # Show all windows (one per subplot)
    plt.ioff()
    plt.show(block=True)

def plot_data_comparison_from_files(real_path: str,
                                    sim_path: str,
                                    num_plots: int = 1,
                                    real_timestamp_col: Optional[str] = 'timestamp',
                                    sim_timestamp_col: Optional[str] = 'timestamp',
                                    sim_timestep: float = 0.1) -> None:
    """Load CSV files and create interactive comparison plots."""
    real_data = pd.read_csv(real_path)
    sim_data = pd.read_csv(sim_path)
    plot_data_comparison(
        real_data,
        sim_data,
        num_plots=num_plots,
        real_timestamp_col=real_timestamp_col,
        sim_timestamp_col=sim_timestamp_col,
        sim_timestep=sim_timestep
    )

if __name__ == "__main__":
    # Example usage
    plot_data_comparison_from_files(
        'examples/imu/real_data.csv',
        'examples/imu/sim_data.csv',
        num_plots=1,
        real_timestamp_col='timestamp',
        sim_timestamp_col=None,
        sim_timestep=0.1
    )
