import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_force_sensor_data(title: str, data: pd.DataFrame, window_size: int | None = None, x_time: np.ndarray = None):
    if x_time is None:
        x_time = np.arange(0, data.shape[0])

    if window_size is not None:
        weights = np.ones(window_size) / window_size
        smoothed_force_z = np.convolve(data["force_z"], weights, mode="same")
        plt.plot(x_time, smoothed_force_z, label="smoothed=2", color="g", marker="o")

    plt.plot(x_time, data["force_x"], label="force_x", color="r")
    plt.plot(x_time, data["force_y"], label="force_y", color="g")
    plt.plot(x_time, data["force_z"], label="force_z", color="b")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(title)
    filename = title.replace(" ", "_").lower() + ".png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
