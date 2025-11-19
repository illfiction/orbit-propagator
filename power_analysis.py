import json
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from analysis.solarpower import compute_generated_power
from satellite.satellite import Satellite


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as config_file:
        return json.load(config_file)


def _build_satellite(config: dict) -> Satellite:
    return Satellite(config["initial_conditions"], config["satellite_properties"])


def _detect_eclipse_windows(flags: np.ndarray) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(flags):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            windows.append((start, idx - 1))
            start = None
    if start is not None:
        windows.append((start, len(flags) - 1))
    return windows


def _plot_power(
    times_s: np.ndarray, power_w: np.ndarray, eclipse_flags: np.ndarray
) -> None:
    times_min = times_s / 60.0
    plt.figure(figsize=(12, 5))
    plt.plot(times_min, power_w, label="Generated Power", color="tab:blue")
    plt.fill_between(
        times_min, 0, power_w, where=~eclipse_flags, color="tab:blue", alpha=0.1
    )
    if np.any(eclipse_flags):
        plt.fill_between(
            times_min,
            0,
            power_w,
            where=eclipse_flags,
            color="tab:red",
            alpha=0.2,
            label="Eclipse",
        )
    plt.xlabel("Time (minutes)")
    plt.ylabel("Power (W)")
    plt.title("Satellite Power Generation Profile")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def run_power_analysis(config_path: str = "config.json") -> None:
    config = _load_config(config_path)
    simulation = config["simulation"]
    dt = simulation["dt_seconds"]
    total_time = simulation["time_span_days"] * 90 * 60
    steps = int(total_time / dt)

    satellite = _build_satellite(config)
    power_config = config.get("power_analysis", {})

    power_profile = np.zeros(steps)
    eclipse_flags = np.zeros(steps, dtype=bool)
    times = np.arange(steps) * dt

    print(
        f"Evaluating solar power over {simulation['time_span_days']} days "
        f"with a timestep of {dt}s..."
    )

    for step in tqdm(range(steps), desc="Propagating Orbit"):
        current_time = step * dt
        satellite.update(current_time, dt)
        power_profile[step], eclipse_flags[step] = compute_generated_power(
            current_time, satellite, power_config
        )

    max_idx = int(np.argmax(power_profile))
    max_power = power_profile[max_idx]
    mean_power = float(np.mean(power_profile))
    total_eclipse_time = float(np.sum(eclipse_flags) * dt)
    eclipse_windows = _detect_eclipse_windows(eclipse_flags)

    print(f"Max power: {max_power:.2f} W at t={times[max_idx] / 60:.2f} min")
    print(f"Mean power: {mean_power:.2f} W")

    if total_eclipse_time > 0:
        print(
            f"Total eclipse duration: {total_eclipse_time / 60:.2f} minutes "
            f"across {len(eclipse_windows)} passes"
        )
        first_start, first_end = eclipse_windows[0]
        print(
            f"First eclipse window: {times[first_start] / 60:.2f} - "
            f"{times[first_end] / 60:.2f} min"
        )
    else:
        print("No eclipse detected during simulation window.")

    _plot_power(times, power_profile, eclipse_flags)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, "config.json")
    run_power_analysis(config_file_path)
