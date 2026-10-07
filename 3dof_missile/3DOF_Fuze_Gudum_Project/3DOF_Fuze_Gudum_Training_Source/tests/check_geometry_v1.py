"""Geometry V1 için hızlı, model gerektirmeyen sayısal kontroller."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from air_to_air_3dof_ppo import (  # noqa: E402
    PPO_GEOMETRY_POLAR,
    PPO_GEOMETRY_ROTATED_NOMINAL,
    PPOEnvironmentConfig,
    build_default_ppo_environment,
)


def check_rotational_invariance() -> float:
    legacy_config = PPOEnvironmentConfig(
        target_maneuver_kind="straight",
        randomize_initial_conditions=True,
    )
    rotated_config = replace(
        legacy_config,
        initial_geometry_mode=PPO_GEOMETRY_ROTATED_NOMINAL,
        global_rotation_limit_deg=180.0,
    )
    legacy = build_default_ppo_environment(legacy_config)
    rotated = build_default_ppo_environment(rotated_config)

    legacy_observation, _ = legacy.reset(seed=20_260_911)
    rotated_observation, _ = rotated.reset(seed=20_260_911)
    maximum_difference = float(
        np.max(np.abs(legacy_observation - rotated_observation))
    )
    if maximum_difference > 2e-6:
        raise AssertionError(
            "Global dönüş ilk observation'ı değiştirdi: "
            f"maksimum fark={maximum_difference}"
        )

    for index in range(100):
        action = np.array(
            [
                0.15 * np.sin(index / 9.0),
                -0.10 * np.cos(index / 13.0),
            ]
        )
        legacy_observation, legacy_reward, legacy_done, legacy_cut, legacy_info = (
            legacy.step(action)
        )
        rotated_observation, rotated_reward, rotated_done, rotated_cut, rotated_info = (
            rotated.step(action)
        )
        observation_difference = float(
            np.max(np.abs(legacy_observation - rotated_observation))
        )
        maximum_difference = max(maximum_difference, observation_difference)
        if observation_difference > 3e-5:
            raise AssertionError(
                f"Global dönüş dinamiği değiştirdi: adım={index}, "
                f"maksimum fark={observation_difference}"
            )
        if abs(legacy_reward - rotated_reward) > 3e-6:
            raise AssertionError("Global dönüş reward'u değiştirdi.")
        legacy_terminal = (legacy_done, legacy_cut, legacy_info["status"])
        rotated_terminal = (rotated_done, rotated_cut, rotated_info["status"])
        if legacy_terminal != rotated_terminal:
            raise AssertionError("Global dönüş terminal sonucunu değiştirdi.")
        if legacy_done or legacy_cut:
            break

    legacy.close()
    rotated.close()
    return maximum_difference


def check_polar_bounds() -> int:
    config = PPOEnvironmentConfig(
        target_maneuver_kind="straight",
        randomize_initial_conditions=True,
        initial_geometry_mode=PPO_GEOMETRY_POLAR,
        polar_range_min_m=4_800.0,
        polar_range_max_m=5_200.0,
        polar_bearing_min_deg=-180.0,
        polar_bearing_max_deg=180.0,
        polar_elevation_min_deg=-10.0,
        polar_elevation_max_deg=10.0,
        polar_target_aspect_min_deg=-30.0,
        polar_target_aspect_max_deg=30.0,
        passed_target_arm_progress_m=100.0,
    )
    environment = build_default_ppo_environment(config)
    occupied_sectors: set[int] = set()

    for seed in range(1_000):
        observation, info = environment.reset(seed=seed)
        if observation.shape != (23,) or not np.all(np.isfinite(observation)):
            raise AssertionError("Polar reset sonlu 23-girdili observation üretmedi.")
        if not 4_800.0 - 1e-7 <= info["initial_range_m"] <= 5_200.0 + 1e-7:
            raise AssertionError("Polar menzil sınırı aşıldı.")
        if not -10.0 - 1e-7 <= info["initial_los_elevation_deg"] <= 10.0 + 1e-7:
            raise AssertionError("Polar elevation sınırı aşıldı.")
        if not -30.0 - 1e-7 <= info["initial_target_aspect_deg"] <= 30.0 + 1e-7:
            raise AssertionError("Polar target-aspect sınırı aşıldı.")
        sector = int((info["initial_relative_bearing_deg"] + 180.0) // 30.0)
        occupied_sectors.add(min(sector, 11))

    environment.close()
    if occupied_sectors != set(range(12)):
        raise AssertionError(f"Eksik 30 derece sektörleri: {occupied_sectors}")
    return len(occupied_sectors)


def check_opening_target_guard() -> float:
    config = PPOEnvironmentConfig(
        target_maneuver_kind="straight",
        randomize_initial_conditions=True,
        missile_position_jitter_m=(0.0, 0.0, 0.0),
        target_position_jitter_m=(0.0, 0.0, 0.0),
        missile_speed_jitter_mps=0.0,
        target_speed_jitter_mps=0.0,
        missile_heading_jitter_deg=0.0,
        target_heading_jitter_deg=0.0,
        missile_flight_path_jitter_deg=0.0,
        target_flight_path_jitter_deg=0.0,
        initial_geometry_mode=PPO_GEOMETRY_POLAR,
        polar_range_min_m=5_000.0,
        polar_range_max_m=5_000.0,
        polar_bearing_min_deg=180.0,
        polar_bearing_max_deg=180.0,
        polar_elevation_min_deg=0.0,
        polar_elevation_max_deg=0.0,
        polar_target_aspect_min_deg=0.0,
        polar_target_aspect_max_deg=0.0,
        passed_target_arm_progress_m=100.0,
    )
    environment = build_default_ppo_environment(config)
    _, info = environment.reset(seed=7)
    initial_closing_speed_mps = float(info["initial_closing_speed_mps"])
    if initial_closing_speed_mps >= 0.0:
        raise AssertionError("Guard testi başlangıçta açılan hedef üretmedi.")

    for _ in range(10):
        _, _, terminated, truncated, info = environment.step(np.zeros(2))
        if terminated or truncated or info["status"] != "flying":
            raise AssertionError("Açılan hedef yanlışlıkla erken sonlandırıldı.")
        if info["closest_approach_armed"]:
            raise AssertionError("Passed-target guard yaklaşma olmadan açıldı.")

    environment.close()
    return initial_closing_speed_mps


def main() -> None:
    maximum_rotation_difference = check_rotational_invariance()
    sector_count = check_polar_bounds()
    opening_speed_mps = check_opening_target_guard()
    print("Geometry V1 kontrolleri başarılı.")
    print("Global dönüş maksimum observation farkı:", maximum_rotation_difference)
    print("Doldurulan 30 derece sektör sayısı      :", sector_count)
    print("Arka hedef ilk closing speed [m/s]       :", opening_speed_mps)


if __name__ == "__main__":
    main()
