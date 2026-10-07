"""Target-sphere başlangıç geometrisi için hızlı, modelsiz sayısal kontroller."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from air_to_air_3dof_ppo import (  # noqa: E402
    PPO_GEOMETRY_TARGET_SPHERE,
    PPOEnvironmentConfig,
    build_default_ppo_environment,
)


def zero_jitter_config(**overrides) -> PPOEnvironmentConfig:
    values = {
        "target_maneuver_kind": "straight",
        "randomize_initial_conditions": True,
        "missile_position_jitter_m": (0.0, 0.0, 0.0),
        "target_position_jitter_m": (0.0, 0.0, 0.0),
        "missile_speed_jitter_mps": 0.0,
        "target_speed_jitter_mps": 0.0,
        "missile_heading_jitter_deg": 0.0,
        "target_heading_jitter_deg": 0.0,
        "missile_flight_path_jitter_deg": 0.0,
        "target_flight_path_jitter_deg": 0.0,
        "initial_geometry_mode": PPO_GEOMETRY_TARGET_SPHERE,
        "target_sphere_range_min_m": 5_000.0,
        "target_sphere_range_max_m": 5_000.0,
        "target_sphere_launch_azimuth_min_deg": -180.0,
        "target_sphere_launch_azimuth_max_deg": 180.0,
        "target_sphere_launch_elevation_min_deg": -90.0,
        "target_sphere_launch_elevation_max_deg": 90.0,
        "target_sphere_aim_azimuth_error_deg": 0.0,
        "target_sphere_aim_elevation_error_deg": 0.0,
        "target_sphere_ground_clearance_m": 100.0,
        "target_sphere_auto_raise_target": True,
        "legacy_replay_probability": 0.0,
        "passed_target_arm_progress_m": 100.0,
    }
    values.update(overrides)
    return PPOEnvironmentConfig(**values)


def check_requested_horizontal_examples() -> None:
    examples = (
        (0.0, np.array([5_000.0, 0.0, 0.0])),
        (
            float(np.degrees(np.arctan2(4_000.0, 3_000.0))),
            np.array([3_000.0, 4_000.0, 0.0]),
        ),
    )

    for launch_azimuth_deg, expected_offset_m in examples:
        config = zero_jitter_config(
            target_sphere_launch_azimuth_min_deg=launch_azimuth_deg,
            target_sphere_launch_azimuth_max_deg=launch_azimuth_deg,
            target_sphere_launch_elevation_min_deg=0.0,
            target_sphere_launch_elevation_max_deg=0.0,
        )
        environment = build_default_ppo_environment(config)
        _, info = environment.reset(seed=123)
        missile_position = np.asarray(info["initial_missile_position_m"])
        target_position = np.asarray(info["initial_target_position_m"])
        actual_offset_m = missile_position - target_position

        if not np.allclose(actual_offset_m, expected_offset_m, atol=1e-7):
            raise AssertionError(
                f"İstenen launch offset üretilmedi: {actual_offset_m}"
            )
        if abs(float(info["initial_range_m"]) - 5_000.0) > 1e-7:
            raise AssertionError("Yatay örneğin başlangıç menzili 5 km değil.")
        if abs(float(info["initial_relative_bearing_deg"])) > 1e-7:
            raise AssertionError("Füze başlangıçta hedefe dönük değil.")
        if abs(float(info["initial_los_elevation_deg"])) > 1e-7:
            raise AssertionError("Yatay örnekte LOS elevation sıfır değil.")
        if abs(float(info["initial_missile_flight_path_deg"])) > 1e-7:
            raise AssertionError("Yatay örnekte füze uçuş-yolu hedefe dönük değil.")
        environment.close()


def check_full_sphere_coverage() -> tuple[int, int, float]:
    config = zero_jitter_config()
    environment = build_default_ppo_environment(config)
    occupied_azimuth_sectors: set[int] = set()
    occupied_equal_area_bands: set[int] = set()
    minimum_missile_altitude_m = np.inf

    for seed in range(2_400):
        observation, info = environment.reset(seed=seed)
        if observation.shape != (23,) or not np.all(np.isfinite(observation)):
            raise AssertionError("Target-sphere sonlu 23-girdili observation üretmedi.")
        if abs(float(info["initial_range_m"]) - 5_000.0) > 1e-6:
            raise AssertionError("Target-sphere yarıçapı 5 km'den saptı.")
        if abs(float(info["initial_relative_bearing_deg"])) > 2e-5:
            raise AssertionError("Füze burnu yatayda hedef LOS'una bakmıyor.")

        los_elevation_deg = float(info["initial_los_elevation_deg"])
        flight_path_deg = float(info["initial_missile_flight_path_deg"])
        if abs(los_elevation_deg - flight_path_deg) > 0.002:
            raise AssertionError("Füze burnu düşeyde hedef LOS'una bakmıyor.")

        launch_azimuth_deg = float(info["initial_launch_azimuth_deg"])
        launch_elevation_deg = float(info["initial_launch_elevation_deg"])
        azimuth_sector = min(
            int((launch_azimuth_deg + 180.0) // 30.0),
            11,
        )
        occupied_azimuth_sectors.add(max(azimuth_sector, 0))

        equal_area_coordinate = (
            np.sin(np.radians(launch_elevation_deg)) + 1.0
        ) / 2.0
        elevation_band = min(int(equal_area_coordinate * 6), 5)
        occupied_equal_area_bands.add(max(elevation_band, 0))

        missile_down_m = float(info["initial_missile_position_m"][2])
        minimum_missile_altitude_m = min(
            minimum_missile_altitude_m,
            -missile_down_m,
        )

    environment.close()
    if occupied_azimuth_sectors != set(range(12)):
        raise AssertionError(
            f"Eksik 30 derece azimut sektörleri: {occupied_azimuth_sectors}"
        )
    if occupied_equal_area_bands != set(range(6)):
        raise AssertionError(
            f"Eksik eşit-alan elevation bantları: {occupied_equal_area_bands}"
        )
    if minimum_missile_altitude_m < 100.0 - 1e-6:
        raise AssertionError("Tam kürenin bir bölümü yer güvenlik payının altında.")
    return (
        len(occupied_azimuth_sectors),
        len(occupied_equal_area_bands),
        float(minimum_missile_altitude_m),
    )


def main() -> None:
    check_requested_horizontal_examples()
    azimuth_count, elevation_count, minimum_altitude_m = (
        check_full_sphere_coverage()
    )
    print("Target Sphere V1 kontrolleri başarılı.")
    print("İstenen iki yatay 5 km örneği           : doğru")
    print("Doldurulan 30 derece azimut sektörü     :", azimuth_count)
    print("Doldurulan eşit-alan elevation bandı    :", elevation_count)
    print("Örneklenen minimum füze irtifası [m]    :", minimum_altitude_m)


if __name__ == "__main__":
    main()
