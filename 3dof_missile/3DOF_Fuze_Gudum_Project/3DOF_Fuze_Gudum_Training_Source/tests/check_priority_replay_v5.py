"""Priority Replay V5 örnekleme ve doğrulama ayrımı kontrolleri."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from air_to_air_3dof_ppo import (  # noqa: E402
    PPO_GEOMETRY_SOURCE_TARGET_SPHERE_PRIORITY,
    PPO_GEOMETRY_TARGET_SPHERE,
    PPOEnvironmentConfig,
    build_default_ppo_environment,
    sample_target_sphere_launch_azimuth_deg,
)
from training.train_ppo import (  # noqa: E402
    make_uniform_evaluation_config,
    normalize_priority_azimuth_windows,
)


PRIORITY_WINDOWS = ((-150.0, -90.0), (120.0, 180.0))
PRIORITY_PROBABILITY = 0.60


def is_in_priority_union(angle_deg: float) -> bool:
    return any(
        minimum_deg <= angle_deg < maximum_deg
        for minimum_deg, maximum_deg in PRIORITY_WINDOWS
    )


def check_zero_probability_preserves_uniform_rng() -> None:
    reference_rng = np.random.default_rng(20260912)
    actual_rng = np.random.default_rng(20260912)
    expected = float(reference_rng.uniform(-180.0, 180.0))
    actual, used_priority = sample_target_sphere_launch_azimuth_deg(
        rng=actual_rng,
        uniform_min_deg=-180.0,
        uniform_max_deg=180.0,
        priority_windows_deg=PRIORITY_WINDOWS,
        priority_probability=0.0,
    )
    if used_priority or not np.isclose(actual, expected, atol=0.0):
        raise AssertionError("Probability=0 eski uniform RNG akışını değiştirdi.")


def check_mixture_statistics() -> tuple[float, float, int]:
    rng = np.random.default_rng(5102026)
    sample_count = 20_000
    priority_branch_count = 0
    priority_region_count = 0
    occupied_sectors: set[int] = set()

    for _ in range(sample_count):
        angle_deg, used_priority = sample_target_sphere_launch_azimuth_deg(
            rng=rng,
            uniform_min_deg=-180.0,
            uniform_max_deg=180.0,
            priority_windows_deg=PRIORITY_WINDOWS,
            priority_probability=PRIORITY_PROBABILITY,
        )
        priority_branch_count += int(used_priority)
        priority_region_count += int(is_in_priority_union(angle_deg))
        occupied_sectors.add(min(int((angle_deg + 180.0) // 30.0), 11))
        if used_priority and not is_in_priority_union(angle_deg):
            raise AssertionError("Priority dalı tanımlı pencerelerin dışına çıktı.")

    priority_branch_fraction = priority_branch_count / sample_count
    priority_region_fraction = priority_region_count / sample_count
    expected_region_fraction = 0.60 + 0.40 * (120.0 / 360.0)
    if abs(priority_branch_fraction - PRIORITY_PROBABILITY) > 0.02:
        raise AssertionError("Priority dal olasılığı beklenen %60'tan saptı.")
    if abs(priority_region_fraction - expected_region_fraction) > 0.02:
        raise AssertionError("Zor bölge toplam yoğunluğu beklenen değerden saptı.")
    if len(occupied_sectors) != 12:
        raise AssertionError("Uniform replay tam çember kapsamını kaybetti.")
    return priority_branch_fraction, priority_region_fraction, len(occupied_sectors)


def check_cli_windows_and_uniform_evaluation() -> None:
    windows = normalize_priority_azimuth_windows(
        raw_windows=[[120.0, 180.0], [-150.0, -90.0]],
        probability=PRIORITY_PROBABILITY,
        launch_min_deg=-180.0,
        launch_max_deg=180.0,
        geometry_mode=PPO_GEOMETRY_TARGET_SPHERE,
    )
    if windows != PRIORITY_WINDOWS:
        raise AssertionError("CLI pencereleri sıralı tuple'a dönüştürülmedi.")

    training_config = PPOEnvironmentConfig(
        initial_geometry_mode=PPO_GEOMETRY_TARGET_SPHERE,
        target_sphere_priority_azimuth_windows_deg=windows,
        target_sphere_priority_probability=PRIORITY_PROBABILITY,
    )
    evaluation_config = make_uniform_evaluation_config(training_config)
    if training_config.target_sphere_priority_probability != 0.60:
        raise AssertionError("Evaluation kopyası training config'i değiştirdi.")
    if evaluation_config.target_sphere_priority_probability != 0.0:
        raise AssertionError("Evaluation priority olasılığı sıfırlanmadı.")
    if evaluation_config.target_sphere_priority_azimuth_windows_deg:
        raise AssertionError("Evaluation priority pencereleri temizlenmedi.")


def check_environment_integration() -> None:
    config = PPOEnvironmentConfig(
        target_maneuver_kind="straight",
        randomize_initial_conditions=True,
        initial_geometry_mode=PPO_GEOMETRY_TARGET_SPHERE,
        target_sphere_range_min_m=5_000.0,
        target_sphere_range_max_m=5_000.0,
        target_sphere_launch_azimuth_min_deg=-180.0,
        target_sphere_launch_azimuth_max_deg=180.0,
        target_sphere_launch_elevation_min_deg=0.0,
        target_sphere_launch_elevation_max_deg=0.0,
        target_sphere_priority_azimuth_windows_deg=PRIORITY_WINDOWS,
        target_sphere_priority_probability=1.0,
    )
    environment = build_default_ppo_environment(config)
    for seed in range(100):
        _, info = environment.reset(seed=700_000 + seed)
        if info["geometry_source"] != PPO_GEOMETRY_SOURCE_TARGET_SPHERE_PRIORITY:
            raise AssertionError("Environment priority örneğini işaretlemedi.")
        if not is_in_priority_union(float(info["initial_launch_azimuth_deg"])):
            raise AssertionError("Environment priority penceresinin dışına çıktı.")
    environment.close()


def main() -> None:
    check_zero_probability_preserves_uniform_rng()
    branch_fraction, region_fraction, sector_count = check_mixture_statistics()
    check_cli_windows_and_uniform_evaluation()
    check_environment_integration()
    print("Priority Replay V5 kontrolleri başarılı.")
    print("Priority dal gerçekleşme oranı        :", f"%{100*branch_fraction:.2f}")
    print("Zor bölgede toplam örnek oranı        :", f"%{100*region_fraction:.2f}")
    print("Uniform replay ile dolan sektör sayısı:", sector_count)
    print("Doğrulama priority oranı              : %0.00")


if __name__ == "__main__":
    main()
