"""Fresh-Start / Predictive-CPA V6 için hızlı deterministik kontroller."""

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
    constant_velocity_closest_approach,
    predictive_cpa_progress_term,
)
from training.train_ppo import (  # noqa: E402
    evaluate_on_fixed_seeds,
    is_validation_better,
)


def target_sphere_config(*, predictive_reward: bool) -> PPOEnvironmentConfig:
    """Tekrarlanabilir yatay 5 km çemberi için test konfigürasyonu."""
    return PPOEnvironmentConfig(
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
        initial_geometry_mode=PPO_GEOMETRY_TARGET_SPHERE,
        target_sphere_range_min_m=5_000.0,
        target_sphere_range_max_m=5_000.0,
        target_sphere_launch_azimuth_min_deg=-180.0,
        target_sphere_launch_azimuth_max_deg=180.0,
        target_sphere_launch_elevation_min_deg=0.0,
        target_sphere_launch_elevation_max_deg=0.0,
        target_sphere_ground_clearance_m=100.0,
        passed_target_arm_progress_m=100.0,
        closest_approach_training=True,
        predictive_cpa_reward_enabled=predictive_reward,
    )


def check_constant_velocity_cpa_math() -> None:
    distance_m, time_s = constant_velocity_closest_approach(
        relative_position_m=np.array([1_000.0, 0.0, 0.0]),
        relative_velocity_mps=np.array([-100.0, 0.0, 0.0]),
        horizon_s=20.0,
    )
    if abs(distance_m) > 1e-9 or abs(time_s - 10.0) > 1e-9:
        raise AssertionError("Head-on CPA hesabı yanlış.")

    distance_m, time_s = constant_velocity_closest_approach(
        relative_position_m=np.array([1_000.0, 0.0, 0.0]),
        relative_velocity_mps=np.array([-100.0, 100.0, 0.0]),
        horizon_s=20.0,
    )
    expected_distance_m = np.sqrt(500.0**2 + 500.0**2)
    if abs(distance_m - expected_distance_m) > 1e-9 or abs(time_s - 5.0) > 1e-9:
        raise AssertionError("Çapraz CPA hesabı yanlış.")

    distance_m, time_s = constant_velocity_closest_approach(
        relative_position_m=np.array([1_000.0, 0.0, 0.0]),
        relative_velocity_mps=np.array([100.0, 0.0, 0.0]),
        horizon_s=20.0,
    )
    if abs(distance_m - 1_000.0) > 1e-9 or abs(time_s) > 1e-9:
        raise AssertionError("Açılan hedefte CPA zamanı geçmişe taşındı.")


def check_progress_term() -> None:
    positive = predictive_cpa_progress_term(500.0, 400.0, 1.0, 100.0, 250.0)
    negative = predictive_cpa_progress_term(400.0, 500.0, 1.0, 100.0, 250.0)
    clipped = predictive_cpa_progress_term(1_000.0, 0.0, 1.0, 100.0, 250.0)
    if abs(positive - 1.0) > 1e-12 or abs(negative + 1.0) > 1e-12:
        raise AssertionError("CPA progress reward işareti yanlış.")
    if abs(clipped - 2.5) > 1e-12:
        raise AssertionError("CPA progress reward kırpması yanlış.")


def check_reward_only_change() -> None:
    environment_off = build_default_ppo_environment(
        target_sphere_config(predictive_reward=False)
    )
    environment_on = build_default_ppo_environment(
        target_sphere_config(predictive_reward=True)
    )
    environment_off.environment_template.hit_radius_m = 75.0
    environment_on.environment_template.hit_radius_m = 75.0

    observation_off, info_off = environment_off.reset(seed=1234)
    observation_on, info_on = environment_on.reset(seed=1234)
    if not np.array_equal(observation_off, observation_on):
        raise AssertionError("Reward seçeneği başlangıç observation'ını değiştirdi.")
    if abs(
        float(info_off["initial_predictive_cpa_distance_m"])
        - float(info_on["initial_predictive_cpa_distance_m"])
    ) > 1e-12:
        raise AssertionError("Reward seçeneği başlangıç CPA'sını değiştirdi.")

    actions = (
        np.array([0.20, -0.10], dtype=np.float32),
        np.array([0.10, 0.05], dtype=np.float32),
        np.array([-0.15, 0.00], dtype=np.float32),
    )
    for action in actions:
        result_off = environment_off.step(action)
        result_on = environment_on.step(action)
        observation_off, reward_off, terminated_off, truncated_off, info_off = (
            result_off
        )
        observation_on, reward_on, terminated_on, truncated_on, info_on = result_on
        if not np.array_equal(observation_off, observation_on):
            raise AssertionError("Predictive reward fizik yörüngesini değiştirdi.")
        if (terminated_off, truncated_off) != (terminated_on, truncated_on):
            raise AssertionError("Predictive reward episode sonlanmasını değiştirdi.")
        cpa_reward = float(info_on["reward_terms"]["predictive_cpa_progress"])
        if abs(float(info_off["reward_terms"]["predictive_cpa_progress"])) > 1e-12:
            raise AssertionError("Kapalı predictive reward sıfır değil.")
        if abs((reward_on - reward_off) - cpa_reward) > 2e-7:
            raise AssertionError("Yeni toplam reward farkı CPA terimine eşit değil.")

    environment_off.close()
    environment_on.close()


class ZeroPolicy:
    """Sektör dağılımı kontrolü için modelsiz deterministik politika."""

    @staticmethod
    def predict(observation, deterministic: bool = True):
        del observation, deterministic
        return np.zeros(2, dtype=np.float32), None


def check_balanced_sector_evaluation() -> None:
    environment = build_default_ppo_environment(
        target_sphere_config(predictive_reward=True)
    )
    environment.environment_template.hit_radius_m = 500.0
    original_bounds = (
        environment.ppo_config.target_sphere_launch_azimuth_min_deg,
        environment.ppo_config.target_sphere_launch_azimuth_max_deg,
    )
    metrics = evaluate_on_fixed_seeds(
        model=ZeroPolicy(),
        environment=environment,
        evaluation_seeds=list(range(24)),
        azimuth_sector_count=12,
    )
    if metrics["sector_episode_counts"] != [2] * 12:
        raise AssertionError("Dengeli doğrulama her sektöre eşit episode vermedi.")
    restored_bounds = (
        environment.ppo_config.target_sphere_launch_azimuth_min_deg,
        environment.ppo_config.target_sphere_launch_azimuth_max_deg,
    )
    if restored_bounds != original_bounds:
        raise AssertionError("Doğrulama sonrasında azimut sınırları geri yüklenmedi.")
    environment.close()


def check_sector_robust_best_selection() -> None:
    incumbent: dict[str, object] = {
        "episode_count": 120.0,
        "success_rate": 0.50,
        "worst_sector_mean_minimum_range_m": 200.0,
        "p90_minimum_range_m": 150.0,
        "mean_minimum_range_m": 100.0,
        "mean_reward": 0.0,
    }
    slightly_more_hits_but_worse_sector = {
        **incumbent,
        "success_rate": 0.51,
        "worst_sector_mean_minimum_range_m": 220.0,
    }
    if is_validation_better(slightly_more_hits_but_worse_sector, incumbent):
        raise AssertionError("Küçük hit farkı kötü sektörü gizleyerek best seçildi.")

    same_band_better_worst_sector = {
        **incumbent,
        "success_rate": 0.49,
        "worst_sector_mean_minimum_range_m": 180.0,
    }
    if not is_validation_better(same_band_better_worst_sector, incumbent):
        raise AssertionError("İyileşen en-kötü sektör checkpoint'e yansımadı.")


def main() -> None:
    check_constant_velocity_cpa_math()
    check_progress_term()
    check_reward_only_change()
    check_balanced_sector_evaluation()
    check_sector_robust_best_selection()
    print("Fresh-Start Predictive-CPA V6 kontrolleri başarılı.")
    print("Sabit-hız CPA matematiği              : doğru")
    print("Erken reward işareti ve kırpması      : doğru")
    print("Reward kapalıyken eski dinamik        : korunuyor")
    print("12 sektörlü doğrulama episode sayısı  : eşit")
    print("Kör-sektör duyarlı best seçimi         : doğru")


if __name__ == "__main__":
    main()
