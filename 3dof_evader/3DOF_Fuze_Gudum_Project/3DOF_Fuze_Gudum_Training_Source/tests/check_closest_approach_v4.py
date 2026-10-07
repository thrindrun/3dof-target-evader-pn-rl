"""Reward V4 closest-approach modu için hızlı modelsiz kontroller."""

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
    smooth_closest_approach_terminal_term,
)
from training.train_ppo import is_validation_better  # noqa: E402


def fixed_head_on_config(closest_approach_training: bool) -> PPOEnvironmentConfig:
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
        target_sphere_launch_azimuth_min_deg=0.0,
        target_sphere_launch_azimuth_max_deg=0.0,
        target_sphere_launch_elevation_min_deg=0.0,
        target_sphere_launch_elevation_max_deg=0.0,
        target_sphere_aim_azimuth_error_deg=0.0,
        target_sphere_aim_elevation_error_deg=0.0,
        target_sphere_ground_clearance_m=100.0,
        target_sphere_auto_raise_target=True,
        legacy_replay_probability=0.0,
        passed_target_arm_progress_m=0.0,
        closest_approach_training=closest_approach_training,
    )


def check_continuous_terminal_term() -> None:
    radius_m = 75.0
    expected = {
        0.0: 50.0,
        37.5: 25.0,
        75.0: 0.0,
        112.5: -25.0,
        150.0: -50.0,
        225.0: -100.0,
        500.0: -100.0,
    }
    for minimum_range_m, expected_term in expected.items():
        actual = smooth_closest_approach_terminal_term(
            minimum_range_m=minimum_range_m,
            success_radius_m=radius_m,
            hit_reward=50.0,
            miss_distance_penalty_weight=50.0,
            miss_distance_penalty_cap=100.0,
        )
        if not np.isclose(actual, expected_term, atol=1e-12):
            raise AssertionError(
                f"Sürekli terminal terimi hatalı: {minimum_range_m=} "
                f"{actual=} {expected_term=}"
            )

    left = smooth_closest_approach_terminal_term(
        75.0 - 1e-3,
        radius_m,
        50.0,
        50.0,
        100.0,
    )
    right = smooth_closest_approach_terminal_term(
        75.0 + 1e-3,
        radius_m,
        50.0,
        50.0,
        100.0,
    )
    if abs(left - right) > 0.002:
        raise AssertionError("Başarı eşiğinde reward sürekliliği bozuldu.")


def check_hit_is_deferred() -> tuple[float, float]:
    legacy_environment = build_default_ppo_environment(
        fixed_head_on_config(closest_approach_training=False)
    )
    legacy_environment.environment_template.hit_radius_m = 4_999.0
    observation, _ = legacy_environment.reset(seed=123)
    observation, _, terminated, truncated, legacy_info = legacy_environment.step(
        np.zeros(2, dtype=np.float32)
    )
    del observation
    if not terminated or truncated or legacy_info["status"] != "hit":
        raise AssertionError("Reward V3 anlık hit davranışı geriye dönük bozuldu.")
    legacy_terminal_time_s = float(legacy_info["terminal_time_s"])
    legacy_environment.close()

    v4_environment = build_default_ppo_environment(
        fixed_head_on_config(closest_approach_training=True)
    )
    v4_environment.environment_template.hit_radius_m = 4_999.0
    observation, _ = v4_environment.reset(seed=123)
    observation, _, terminated, truncated, first_v4_info = v4_environment.step(
        np.zeros(2, dtype=np.float32)
    )
    if terminated or truncated:
        raise AssertionError("V4 başarı eşiğinde episode'u erken bitirdi.")
    if not first_v4_info["is_success"]:
        raise AssertionError("V4 eşik geçişini başarı olarak kaydetmedi.")

    step_count = 1
    final_info = first_v4_info
    while not (terminated or truncated) and step_count < 500:
        observation, _, terminated, truncated, final_info = v4_environment.step(
            np.zeros(2, dtype=np.float32)
        )
        step_count += 1
    del observation

    if not (terminated or truncated):
        raise AssertionError("V4 episode'u en yakın geçişten sonra bitmedi.")
    if final_info["status"] != "hit":
        raise AssertionError(
            "V4 başarılı closest-approach episode'unu hit olarak işaretlemedi."
        )
    if not final_info["is_success"]:
        raise AssertionError("V4 tamamlanan episode başarısını kaybetti.")
    if float(final_info["terminal_time_s"]) <= legacy_terminal_time_s:
        raise AssertionError("V4 terminal bölgesini gözlemlemeden sona erdi.")
    if "terminal_accuracy" not in final_info["reward_terms"]:
        raise AssertionError("V4 reward_terms terminal_accuracy içermiyor.")

    v4_terminal_time_s = float(final_info["terminal_time_s"])
    v4_environment.close()
    return legacy_terminal_time_s, v4_terminal_time_s


def metrics(
    success_rate: float,
    p90_m: float,
    mean_m: float,
    reward: float = 0.0,
) -> dict[str, float]:
    return {
        "episode_count": 300.0,
        "success_rate": success_rate,
        "p90_minimum_range_m": p90_m,
        "mean_minimum_range_m": mean_m,
        "mean_reward": reward,
    }


def check_robust_best_selection() -> None:
    incumbent = metrics(0.68, 115.0, 82.7)
    one_extra_hit_but_worse_tail = metrics(
        0.68 + 1.0 / 300.0,
        130.0,
        90.0,
    )
    if is_validation_better(one_extra_hit_but_worse_tail, incumbent):
        raise AssertionError("Tek fazla hit kötü P90'ı gizleyerek best seçildi.")

    same_success_better_tail = metrics(0.68, 110.0, 82.0)
    if not is_validation_better(same_success_better_tail, incumbent):
        raise AssertionError("Daha iyi P90'a sahip checkpoint seçilmedi.")

    several_extra_hits = metrics(0.69, 130.0, 90.0)
    if not is_validation_better(several_extra_hits, incumbent):
        raise AssertionError("Anlamlı başarı artışı checkpoint seçimine yansımadı.")


def main() -> None:
    check_continuous_terminal_term()
    legacy_time_s, v4_time_s = check_hit_is_deferred()
    check_robust_best_selection()
    print("Closest-Approach V4 kontrolleri başarılı.")
    print("Reward eşiği sürekliliği              : doğru")
    print("Eski V3 hit sonlandırması              : korunuyor")
    print("V4 hit sonrası terminal gözlem süresi  : doğru")
    print("Sağlam best-checkpoint seçimi          : doğru")
    print("V3 / V4 terminal zamanı [s]            :", legacy_time_s, "/", v4_time_s)


if __name__ == "__main__":
    main()
