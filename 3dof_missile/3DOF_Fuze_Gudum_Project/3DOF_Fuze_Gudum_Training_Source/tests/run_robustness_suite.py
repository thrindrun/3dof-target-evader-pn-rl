"""
3DOF PPO Robustness Test Suite
==============================

VSCode / terminal kullanım örneği:

    python tests/run_robustness_suite.py ^
        --stage fresh_cpa_v6_hardpocket_7p5m_colab_refine01

PowerShell / bash tek satır:

    python tests/run_robustness_suite.py --stage fresh_cpa_v6_hardpocket_7p5m_colab_refine01

Hızlı smoke değerlendirmesi:

    python tests/run_robustness_suite.py --stage fresh_cpa_v6_hardpocket_7p5m_colab_refine01 --quick

Sadece belirli testler:

    python tests/run_robustness_suite.py --stage fresh_cpa_v6_hardpocket_7p5m_colab_refine01 --tests training_domain large_mc maneuver constant_turn

Beklenen proje yapısı:

3DOF_Fuze_Gudum_Project/
├── air_to_air_3dof_core.py
└── 3DOF_Fuze_Gudum_Training_Source/
    ├── air_to_air_3dof_ppo.py
    ├── models/
    │   └── <MODEL_STAGE>/
    │       ├── best_model.zip
    │       └── training_metadata.json
    └── tests/
        └── run_robustness_suite.py

Sonuçlar varsayılan olarak:

    results/tests/<MODEL_STAGE>/

altına CSV ve JSON olarak kaydedilir.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from stable_baselines3 import PPO


DEFAULT_MODEL_STAGE = "fresh_cpa_v6_hardpocket_7p5m_colab_refine01"

SOURCE_DIR = Path(__file__).resolve().parents[1]
OUTER_PROJECT_DIR = SOURCE_DIR.parent

for path in (OUTER_PROJECT_DIR, SOURCE_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import air_to_air_3dof_core as core  # noqa: E402
import air_to_air_3dof_ppo as ppo  # noqa: E402


ALL_TEST_NAMES = (
    "training_domain",
    "large_mc",
    "elevation",
    "range",
    "aim_error",
    "randomization",
    "maneuver",
    "constant_turn",
    "constant_turn_3d",
    "control_period",
    "action_noise",
)

ELEVATION_BANDS = (
    (-90.0, -60.0),
    (-60.0, -40.0),
    (-40.0, -20.0),
    (-20.0, 0.0),
    (0.0, 20.0),
    (20.0, 40.0),
    (40.0, 60.0),
    (60.0, 90.0),
)

RANGES_M = (3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0)

AIM_ERROR_CASES_DEG = (
    (0.0, 0.0),
    (2.0, 2.0),
    (5.0, 5.0),
    (10.0, 10.0),
    (20.0, 20.0),
    (30.0, 30.0),
)

RANDOMIZATION_SCALES = (0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00)

MANEUVERS = ("straight", "constant_turn", "sinusoidal", "random_smooth")

TURN_RATES_DEG_S = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0)

TURN_3D_CASES_DEG_S = (
    (4.0, 0.0),
    (8.0, 0.0),
    (12.0, 0.0),
    (16.0, 0.0),
    (8.0, 3.0),
    (12.0, 4.5),
    (16.0, 6.0),
)

CONTROL_PERIODS_S = (0.08, 0.10, 0.12, 0.14, 0.20)

ACTION_NOISE_STDS = (0.00, 0.01, 0.02, 0.05, 0.10)

DEFAULT_THRESHOLDS_M = (7.5, 5.0, 2.5, 1.0, 0.5, 0.25, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO modeli için kapsamlı 3DOF robustness test paketi."
    )
    parser.add_argument(
        "--stage",
        default=DEFAULT_MODEL_STAGE,
        help="models/<stage>/best_model.zip ve training_metadata.json kullanılacak stage adı.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="İsteğe bağlı doğrudan best_model.zip yolu.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="İsteğe bağlı training_metadata.json yolu.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Sonuç kök klasörü. Varsayılan: <training_source>/results/tests/<stage>",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=ALL_TEST_NAMES,
        default=list(ALL_TEST_NAMES),
        help="Çalıştırılacak test grupları.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Episode sayılarını azaltan hızlı doğrulama modu.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Stable-Baselines3 model device değeri. Varsayılan: cpu",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(
    success_count: int,
    episode_count: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if episode_count <= 0:
        return math.nan, math.nan

    p = success_count / episode_count
    denominator = 1.0 + z * z / episode_count
    center = (p + z * z / (2.0 * episode_count)) / denominator
    half_width = (
        z
        * math.sqrt(
            (p * (1.0 - p) / episode_count)
            + (z * z / (4.0 * episode_count * episode_count))
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def apply_test_randomization_scale(
    config: ppo.PPOEnvironmentConfig,
    scale: float,
) -> None:
    scale = float(scale)

    config.missile_position_jitter_m = tuple(
        scale * value for value in config.missile_position_jitter_m
    )
    config.target_position_jitter_m = tuple(
        scale * value for value in config.target_position_jitter_m
    )
    config.missile_speed_jitter_mps *= scale
    config.target_speed_jitter_mps *= scale
    config.missile_heading_jitter_deg *= scale
    config.target_heading_jitter_deg *= scale
    config.missile_flight_path_jitter_deg *= scale
    config.target_flight_path_jitter_deg *= scale


def make_test_config(
    metadata: dict[str, Any],
    *,
    maneuver: str = "straight",
    elevation_bounds: tuple[float, float] | None = None,
    range_bounds: tuple[float, float] | None = None,
    aim_error_deg: tuple[float, float] | None = None,
    randomization_scale: float | None = None,
    control_period_s: float = 0.10,
) -> ppo.PPOEnvironmentConfig:
    if elevation_bounds is None:
        elevation_bounds = (
            float(metadata["target_sphere_launch_elevation_min_deg"]),
            float(metadata["target_sphere_launch_elevation_max_deg"]),
        )

    if range_bounds is None:
        range_bounds = (
            float(metadata["target_sphere_range_min_m"]),
            float(metadata["target_sphere_range_max_m"]),
        )

    if aim_error_deg is None:
        aim_error_deg = (
            float(metadata.get("target_sphere_aim_azimuth_error_deg", 0.0)),
            float(metadata.get("target_sphere_aim_elevation_error_deg", 0.0)),
        )

    if randomization_scale is None:
        randomization_scale = float(metadata.get("randomization_scale", 0.5))

    config = ppo.PPOEnvironmentConfig(
        control_period_s=float(control_period_s),
        target_maneuver_kind=maneuver,
        randomize_initial_conditions=bool(
            metadata.get("randomize_initial_conditions", True)
        ),
        initial_geometry_mode=metadata.get(
            "initial_geometry_mode",
            ppo.PPO_GEOMETRY_TARGET_SPHERE,
        ),
        target_sphere_range_min_m=float(range_bounds[0]),
        target_sphere_range_max_m=float(range_bounds[1]),
        target_sphere_launch_azimuth_min_deg=float(
            metadata.get("target_sphere_launch_azimuth_min_deg", -180.0)
        ),
        target_sphere_launch_azimuth_max_deg=float(
            metadata.get("target_sphere_launch_azimuth_max_deg", 180.0)
        ),
        target_sphere_launch_elevation_min_deg=float(elevation_bounds[0]),
        target_sphere_launch_elevation_max_deg=float(elevation_bounds[1]),
        target_sphere_priority_azimuth_windows_deg=(),
        target_sphere_priority_probability=0.0,
        target_sphere_aim_azimuth_error_deg=float(aim_error_deg[0]),
        target_sphere_aim_elevation_error_deg=float(aim_error_deg[1]),
        target_sphere_ground_clearance_m=float(
            metadata.get("target_sphere_ground_clearance_m", 100.0)
        ),
        target_sphere_auto_raise_target=bool(
            metadata.get("target_sphere_auto_raise_target", True)
        ),
        legacy_replay_probability=0.0,
        passed_target_arm_progress_m=float(
            metadata.get("passed_target_arm_progress_m", 100.0)
        ),
        closest_approach_training=bool(
            metadata.get("closest_approach_training", True)
        ),
        predictive_cpa_reward_enabled=bool(
            metadata.get("predictive_cpa_reward_enabled", True)
        ),
        predictive_cpa_progress_weight=float(
            metadata.get("predictive_cpa_progress_weight", 1.0)
        ),
        predictive_cpa_progress_scale_m=float(
            metadata.get("predictive_cpa_progress_scale_m", 100.0)
        ),
        predictive_cpa_progress_clip_m=float(
            metadata.get("predictive_cpa_progress_clip_m", 250.0)
        ),
        predictive_cpa_horizon_s=float(
            metadata.get("predictive_cpa_horizon_s", 35.0)
        ),
        randomize_target_maneuver_seed=True,
    )

    metadata_to_config = {
        "reward_scale": "reward_scale",
        "hit_reward_unscaled": "hit_reward",
        "passed_target_penalty_unscaled": "passed_target_penalty",
        "miss_distance_penalty_weight_unscaled": "miss_distance_penalty_weight",
        "miss_distance_penalty_cap_unscaled": "miss_distance_penalty_cap",
        "range_progress_weight": "range_progress_weight",
        "range_progress_scale_m": "range_progress_scale_m",
        "minimum_range_progress_weight": "minimum_range_progress_weight",
        "minimum_range_progress_scale_m": "minimum_range_progress_scale_m",
        "control_effort_penalty": "control_effort_penalty",
        "time_penalty": "time_penalty",
    }

    for metadata_key, config_attribute in metadata_to_config.items():
        if metadata_key in metadata and hasattr(config, config_attribute):
            setattr(
                config,
                config_attribute,
                float(metadata[metadata_key]),
            )

    apply_test_randomization_scale(config, float(randomization_scale))
    return config


def build_test_env(
    metadata: dict[str, Any],
    obs_version: str,
    *,
    maneuver: str = "straight",
    elevation_bounds: tuple[float, float] | None = None,
    range_bounds: tuple[float, float] | None = None,
    aim_error_deg: tuple[float, float] | None = None,
    randomization_scale: float | None = None,
    control_period_s: float = 0.10,
    maneuver_rates: tuple[float, float] | None = None,
):
    config = make_test_config(
        metadata,
        maneuver=maneuver,
        elevation_bounds=elevation_bounds,
        range_bounds=range_bounds,
        aim_error_deg=aim_error_deg,
        randomization_scale=randomization_scale,
        control_period_s=control_period_s,
    )

    rl_config = core.RLInterfaceConfig(
        observation_version=obs_version,
        deterministic=True,
        control_period_s=float(control_period_s),
    )

    env = ppo.build_default_ppo_environment(
        ppo_environment_config=config,
        rl_interface_config=rl_config,
    )
    env.environment_template.hit_radius_m = float(metadata["hit_radius_m"])

    if maneuver_rates is not None:
        heading_rate, flight_path_rate = maneuver_rates
        target_template = next(
            vehicle
            for vehicle in env.vehicle_templates
            if vehicle.role == "aircraft"
        )
        target_template.maneuver.heading_rate_deg_s = float(heading_rate)
        target_template.maneuver.flight_path_rate_deg_s = float(
            flight_path_rate
        )

    return env


def evaluate_test(
    model: PPO,
    env,
    *,
    episodes: int,
    seed_start: int,
    sector_count: int = 12,
    thresholds_m: Iterable[float] = DEFAULT_THRESHOLDS_M,
    action_noise_std: float = 0.0,
    progress_every: int = 100,
) -> dict[str, Any]:
    rewards: list[float] = []
    minimum_ranges: list[float] = []
    terminal_times: list[float] = []
    statuses: dict[str, int] = {}
    hit_count = 0

    sector_hits = np.zeros(sector_count, dtype=int)
    sector_counts = np.zeros(sector_count, dtype=int)
    sector_ranges: list[list[float]] = [
        [] for _ in range(sector_count)
    ]

    original_bounds: tuple[float, float] | None = None
    if sector_count > 0:
        original_bounds = (
            float(env.ppo_config.target_sphere_launch_azimuth_min_deg),
            float(env.ppo_config.target_sphere_launch_azimuth_max_deg),
        )
        full_width = original_bounds[1] - original_bounds[0]
        sector_width = full_width / sector_count

    noise_rng = np.random.default_rng(int(seed_start) + 9_000_000)

    try:
        for episode_index in range(int(episodes)):
            seed = int(seed_start) + episode_index
            sector_index: int | None = None

            if sector_count > 0:
                assert original_bounds is not None
                sector_index = episode_index % sector_count
                lower = (
                    original_bounds[0]
                    + sector_index * sector_width
                )
                env.ppo_config.target_sphere_launch_azimuth_min_deg = lower
                env.ppo_config.target_sphere_launch_azimuth_max_deg = (
                    lower + sector_width
                )

            observation, info = env.reset(seed=seed)
            terminated = False
            truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                action = np.asarray(action, dtype=np.float32)

                if action_noise_std > 0.0:
                    action = action + noise_rng.normal(
                        0.0,
                        float(action_noise_std),
                        size=action.shape,
                    )
                    action = np.clip(
                        action,
                        -1.0,
                        1.0,
                    ).astype(np.float32)

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                total_reward += float(reward)

            minimum_range = float(info["minimum_range_m"])
            success = bool(
                info.get(
                    "is_success",
                    info.get("status") == "hit",
                )
            )
            status = str(info.get("status", "unknown"))

            rewards.append(total_reward)
            minimum_ranges.append(minimum_range)
            terminal_times.append(float(info["terminal_time_s"]))
            statuses[status] = statuses.get(status, 0) + 1
            hit_count += int(success)

            if sector_index is not None:
                sector_counts[sector_index] += 1
                sector_hits[sector_index] += int(success)
                sector_ranges[sector_index].append(minimum_range)

            completed = episode_index + 1
            if (
                progress_every > 0
                and (
                    completed % progress_every == 0
                    or completed == episodes
                )
            ):
                current_rate = 100.0 * hit_count / completed
                print(
                    f"  {completed:5d}/{episodes:5d} | "
                    f"success={current_rate:6.2f}%"
                )

    finally:
        if original_bounds is not None:
            (
                env.ppo_config.target_sphere_launch_azimuth_min_deg,
                env.ppo_config.target_sphere_launch_azimuth_max_deg,
            ) = original_bounds

    minimum_ranges_array = np.asarray(minimum_ranges, dtype=float)
    rewards_array = np.asarray(rewards, dtype=float)
    terminal_times_array = np.asarray(terminal_times, dtype=float)

    ci_low, ci_high = wilson_interval(hit_count, int(episodes))

    result: dict[str, Any] = {
        "episodes": int(episodes),
        "success_count": int(hit_count),
        "success_rate": float(hit_count / episodes),
        "success_ci95_low": float(ci_low),
        "success_ci95_high": float(ci_high),
        "mean_minimum_range_m": float(np.mean(minimum_ranges_array)),
        "median_minimum_range_m": float(np.median(minimum_ranges_array)),
        "p90_minimum_range_m": float(
            np.percentile(minimum_ranges_array, 90.0)
        ),
        "p95_minimum_range_m": float(
            np.percentile(minimum_ranges_array, 95.0)
        ),
        "worst_minimum_range_m": float(np.max(minimum_ranges_array)),
        "mean_reward": float(np.mean(rewards_array)),
        "mean_terminal_time_s": float(np.mean(terminal_times_array)),
        "statuses": statuses,
        "threshold_rates": {
            str(float(threshold)): float(
                np.mean(minimum_ranges_array <= float(threshold))
            )
            for threshold in thresholds_m
        },
    }

    if sector_count > 0:
        if np.any(sector_counts == 0):
            raise RuntimeError(
                "En az bir azimut sektörü episode almadı."
            )

        sector_success = sector_hits / sector_counts
        sector_means = np.asarray(
            [float(np.mean(values)) for values in sector_ranges],
            dtype=float,
        )

        assert original_bounds is not None
        sector_bounds = []
        for index in range(sector_count):
            lower = original_bounds[0] + index * sector_width
            sector_bounds.append(
                [float(lower), float(lower + sector_width)]
            )

        result.update(
            {
                "azimuth_sector_count": int(sector_count),
                "sector_azimuth_bounds_deg": sector_bounds,
                "sector_episode_counts": sector_counts.tolist(),
                "sector_success_rates": sector_success.tolist(),
                "sector_mean_minimum_ranges_m": sector_means.tolist(),
                "worst_sector_success_rate": float(
                    np.min(sector_success)
                ),
                "worst_sector_mean_minimum_range_m": float(
                    np.max(sector_means)
                ),
            }
        )

    return result


def result_row(
    test_name: str,
    case_name: str,
    result: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "test": test_name,
        "case": case_name,
        **extra,
        "episodes": result["episodes"],
        "success_count": result["success_count"],
        "success_percent": 100.0 * result["success_rate"],
        "success_ci95_low_percent": 100.0 * result["success_ci95_low"],
        "success_ci95_high_percent": 100.0 * result["success_ci95_high"],
        "mean_minimum_range_m": result["mean_minimum_range_m"],
        "median_minimum_range_m": result["median_minimum_range_m"],
        "p90_minimum_range_m": result["p90_minimum_range_m"],
        "p95_minimum_range_m": result["p95_minimum_range_m"],
        "worst_minimum_range_m": result["worst_minimum_range_m"],
        "mean_reward": result["mean_reward"],
        "mean_terminal_time_s": result["mean_terminal_time_s"],
    }

    if "worst_sector_success_rate" in result:
        row["worst_sector_success_percent"] = (
            100.0 * result["worst_sector_success_rate"]
        )
        row["worst_sector_mean_minimum_range_m"] = result[
            "worst_sector_mean_minimum_range_m"
        ]

    for threshold, rate in result["threshold_rates"].items():
        row[f"success_le_{threshold}_m_percent"] = 100.0 * rate

    return row


def print_result(title: str, result: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(
        f"Success      : {result['success_count']}/{result['episodes']} "
        f"({100.0 * result['success_rate']:.2f}%)"
    )
    print(
        f"95% CI       : "
        f"{100.0 * result['success_ci95_low']:.2f}% .. "
        f"{100.0 * result['success_ci95_high']:.2f}%"
    )
    print(
        f"Mean min     : {result['mean_minimum_range_m']:.4f} m"
    )
    print(
        f"Median min   : {result['median_minimum_range_m']:.4f} m"
    )
    print(
        f"P90 / P95    : "
        f"{result['p90_minimum_range_m']:.4f} / "
        f"{result['p95_minimum_range_m']:.4f} m"
    )
    print(
        f"Worst        : {result['worst_minimum_range_m']:.4f} m"
    )

    if "worst_sector_success_rate" in result:
        print(
            f"Worst sector : "
            f"{100.0 * result['worst_sector_success_rate']:.2f}% "
            f"| mean="
            f"{result['worst_sector_mean_minimum_range_m']:.4f} m"
        )

    print("Thresholds:")
    for threshold, rate in result["threshold_rates"].items():
        print(
            f"  <= {float(threshold):6.2f} m : "
            f"{100.0 * rate:6.2f}%"
        )


def run_case(
    *,
    model: PPO,
    metadata: dict[str, Any],
    obs_version: str,
    title: str,
    case_name: str,
    episodes: int,
    seed_start: int,
    sector_count: int = 12,
    maneuver: str = "straight",
    elevation_bounds: tuple[float, float] | None = None,
    range_bounds: tuple[float, float] | None = None,
    aim_error_deg: tuple[float, float] | None = None,
    randomization_scale: float | None = None,
    control_period_s: float = 0.10,
    maneuver_rates: tuple[float, float] | None = None,
    action_noise_std: float = 0.0,
    progress_every: int = 100,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print("\n" + "#" * 78)
    print(title, "-", case_name)
    print("#" * 78)

    env = build_test_env(
        metadata,
        obs_version,
        maneuver=maneuver,
        elevation_bounds=elevation_bounds,
        range_bounds=range_bounds,
        aim_error_deg=aim_error_deg,
        randomization_scale=randomization_scale,
        control_period_s=control_period_s,
        maneuver_rates=maneuver_rates,
    )

    try:
        result = evaluate_test(
            model,
            env,
            episodes=episodes,
            seed_start=seed_start,
            sector_count=sector_count,
            action_noise_std=action_noise_std,
            progress_every=progress_every,
        )
    finally:
        env.close()

    print_result(f"{title} - {case_name}", result)

    row = result_row(
        title,
        case_name,
        result,
        **(extra or {}),
    )
    return result, row


def main() -> int:
    args = parse_args()

    model_stage = str(args.stage)

    if args.model is None:
        model_path = (
            SOURCE_DIR
            / "models"
            / model_stage
            / "best_model.zip"
        )
    else:
        model_path = Path(args.model).expanduser().resolve()

    if args.metadata is None:
        metadata_path = model_path.parent / "training_metadata.json"
    else:
        metadata_path = Path(args.metadata).expanduser().resolve()

    if args.results_dir is None:
        results_dir = (
            SOURCE_DIR
            / "results"
            / "tests"
            / model_stage
        )
    else:
        results_dir = Path(args.results_dir).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata bulunamadı: {metadata_path}")

    results_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(metadata_path)
    model = PPO.load(str(model_path), device=args.device)
    obs_version = core.infer_policy_observation_version(model)

    print("=" * 78)
    print("3DOF PPO ROBUSTNESS SUITE")
    print("=" * 78)
    print("Source dir   :", SOURCE_DIR)
    print("Stage        :", model_stage)
    print("Model        :", model_path)
    print("Metadata     :", metadata_path)
    print("Results      :", results_dir)
    print("Observation  :", obs_version, model.observation_space.shape)
    print("Hit radius   :", metadata.get("hit_radius_m"), "m")
    print(
        "Train elev.  :",
        metadata.get("target_sphere_launch_elevation_min_deg"),
        "..",
        metadata.get("target_sphere_launch_elevation_max_deg"),
        "deg",
    )
    print(
        "Train range  :",
        metadata.get("target_sphere_range_min_m"),
        "..",
        metadata.get("target_sphere_range_max_m"),
        "m",
    )
    print("Rand scale   :", metadata.get("randomization_scale"))
    print("Tests        :", ", ".join(args.tests))
    print("Quick mode   :", args.quick)

    if args.quick:
        training_episodes = 120
        large_mc_episodes = 240
        sweep_episodes = 120
        maneuver_episodes = 120
        progress_every = 60
    else:
        training_episodes = int(
            metadata.get("evaluation_episode_count", 600)
        )
        large_mc_episodes = 3000
        sweep_episodes = 360
        maneuver_episodes = 600
        progress_every = 100

    sector_count = int(
        metadata.get("evaluation_azimuth_sector_count", 12)
    )
    training_seed = int(
        metadata.get("evaluation_first_seed", 810000)
    )

    all_rows: list[dict[str, Any]] = []
    detailed_results: dict[str, Any] = {
        "model_stage": model_stage,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "observation_version": obs_version,
        "quick_mode": bool(args.quick),
        "tests": {},
    }

    if "training_domain" in args.tests:
        result, row = run_case(
            model=model,
            metadata=metadata,
            obs_version=obs_version,
            title="training_domain",
            case_name="metadata_domain",
            episodes=training_episodes,
            seed_start=training_seed,
            sector_count=sector_count,
            progress_every=progress_every,
        )
        detailed_results["tests"]["training_domain"] = result
        all_rows.append(row)
        save_json(results_dir / "training_domain.json", result)

    if "large_mc" in args.tests:
        result, row = run_case(
            model=model,
            metadata=metadata,
            obs_version=obs_version,
            title="large_mc",
            case_name="independent_mc",
            episodes=large_mc_episodes,
            seed_start=1_000_000,
            sector_count=12,
            progress_every=progress_every,
        )
        detailed_results["tests"]["large_mc"] = result
        all_rows.append(row)
        save_json(results_dir / "large_mc.json", result)

    if "elevation" in args.tests:
        rows = []
        detailed_cases = []
        for index, bounds in enumerate(ELEVATION_BANDS):
            result, row = run_case(
                model=model,
                metadata=metadata,
                obs_version=obs_version,
                title="elevation",
                case_name=f"{bounds[0]:g}_to_{bounds[1]:g}_deg",
                episodes=sweep_episodes,
                seed_start=1_100_000 + index * 10_000,
                sector_count=12,
                elevation_bounds=bounds,
                progress_every=progress_every,
                extra={
                    "elevation_min_deg": bounds[0],
                    "elevation_max_deg": bounds[1],
                },
            )
            rows.append(row)
            all_rows.append(row)
            detailed_cases.append(
                {
                    "elevation_bounds_deg": list(bounds),
                    "metrics": result,
                }
            )
        write_csv(results_dir / "elevation_sweep.csv", rows)
        detailed_results["tests"]["elevation"] = detailed_cases

    if "range" in args.tests:
        rows = []
        detailed_cases = []
        for index, range_m in enumerate(RANGES_M):
            result, row = run_case(
                model=model,
                metadata=metadata,
                obs_version=obs_version,
                title="range",
                case_name=f"{range_m:g}_m",
                episodes=sweep_episodes,
                seed_start=1_200_000 + index * 10_000,
                sector_count=12,
                range_bounds=(range_m, range_m),
                progress_every=progress_every,
                extra={"range_m": range_m},
            )
            rows.append(row)
            all_rows.append(row)
            detailed_cases.append(
                {"range_m": range_m, "metrics": result}
            )
        write_csv(results_dir / "range_sweep.csv", rows)
        detailed_results["tests"]["range"] = detailed_cases

    if "aim_error" in args.tests:
        rows = []
        detailed_cases = []
        for index, errors in enumerate(AIM_ERROR_CASES_DEG):
            az_error, el_error = errors
            result, row = run_case(
                model=model,
                metadata=metadata,
                obs_version=obs_version,
                title="aim_error",
                case_name=f"az{az_error:g}_el{el_error:g}_deg",
                episodes=sweep_episodes,
                seed_start=1_300_000 + index * 10_000,
                sector_count=12,
                aim_error_deg=errors,
                progress_every=progress_every,
                extra={
                    "aim_azimuth_error_deg": az_error,
                    "aim_elevation_error_deg": el_error,
                },
            )
            rows.append(row)
            all_rows.append(row)
            detailed_cases.append(
                {"aim_error_deg": list(errors), "metrics": result}
            )
        write_csv(results_dir / "aim_error_sweep.csv", rows)
        detailed_results["tests"]["aim_error"] = detailed_cases

    if "randomization" in args.tests:
        rows = []
        detailed_cases = []
        for index, scale in enumerate(RANDOMIZATION_SCALES):
            result, row = run_case(
                model=model,
                metadata=metadata,
                obs_version=obs_version,
                title="randomization",
                case_name=f"scale_{scale:g}",
                episodes=sweep_episodes,
                seed_start=1_400_000 + index * 10_000,
                sector_count=12,
                randomization_scale=scale,
                progress_every=progress_every,
                extra={"randomization_scale": scale},
            )
            rows.append(row)
            all_rows.append(row)
            detailed_cases.append(
                {"randomization_scale": scale, "metrics": result}
            )
        write_csv(results_dir / "randomization_sweep.csv", rows)
        detailed_results["tests"]["randomization"] = detailed_cases

    if "maneuver" in args.tests:
        rows = []
        detailed_cases = []
        for index, maneuver in enumerate(MANEUVERS):
            result, row = run_case(
                model=model,
                metadata=metadata,
                obs_version=obs_version,
                title="maneuver",
                case_name=maneuver,
                episodes=maneuver_episodes,
                seed_start=1_500_000 + index * 10_000,
                sector_count=12,
                maneuver=maneuver,
                progress_every=progress_every,
                extra={"maneuver": maneuver},
            )
            rows.append(row)
            all_rows.append(row)
            detailed_cases.append(
                {"maneuver": maneuver, "metrics": result}
            )
        write_csv(results_dir / "maneuver_sweep.csv", rows)
        detailed_results["tests"]["maneuver"] = detailed_cases

    if "constant_turn" in args.tests:
        rows = []
        detailed_cases = []
        for index, rate in enumerate(TURN_RATES_DEG_S):
            result, row = run_case(
                model=model,
                metadata=metadata,
                obs_version=obs_version,
                title="constant_turn",
                case_name=f"heading_{rate:g}_deg_s",
                episodes=sweep_episodes,
                seed_start=1_600_000 + index * 10_000,
                sector_count=12,
                maneuver="constant_turn",
                maneuver_rates=(rate, 0.0),
                progress_every=progress_every,
                extra={
                    "heading_rate_deg_s": rate,
                    "flight_path_rate_deg_s": 0.0,
                },
            )
            rows.append(row)
            all_rows.append(row)
            detailed_cases.append(
                {
                    "heading_rate_deg_s": rate,
                    "flight_path_rate_deg_s": 0.0,
                    "metrics": result,
                }
            )
        write_csv(results_dir / "constant_turn_sweep.csv", rows)
        detailed_results["tests"]["constant_turn"] = detailed_cases

    if "constant_turn_3d" in args.tests:
        rows = []
        detailed_cases = []
        for index, rates in enumerate(TURN_3D_CASES_DEG_S):
            heading_rate, flight_rate = rates
            result, row = run_case(
                model=model,
                metadata=metadata,
                obs_version=obs_version,
                title="constant_turn_3d",
                case_name=(
                    f"heading_{heading_rate:g}_"
                    f"flight_{flight_rate:g}_deg_s"
                ),
                episodes=sweep_episodes,
                seed_start=1_700_000 + index * 10_000,
                sector_count=12,
                maneuver="constant_turn",
                maneuver_rates=rates,
                progress_every=progress_every,
                extra={
                    "heading_rate_deg_s": heading_rate,
                    "flight_path_rate_deg_s": flight_rate,
                },
            )
            rows.append(row)
            all_rows.append(row)
            detailed_cases.append(
                {
                    "heading_rate_deg_s": heading_rate,
                    "flight_path_rate_deg_s": flight_rate,
                    "metrics": result,
                }
            )
        write_csv(results_dir / "constant_turn_3d.csv", rows)
        detailed_results["tests"]["constant_turn_3d"] = detailed_cases

    if "control_period" in args.tests:
        rows = []
        detailed_cases = []
        for index, period_s in enumerate(CONTROL_PERIODS_S):
            result, row = run_case(
                model=model,
                metadata=metadata,
                obs_version=obs_version,
                title="control_period",
                case_name=f"{period_s:g}_s",
                episodes=sweep_episodes,
                seed_start=1_800_000 + index * 10_000,
                sector_count=12,
                control_period_s=period_s,
                progress_every=progress_every,
                extra={
                    "control_period_s": period_s,
                    "control_frequency_hz": 1.0 / period_s,
                },
            )
            rows.append(row)
            all_rows.append(row)
            detailed_cases.append(
                {
                    "control_period_s": period_s,
                    "control_frequency_hz": 1.0 / period_s,
                    "metrics": result,
                }
            )
        write_csv(results_dir / "control_period_sweep.csv", rows)
        detailed_results["tests"]["control_period"] = detailed_cases

    if "action_noise" in args.tests:
        rows = []
        detailed_cases = []
        for index, noise_std in enumerate(ACTION_NOISE_STDS):
            result, row = run_case(
                model=model,
                metadata=metadata,
                obs_version=obs_version,
                title="action_noise",
                case_name=f"std_{noise_std:g}",
                episodes=sweep_episodes,
                seed_start=1_900_000 + index * 10_000,
                sector_count=12,
                action_noise_std=noise_std,
                progress_every=progress_every,
                extra={"action_noise_std": noise_std},
            )
            rows.append(row)
            all_rows.append(row)
            detailed_cases.append(
                {
                    "action_noise_std": noise_std,
                    "metrics": result,
                }
            )
        write_csv(results_dir / "action_noise_sweep.csv", rows)
        detailed_results["tests"]["action_noise"] = detailed_cases

    write_csv(results_dir / "all_test_cases.csv", all_rows)
    save_json(
        results_dir / "robustness_suite_summary.json",
        detailed_results,
    )

    print("\n" + "=" * 78)
    print("ROBUSTNESS SUITE TAMAMLANDI")
    print("=" * 78)
    print("Sonuç klasörü:")
    print(results_dir)
    print("\nAna CSV:")
    print(results_dir / "all_test_cases.csv")
    print("\nDetaylı JSON:")
    print(results_dir / "robustness_suite_summary.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
