r"""
PPO modelini geniş başlangıç geometrilerinde ve açı sektörlerinde değerlendirir.

Bu araç eğitim yapmaz. ``rotated_nominal`` modu aynı karşılaşmayı global N/E
düzleminde döndürerek koordinat invariance'ını, ``polar`` hedefi füzenin
çevresinde örneklemeyi, ``target_sphere`` ise hedef sabit merkezdeyken füzeyi
hedef-merkezli 3B kabukta örneklemeyi ölçer.

Örnek tam çember testi (PowerShell):

    python .\tests\evaluate_ppo_geometry.py `
        --model .\models\obs_v2_random50_9m_1m\best_model.zip `
        --episodes 360 --geometry-mode polar `
        --bearing-min-deg -180 --bearing-max-deg 180
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from air_to_air_3dof_ppo import (  # noqa: E402
    PPO_GEOMETRY_MODES,
    PPO_GEOMETRY_POLAR,
    PPO_GEOMETRY_ROTATED_NOMINAL,
    PPO_GEOMETRY_TARGET_SPHERE,
    PPOEnvironmentConfig,
    RLInterfaceConfig,
    build_default_ppo_environment,
    infer_policy_observation_version,
    rl_observation_size,
)


DEFAULT_THRESHOLDS_M = [
    1_000.0,
    500.0,
    200.0,
    100.0,
    50.0,
    25.0,
    10.0,
]


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Değer sıfırdan büyük olmalıdır.")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("Değer sonlu ve sıfırdan büyük olmalıdır.")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("Değer sonlu ve negatif olmayan olmalıdır.")
    return number


def finite_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise argparse.ArgumentTypeError("Değer sonlu olmalıdır.")
    return number


def probability_float(value: str) -> float:
    number = finite_float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("Olasılık [0, 1] içinde olmalıdır.")
    return number


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO modelini 5 km geniş başlangıç geometrisinde test et",
    )
    parser.add_argument("--model", required=True, help="PPO .zip model yolu")
    parser.add_argument("--episodes", type=positive_int, default=360)
    parser.add_argument("--seed", type=int, default=300_000)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=positive_float,
        default=DEFAULT_THRESHOLDS_M,
        help="Başarı oranı hesaplanacak minimum-menzil eşikleri [m]",
    )
    parser.add_argument(
        "--maneuver",
        choices=("straight", "constant_turn", "sinusoidal", "random_smooth"),
        default="straight",
    )
    parser.add_argument(
        "--randomization-scale",
        type=nonnegative_float,
        default=0.50,
        help="Eski konum/hız/açı jitter'larının çarpanı. Varsayılan: 0.50",
    )
    parser.add_argument(
        "--geometry-mode",
        choices=PPO_GEOMETRY_MODES,
        default=PPO_GEOMETRY_POLAR,
    )
    parser.add_argument(
        "--global-rotation-limit-deg",
        type=nonnegative_float,
        default=180.0,
    )
    parser.add_argument("--range-min-m", type=positive_float, default=4_800.0)
    parser.add_argument("--range-max-m", type=positive_float, default=5_200.0)
    parser.add_argument("--bearing-min-deg", type=finite_float, default=-180.0)
    parser.add_argument("--bearing-max-deg", type=finite_float, default=180.0)
    parser.add_argument("--elevation-min-deg", type=finite_float, default=1.0)
    parser.add_argument("--elevation-max-deg", type=finite_float, default=6.0)
    parser.add_argument(
        "--target-aspect-min-deg",
        type=finite_float,
        default=-10.0,
    )
    parser.add_argument(
        "--target-aspect-max-deg",
        type=finite_float,
        default=10.0,
    )
    parser.add_argument(
        "--launch-azimuth-min-deg",
        type=finite_float,
        default=-180.0,
    )
    parser.add_argument(
        "--launch-azimuth-max-deg",
        type=finite_float,
        default=180.0,
    )
    parser.add_argument(
        "--launch-elevation-min-deg",
        type=finite_float,
        default=-90.0,
    )
    parser.add_argument(
        "--launch-elevation-max-deg",
        type=finite_float,
        default=90.0,
    )
    parser.add_argument(
        "--aim-azimuth-error-deg",
        type=nonnegative_float,
        default=0.0,
    )
    parser.add_argument(
        "--aim-elevation-error-deg",
        type=nonnegative_float,
        default=0.0,
    )
    parser.add_argument(
        "--sphere-ground-clearance-m",
        type=nonnegative_float,
        default=100.0,
    )
    parser.add_argument(
        "--sphere-auto-raise-target",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--legacy-replay-probability",
        type=probability_float,
        default=0.0,
        help="Saf geometri testinde 0 bırakın.",
    )
    parser.add_argument(
        "--passed-target-arm-progress-m",
        type=nonnegative_float,
        default=100.0,
        help="İlk açılan/arkadaki hedeflerin erken bitmemesi için varsayılan 100 m",
    )
    parser.add_argument(
        "--sector-count",
        type=positive_int,
        default=12,
        help="Açı tablosundaki eşit sektör sayısı. Varsayılan: 12",
    )
    parser.add_argument(
        "--sector-threshold-m",
        type=positive_float,
        default=200.0,
        help="Sektör tablosunda raporlanacak başarı eşiği [m]",
    )
    parser.add_argument(
        "--sector-basis",
        choices=("auto", "global", "relative", "launch"),
        default="auto",
        help=(
            "Sektör açısı: global LOS, füze burnuna göre relative bearing veya "
            "hedef->füze launch azimutu"
        ),
    )
    parser.add_argument(
        "--elevation-band-count",
        type=positive_int,
        default=6,
        help="target_sphere eşit-alan elevation bant sayısı. Varsayılan: 6",
    )
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def resolve_model_path(model_argument: str) -> Path:
    model_path = Path(model_argument)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    candidates = [model_path]
    if model_path.suffix.lower() != ".zip":
        candidates.append(model_path.with_suffix(".zip"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Test edilecek model bulunamadı. Kontrol edilen yollar:\n" + checked
    )


def safe_file_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "model"


def apply_randomization_scale(
    config: PPOEnvironmentConfig,
    scale: float,
) -> None:
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


def wrap_angle_deg(angle_deg: float) -> float:
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def sector_index(angle_deg: float, sector_count: int) -> int:
    normalized = wrap_angle_deg(angle_deg)
    width_deg = 360.0 / sector_count
    return min(int((normalized + 180.0) // width_deg), sector_count - 1)


def sector_label(index: int, sector_count: int) -> str:
    width_deg = 360.0 / sector_count
    lower = -180.0 + index * width_deg
    upper = lower + width_deg
    return f"[{lower:g}, {upper:g})"


def elevation_band_index(elevation_deg: float, band_count: int) -> int:
    """Küre yüzeyini sin(elevation) uzayında eşit alanlı bantlara ayırır."""
    normalized = (np.sin(np.radians(elevation_deg)) + 1.0) / 2.0
    return min(max(int(normalized * band_count), 0), band_count - 1)


def elevation_band_label(index: int, band_count: int) -> str:
    lower_sine = -1.0 + 2.0 * index / band_count
    upper_sine = -1.0 + 2.0 * (index + 1) / band_count
    lower_deg = float(np.degrees(np.arcsin(np.clip(lower_sine, -1.0, 1.0))))
    upper_deg = float(np.degrees(np.arcsin(np.clip(upper_sine, -1.0, 1.0))))
    return f"[{lower_deg:.1f}, {upper_deg:.1f})"


def validate_ranges(args: argparse.Namespace) -> None:
    pairs = (
        (args.range_min_m, args.range_max_m, "range"),
        (args.bearing_min_deg, args.bearing_max_deg, "bearing"),
        (args.elevation_min_deg, args.elevation_max_deg, "elevation"),
        (
            args.target_aspect_min_deg,
            args.target_aspect_max_deg,
            "target-aspect",
        ),
        (
            args.launch_azimuth_min_deg,
            args.launch_azimuth_max_deg,
            "launch-azimuth",
        ),
        (
            args.launch_elevation_min_deg,
            args.launch_elevation_max_deg,
            "launch-elevation",
        ),
    )
    for minimum, maximum, label in pairs:
        if maximum < minimum:
            raise ValueError(f"--{label}-max, --{label}-min'den küçük olamaz.")
    if not (
        -90.0 <= args.launch_elevation_min_deg <= 90.0
        and -90.0 <= args.launch_elevation_max_deg <= 90.0
    ):
        raise ValueError("Launch elevation sınırları [-90, 90] içinde olmalıdır.")


def main() -> None:
    args = parse_arguments()
    validate_ranges(args)
    model_path = resolve_model_path(args.model)
    deterministic = not args.stochastic
    thresholds_m = sorted({float(value) for value in args.thresholds}, reverse=True)

    model = PPO.load(str(model_path), device="cpu")
    observation_version = infer_policy_observation_version(model)

    environment_config = PPOEnvironmentConfig(
        target_maneuver_kind=args.maneuver,
        randomize_initial_conditions=True,
        control_period_s=0.10,
        initial_geometry_mode=args.geometry_mode,
        global_rotation_limit_deg=args.global_rotation_limit_deg,
        polar_range_min_m=args.range_min_m,
        polar_range_max_m=args.range_max_m,
        polar_bearing_min_deg=args.bearing_min_deg,
        polar_bearing_max_deg=args.bearing_max_deg,
        polar_elevation_min_deg=args.elevation_min_deg,
        polar_elevation_max_deg=args.elevation_max_deg,
        polar_target_aspect_min_deg=args.target_aspect_min_deg,
        polar_target_aspect_max_deg=args.target_aspect_max_deg,
        target_sphere_range_min_m=args.range_min_m,
        target_sphere_range_max_m=args.range_max_m,
        target_sphere_launch_azimuth_min_deg=args.launch_azimuth_min_deg,
        target_sphere_launch_azimuth_max_deg=args.launch_azimuth_max_deg,
        target_sphere_launch_elevation_min_deg=args.launch_elevation_min_deg,
        target_sphere_launch_elevation_max_deg=args.launch_elevation_max_deg,
        target_sphere_aim_azimuth_error_deg=args.aim_azimuth_error_deg,
        target_sphere_aim_elevation_error_deg=args.aim_elevation_error_deg,
        target_sphere_ground_clearance_m=args.sphere_ground_clearance_m,
        target_sphere_auto_raise_target=args.sphere_auto_raise_target,
        legacy_replay_probability=args.legacy_replay_probability,
        passed_target_arm_progress_m=args.passed_target_arm_progress_m,
    )
    apply_randomization_scale(environment_config, args.randomization_scale)

    environment = build_default_ppo_environment(
        ppo_environment_config=environment_config,
        rl_interface_config=RLInterfaceConfig(
            observation_version=observation_version,
        ),
    )
    environment.environment_template.hit_radius_m = 1e-6

    if args.sector_basis == "auto":
        if args.geometry_mode == PPO_GEOMETRY_TARGET_SPHERE:
            sector_basis = "launch"
        elif args.geometry_mode == PPO_GEOMETRY_ROTATED_NOMINAL:
            sector_basis = "global"
        else:
            sector_basis = "relative"
    else:
        sector_basis = args.sector_basis

    print()
    print("=" * 88)
    print("PPO GENİŞ-GEOMETRİ TESTİ")
    print("=" * 88)
    print("Model                 :", model_path)
    print(
        "Observation           :",
        observation_version,
        f"({rl_observation_size(observation_version)} girdi)",
    )
    print("Episode / seed aralığı:", args.episodes, "/", end=" ")
    print(f"{args.seed}-{args.seed + args.episodes - 1}")
    print("Geometri modu         :", args.geometry_mode)
    if args.geometry_mode == PPO_GEOMETRY_ROTATED_NOMINAL:
        print("Global dönüş sınırı   :", args.global_rotation_limit_deg, "deg")
    elif args.geometry_mode == PPO_GEOMETRY_POLAR:
        print(
            "Menzil aralığı        :",
            f"{args.range_min_m:g}..{args.range_max_m:g} m",
        )
        print(
            "Bağıl bearing aralığı :",
            f"{args.bearing_min_deg:g}..{args.bearing_max_deg:g} deg",
        )
        print(
            "Elevation / aspect    :",
            f"{args.elevation_min_deg:g}..{args.elevation_max_deg:g} /",
            f"{args.target_aspect_min_deg:g}..{args.target_aspect_max_deg:g} deg",
        )
    elif args.geometry_mode == PPO_GEOMETRY_TARGET_SPHERE:
        print(
            "Küre menzil aralığı   :",
            f"{args.range_min_m:g}..{args.range_max_m:g} m",
        )
        print(
            "Launch azimut aralığı :",
            f"{args.launch_azimuth_min_deg:g}.."
            f"{args.launch_azimuth_max_deg:g} deg",
        )
        print(
            "Launch elevation      :",
            f"{args.launch_elevation_min_deg:g}.."
            f"{args.launch_elevation_max_deg:g} deg",
        )
        print(
            "Başlangıç nişan hatası:",
            f"az ±{args.aim_azimuth_error_deg:g} / "
            f"el ±{args.aim_elevation_error_deg:g} deg",
        )
        print("Küre yer payı         :", args.sphere_ground_clearance_m, "m")
        print("Hedef irtifa düzeltme :", args.sphere_auto_raise_target)
    print("Hedef / jitter ölçeği :", args.maneuver, "/", args.randomization_scale)
    print("Passed-target arm      :", args.passed_target_arm_progress_m, "m")
    print("Sektör temeli / eşiği :", sector_basis, "/", args.sector_threshold_m, "m")
    print("Politika modu          :", "deterministic" if deterministic else "stochastic")
    print("=" * 88)

    rows: list[dict[str, object]] = []
    for episode_index in range(args.episodes):
        episode_seed = args.seed + episode_index
        observation, initial_info = environment.reset(seed=episode_seed)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, final_info = environment.step(
                action
            )
            total_reward += float(reward)

        sector_field = {
            "global": "initial_los_azimuth_deg",
            "relative": "initial_relative_bearing_deg",
            "launch": "initial_launch_azimuth_deg",
        }[sector_basis]
        sector_angle_deg = float(initial_info[sector_field])
        index = sector_index(sector_angle_deg, args.sector_count)
        elevation_index = elevation_band_index(
            float(initial_info["initial_launch_elevation_deg"]),
            args.elevation_band_count,
        )
        minimum_range_m = float(final_info["minimum_range_m"])
        missile_position = initial_info["initial_missile_position_m"]
        target_position = initial_info["initial_target_position_m"]
        row: dict[str, object] = {
            "episode": episode_index + 1,
            "seed": episode_seed,
            "geometry_source": str(initial_info["geometry_source"]),
            "initial_missile_north_m": float(missile_position[0]),
            "initial_missile_east_m": float(missile_position[1]),
            "initial_missile_down_m": float(missile_position[2]),
            "initial_target_north_m": float(target_position[0]),
            "initial_target_east_m": float(target_position[1]),
            "initial_target_down_m": float(target_position[2]),
            "initial_missile_speed_mps": float(
                initial_info["initial_missile_speed_mps"]
            ),
            "initial_target_speed_mps": float(
                initial_info["initial_target_speed_mps"]
            ),
            "initial_missile_heading_deg": float(
                initial_info["initial_missile_heading_deg"]
            ),
            "initial_missile_flight_path_deg": float(
                initial_info["initial_missile_flight_path_deg"]
            ),
            "initial_target_heading_deg": float(
                initial_info["initial_target_heading_deg"]
            ),
            "initial_range_m": float(initial_info["initial_range_m"]),
            "initial_closing_speed_mps": float(
                initial_info["initial_closing_speed_mps"]
            ),
            "initial_los_azimuth_deg": float(
                initial_info["initial_los_azimuth_deg"]
            ),
            "initial_relative_bearing_deg": float(
                initial_info["initial_relative_bearing_deg"]
            ),
            "initial_los_elevation_deg": float(
                initial_info["initial_los_elevation_deg"]
            ),
            "initial_target_aspect_deg": float(
                initial_info["initial_target_aspect_deg"]
            ),
            "initial_launch_azimuth_deg": float(
                initial_info["initial_launch_azimuth_deg"]
            ),
            "initial_launch_elevation_deg": float(
                initial_info["initial_launch_elevation_deg"]
            ),
            "target_sphere_altitude_shift_m": float(
                initial_info["target_sphere_altitude_shift_m"]
            ),
            "sector_index": index,
            "sector": sector_label(index, args.sector_count),
            "elevation_band_index": elevation_index,
            "elevation_band": elevation_band_label(
                elevation_index,
                args.elevation_band_count,
            ),
            "status": str(final_info["status"]),
            "minimum_range_m": minimum_range_m,
            "terminal_time_s": float(final_info["terminal_time_s"]),
            "total_reward": total_reward,
        }
        for threshold_m in thresholds_m:
            row[f"hit_{threshold_m:g}m"] = int(minimum_range_m <= threshold_m)
        rows.append(row)

        completed = episode_index + 1
        progress_interval = max(args.episodes // 20, 1)
        if completed % progress_interval == 0 or completed == args.episodes:
            print(f"Tamamlanan episode: {completed}/{args.episodes}")

    minimum_ranges_m = np.asarray(
        [float(row["minimum_range_m"]) for row in rows],
        dtype=float,
    )
    terminal_times_s = np.asarray(
        [float(row["terminal_time_s"]) for row in rows],
        dtype=float,
    )
    status_counts = Counter(str(row["status"]) for row in rows)
    success_rates = {
        f"{threshold_m:g}m": float(
            100.0 * np.mean(minimum_ranges_m <= threshold_m)
        )
        for threshold_m in thresholds_m
    }

    sector_summaries: list[dict[str, object]] = []
    for index in range(args.sector_count):
        values = np.asarray(
            [
                float(row["minimum_range_m"])
                for row in rows
                if int(row["sector_index"]) == index
            ],
            dtype=float,
        )
        if values.size == 0:
            continue
        sector_summaries.append(
            {
                "sector_index": index,
                "sector_deg": sector_label(index, args.sector_count),
                "episode_count": int(values.size),
                "success_rate_percent": float(
                    100.0 * np.mean(values <= args.sector_threshold_m)
                ),
                "mean_minimum_range_m": float(np.mean(values)),
                "p90_minimum_range_m": float(np.percentile(values, 90.0)),
                "maximum_minimum_range_m": float(np.max(values)),
            }
        )

    elevation_band_summaries: list[dict[str, object]] = []
    for index in range(args.elevation_band_count):
        values = np.asarray(
            [
                float(row["minimum_range_m"])
                for row in rows
                if int(row["elevation_band_index"]) == index
            ],
            dtype=float,
        )
        if values.size == 0:
            continue
        elevation_band_summaries.append(
            {
                "band_index": index,
                "elevation_deg": elevation_band_label(
                    index,
                    args.elevation_band_count,
                ),
                "episode_count": int(values.size),
                "success_rate_percent": float(
                    100.0 * np.mean(values <= args.sector_threshold_m)
                ),
                "mean_minimum_range_m": float(np.mean(values)),
                "p90_minimum_range_m": float(np.percentile(values, 90.0)),
                "maximum_minimum_range_m": float(np.max(values)),
            }
        )

    statistics = {
        "minimum": float(np.min(minimum_ranges_m)),
        "mean": float(np.mean(minimum_ranges_m)),
        "median": float(np.median(minimum_ranges_m)),
        "p90": float(np.percentile(minimum_ranges_m, 90.0)),
        "p95": float(np.percentile(minimum_ranges_m, 95.0)),
        "maximum": float(np.max(minimum_ranges_m)),
    }
    summary = {
        "model": str(model_path),
        "observation_version": observation_version,
        "observation_size": rl_observation_size(observation_version),
        "episodes": args.episodes,
        "first_seed": args.seed,
        "last_seed": args.seed + args.episodes - 1,
        "maneuver": args.maneuver,
        "deterministic": deterministic,
        "randomization_scale": args.randomization_scale,
        "geometry": {
            "mode": args.geometry_mode,
            "global_rotation_limit_deg": args.global_rotation_limit_deg,
            "range_min_m": args.range_min_m,
            "range_max_m": args.range_max_m,
            "bearing_min_deg": args.bearing_min_deg,
            "bearing_max_deg": args.bearing_max_deg,
            "elevation_min_deg": args.elevation_min_deg,
            "elevation_max_deg": args.elevation_max_deg,
            "target_aspect_min_deg": args.target_aspect_min_deg,
            "target_aspect_max_deg": args.target_aspect_max_deg,
            "launch_azimuth_min_deg": args.launch_azimuth_min_deg,
            "launch_azimuth_max_deg": args.launch_azimuth_max_deg,
            "launch_elevation_min_deg": args.launch_elevation_min_deg,
            "launch_elevation_max_deg": args.launch_elevation_max_deg,
            "aim_azimuth_error_deg": args.aim_azimuth_error_deg,
            "aim_elevation_error_deg": args.aim_elevation_error_deg,
            "sphere_ground_clearance_m": args.sphere_ground_clearance_m,
            "sphere_auto_raise_target": args.sphere_auto_raise_target,
            "legacy_replay_probability": args.legacy_replay_probability,
            "passed_target_arm_progress_m": args.passed_target_arm_progress_m,
        },
        "minimum_range_statistics_m": statistics,
        "mean_terminal_time_s": float(np.mean(terminal_times_s)),
        "status_counts": dict(status_counts),
        "success_rates_percent": success_rates,
        "sector_basis": sector_basis,
        "sector_threshold_m": args.sector_threshold_m,
        "sector_summaries": sector_summaries,
        "elevation_band_count": args.elevation_band_count,
        "elevation_band_summaries": elevation_band_summaries,
    }

    output_directory = PROJECT_ROOT / "results" / "geometry"
    output_directory.mkdir(parents=True, exist_ok=True)
    model_tag = safe_file_part(model_path.parent.name + "_" + model_path.stem)
    mode_tag = "deterministic" if deterministic else "stochastic"
    if args.geometry_mode == PPO_GEOMETRY_TARGET_SPHERE:
        geometry_descriptor = (
            f"{args.geometry_mode}_az{args.launch_azimuth_min_deg:g}to"
            f"{args.launch_azimuth_max_deg:g}_el"
            f"{args.launch_elevation_min_deg:g}to"
            f"{args.launch_elevation_max_deg:g}"
        )
    else:
        geometry_descriptor = (
            f"{args.geometry_mode}_b{args.bearing_min_deg:g}to"
            f"{args.bearing_max_deg:g}"
        )
    geometry_tag = safe_file_part(geometry_descriptor)
    output_stem = (
        f"{model_tag}_{geometry_tag}_{args.episodes}eps_"
        f"seed{args.seed}_{mode_tag}"
    )
    csv_path = output_directory / f"{output_stem}.csv"
    json_path = output_directory / f"{output_stem}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("GENİŞ-GEOMETRİ TEST SONUCU")
    print("=" * 88)
    for threshold_m in thresholds_m:
        label = f"{threshold_m:g}m"
        hits = int(np.count_nonzero(minimum_ranges_m <= threshold_m))
        print(
            f"{threshold_m:8g} m başarı : {hits:4d}/{args.episodes} "
            f"(%{success_rates[label]:6.2f})"
        )
    print("-" * 88)
    print("Minimum / ortalama [m]:", f"{statistics['minimum']:.3f} / {statistics['mean']:.3f}")
    print("Medyan / P90 / P95 [m]:", end=" ")
    print(
        f"{statistics['median']:.3f} / {statistics['p90']:.3f} / "
        f"{statistics['p95']:.3f}"
    )
    print("Maksimum menzil [m]   :", f"{statistics['maximum']:.3f}")
    print("Durum sayıları        :", dict(status_counts))
    print("-" * 88)
    print(
        f"SEKTÖRLER — {sector_basis} açı — {args.sector_threshold_m:g} m başarı"
    )
    print(
        f"{'Sektör [deg]':>18} {'N':>5} {'Başarı':>10} "
        f"{'Ort.[m]':>10} {'P90[m]':>10} {'Maks.[m]':>10}"
    )
    for item in sector_summaries:
        print(
            f"{str(item['sector_deg']):>18} "
            f"{int(item['episode_count']):5d} "
            f"%{float(item['success_rate_percent']):8.2f} "
            f"{float(item['mean_minimum_range_m']):10.2f} "
            f"{float(item['p90_minimum_range_m']):10.2f} "
            f"{float(item['maximum_minimum_range_m']):10.2f}"
        )
    if args.geometry_mode == PPO_GEOMETRY_TARGET_SPHERE:
        print("-" * 88)
        print(
            f"EŞİT-ALAN ELEVATION BANTLARI — {args.sector_threshold_m:g} m başarı"
        )
        print(
            f"{'Elevation [deg]':>18} {'N':>5} {'Başarı':>10} "
            f"{'Ort.[m]':>10} {'P90[m]':>10} {'Maks.[m]':>10}"
        )
        for item in elevation_band_summaries:
            print(
                f"{str(item['elevation_deg']):>18} "
                f"{int(item['episode_count']):5d} "
                f"%{float(item['success_rate_percent']):8.2f} "
                f"{float(item['mean_minimum_range_m']):10.2f} "
                f"{float(item['p90_minimum_range_m']):10.2f} "
                f"{float(item['maximum_minimum_range_m']):10.2f}"
            )
    print("CSV episode verisi    :", csv_path)
    print("JSON özet              :", json_path)
    print("=" * 88)

    environment.close()


if __name__ == "__main__":
    main()
