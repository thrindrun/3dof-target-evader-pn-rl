from __future__ import annotations

"""Deterministic Monte Carlo evaluation for a trained evader policy."""

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from air_to_air_3dof_evader import (  # noqa: E402
    EVADER_ACTION_SIZE,
    EVADER_OBSERVATION_SIZE,
    EvaderEnvironmentConfig,
    build_default_evader_environment,
    plot_evader_episode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an evader PPO model against fixed Vector PN.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=2_000_000)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--randomization-scale", type=float, default=1.0)
    parser.add_argument("--pn-navigation-constant", type=float, default=4.0)
    parser.add_argument("--pn-max-command-g", type=float, default=30.0)
    parser.add_argument(
        "--missile-actuator-time-constant-s",
        type=float,
        default=0.0,
        help=(
            "First-order missile acceleration-response time constant. "
            "Use 0.0 for the original instantaneous missile."
        ),
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def resolve_model_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = SOURCE_ROOT / path
    candidates = [path]
    if path.suffix.lower() != ".zip":
        candidates.append(path.with_suffix(".zip"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError("Model not found:\n" + checked)


def apply_randomization_scale(
    config: EvaderEnvironmentConfig,
    scale: float,
) -> None:
    if scale < 0.0:
        raise ValueError("randomization-scale cannot be negative.")
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


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.missile_actuator_time_constant_s < 0.0:
        raise ValueError(
            "--missile-actuator-time-constant-s cannot be negative."
        )

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError("Install stable-baselines3 and torch first.") from exc

    model_path = resolve_model_path(args.model)
    model = PPO.load(str(model_path), device=args.device)
    if tuple(model.observation_space.shape) != (EVADER_OBSERVATION_SIZE,):
        raise ValueError(
            "This is not an evader_v1 model: expected observation shape "
            f"{(EVADER_OBSERVATION_SIZE,)}, got {model.observation_space.shape}."
        )
    if tuple(model.action_space.shape) != (EVADER_ACTION_SIZE,):
        raise ValueError(
            f"Expected action shape {(EVADER_ACTION_SIZE,)}, got "
            f"{model.action_space.shape}."
        )

    if not args.randomize and args.episodes > 1 and not args.stochastic:
        print(
            "Note: fixed geometry + deterministic policy produces the same "
            "episode for every seed. One episode is sufficient for E1.\n"
        )

    config = EvaderEnvironmentConfig(
        pn_navigation_constant=args.pn_navigation_constant,
        pn_max_command_g=args.pn_max_command_g,
        missile_actuator_time_constant_s=(
            args.missile_actuator_time_constant_s
        ),
        randomize_initial_conditions=args.randomize,
        record_history=args.plot,
    )
    apply_randomization_scale(config, args.randomization_scale)
    environment = build_default_evader_environment(
        config=config,
        hit_radius_m=args.radius,
    )

    rows: list[dict[str, Any]] = []
    plotted_result = None
    try:
        for episode_index in range(args.episodes):
            episode_seed = args.first_seed + episode_index
            observation, _ = environment.reset(seed=episode_seed)
            terminated = False
            truncated = False
            episode_reward = 0.0
            squared_action_sum = 0.0
            action_count = 0
            info: dict[str, Any] = {}

            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                squared_action_sum += float(np.mean(np.asarray(action) ** 2))
                action_count += 1
                observation, reward, terminated, truncated, info = (
                    environment.step(action)
                )
                episode_reward += float(reward)

            row = {
                "episode": episode_index,
                "seed": episode_seed,
                "status": str(info["status"]),
                "survived": bool(info["survived"]),
                "minimum_range_m": float(info["minimum_range_m"]),
                "terminal_time_s": float(info["time_s"]),
                "episode_reward": float(episode_reward),
                "mean_squared_action": (
                    squared_action_sum / max(action_count, 1)
                ),
            }
            rows.append(row)
            if args.plot and episode_index == 0:
                plotted_result = environment.episode_result()

            if (
                episode_index == 0
                or (episode_index + 1) % 100 == 0
                or episode_index + 1 == args.episodes
            ):
                print(f"Evaluated {episode_index + 1}/{args.episodes}")
    finally:
        environment.close()

    minimum_ranges = np.asarray(
        [row["minimum_range_m"] for row in rows],
        dtype=float,
    )
    rewards = np.asarray([row["episode_reward"] for row in rows], dtype=float)
    survival_flags = np.asarray([row["survived"] for row in rows], dtype=bool)
    status_counts = Counter(row["status"] for row in rows)
    summary = {
        "model": str(model_path),
        "episodes": args.episodes,
        "first_seed": args.first_seed,
        "last_seed": args.first_seed + args.episodes - 1,
        "deterministic": not args.stochastic,
        "randomize_initial_conditions": args.randomize,
        "randomization_scale": args.randomization_scale,
        "hit_radius_m": args.radius,
        "pn_navigation_constant": args.pn_navigation_constant,
        "pn_max_command_g": args.pn_max_command_g,
        "missile_actuator_time_constant_s": (
            args.missile_actuator_time_constant_s
        ),
        "survival_rate": float(np.mean(survival_flags)),
        "intercept_rate": float(
            np.mean([row["status"] == "intercepted" for row in rows])
        ),
        "minimum_range_statistics_m": {
            "minimum": float(np.min(minimum_ranges)),
            "mean": float(np.mean(minimum_ranges)),
            "median": float(np.median(minimum_ranges)),
            "p10": percentile(minimum_ranges, 10),
            "p90": percentile(minimum_ranges, 90),
            "maximum": float(np.max(minimum_ranges)),
        },
        "mean_episode_reward": float(np.mean(rewards)),
        "status_counts": dict(status_counts),
        "environment_config": asdict(config),
    }

    print("\n" + "=" * 72)
    print("EVADER EVALUATION")
    print("=" * 72)
    print("Model            :", model_path)
    print("Episodes         :", args.episodes)
    print(
        "PN configuration :",
        f"N={args.pn_navigation_constant:g}, "
        f"max={args.pn_max_command_g:g} g",
    )
    print(
        "Missile lag tau  :",
        f"{args.missile_actuator_time_constant_s:g} s",
    )
    print("Survival rate    :", f"{100.0 * summary['survival_rate']:.2f}%")
    print("Intercept rate   :", f"{100.0 * summary['intercept_rate']:.2f}%")
    print("Mean min range   :", f"{np.mean(minimum_ranges):.3f} m")
    print("Median min range :", f"{np.median(minimum_ranges):.3f} m")
    print("P10 min range    :", f"{percentile(minimum_ranges, 10):.3f} m")
    print("Mean reward      :", f"{np.mean(rewards):.3f}")
    print("Statuses         :", dict(status_counts))
    print("=" * 72)

    if args.save_dir is not None:
        save_directory = Path(args.save_dir).expanduser()
        if not save_directory.is_absolute():
            save_directory = SOURCE_ROOT / save_directory
        save_directory.mkdir(parents=True, exist_ok=True)

        csv_path = save_directory / "evader_episodes.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        summary_path = save_directory / "evader_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("Saved CSV        :", csv_path)
        print("Saved summary    :", summary_path)

    if plotted_result is not None:
        _ = plot_evader_episode(plotted_result, show=True)


if __name__ == "__main__":
    main()
