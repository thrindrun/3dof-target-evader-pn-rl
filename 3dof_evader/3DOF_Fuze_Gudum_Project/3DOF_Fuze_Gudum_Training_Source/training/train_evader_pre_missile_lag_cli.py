from __future__ import annotations

"""Train a PPO target aircraft against the project's fixed Vector-PN missile.

This script is the single training entry point for both VS Code and Colab.
The notebook should construct a command and call this file; it should not
contain another copy of the training implementation.
"""

import argparse
import csv
import json
import multiprocessing as mp
import sys
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
    EVADER_OBSERVATION_VERSION,
    EvaderEnvironmentConfig,
    build_default_evader_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RL aircraft evader against fixed Vector PN.",
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--allow-existing", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--action-std", type=float, default=0.35)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.20)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-envs", type=int, default=4)

    parser.add_argument("--control-period-s", type=float, default=0.10)
    parser.add_argument("--pn-navigation-constant", type=float, default=4.0)
    parser.add_argument(
        "--pn-max-command-g",
        type=float,
        default=5.0,
        help="E1 starts at 5 g; later continuation stages increase this value.",
    )
    parser.add_argument("--max-heading-rate-deg-s", type=float, default=16.0)
    parser.add_argument(
        "--max-flight-path-rate-deg-s",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--actuator-time-constant-s",
        type=float,
        default=0.25,
    )

    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--randomization-scale", type=float, default=1.0)

    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=1_000_000)
    parser.add_argument("--eval-frequency", type=int, default=10_000)
    parser.add_argument("--checkpoint-frequency", type=int, default=50_000)
    return parser.parse_args()


def resolve_path(path_text: str | None) -> Path | None:
    if path_text is None:
        return None
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
    raise FileNotFoundError("Source evader model not found:\n" + checked)


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


def build_environment(
    config: EvaderEnvironmentConfig,
    hit_radius_m: float,
):
    return build_default_evader_environment(
        config=config,
        hit_radius_m=hit_radius_m,
    )


def evaluate_on_fixed_seeds(
    model,
    config: EvaderEnvironmentConfig,
    hit_radius_m: float,
    seeds: list[int],
) -> dict[str, Any]:
    evaluation_config = EvaderEnvironmentConfig(**asdict(config))
    evaluation_config.record_history = False
    environment = build_default_evader_environment(
        config=evaluation_config,
        hit_radius_m=hit_radius_m,
    )

    episode_rows: list[dict[str, Any]] = []
    try:
        for episode_seed in seeds:
            observation, _ = environment.reset(seed=episode_seed)
            terminated = False
            truncated = False
            total_reward = 0.0
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = (
                    environment.step(action)
                )
                total_reward += float(reward)
            episode_rows.append(
                {
                    "seed": int(episode_seed),
                    "status": str(info["status"]),
                    "survived": bool(info["survived"]),
                    "minimum_range_m": float(info["minimum_range_m"]),
                    "terminal_time_s": float(info["time_s"]),
                    "episode_reward": float(total_reward),
                }
            )
    finally:
        environment.close()

    minimum_ranges = np.asarray(
        [row["minimum_range_m"] for row in episode_rows],
        dtype=float,
    )
    rewards = np.asarray(
        [row["episode_reward"] for row in episode_rows],
        dtype=float,
    )
    survival_rate = float(
        np.mean([row["survived"] for row in episode_rows])
    )
    return {
        "episodes": len(episode_rows),
        "survival_rate": survival_rate,
        "mean_minimum_range_m": float(np.mean(minimum_ranges)),
        "median_minimum_range_m": float(np.median(minimum_ranges)),
        "p10_minimum_range_m": float(np.percentile(minimum_ranges, 10)),
        "mean_episode_reward": float(np.mean(rewards)),
        "rows": episode_rows,
    }


def score_evaluation(metrics: dict[str, Any]) -> tuple[float, float, float]:
    """Lexicographic score: survival first, then miss distance, then reward."""

    return (
        float(metrics["survival_rate"]),
        float(metrics["mean_minimum_range_m"]),
        float(metrics["mean_episode_reward"]),
    )


def main() -> None:
    args = parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive.")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive.")
    if args.eval_frequency <= 0 or args.checkpoint_frequency <= 0:
        raise ValueError("Evaluation/checkpoint frequencies must be positive.")
    if not (0.0 < args.action_std <= 2.0):
        raise ValueError("--action-std must be in (0, 2].")
    if not args.stage or any(character in args.stage for character in "/\\"):
        raise ValueError("--stage must be one directory-safe name.")

    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.utils import get_schedule_fn
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError as exc:
        raise ImportError(
            "Install gymnasium, stable-baselines3, torch and tensorboard first."
        ) from exc

    output_root = (
        Path(args.output_root).expanduser()
        if args.output_root is not None
        else SOURCE_ROOT / "models" / "evader"
    )
    if not output_root.is_absolute():
        output_root = SOURCE_ROOT / output_root
    stage_directory = output_root / args.stage
    if stage_directory.exists() and any(stage_directory.iterdir()):
        if not args.allow_existing:
            raise FileExistsError(
                f"Stage directory is not empty: {stage_directory}\n"
                "Use a new stage name or pass --allow-existing deliberately."
            )
    stage_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = stage_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    tensorboard_directory = stage_directory / "tensorboard"
    tensorboard_directory.mkdir(parents=True, exist_ok=True)

    source_model = resolve_path(args.source)
    config = EvaderEnvironmentConfig(
        control_period_s=args.control_period_s,
        pn_navigation_constant=args.pn_navigation_constant,
        pn_max_command_g=args.pn_max_command_g,
        max_heading_rate_deg_s=args.max_heading_rate_deg_s,
        max_flight_path_rate_deg_s=args.max_flight_path_rate_deg_s,
        actuator_time_constant_s=args.actuator_time_constant_s,
        randomize_initial_conditions=args.randomize,
        record_history=False,
    )
    apply_randomization_scale(config, args.randomization_scale)

    vector_class = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    training_environment = make_vec_env(
        build_environment,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={
            "config": config,
            "hit_radius_m": args.radius,
        },
        vec_env_cls=vector_class,
    )

    try:
        if source_model is None:
            policy_kwargs = {
                "activation_fn": torch.nn.Tanh,
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            }
            model = PPO(
                "MlpPolicy",
                training_environment,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                seed=args.seed,
                device=args.device,
                verbose=1,
                tensorboard_log=str(tensorboard_directory),
                policy_kwargs=policy_kwargs,
            )
            continuation = False
        else:
            model = PPO.load(
                str(source_model),
                env=training_environment,
                device=args.device,
            )
            expected_observation_shape = (EVADER_OBSERVATION_SIZE,)
            expected_action_shape = (EVADER_ACTION_SIZE,)
            if tuple(model.observation_space.shape) != expected_observation_shape:
                raise ValueError(
                    "Source model is not an evader_v1 model. "
                    f"Expected observation {expected_observation_shape}, got "
                    f"{model.observation_space.shape}. Missile PPO models cannot "
                    "be used as evader source models."
                )
            if tuple(model.action_space.shape) != expected_action_shape:
                raise ValueError(
                    f"Expected action {expected_action_shape}, got "
                    f"{model.action_space.shape}."
                )
            model.learning_rate = args.learning_rate
            model.lr_schedule = get_schedule_fn(args.learning_rate)
            for parameter_group in model.policy.optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate
            model.tensorboard_log = str(tensorboard_directory)
            continuation = True

        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(args.action_std)))

        initial_num_timesteps = int(model.num_timesteps)
        model.save(stage_directory / "initial_model")

        evaluation_seeds = list(
            range(args.eval_seed, args.eval_seed + args.eval_episodes)
        )
        evaluation_csv = stage_directory / "evaluation_history.csv"

        class FixedSeedEvaluationCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)
                self.last_evaluation_timestep = initial_num_timesteps
                self.best_score = (-np.inf, -np.inf, -np.inf)

            def run_evaluation(self) -> dict[str, Any]:
                metrics = evaluate_on_fixed_seeds(
                    model=self.model,
                    config=config,
                    hit_radius_m=args.radius,
                    seeds=evaluation_seeds,
                )
                score = score_evaluation(metrics)
                improved = score > self.best_score
                if improved:
                    self.best_score = score
                    self.model.save(stage_directory / "best_model")

                row = {
                    "timesteps": int(self.num_timesteps),
                    "survival_rate": metrics["survival_rate"],
                    "mean_minimum_range_m": metrics[
                        "mean_minimum_range_m"
                    ],
                    "median_minimum_range_m": metrics[
                        "median_minimum_range_m"
                    ],
                    "p10_minimum_range_m": metrics[
                        "p10_minimum_range_m"
                    ],
                    "mean_episode_reward": metrics[
                        "mean_episode_reward"
                    ],
                    "new_best": improved,
                }
                write_header = not evaluation_csv.exists()
                with evaluation_csv.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

                print(
                    "\n[evader evaluation] "
                    f"steps={self.num_timesteps} | "
                    f"survival={100.0 * metrics['survival_rate']:.2f}% | "
                    f"mean d_min={metrics['mean_minimum_range_m']:.3f} m | "
                    f"reward={metrics['mean_episode_reward']:.3f} | "
                    f"new_best={improved}"
                )
                return metrics

            def _on_step(self) -> bool:
                if (
                    self.num_timesteps - self.last_evaluation_timestep
                    >= args.eval_frequency
                ):
                    self.run_evaluation()
                    self.last_evaluation_timestep = self.num_timesteps
                return True

        evaluation_callback = FixedSeedEvaluationCallback()
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.checkpoint_frequency // args.n_envs, 1),
            save_path=str(checkpoint_directory),
            name_prefix="evader",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks = CallbackList([evaluation_callback, checkpoint_callback])

        metadata = {
            "stage": args.stage,
            "observation_version": EVADER_OBSERVATION_VERSION,
            "observation_size": EVADER_OBSERVATION_SIZE,
            "action_size": EVADER_ACTION_SIZE,
            "source_model": str(source_model) if source_model else None,
            "continuation": continuation,
            "initial_num_timesteps": initial_num_timesteps,
            "additional_timesteps_requested": args.steps,
            "hit_radius_m": args.radius,
            "environment_config": asdict(config),
            "training_arguments": vars(args),
            "evaluation_seeds": evaluation_seeds,
        }
        (stage_directory / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("=" * 72)
        print("EVADER PPO TRAINING")
        print("=" * 72)
        print("Stage             :", args.stage)
        print("Output            :", stage_directory)
        print("Source            :", source_model or "fresh policy")
        print("Observation       :", (EVADER_OBSERVATION_SIZE,))
        print("Action            :", (EVADER_ACTION_SIZE,))
        print("Fixed Vector PN   : N =", args.pn_navigation_constant)
        print("Randomization     :", args.randomize)
        print("Parallel envs     :", args.n_envs)
        print("Additional steps  :", args.steps)
        print("=" * 72)

        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            reset_num_timesteps=not continuation,
            tb_log_name=args.stage,
            progress_bar=False,
        )
        model.save(stage_directory / "final_model")

        final_metrics = evaluation_callback.run_evaluation()
        metadata["final_num_timesteps"] = int(model.num_timesteps)
        metadata["final_evaluation"] = {
            key: value
            for key, value in final_metrics.items()
            if key != "rows"
        }
        metadata["best_score"] = list(evaluation_callback.best_score)
        (stage_directory / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("\nTraining complete.")
        print("Best model :", stage_directory / "best_model.zip")
        print("Final model:", stage_directory / "final_model.zip")
    finally:
        training_environment.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
