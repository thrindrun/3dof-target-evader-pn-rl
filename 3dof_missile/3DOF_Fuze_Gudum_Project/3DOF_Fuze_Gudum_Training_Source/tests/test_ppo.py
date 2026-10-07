"""
Eğitilmiş PPO modelini sabit, düz hedef senaryosunda test eder.

Örnek kullanım:

    python tests/test_ppo.py --model models/1500m/best_model.zip --radius 1500

Daha zor bir vurma yarıçapı denemek için yalnızca --radius değiştirilir:

    python tests/test_ppo.py --model models/1500m/best_model.zip --radius 1000

Test sonunda sayısal geçmiş .npz dosyasına, grafik ise .png dosyasına kaydedilir.
Grafik aynı zamanda ekranda açılır.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from air_to_air_3dof_ppo import (  # noqa: E402
    POSITION,
    PPOEnvironmentConfig,
    RLInterfaceConfig,
    build_default_ppo_environment,
    infer_policy_observation_version,
    rl_observation_size,
)


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("Vurma yarıçapı sıfırdan büyük olmalıdır.")
    return number


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eğitilmiş PPO füze güdüm modelini test et",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Test edilecek PPO .zip modelinin yolu",
    )
    parser.add_argument(
        "--radius",
        type=positive_float,
        default=1500.0,
        help="Test vurma yarıçapı [m]. Varsayılan: 1500",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Test seed değeri. Varsayılan: 42",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Deterministik ortalama aksiyon yerine rastgele aksiyon örnekle",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Grafiği ekranda açma; yalnızca PNG olarak kaydet",
    )
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
    """Sonuç dosya isminde güvenle kullanılabilecek kısa metin üretir."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "model"


def main() -> None:
    args = parse_arguments()
    model_path = resolve_model_path(args.model)

    test_model = PPO.load(
        str(model_path),
        device="cpu",
    )
    observation_version = infer_policy_observation_version(test_model)

    environment_config = PPOEnvironmentConfig(
        target_maneuver_kind="straight",
        randomize_initial_conditions=False,
        control_period_s=0.10,
    )

    evaluation_environment = build_default_ppo_environment(
        ppo_environment_config=environment_config,
        rl_interface_config=RLInterfaceConfig(
            observation_version=observation_version,
        ),
    )
    evaluation_environment.environment_template.hit_radius_m = args.radius

    observation, info = evaluation_environment.reset(seed=args.seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    last_action = np.zeros(2, dtype=np.float32)

    time_history = [float(info["time_s"])]
    range_history = [float(info["range_m"])]
    command_history = [0.0]
    reward_history = [0.0]

    missile_position_history = [
        evaluation_environment.states[
            evaluation_environment.missile_name
        ][POSITION].copy()
    ]
    aircraft_position_history = [
        evaluation_environment.states[
            evaluation_environment.target_name
        ][POSITION].copy()
    ]

    while not (terminated or truncated):
        # Burada yüklediğimiz değişkenin adı özellikle test_model'dir. Önceki
        # Colab testlerinde yaşanan eski model değişkeni karışıklığını önler.
        action, _ = test_model.predict(
            observation,
            deterministic=not args.stochastic,
        )

        observation, reward, terminated, truncated, info = (
            evaluation_environment.step(action)
        )

        total_reward += float(reward)
        last_action = np.asarray(action, dtype=np.float32).copy()

        time_history.append(float(info["time_s"]))
        range_history.append(float(info["range_m"]))
        command_history.append(float(info["command_g"]))
        reward_history.append(float(reward))

        missile_position_history.append(
            evaluation_environment.states[
                evaluation_environment.missile_name
            ][POSITION].copy()
        )
        aircraft_position_history.append(
            evaluation_environment.states[
                evaluation_environment.target_name
            ][POSITION].copy()
        )

    time_history = np.asarray(time_history, dtype=float)
    range_history = np.asarray(range_history, dtype=float)
    command_history = np.asarray(command_history, dtype=float)
    reward_history = np.asarray(reward_history, dtype=float)
    missile_position_history = np.asarray(missile_position_history, dtype=float)
    aircraft_position_history = np.asarray(aircraft_position_history, dtype=float)

    results_directory = PROJECT_ROOT / "results"
    results_directory.mkdir(parents=True, exist_ok=True)

    model_tag = safe_file_part(model_path.parent.name or model_path.stem)
    radius_tag = f"{args.radius:g}m"
    mode_tag = "stochastic" if args.stochastic else "deterministic"
    result_stem = f"{model_tag}_{radius_tag}_seed{args.seed}_{mode_tag}"

    data_path = results_directory / f"{result_stem}.npz"
    figure_path = results_directory / f"{result_stem}.png"

    np.savez_compressed(
        data_path,
        time_s=time_history,
        range_m=range_history,
        command_g=command_history,
        reward=reward_history,
        missile_position_ned_m=missile_position_history,
        aircraft_position_ned_m=aircraft_position_history,
        status=np.asarray(info["status"]),
        minimum_range_m=np.asarray(info["minimum_range_m"]),
        total_reward=np.asarray(total_reward),
    )

    figure = plt.figure(figsize=(15, 9))

    range_axis = figure.add_subplot(2, 2, 1)
    range_axis.plot(time_history, range_history, color="tab:blue")
    range_axis.axhline(
        args.radius,
        color="red",
        linestyle="--",
        label=f"Vurma sınırı: {args.radius:g} m",
    )
    range_axis.set_xlabel("Zaman [s]")
    range_axis.set_ylabel("Menzil [m]")
    range_axis.set_title("Füze-hedef menzili")
    range_axis.grid(True)
    range_axis.legend()

    command_axis = figure.add_subplot(2, 2, 2)
    command_axis.plot(time_history, command_history, color="tab:orange")
    command_axis.set_xlabel("Zaman [s]")
    command_axis.set_ylabel("Normal ivme komutu [g]")
    command_axis.set_title("PPO komut büyüklüğü")
    command_axis.grid(True)

    reward_axis = figure.add_subplot(2, 2, 3)
    reward_axis.plot(time_history, np.cumsum(reward_history), color="tab:green")
    reward_axis.set_xlabel("Zaman [s]")
    reward_axis.set_ylabel("Kümülatif reward")
    reward_axis.set_title("Episode reward gelişimi")
    reward_axis.grid(True)

    trajectory_axis = figure.add_subplot(2, 2, 4, projection="3d")
    trajectory_axis.plot(
        missile_position_history[:, 0],
        missile_position_history[:, 1],
        missile_position_history[:, 2],
        label="Füze",
        color="tab:red",
    )
    trajectory_axis.plot(
        aircraft_position_history[:, 0],
        aircraft_position_history[:, 1],
        aircraft_position_history[:, 2],
        label="Uçak",
        color="tab:blue",
    )
    trajectory_axis.scatter(
        missile_position_history[0, 0],
        missile_position_history[0, 1],
        missile_position_history[0, 2],
        color="darkred",
        marker="o",
        label="Füze başlangıç",
    )
    trajectory_axis.scatter(
        aircraft_position_history[0, 0],
        aircraft_position_history[0, 1],
        aircraft_position_history[0, 2],
        color="navy",
        marker="o",
        label="Uçak başlangıç",
    )
    trajectory_axis.set_xlabel("North [m]")
    trajectory_axis.set_ylabel("East [m]")
    trajectory_axis.set_zlabel("Down [m]")
    trajectory_axis.set_title("3B NED yörüngeleri")
    trajectory_axis.invert_zaxis()
    trajectory_axis.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0.0,
    fontsize=9,
)

    figure.suptitle(
        f"PPO testi — {info['status']} — minimum menzil "
        f"{float(info['minimum_range_m']):.2f} m",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")

    print()
    print("=" * 72)
    print("PPO TEST SONUCU")
    print("=" * 72)
    print("Test edilen model       :", model_path)
    print(
        "Observation            :",
        observation_version,
        f"({rl_observation_size(observation_version)} girdi)",
    )
    print("Test modu               :", mode_tag)
    print("Vurma yarıçapı          :", args.radius, "m")
    print("Sonuç                   :", info["status"])
    print("Minimum menzil          :", info["minimum_range_m"], "m")
    print("Sınır dışında kalan     :", info.get("miss_distance_m", 0.0), "m")
    print("Son menzil              :", info["range_m"], "m")
    print("Terminal zamanı         :", info["terminal_time_s"], "s")
    print("Toplam reward           :", total_reward)
    print("Son aksiyon             :", last_action)
    print("Son reward terimleri    :", info.get("reward_terms", {}))
    print("Veri dosyası            :", data_path)
    print("Grafik dosyası          :", figure_path)
    print("=" * 72)

    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)

    evaluation_environment.close()


if __name__ == "__main__":
    main()
