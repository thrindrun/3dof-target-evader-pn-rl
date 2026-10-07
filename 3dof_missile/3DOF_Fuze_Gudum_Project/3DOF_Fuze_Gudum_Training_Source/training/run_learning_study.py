"""Kontrollü PPO reward ve observation öğrenme çalışmasını sıfırdan yürütür.

Bu araç başarılı üretim modelini yüklemez. Her deney yeni actor, yeni critic ve
``num_timesteps == 0`` ile başlar. Reward fazında yalnız reward profili;
observation fazında yalnız observation boyutu değiştirilir. V2 varsayılanları,
V1 pilotunda görülen tavan etkisini azaltmak için 100 m ve +/-30 derece launch
elevation kullanır.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "training" / "train_ppo.py"
PLOT_SCRIPT = PROJECT_ROOT / "tests" / "plot_learning_study.py"


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Değer sıfırdan büyük olmalıdır.")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("Değer sıfırdan büyük olmalıdır.")
    return number


def safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value.strip()
    )
    if not cleaned:
        raise argparse.ArgumentTypeError("Etiket boş olamaz.")
    return cleaned


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Altı reward ve üç observation deneyini, ağırlık aktarmadan sıfırdan "
            "eğitir."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("reward", "observation", "all"),
        default="all",
        help="Çalıştırılacak deney grubu. Varsayılan: all",
    )
    parser.add_argument(
        "--tag",
        type=safe_name,
        default="learning_study_v2",
        help="Stage ve rapor klasörlerinde kullanılacak çalışma etiketi.",
    )
    parser.add_argument(
        "--reward-steps",
        type=positive_int,
        default=200_000,
        help="Her reward deneyi için istenen PPO adımı. Varsayılan: 200000",
    )
    parser.add_argument(
        "--observation-steps",
        type=positive_int,
        default=150_000,
        help="Her observation deneyi için istenen PPO adımı. Varsayılan: 150000",
    )
    parser.add_argument(
        "--n-envs",
        type=positive_int,
        default=12,
        help="Paralel ortam sayısı. i9-12900H için varsayılan: 12",
    )
    parser.add_argument(
        "--eval-episodes",
        type=positive_int,
        default=120,
        help="Her sabit-seed ölçümündeki episode sayısı. Varsayılan: 120",
    )
    parser.add_argument(
        "--eval-frequency",
        type=positive_int,
        default=20_000,
        help="Öğrenme eğrisi ölçüm aralığı [PPO adımı]. Varsayılan: 20000",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-eval-seed", type=int, default=900_000)
    parser.add_argument("--observation-eval-seed", type=int, default=910_000)
    parser.add_argument(
        "--reward-radius-m",
        type=positive_float,
        default=100.0,
        help="Reward fazının başarı yarıçapı [m]. V2 varsayılanı: 100",
    )
    parser.add_argument(
        "--reward-elevation-min-deg",
        type=float,
        default=-30.0,
        help="Reward fazı minimum launch elevation [deg]. Varsayılan: -30",
    )
    parser.add_argument(
        "--reward-elevation-max-deg",
        type=float,
        default=30.0,
        help="Reward fazı maksimum launch elevation [deg]. Varsayılan: 30",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Komutları ve manifesti üretir; eğitim başlatmaz.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Eğitimlerin ardından otomatik grafik üretimini atlar.",
    )
    return parser.parse_args()


def reward_experiments(
    tag: str,
    steps: int,
    eval_seed: int,
    radius_m: float = 100.0,
    elevation_min_deg: float = -30.0,
    elevation_max_deg: float = 30.0,
) -> list[dict]:
    profiles = (
        (
            "01_weak_binary",
            "weak_binary",
            "Yalnız terminal hit=+1 / miss=-1; ara adımlarda bilgi yok.",
        ),
        (
            "02_sparse",
            "sparse",
            "Terminal ölçeği ve küçük zaman/kontrol maliyetleri kalibre edilir.",
        ),
        (
            "03_range",
            "range_progress",
            "Ajan ilk kez anlık menzil azalmasıyla yön bilgisi alır.",
        ),
        (
            "04_best_range",
            "best_range",
            "Episode içindeki yeni minimum menzil rekoru ayrıca ödüllendirilir.",
        ),
        (
            "05_terminal_distance",
            "terminal_distance",
            "Kaçırmalar minimum mesafelerine göre sürekli ayrıştırılır.",
        ),
        (
            "06_predictive_cpa",
            "predictive_cpa",
            "Tahminî CPA ilerlemesi ve sürekli terminal doğruluğu eklenir.",
        ),
    )
    return [
        {
            "phase": "reward",
            "stage": f"study_{tag}_reward_{order}_obs10",
            "label": profile,
            "design_intent": design_intent,
            "reward_profile": profile,
            "observation_version": "legacy10",
            "radius_m": float(radius_m),
            "steps": steps,
            "launch_elevation_min_deg": float(elevation_min_deg),
            "launch_elevation_max_deg": float(elevation_max_deg),
            "eval_seed": eval_seed,
        }
        for order, profile, design_intent in profiles
    ]


def observation_experiments(
    tag: str,
    steps: int,
    eval_seed: int,
) -> list[dict]:
    observations = (
        ("01_obs10", "legacy10"),
        ("02_obs13", "markov13"),
        ("03_obs23", "v2"),
    )
    return [
        {
            "phase": "observation",
            "stage": f"study_{tag}_observation_{order}",
            "label": observation_version,
            "design_intent": (
                "Aynı predictive-CPA reward altında yalnız politika girdisi "
                "değiştirilir."
            ),
            "reward_profile": "predictive_cpa",
            "observation_version": observation_version,
            "radius_m": 100.0,
            "steps": steps,
            "launch_elevation_min_deg": -30.0,
            "launch_elevation_max_deg": 30.0,
            "eval_seed": eval_seed,
        }
        for order, observation_version in observations
    ]


def command_for_experiment(experiment: dict, args: argparse.Namespace) -> list[str]:
    """Bütün kontrol değişkenlerini açıkça yazan yeniden üretilebilir komut."""
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--stage",
        str(experiment["stage"]),
        "--radius",
        str(experiment["radius_m"]),
        "--steps",
        str(experiment["steps"]),
        "--fresh-start",
        "--record-initial-evaluation",
        "--observation-version",
        str(experiment["observation_version"]),
        "--reward-profile",
        str(experiment["reward_profile"]),
        "--learning-rate",
        "0.0003",
        "--action-std",
        "0.5",
        "--seed",
        str(args.seed),
        "--randomize",
        "--randomization-scale",
        "0.5",
        "--geometry-mode",
        "target_sphere",
        "--range-min-m",
        "5000",
        "--range-max-m",
        "5000",
        "--launch-azimuth-min-deg",
        "-180",
        "--launch-azimuth-max-deg",
        "180",
        "--launch-elevation-min-deg",
        str(experiment["launch_elevation_min_deg"]),
        "--launch-elevation-max-deg",
        str(experiment["launch_elevation_max_deg"]),
        "--aim-azimuth-error-deg",
        "0",
        "--aim-elevation-error-deg",
        "0",
        "--sphere-ground-clearance-m",
        "100",
        "--sphere-auto-raise-target",
        "--legacy-replay-probability",
        "0",
        "--passed-target-arm-progress-m",
        "100",
        "--n-envs",
        str(args.n_envs),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-seed",
        str(experiment["eval_seed"]),
        "--eval-frequency",
        str(args.eval_frequency),
        "--eval-sector-count",
        "12",
    ]


def command_text(command: list[str]) -> str:
    """Windows PowerShell'de de okunabilir bir komut satırı üretir."""
    return subprocess.list2cmdline(command)


def experiment_state(stage: str) -> str:
    model_directory = PROJECT_ROOT / "models" / stage
    checkpoint_directory = PROJECT_ROOT / "checkpoints" / stage
    evaluation_directory = PROJECT_ROOT / "logs" / "evaluation" / stage
    required = (
        model_directory / "best_model.zip",
        model_directory / "final_model.zip",
        model_directory / "training_metadata.json",
        evaluation_directory / "fixed_seed_evaluations.jsonl",
    )
    if all(path.is_file() for path in required):
        return "complete"

    directories = (model_directory, checkpoint_directory, evaluation_directory)
    if any(directory.exists() and any(directory.iterdir()) for directory in directories):
        return "partial"
    return "new"


def load_manifest(path: Path, tag: str) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 2,
        "tag": tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scientific_rule": (
            "Her koşul bağımsız actor/critic ile num_timesteps=0'dan başlar; "
            "hiçbir başarılı model kaynak olarak kullanılmaz."
        ),
        "experiments": [],
    }


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def upsert_manifest_experiment(manifest: dict, record: dict) -> None:
    experiments = manifest["experiments"]
    for index, existing in enumerate(experiments):
        if existing.get("stage") == record["stage"]:
            experiments[index] = record
            return
    experiments.append(record)


def main() -> None:
    args = parse_arguments()
    if args.eval_episodes % 12 != 0:
        raise ValueError(
            "--eval-episodes, dengeli 12 azimut sektörü için 12'nin tam katı "
            "olmalıdır."
        )
    if args.reward_elevation_max_deg < args.reward_elevation_min_deg:
        raise ValueError(
            "--reward-elevation-max-deg, minimum değerden küçük olamaz."
        )
    if not (
        -90.0 <= args.reward_elevation_min_deg <= 90.0
        and -90.0 <= args.reward_elevation_max_deg <= 90.0
    ):
        raise ValueError("Reward elevation sınırları [-90, 90] içinde olmalıdır.")

    experiments: list[dict] = []
    if args.phase in {"reward", "all"}:
        experiments.extend(
            reward_experiments(
                args.tag,
                args.reward_steps,
                args.reward_eval_seed,
                radius_m=args.reward_radius_m,
                elevation_min_deg=args.reward_elevation_min_deg,
                elevation_max_deg=args.reward_elevation_max_deg,
            )
        )
    if args.phase in {"observation", "all"}:
        experiments.extend(
            observation_experiments(
                args.tag,
                args.observation_steps,
                args.observation_eval_seed,
            )
        )

    study_directory = PROJECT_ROOT / "results" / "learning_study" / args.tag
    manifest_path = study_directory / "study_manifest.json"
    manifest = load_manifest(manifest_path, args.tag)
    manifest["requested_phase"] = args.phase
    manifest["common_controls"] = {
        "fresh_start": True,
        "source_model": None,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "learning_rate": 3e-4,
        "initial_action_std": 0.5,
        "target_maneuver": "straight",
        "target_sphere_range_m": [5000.0, 5000.0],
        "launch_azimuth_deg": [-180.0, 180.0],
        "randomization_scale": 0.5,
        "evaluation_episodes": args.eval_episodes,
        "evaluation_frequency_timesteps": args.eval_frequency,
        "evaluation_azimuth_sectors": 12,
        "reward_phase": {
            "observation_version": "legacy10",
            "hit_radius_m": args.reward_radius_m,
            "launch_elevation_deg": [
                args.reward_elevation_min_deg,
                args.reward_elevation_max_deg,
            ],
        },
        "observation_phase": {
            "reward_profile": "predictive_cpa",
            "hit_radius_m": 100.0,
            "launch_elevation_deg": [-30.0, 30.0],
        },
    }

    print("=" * 88)
    print("PPO ÖĞRENME ÇALIŞMASI — HER DENEY SIFIRDAN")
    print("=" * 88)
    print("Etiket             :", args.tag)
    print("Deney sayısı       :", len(experiments))
    print("Kaynak model       : YOK")
    print("Paralel ortam      :", args.n_envs)
    if args.phase in {"reward", "all"}:
        print("Reward geometrisi  :", end=" ")
        print(
            f"R={args.reward_radius_m:g} m, "
            f"el={args.reward_elevation_min_deg:g}.."
            f"{args.reward_elevation_max_deg:g} deg"
        )
    print("Dry-run            :", args.dry_run)
    print("=" * 88)

    for sequence, experiment in enumerate(experiments, start=1):
        stage = str(experiment["stage"])
        state = experiment_state(stage)
        command = command_for_experiment(experiment, args)
        record = {
            **experiment,
            "sequence": sequence,
            "command": command,
            "command_text": command_text(command),
            "detected_state": state,
        }

        print()
        print(f"[{sequence}/{len(experiments)}] {stage}")
        print(command_text(command))

        if state == "complete":
            record["status"] = "skipped_complete"
            upsert_manifest_experiment(manifest, record)
            save_manifest(manifest_path, manifest)
            print("Durum              : tamamlanmış; güvenle atlandı.")
            continue
        if state == "partial" and not args.dry_run:
            record["status"] = "blocked_partial_output"
            upsert_manifest_experiment(manifest, record)
            save_manifest(manifest_path, manifest)
            raise RuntimeError(
                f"{stage} için yarım/eski çıktı bulundu. Hiçbir dosya silinmedi. "
                "Yeni bir --tag seçin veya klasörleri elle inceleyin."
            )
        if args.dry_run:
            record["status"] = "planned"
            upsert_manifest_experiment(manifest, record)
            save_manifest(manifest_path, manifest)
            continue

        record["status"] = "running"
        record["started_at"] = datetime.now().isoformat(timespec="seconds")
        upsert_manifest_experiment(manifest, record)
        save_manifest(manifest_path, manifest)
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            record["status"] = "failed"
            record["return_code"] = error.returncode
            record["finished_at"] = datetime.now().isoformat(timespec="seconds")
            upsert_manifest_experiment(manifest, record)
            save_manifest(manifest_path, manifest)
            raise
        record["status"] = "complete"
        record["finished_at"] = datetime.now().isoformat(timespec="seconds")
        upsert_manifest_experiment(manifest, record)
        save_manifest(manifest_path, manifest)

    if not args.no_plots and not args.dry_run:
        plot_command = [
            sys.executable,
            str(PLOT_SCRIPT),
            "--tag",
            args.tag,
            "--phase",
            args.phase,
        ]
        print()
        print("Grafikler oluşturuluyor:")
        print(command_text(plot_command))
        subprocess.run(plot_command, cwd=PROJECT_ROOT, check=True)

    print()
    print("Manifest:", manifest_path)
    if args.dry_run:
        print("Dry-run tamamlandı; hiçbir eğitim başlatılmadı.")
    else:
        print("Seçilen öğrenme çalışması tamamlandı.")


if __name__ == "__main__":
    main()
