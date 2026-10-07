"""Reward/observation öğrenme çalışmasının karşılaştırmalı grafiklerini üretir."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        description="PPO öğrenme çalışması grafiklerini üretir."
    )
    parser.add_argument("--tag", type=safe_name, default="learning_study_v2")
    parser.add_argument(
        "--phase",
        choices=("reward", "observation", "all"),
        default="all",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="PNG kaydından sonra matplotlib pencerelerini de aç.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Geçersiz JSONL satırı: {path}:{line_number}"
                ) from error
    return records


def experiment_label(experiment: dict) -> str:
    if experiment["phase"] == "reward":
        labels = {
            "weak_binary": "Naif +1/-1",
            "sparse": "Sparse terminal",
            "range_progress": "Anlık menzil",
            "best_range": "Best menzil",
            "terminal_distance": "Terminal mesafe",
            "predictive_cpa": "Predictive CPA",
        }
        profile_label = labels.get(
            experiment["reward_profile"],
            experiment["label"],
        )
        sequence = experiment.get("sequence")
        return f"R{int(sequence)} {profile_label}" if sequence else profile_label
    labels = {
        "legacy10": "10 girdi",
        "markov13": "13 girdi",
        "v2": "23 girdi",
    }
    return labels.get(experiment["observation_version"], experiment["label"])


def load_experiments(manifest: dict, phase: str) -> list[dict]:
    experiments: list[dict] = []
    for manifest_experiment in manifest.get("experiments", []):
        if manifest_experiment.get("phase") != phase:
            continue
        stage = manifest_experiment["stage"]
        history_path = (
            PROJECT_ROOT
            / "logs"
            / "evaluation"
            / stage
            / "fixed_seed_evaluations.jsonl"
        )
        records = read_jsonl(history_path)
        if not records:
            print(f"UYARI: değerlendirme kaydı bulunamadı, atlandı: {stage}")
            continue
        experiment = dict(manifest_experiment)
        metadata_path = PROJECT_ROOT / "models" / stage / "training_metadata.json"
        experiment["metadata"] = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        experiment["records"] = sorted(
            records,
            key=lambda record: int(record["num_timesteps"]),
        )
        experiment["display_label"] = experiment_label(experiment)
        experiments.append(experiment)
    return sorted(experiments, key=lambda item: int(item.get("sequence", 0)))


def configure_axis(axis, title: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("PPO adımı")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.30)


def plot_learning_curves(experiments: list[dict], phase: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    metrics = (
        ("success_rate", "Sabit-seed başarı", "Başarı [%]", 100.0),
        (
            "mean_minimum_range_m",
            "Ortalama minimum menzil",
            "Menzil [m]",
            1.0,
        ),
        ("p90_minimum_range_m", "P90 minimum menzil", "Menzil [m]", 1.0),
        ("mean_reward", "Ortalama episode reward", "Reward", 1.0),
    )
    for axis, (metric, title, ylabel, scale) in zip(
        axes.flat,
        metrics,
        strict=True,
    ):
        for experiment in experiments:
            records = experiment["records"]
            timesteps = [int(record["num_timesteps"]) for record in records]
            values = [scale * float(record[metric]) for record in records]
            axis.plot(
                timesteps,
                values,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                label=experiment["display_label"],
            )
        configure_axis(axis, title, ylabel)
        if "minimum_range" in metric:
            axis.set_yscale("symlog", linthresh=1.0)
        if metric == "success_rate":
            axis.set_ylim(-2.0, 102.0)

    axes[0, 0].legend(loc="best", fontsize=9)
    phase_title = (
        "Reward geliştirme dizisi — 10 girdi ve aynı senaryolar"
        if phase == "reward"
        else "Observation ablation — aynı predictive-CPA reward ve senaryolar"
    )
    note = (
        "Reward değerleri farklı tanımlara sahip olduğundan son panel her çizgi "
        "içinde öğrenme yönünü gösterir; weak_binary varsa ölçeği de farklıdır "
        "ve mutlak reward'lar doğrudan sıralanmaz."
        if phase == "reward"
        else "Bütün koşullar sıfırdan, aynı PPO hiperparametreleriyle başlamıştır."
    )
    figure.suptitle(f"{phase_title}\n{note}", fontsize=14)
    figure.savefig(output, dpi=180)
    print("Grafik:", output)


def plot_reward_components(experiments: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    term_labels = {
        "range_progress": "Anlık menzil",
        "minimum_range_progress": "Best menzil",
        "predictive_cpa_progress": "CPA progress",
        "control_effort": "Kontrol",
        "time": "Zaman",
        "terminal": "Terminal",
        "terminal_accuracy": "Terminal doğruluk",
        "miss_distance": "Iskalama mesafesi",
    }
    term_names = tuple(term_labels)
    x_positions = np.arange(len(experiments), dtype=float)
    positive_bottom = np.zeros(len(experiments), dtype=float)
    negative_bottom = np.zeros(len(experiments), dtype=float)

    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for term_name in term_names:
        values = np.asarray(
            [
                float(
                    experiment["records"][-1]
                    .get("mean_reward_terms", {})
                    .get(term_name, 0.0)
                )
                for experiment in experiments
            ],
            dtype=float,
        )
        bottoms = np.where(values >= 0.0, positive_bottom, negative_bottom)
        axis.bar(
            x_positions,
            values,
            bottom=bottoms,
            label=term_labels[term_name],
        )
        positive_bottom += np.where(values >= 0.0, values, 0.0)
        negative_bottom += np.where(values < 0.0, values, 0.0)

    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xticks(
        x_positions,
        [experiment["display_label"] for experiment in experiments],
        rotation=15,
        ha="right",
    )
    axis.set_ylabel("Ortalama episode katkısı (ölçeklenmiş reward)")
    axis.set_title("Son değerlendirmede reward bileşenlerinin katkısı")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(ncol=2, fontsize=9)
    figure.savefig(output, dpi=180)
    print("Grafik:", output)


def tensorboard_scalar_series(stage: str, tag: str) -> tuple[list[int], list[float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return [], []

    tensorboard_root = PROJECT_ROOT / "logs" / "tensorboard"
    event_files = list(
        tensorboard_root.glob(f"PPO_{stage}*/**/events.out.tfevents.*")
    )
    if not event_files:
        return [], []
    newest_directory = max(event_files, key=lambda path: path.stat().st_mtime).parent
    accumulator = EventAccumulator(
        str(newest_directory),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return [], []
    events = accumulator.Scalars(tag)
    return [int(event.step) for event in events], [float(event.value) for event in events]


def plot_tensorboard_diagnostics(
    experiments: list[dict],
    phase: str,
    output: Path,
) -> bool:
    import matplotlib.pyplot as plt

    diagnostic_tags = (
        ("train/value_loss", "Critic value loss"),
        ("train/policy_gradient_loss", "Policy-gradient loss"),
        ("train/explained_variance", "Explained variance"),
        ("train/entropy_loss", "Entropy loss"),
        ("train/std", "Action dağılımı std"),
        ("train/approx_kl", "Yaklaşık KL"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    plotted_anything = False
    for axis, (tag, title) in zip(axes.flat, diagnostic_tags, strict=True):
        for experiment in experiments:
            steps, values = tensorboard_scalar_series(experiment["stage"], tag)
            if not steps:
                continue
            plotted_anything = True
            axis.plot(
                steps,
                values,
                linewidth=1.5,
                label=experiment["display_label"],
            )
        axis.set_title(title)
        axis.set_xlabel("PPO adımı")
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    figure.suptitle(f"PPO optimizasyon tanıları — {phase}", fontsize=14)
    if not plotted_anything:
        plt.close(figure)
        return False
    figure.savefig(output, dpi=180)
    print("Grafik:", output)
    return True


def write_csv_outputs(all_experiments: list[dict], output_directory: Path) -> None:
    term_names = (
        "range_progress",
        "minimum_range_progress",
        "predictive_cpa_progress",
        "control_effort",
        "time",
        "terminal",
        "terminal_accuracy",
        "miss_distance",
    )
    curve_path = output_directory / "learning_curves.csv"
    fieldnames = [
        "phase",
        "stage",
        "label",
        "reward_profile",
        "observation_version",
        "num_timesteps",
        "record_label",
        "episode_count",
        "success_rate",
        "mean_reward",
        "std_reward",
        "mean_minimum_range_m",
        "median_minimum_range_m",
        "p90_minimum_range_m",
        "mean_terminal_time_s",
        *[f"reward_term_{name}" for name in term_names],
    ]
    rows: list[dict] = []
    for experiment in all_experiments:
        for record in experiment["records"]:
            row = {
                "phase": experiment["phase"],
                "stage": experiment["stage"],
                "label": experiment["display_label"],
                "reward_profile": experiment["reward_profile"],
                "observation_version": experiment["observation_version"],
                "num_timesteps": int(record["num_timesteps"]),
                "record_label": record["label"],
                "episode_count": record["episode_count"],
                "success_rate": record["success_rate"],
                "mean_reward": record["mean_reward"],
                "std_reward": record["std_reward"],
                "mean_minimum_range_m": record["mean_minimum_range_m"],
                "median_minimum_range_m": record["median_minimum_range_m"],
                "p90_minimum_range_m": record["p90_minimum_range_m"],
                "mean_terminal_time_s": record["mean_terminal_time_s"],
            }
            reward_terms = record.get("mean_reward_terms", {})
            for term_name in term_names:
                row[f"reward_term_{term_name}"] = reward_terms.get(term_name, 0.0)
            rows.append(row)
    with curve_path.open("w", newline="", encoding="utf-8-sig") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Veri  :", curve_path)

    summary_path = output_directory / "final_summary.csv"
    summary_fields = [
        "phase",
        "stage",
        "label",
        "requested_steps",
        "actual_model_timesteps",
        "last_evaluation_timesteps",
        "initial_success_rate",
        "success_rate",
        "success_change_percentage_points",
        "initial_mean_minimum_range_m",
        "mean_minimum_range_m",
        "mean_minimum_range_improvement_m",
        "initial_p90_minimum_range_m",
        "p90_minimum_range_m",
        "p90_minimum_range_improvement_m",
        "worst_sector_success_rate",
        "worst_sector_mean_minimum_range_m",
        "mean_reward",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as destination:
        writer = csv.DictWriter(destination, fieldnames=summary_fields)
        writer.writeheader()
        for experiment in all_experiments:
            initial_record = experiment["records"][0]
            final_record = experiment["records"][-1]
            initial_success = float(initial_record["success_rate"])
            final_success = float(final_record["success_rate"])
            initial_mean_range = float(
                initial_record["mean_minimum_range_m"]
            )
            final_mean_range = float(final_record["mean_minimum_range_m"])
            initial_p90_range = float(initial_record["p90_minimum_range_m"])
            final_p90_range = float(final_record["p90_minimum_range_m"])
            writer.writerow(
                {
                    "phase": experiment["phase"],
                    "stage": experiment["stage"],
                    "label": experiment["display_label"],
                    "requested_steps": experiment["steps"],
                    "actual_model_timesteps": experiment["metadata"].get(
                        "ending_timesteps",
                        final_record["num_timesteps"],
                    ),
                    "last_evaluation_timesteps": final_record["num_timesteps"],
                    "initial_success_rate": initial_success,
                    "success_rate": final_success,
                    "success_change_percentage_points": (
                        100.0 * (final_success - initial_success)
                    ),
                    "initial_mean_minimum_range_m": initial_mean_range,
                    "mean_minimum_range_m": final_mean_range,
                    "mean_minimum_range_improvement_m": (
                        initial_mean_range - final_mean_range
                    ),
                    "initial_p90_minimum_range_m": initial_p90_range,
                    "p90_minimum_range_m": final_p90_range,
                    "p90_minimum_range_improvement_m": (
                        initial_p90_range - final_p90_range
                    ),
                    "worst_sector_success_rate": final_record.get(
                        "worst_sector_success_rate"
                    ),
                    "worst_sector_mean_minimum_range_m": final_record.get(
                        "worst_sector_mean_minimum_range_m"
                    ),
                    "mean_reward": final_record["mean_reward"],
                }
            )
    print("Özet :", summary_path)


def write_reward_diagnostics(
    all_experiments: list[dict],
    output_directory: Path,
) -> None:
    """Önceden tanımlanan tavan/aşırı-zorluk kurallarını makinece uygular."""
    reward_experiments = [
        experiment
        for experiment in all_experiments
        if experiment.get("phase") == "reward"
    ]
    if not reward_experiments:
        return

    profiles: list[dict] = []
    for experiment in reward_experiments:
        initial = experiment["records"][0]
        final = experiment["records"][-1]
        profiles.append(
            {
                "stage": experiment["stage"],
                "label": experiment["display_label"],
                "initial_success_rate": float(initial["success_rate"]),
                "final_success_rate": float(final["success_rate"]),
                "success_change_percentage_points": 100.0
                * (
                    float(final["success_rate"])
                    - float(initial["success_rate"])
                ),
                "mean_minimum_range_improvement_m": (
                    float(initial["mean_minimum_range_m"])
                    - float(final["mean_minimum_range_m"])
                ),
                "p90_minimum_range_improvement_m": (
                    float(initial["p90_minimum_range_m"])
                    - float(final["p90_minimum_range_m"])
                ),
            }
        )

    final_success_rates = [item["final_success_rate"] for item in profiles]
    mean_improvements = [
        item["mean_minimum_range_improvement_m"] for item in profiles
    ]
    ceiling_effect = all(value >= 0.95 for value in final_success_rates)
    uniformly_too_hard = all(value <= 0.05 for value in final_success_rates) and all(
        value <= 1.0 for value in mean_improvements
    )
    if ceiling_effect:
        classification = "ceiling_effect"
        recommendation = (
            "Bu sonuçları koruyun; yeni tag ile yalnız reward radius'u 50 m "
            "yaparak ayrı V3 pilotu çalıştırın."
        )
    elif uniformly_too_hard:
        classification = "uniformly_too_hard"
        recommendation = (
            "Bu sonuçları koruyun; yeni tag ile radius=200 m veya elevation="
            "-15..15 deg kullanarak ayrı kalibrasyon pilotu çalıştırın."
        )
    else:
        classification = "informative_separation"
        recommendation = (
            "Reward profilleri fiziksel metriklerde ayrışıyor; zorluğu sonuç "
            "sonrası değiştirmeden bu V2'yi raporlayın ve sonra observation "
            "fazına geçin."
        )

    diagnostics = {
        "classification": classification,
        "ceiling_effect_all_final_success_ge_95_percent": ceiling_effect,
        "uniformly_too_hard_rule_triggered": uniformly_too_hard,
        "maximum_final_success_gap_percentage_points": 100.0
        * (max(final_success_rates) - min(final_success_rates)),
        "recommendation": recommendation,
        "profiles": profiles,
        "caution": (
            "Mutlak reward büyüklükleri profiller arasında sıralama ölçütü "
            "değildir; başarı ve minimum-menzil metriklerini kullanın."
        ),
    }
    diagnostics_path = output_directory / "reward_study_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Tanı  :", diagnostics_path)


def main() -> None:
    args = parse_arguments()
    if not args.show:
        matplotlib.use("Agg")

    study_directory = PROJECT_ROOT / "results" / "learning_study" / args.tag
    manifest_path = study_directory / "study_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Çalışma manifesti bulunamadı: {manifest_path}\n"
            "Önce training/run_learning_study.py aracını çalıştırın."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    figure_directory = study_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)

    selected_phases = (
        ("reward", "observation") if args.phase == "all" else (args.phase,)
    )
    plotted_experiments: list[dict] = []
    for phase in selected_phases:
        experiments = load_experiments(manifest, phase)
        if not experiments:
            print(f"UYARI: {phase} fazı için çizilecek tamamlanmış kayıt yok.")
            continue
        plotted_experiments.extend(experiments)
        plot_learning_curves(
            experiments,
            phase,
            figure_directory / f"{phase}_learning_curves.png",
        )
        if phase == "reward":
            plot_reward_components(
                experiments,
                figure_directory / "reward_final_components.png",
            )
        diagnostics_created = plot_tensorboard_diagnostics(
            experiments,
            phase,
            figure_directory / f"{phase}_ppo_diagnostics.png",
        )
        if not diagnostics_created:
            print(
                "UYARI: TensorBoard scalarları bulunamadı; PPO tanı grafiği "
                f"oluşturulmadı ({phase})."
            )

    if not plotted_experiments:
        raise RuntimeError("Seçilen fazlarda kullanılabilir değerlendirme yok.")

    # Reward ve observation fazları ayrı günlerde çalıştırılsa bile ortak CSV
    # daha önce tamamlanmış fazı kaybetmemelidir. Manifestte veri bulunan bütün
    # fazları yeniden toplayıp birleşik tabloyu yazarız.
    csv_experiments: list[dict] = []
    for available_phase in ("reward", "observation"):
        csv_experiments.extend(load_experiments(manifest, available_phase))
    write_csv_outputs(csv_experiments, study_directory)
    write_reward_diagnostics(csv_experiments, study_directory)

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
