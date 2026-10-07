"""
Yerel bilgisayarda PPO eğitimi başlatma / curriculum'a devam etme aracı.

Bu dosya doğrudan çalıştırılır. Ana 3DOF simülasyon kodunu değiştirmez;
air_to_air_3dof_ppo.py içindeki Gymnasium ortamını kullanır.

İlk, sıfırdan eğitim örneği:

    python training/train_ppo.py --stage 1500m --radius 1500 --steps 50000

Daha sonraki bir checkpoint'ten devam etme örneği:

    python training/train_ppo.py --stage 1000m --radius 1000 --steps 50000 \
        --source models/1500m/best_model.zip --learning-rate 0.0001 \
        --action-std 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import SubprocVecEnv


# Bu dosya training klasöründedir. parents[1] proje ana klasörünü verir:
#
#   C:\Projects\missile_guidance
#
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ana simülasyon dosyasının import edilebilmesi için proje kökünü Python arama
# yoluna ekliyoruz. Böylece dosyayı hangi terminal klasöründen çalıştırırsak
# çalıştıralım import doğru yerde aranır.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from air_to_air_3dof_ppo import (  # noqa: E402
    PPO_GEOMETRY_LEGACY,
    PPO_GEOMETRY_MODES,
    PPO_GEOMETRY_POLAR,
    PPO_GEOMETRY_ROTATED_NOMINAL,
    PPO_GEOMETRY_TARGET_SPHERE,
    PPO_REWARD_PROFILES,
    PPO_REWARD_PROFILE_CURRENT,
    PPO_REWARD_TERM_NAMES,
    PPOEnvironmentConfig,
    RLInterfaceConfig,
    RL_OBSERVATION_LEGACY10,
    RL_OBSERVATION_MARKOV13,
    RL_OBSERVATION_V2,
    apply_ppo_reward_profile,
    build_default_ppo_environment,
    infer_policy_observation_version,
    ppo_reward_profile_description,
    rl_observation_size,
)


def positive_float(value: str) -> float:
    """Komut satırından gelen değerin pozitif float olduğunu doğrular."""
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("Değer sıfırdan büyük olmalıdır.")
    return number


def nonnegative_float(value: str) -> float:
    """Komut satırından gelen değerin negatif olmayan float olduğunu doğrular."""
    number = float(value)
    if number < 0.0:
        raise argparse.ArgumentTypeError("Değer negatif olamaz.")
    return number


def finite_float(value: str) -> float:
    """Komut satırından gelen değerin sonlu float olduğunu doğrular."""
    number = float(value)
    if not np.isfinite(number):
        raise argparse.ArgumentTypeError("Değer sonlu olmalıdır.")
    return number


def probability_float(value: str) -> float:
    """Komut satırından gelen değerin [0, 1] olasılığı olduğunu doğrular."""
    number = finite_float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("Olasılık [0, 1] içinde olmalıdır.")
    return number


def positive_int(value: str) -> int:
    """Komut satırından gelen değerin pozitif int olduğunu doğrular."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Değer sıfırdan büyük olmalıdır.")
    return number


def nonnegative_int(value: str) -> int:
    """Komut satırından gelen değerin negatif olmayan int olduğunu doğrular."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Değer negatif olamaz.")
    return number


def safe_stage_name(value: str) -> str:
    """Stage adını güvenli bir klasör adına dönüştürür."""
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value.strip()
    )
    if not cleaned:
        raise argparse.ArgumentTypeError("Stage adı boş olamaz.")
    return cleaned


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3DOF füze güdümü için PPO curriculum eğitimi",
    )

    parser.add_argument(
        "--stage",
        type=safe_stage_name,
        default="1500m",
        help="Kayıtlarda kullanılacak aşama adı. Varsayılan: 1500m",
    )
    parser.add_argument(
        "--radius",
        type=positive_float,
        default=1500.0,
        help="Eğitim vurma yarıçapı [m]. Varsayılan: 1500",
    )
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=50_000,
        help="Bu çalıştırmada eklenecek PPO adımı. Varsayılan: 50000",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Devam eğitiminde yüklenecek .zip model yolu. İlk eğitimde verilmez.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help=(
            "Bu stage'in yeni PPO ağıyla sıfırdan başlamasını açıkça doğrular. "
            "--source ile birlikte kullanılamaz."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=positive_float,
        default=None,
        help="Öğrenme oranı. Sıfırdan eğitimde 3e-4, devamda 1e-4 kullanılır.",
    )
    parser.add_argument(
        "--action-std",
        type=positive_float,
        default=None,
        help="Başlangıç aksiyon std. Sıfırdan eğitimde 0.50, devamda 0.20.",
    )
    parser.add_argument(
        "--reset-critic",
        action="store_true",
        help=(
            "Kaynak modelin actor/politika ağırlıklarını korur fakat value "
            "network (critic) ağırlıklarını yeniden başlatır. Reward ölçeği "
            "değiştirildikten sonraki ilk devam eğitiminde kullanılmalıdır."
        ),
    )
    parser.add_argument(
        "--observation-version",
        choices=(
            RL_OBSERVATION_LEGACY10,
            RL_OBSERVATION_MARKOV13,
            RL_OBSERVATION_V2,
        ),
        default=RL_OBSERVATION_V2,
        help=(
            "Kullanılacak RL observation sürümü: legacy10, fiziksel Markov "
            "tamamlamalı markov13 veya tam v2 (23 girdi)."
        ),
    )
    parser.add_argument(
        "--reward-profile",
        choices=PPO_REWARD_PROFILES,
        default=PPO_REWARD_PROFILE_CURRENT,
        help=(
            "Kontrollü reward-ablation profili. 'current' mevcut CLI "
            "ayarlarını korur; diğer profiller yalnız --fresh-start ile "
            "kullanılır."
        ),
    )
    parser.add_argument(
        "--record-initial-evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Sıfırdan oluşturulan rastgele politikayı PPO güncellemesi öncesinde "
            "aynı sabit seed'lerde ölçüp adım 0 kaydı oluştur."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgelelik seed değeri. Varsayılan: 42",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Başlangıç koşullarını rastgeleleştir. İlk curriculum'da kullanmayın.",
    )
    parser.add_argument(
        "--randomization-scale",
        type=nonnegative_float,
        default=1.0,
        help=(
            "PPOEnvironmentConfig jitter değerlerinin ortak çarpanı. "
            "0.25=%%25, 0.50=%%50, 1.0=tam randomizasyon. Varsayılan: 1.0"
        ),
    )
    parser.add_argument(
        "--geometry-mode",
        choices=PPO_GEOMETRY_MODES,
        default=PPO_GEOMETRY_LEGACY,
        help=(
            "Başlangıç geometrisi: legacy eski jitter, rotated_nominal aynı "
            "karşılaşmanın global dönüşü, polar hedef-çevresi curriculum'u, "
            "target_sphere hedef merkezli füze fırlatma küresi."
        ),
    )
    parser.add_argument(
        "--global-rotation-limit-deg",
        type=nonnegative_float,
        default=180.0,
        help="rotated_nominal yaw sınırı [deg]. Varsayılan: 180",
    )
    parser.add_argument(
        "--range-min-m",
        type=positive_float,
        default=4_800.0,
        help="polar/target_sphere başlangıç 3B menzil alt sınırı [m]",
    )
    parser.add_argument(
        "--range-max-m",
        type=positive_float,
        default=5_200.0,
        help="polar/target_sphere başlangıç 3B menzil üst sınırı [m]",
    )
    parser.add_argument(
        "--bearing-min-deg",
        type=finite_float,
        default=-15.0,
        help="polar LOS'un füze burnuna göre minimum yatay açısı [deg]",
    )
    parser.add_argument(
        "--bearing-max-deg",
        type=finite_float,
        default=15.0,
        help="polar LOS'un füze burnuna göre maksimum yatay açısı [deg]",
    )
    parser.add_argument(
        "--elevation-min-deg",
        type=finite_float,
        default=1.0,
        help="polar minimum LOS yükseliş açısı [deg]. Varsayılan: 1",
    )
    parser.add_argument(
        "--elevation-max-deg",
        type=finite_float,
        default=6.0,
        help="polar maksimum LOS yükseliş açısı [deg]. Varsayılan: 6",
    )
    parser.add_argument(
        "--target-aspect-min-deg",
        type=finite_float,
        default=-10.0,
        help="head-on doğrultusuna göre minimum hedef heading sapması [deg]",
    )
    parser.add_argument(
        "--target-aspect-max-deg",
        type=finite_float,
        default=10.0,
        help="head-on doğrultusuna göre maksimum hedef heading sapması [deg]",
    )
    parser.add_argument(
        "--launch-azimuth-min-deg",
        type=finite_float,
        default=-180.0,
        help="target_sphere hedef->füze global azimut alt sınırı [deg]",
    )
    parser.add_argument(
        "--launch-azimuth-max-deg",
        type=finite_float,
        default=180.0,
        help="target_sphere hedef->füze global azimut üst sınırı [deg]",
    )
    parser.add_argument(
        "--launch-elevation-min-deg",
        type=finite_float,
        default=-90.0,
        help="target_sphere hedef->füze elevation alt sınırı [deg]",
    )
    parser.add_argument(
        "--launch-elevation-max-deg",
        type=finite_float,
        default=90.0,
        help="target_sphere hedef->füze elevation üst sınırı [deg]",
    )
    parser.add_argument(
        "--aim-azimuth-error-deg",
        type=nonnegative_float,
        default=0.0,
        help="target_sphere başlangıç yatay nişan hata sınırı ±deg",
    )
    parser.add_argument(
        "--aim-elevation-error-deg",
        type=nonnegative_float,
        default=0.0,
        help="target_sphere başlangıç düşey nişan hata sınırı ±deg",
    )
    parser.add_argument(
        "--sphere-ground-clearance-m",
        type=nonnegative_float,
        default=100.0,
        help="target_sphere en alçak fırlatma noktası yer payı [m]",
    )
    parser.add_argument(
        "--sphere-auto-raise-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Tam küre yerle kesişirse hedef merkez irtifasını otomatik yükselt. "
            "Varsayılan: açık"
        ),
    )
    parser.add_argument(
        "--legacy-replay-probability",
        type=probability_float,
        default=0.0,
        help="polar eğitimde eski dar geometrinin tekrar olasılığı. Varsayılan: 0",
    )
    parser.add_argument(
        "--passed-target-arm-progress-m",
        type=nonnegative_float,
        default=0.0,
        help=(
            "Passed-target sonlandırmasını açmadan önce gereken yaklaşma [m]. "
            "Arka hedef curriculum'u için 100 önerilir; legacy için 0."
        ),
    )
    parser.add_argument(
        "--closest-approach-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reward V4 modu: curriculum yarıçapına girildiğinde episode'u "
            "bitirme; gerçek en yakın geçişe kadar sürdür ve minimum menzili "
            "kesintisiz terminal reward ile puanla. Varsayılan: kapalı"
        ),
    )
    parser.add_argument(
        "--predictive-cpa-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reward V6: sabit-bağıl-hız tahminî CPA mesafesindeki iyileşmeyi "
            "her adımda ödüllendir. --closest-approach-training gerektirir."
        ),
    )
    parser.add_argument(
        "--predictive-cpa-weight",
        type=nonnegative_float,
        default=1.0,
        help="Tahminî CPA ilerleme reward ağırlığı. Varsayılan: 1.0",
    )
    parser.add_argument(
        "--predictive-cpa-scale-m",
        type=positive_float,
        default=100.0,
        help="CPA iyileşmesinin boyutsuzlaştırma ölçeği [m]. Varsayılan: 100",
    )
    parser.add_argument(
        "--predictive-cpa-clip-m",
        type=positive_float,
        default=250.0,
        help="Tek adımlık CPA değişimi kırpma sınırı [m]. Varsayılan: 250",
    )
    parser.add_argument(
        "--predictive-cpa-horizon-s",
        type=positive_float,
        default=35.0,
        help="Sabit hız CPA tahmin ufku [s]. Varsayılan: 35",
    )
    parser.add_argument(
        "--priority-azimuth-window",
        action="append",
        nargs=2,
        type=finite_float,
        default=[],
        metavar=("MIN_DEG", "MAX_DEG"),
        help=(
            "Priority Replay V5 için öncelikli target_sphere launch azimut "
            "penceresi. Birden çok ayrık pencere için argümanı tekrarlayın."
        ),
    )
    parser.add_argument(
        "--priority-azimuth-probability",
        type=probability_float,
        default=0.0,
        help=(
            "Episode'un priority pencereler birleşiminden seçilme olasılığı. "
            "Kalan episode'lar ana launch aralığından uniform gelir. "
            "Varsayılan: 0"
        ),
    )
    parser.add_argument(
        "--n-envs",
        type=positive_int,
        default=1,
        help=(
            "Paralel eğitim ortamı sayısı. Varsayılan: 1. "
            "i9-12900H için geniş-geometri eğitim önerisi: 8-12"
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=positive_int,
        default=20,
        help=(
            "Her checkpoint değerlendirmesinde kullanılacak sabit doğrulama "
            "episode sayısı. Varsayılan: 20"
        ),
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=10_000,
        help=(
            "Sabit doğrulama seed grubunun ilk değeri. Varsayılan: 10000. "
            "Toplu nihai test seed'lerinden farklı tutulmalıdır."
        ),
    )
    parser.add_argument(
        "--eval-frequency",
        type=positive_int,
        default=5_000,
        help=(
            "Sabit-seed doğrulamalar arasındaki PPO adımı. Hızlı geniş-geometri "
            "eğitimi için 20000 önerilir. Varsayılan: 5000"
        ),
    )
    parser.add_argument(
        "--eval-sector-count",
        type=nonnegative_int,
        default=0,
        help=(
            "Target-sphere doğrulamasını bu kadar eşit azimut sektörüne "
            "dağıt. 0 eski uniform-seed davranışıdır; yeni geniş geometri "
            "eğitimi için 12 önerilir."
        ),
    )

    return parser.parse_args()


def resolve_source_path(source: str | None) -> Path | None:
    """Göreli model yolunu proje köküne göre çözer ve varlığını kontrol eder."""
    if source is None:
        return None

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path

    # Stable-Baselines3, uzantı verilmezse .zip ekleyebilir. Kullanıcı iki
    # biçimden birini yazabilsin diye ikisini de kontrol ediyoruz.
    candidates = [source_path]
    if source_path.suffix.lower() != ".zip":
        candidates.append(source_path.with_suffix(".zip"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Kaynak PPO modeli bulunamadı. Kontrol edilen yollar:\n" + checked
    )


def normalize_priority_azimuth_windows(
    raw_windows: list[list[float]],
    probability: float,
    launch_min_deg: float,
    launch_max_deg: float,
    geometry_mode: str,
) -> tuple[tuple[float, float], ...]:
    """CLI pencerelerini doğrular, sıralar ve değişmez tuple'a çevirir."""
    windows = tuple(
        sorted(
            (
                (float(minimum_deg), float(maximum_deg))
                for minimum_deg, maximum_deg in raw_windows
            ),
            key=lambda item: item[0],
        )
    )
    priority_enabled = probability > 0.0
    if priority_enabled != bool(windows):
        raise ValueError(
            "Priority Replay için hem --priority-azimuth-probability hem de "
            "en az bir --priority-azimuth-window birlikte verilmelidir."
        )
    if priority_enabled and geometry_mode != PPO_GEOMETRY_TARGET_SPHERE:
        raise ValueError(
            "Priority azimut örneklemesi yalnız --geometry-mode target_sphere "
            "ile kullanılabilir."
        )

    previous_maximum_deg: float | None = None
    for minimum_deg, maximum_deg in windows:
        if maximum_deg <= minimum_deg:
            raise ValueError(
                "Priority pencere maksimumu minimumdan büyük olmalıdır."
            )
        if minimum_deg < launch_min_deg or maximum_deg > launch_max_deg:
            raise ValueError(
                "Priority pencereleri ana launch azimut aralığı içinde kalmalıdır."
            )
        if (
            previous_maximum_deg is not None
            and minimum_deg < previous_maximum_deg
        ):
            raise ValueError("Priority azimut pencereleri çakışmamalıdır.")
        previous_maximum_deg = maximum_deg
    return windows


def make_uniform_evaluation_config(
    training_config: PPOEnvironmentConfig,
) -> PPOEnvironmentConfig:
    """Priority eğitime rağmen doğrulamayı tarafsız uniform dağılımda tutar."""
    return replace(
        training_config,
        target_sphere_priority_azimuth_windows_deg=(),
        target_sphere_priority_probability=0.0,
    )


def build_curriculum_environment(
    ppo_environment_config: PPOEnvironmentConfig,
    hit_radius_m: float,
    observation_version: str = RL_OBSERVATION_V2,
):
    """
    Tek bir curriculum ortamı oluşturur.

    Bu fonksiyonun dosyanın üst seviyesinde bulunması önemlidir. Windows'ta
    SubprocVecEnv her paralel ortamı ayrı Python sürecinde başlatır ve ortam
    kurucu fonksiyonunu yeni sürece aktarır. İç içe yazılmış bir fonksiyon veya
    doğrudan oluşturulmuş bir ortam bazı Windows kurulumlarında aktarılamaz.

    Monitor sarmalaması burada yapılmaz; make_vec_env her eğitim ortamını
    otomatik olarak Monitor ile sarar.
    """
    environment = build_default_ppo_environment(
        ppo_environment_config=ppo_environment_config,
        rl_interface_config=RLInterfaceConfig(
            observation_version=observation_version,
        ),
    )
    environment.environment_template.hit_radius_m = hit_radius_m
    return environment


def apply_randomization_scale(
    config: PPOEnvironmentConfig,
    scale: float,
) -> None:
    """Başlangıç randomizasyon aralıklarını ortak bir katsayıyla ölçekler."""
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


def evaluate_on_fixed_seeds(
    model: PPO,
    environment,
    evaluation_seeds: list[int],
    azimuth_sector_count: int = 0,
) -> dict[str, object]:
    """Bir politikayı her seferinde aynı doğrulama senaryolarında ölçer."""
    episode_rewards: list[float] = []
    minimum_ranges_m: list[float] = []
    terminal_times_s: list[float] = []
    reward_term_episode_values: dict[str, list[float]] = {
        name: [] for name in PPO_REWARD_TERM_NAMES
    }
    hit_count = 0
    sector_hit_counts = np.zeros(azimuth_sector_count, dtype=int)
    sector_episode_counts = np.zeros(azimuth_sector_count, dtype=int)
    sector_minimum_ranges_m: list[list[float]] = [
        [] for _ in range(azimuth_sector_count)
    ]

    original_azimuth_bounds: tuple[float, float] | None = None
    if azimuth_sector_count > 0:
        original_azimuth_bounds = (
            environment.ppo_config.target_sphere_launch_azimuth_min_deg,
            environment.ppo_config.target_sphere_launch_azimuth_max_deg,
        )
        full_azimuth_width_deg = (
            original_azimuth_bounds[1] - original_azimuth_bounds[0]
        )
        sector_width_deg = full_azimuth_width_deg / azimuth_sector_count

    try:
        for episode_index, episode_seed in enumerate(evaluation_seeds):
            sector_index: int | None = None
            if original_azimuth_bounds is not None:
                sector_index = episode_index % azimuth_sector_count
                sector_minimum_deg = (
                    original_azimuth_bounds[0] + sector_index * sector_width_deg
                )
                environment.ppo_config.target_sphere_launch_azimuth_min_deg = (
                    sector_minimum_deg
                )
                environment.ppo_config.target_sphere_launch_azimuth_max_deg = (
                    sector_minimum_deg + sector_width_deg
                )

            observation, info = environment.reset(seed=episode_seed)
            terminated = False
            truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                observation, reward, terminated, truncated, info = (
                    environment.step(action)
                )
                total_reward += float(reward)

            minimum_range_m = float(info["minimum_range_m"])
            is_success = bool(
                info.get("is_success", info["status"] == "hit")
            )
            episode_rewards.append(total_reward)
            minimum_ranges_m.append(minimum_range_m)
            terminal_times_s.append(float(info["terminal_time_s"]))
            episode_reward_terms = info.get("episode_reward_terms", {})
            for term_name in PPO_REWARD_TERM_NAMES:
                reward_term_episode_values[term_name].append(
                    float(episode_reward_terms.get(term_name, 0.0))
                )
            # Reward V4 episode'u başarı eşiğinde hemen bitirmez. Bu nedenle
            # başarı yalnız status metnine değil açık is_success bilgisine dayanır.
            hit_count += int(is_success)
            if sector_index is not None:
                sector_episode_counts[sector_index] += 1
                sector_hit_counts[sector_index] += int(is_success)
                sector_minimum_ranges_m[sector_index].append(minimum_range_m)
    finally:
        if original_azimuth_bounds is not None:
            (
                environment.ppo_config.target_sphere_launch_azimuth_min_deg,
                environment.ppo_config.target_sphere_launch_azimuth_max_deg,
            ) = original_azimuth_bounds

    rewards = np.asarray(episode_rewards, dtype=float)
    minimum_ranges = np.asarray(minimum_ranges_m, dtype=float)
    terminal_times = np.asarray(terminal_times_s, dtype=float)

    metrics: dict[str, object] = {
        "episode_count": float(len(evaluation_seeds)),
        "success_rate": float(hit_count / len(evaluation_seeds)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_minimum_range_m": float(np.mean(minimum_ranges)),
        "median_minimum_range_m": float(np.median(minimum_ranges)),
        "p90_minimum_range_m": float(np.percentile(minimum_ranges, 90.0)),
        "mean_terminal_time_s": float(np.mean(terminal_times)),
        "mean_reward_terms": {
            term_name: float(np.mean(values))
            for term_name, values in reward_term_episode_values.items()
        },
    }
    if azimuth_sector_count > 0:
        sector_success_rates = [
            float(hit_count_value / episode_count_value)
            for hit_count_value, episode_count_value in zip(
                sector_hit_counts,
                sector_episode_counts,
                strict=True,
            )
        ]
        sector_mean_ranges = [
            float(np.mean(values)) for values in sector_minimum_ranges_m
        ]
        metrics.update(
            {
                "azimuth_sector_count": int(azimuth_sector_count),
                "sector_azimuth_bounds_deg": [
                    [
                        float(original_azimuth_bounds[0] + index * sector_width_deg),
                        float(
                            original_azimuth_bounds[0]
                            + (index + 1) * sector_width_deg
                        ),
                    ]
                    for index in range(azimuth_sector_count)
                ],
                "sector_episode_counts": sector_episode_counts.tolist(),
                "sector_success_rates": sector_success_rates,
                "sector_mean_minimum_ranges_m": sector_mean_ranges,
                "worst_sector_success_rate": float(min(sector_success_rates)),
                "worst_sector_mean_minimum_range_m": float(
                    max(sector_mean_ranges)
                ),
            }
        )
    return metrics


def is_validation_better(
    candidate: dict[str, object],
    incumbent: dict[str, object],
) -> bool:
    """Küçük başarı oynamalarının kötü sektörleri/kuyruğu gizlemesini engeller.

    İki yüzde puana kadar başarı farkı örneklem oynaklığı kabul edilir. Dengeli
    azimut doğrulaması açıksa önce en kötü sektörün ortalama minimum menzili,
    ardından genel P90 ve ortalama karşılaştırılır. Böylece birkaç şanslı hit
    uğruna kör sektörleri bozan checkpoint ``best`` seçilmez.
    """
    episode_count = max(int(candidate.get("episode_count", 1.0)), 1)
    sector_robust_selection = (
        "worst_sector_mean_minimum_range_m" in candidate
        and "worst_sector_mean_minimum_range_m" in incumbent
    )
    if sector_robust_selection:
        success_tolerance = max(2.0 / episode_count, 0.02) + 1e-12
    else:
        # V4/V5 ve eski komutlarda mevcut seçim davranışını koru.
        success_tolerance = 1.0 / episode_count + 1e-12
    success_delta = float(candidate["success_rate"]) - float(
        incumbent["success_rate"]
    )
    if success_delta > success_tolerance:
        return True
    if success_delta < -success_tolerance:
        return False

    candidate_worst_sector_mean = candidate.get(
        "worst_sector_mean_minimum_range_m"
    )
    incumbent_worst_sector_mean = incumbent.get(
        "worst_sector_mean_minimum_range_m"
    )
    if (
        candidate_worst_sector_mean is not None
        and incumbent_worst_sector_mean is not None
    ):
        worst_sector_delta_m = float(candidate_worst_sector_mean) - float(
            incumbent_worst_sector_mean
        )
        if worst_sector_delta_m < -0.25:
            return True
        if worst_sector_delta_m > 0.25:
            return False

    p90_delta_m = float(candidate["p90_minimum_range_m"]) - float(
        incumbent["p90_minimum_range_m"]
    )
    if p90_delta_m < -0.05:
        return True
    if p90_delta_m > 0.05:
        return False

    mean_delta_m = float(candidate["mean_minimum_range_m"]) - float(
        incumbent["mean_minimum_range_m"]
    )
    if mean_delta_m < -0.01:
        return True
    if mean_delta_m > 0.01:
        return False

    return float(candidate["mean_reward"]) > float(incumbent["mean_reward"])


class FixedSeedEvalCallback(BaseCallback):
    """
    PPO checkpoint'lerini aynı seed grubu üzerinde adil biçimde karşılaştırır.

    Stable-Baselines3 EvalCallback rastgele ortamda her değerlendirmede farklı
    episode'lar görebilir. Küçük bir örneklemde şanslı checkpoint'in best_model
    seçilmesini önlemek için bu callback her seferinde aynı seed'leri kullanır.
    """

    def __init__(
        self,
        evaluation_environment,
        evaluation_seeds: list[int],
        azimuth_sector_count: int,
        evaluation_frequency_timesteps: int,
        best_model_path: Path,
        log_directory: Path,
        initial_best_metrics: dict[str, object] | None = None,
        initial_metrics_label: str = "source_baseline",
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.evaluation_environment = evaluation_environment
        self.evaluation_seeds = evaluation_seeds
        self.azimuth_sector_count = azimuth_sector_count
        self.evaluation_frequency_timesteps = evaluation_frequency_timesteps
        self.best_model_path = best_model_path
        self.log_directory = log_directory
        self.best_metrics = initial_best_metrics
        self.initial_metrics_label = initial_metrics_label
        self.next_evaluation_timestep = 0
        self.last_evaluation_timestep: int | None = None
        self.history_path = log_directory / "fixed_seed_evaluations.jsonl"
        self.best_metrics_path = log_directory / "best_validation_metrics.json"

    def _write_record(
        self,
        label: str,
        metrics: dict[str, object],
    ) -> None:
        record = {
            "label": label,
            "num_timesteps": int(self.model.num_timesteps),
            **metrics,
        }
        with self.history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _on_training_start(self) -> None:
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.next_evaluation_timestep = (
            int(self.model.num_timesteps) + self.evaluation_frequency_timesteps
        )
        if self.best_metrics is not None:
            self._write_record(self.initial_metrics_label, self.best_metrics)
            self.last_evaluation_timestep = int(self.model.num_timesteps)
            self.best_metrics_path.write_text(
                json.dumps(
                    {
                        "num_timesteps": int(self.model.num_timesteps),
                        **self.best_metrics,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    def _evaluate_and_record(self, label: str) -> dict[str, object]:
        metrics = evaluate_on_fixed_seeds(
            model=self.model,
            environment=self.evaluation_environment,
            evaluation_seeds=self.evaluation_seeds,
            azimuth_sector_count=self.azimuth_sector_count,
        )
        self._write_record(label, metrics)
        self.last_evaluation_timestep = int(self.model.num_timesteps)

        if self.verbose:
            print()
            print("-" * 72)
            print("SABİT-SEED DOĞRULAMA", f"[{label}]")
            print("PPO adımı              :", self.model.num_timesteps)
            print(
                "Başarı oranı           :",
                f"%{100.0 * float(metrics['success_rate']):.2f}",
            )
            print(
                "Ortalama minimum menzil:",
                f"{float(metrics['mean_minimum_range_m']):.3f} m",
            )
            print(
                "P90 minimum menzil     :",
                f"{float(metrics['p90_minimum_range_m']):.3f} m",
            )
            print(
                "Ortalama reward        :",
                f"{float(metrics['mean_reward']):.3f} +/- "
                f"{float(metrics['std_reward']):.3f}",
            )
            if self.azimuth_sector_count > 0:
                sector_rates = metrics["sector_success_rates"]
                print(
                    "Sektör başarıları [%] :",
                    " ".join(
                        f"{100.0 * float(value):.0f}"
                        for value in sector_rates
                    ),
                )
                print(
                    "En kötü sektör ort.   :",
                    f"{float(metrics['worst_sector_mean_minimum_range_m']):.3f} m",
                )

        self.logger.record(
            "fixed_eval/success_rate",
            float(metrics["success_rate"]),
        )
        self.logger.record(
            "fixed_eval/mean_minimum_range_m",
            float(metrics["mean_minimum_range_m"]),
        )
        self.logger.record(
            "fixed_eval/median_minimum_range_m",
            float(metrics["median_minimum_range_m"]),
        )
        self.logger.record(
            "fixed_eval/p90_minimum_range_m",
            float(metrics["p90_minimum_range_m"]),
        )
        self.logger.record(
            "fixed_eval/mean_reward",
            float(metrics["mean_reward"]),
        )
        self.logger.record(
            "fixed_eval/std_reward",
            float(metrics["std_reward"]),
        )
        self.logger.record(
            "fixed_eval/mean_terminal_time_s",
            float(metrics["mean_terminal_time_s"]),
        )
        for term_name, term_value in metrics["mean_reward_terms"].items():
            self.logger.record(
                f"fixed_eval/reward_term_{term_name}",
                float(term_value),
            )

        is_better = (
            self.best_metrics is None
            or is_validation_better(metrics, self.best_metrics)
        )
        if is_better:
            self.best_metrics = metrics.copy()
            self.model.save(str(self.best_model_path))
            self.best_metrics_path.write_text(
                json.dumps(
                    {
                        "num_timesteps": int(self.model.num_timesteps),
                        **metrics,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            if self.verbose:
                print("Yeni sabit-seed best model kaydedildi.")

        if self.verbose:
            print("-" * 72)
        return metrics

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_evaluation_timestep:
            return True

        self._evaluate_and_record("candidate")
        while self.next_evaluation_timestep <= self.num_timesteps:
            self.next_evaluation_timestep += self.evaluation_frequency_timesteps
        return True

    def _on_training_end(self) -> None:
        # İstenen adım rollout sınırının ortasına denk gelebilir. Periyodik son
        # ölçüm ile gerçek final politika farklıysa final modeli de mutlaka aynı
        # sabit seed grubunda ölç ve best seçiminden geçir.
        if self.last_evaluation_timestep != int(self.model.num_timesteps):
            self._evaluate_and_record("final_policy")


def main() -> None:
    args = parse_arguments()
    if args.range_max_m < args.range_min_m:
        raise ValueError("--range-max-m, --range-min-m'den küçük olamaz.")
    if args.bearing_max_deg < args.bearing_min_deg:
        raise ValueError("--bearing-max-deg, --bearing-min-deg'den küçük olamaz.")
    if args.elevation_max_deg < args.elevation_min_deg:
        raise ValueError("--elevation-max-deg, --elevation-min-deg'den küçük olamaz.")
    if args.target_aspect_max_deg < args.target_aspect_min_deg:
        raise ValueError(
            "--target-aspect-max-deg, --target-aspect-min-deg'den küçük olamaz."
        )
    if args.launch_azimuth_max_deg < args.launch_azimuth_min_deg:
        raise ValueError(
            "--launch-azimuth-max-deg, minimum değerden küçük olamaz."
        )
    if args.launch_elevation_max_deg < args.launch_elevation_min_deg:
        raise ValueError(
            "--launch-elevation-max-deg, minimum değerden küçük olamaz."
        )
    if not (
        -90.0 <= args.launch_elevation_min_deg <= 90.0
        and -90.0 <= args.launch_elevation_max_deg <= 90.0
    ):
        raise ValueError("Launch elevation sınırları [-90, 90] içinde olmalıdır.")
    priority_azimuth_windows = normalize_priority_azimuth_windows(
        raw_windows=args.priority_azimuth_window,
        probability=args.priority_azimuth_probability,
        launch_min_deg=args.launch_azimuth_min_deg,
        launch_max_deg=args.launch_azimuth_max_deg,
        geometry_mode=args.geometry_mode,
    )
    if args.fresh_start and args.source is not None:
        raise ValueError("--fresh-start ile --source birlikte kullanılamaz.")
    if (
        args.reward_profile != PPO_REWARD_PROFILE_CURRENT
        and not args.fresh_start
    ):
        raise ValueError(
            "Kontrollü reward profilleri yalnız --fresh-start ile çalıştırılır. "
            "Bu koruma, eski bir modelden gizli ağırlık aktarılmasını engeller."
        )
    if (
        args.reward_profile != PPO_REWARD_PROFILE_CURRENT
        and (args.closest_approach_training or args.predictive_cpa_reward)
    ):
        raise ValueError(
            "--reward-profile kullanılırken closest/CPA bayraklarını ayrıca "
            "vermeyin; profil gerekli reward alanlarını tek başına belirler."
        )
    if args.predictive_cpa_reward and not args.closest_approach_training:
        raise ValueError(
            "--predictive-cpa-reward, --closest-approach-training gerektirir."
        )
    if args.eval_sector_count > 0:
        if args.geometry_mode != PPO_GEOMETRY_TARGET_SPHERE:
            raise ValueError(
                "--eval-sector-count yalnız --geometry-mode target_sphere "
                "ile kullanılabilir."
            )
        if args.eval_episodes < args.eval_sector_count:
            raise ValueError(
                "--eval-episodes, --eval-sector-count değerinden küçük olamaz."
            )
        if args.eval_episodes % args.eval_sector_count != 0:
            raise ValueError(
                "Dengeli doğrulama için --eval-episodes değeri "
                "--eval-sector-count değerinin tam katı olmalıdır."
            )
    source_path = resolve_source_path(args.source)
    continuing_training = source_path is not None
    if args.record_initial_evaluation and continuing_training:
        raise ValueError(
            "--record-initial-evaluation yalnız sıfırdan eğitimde kullanılır."
        )
    if args.reset_critic and not continuing_training:
        raise ValueError(
            "--reset-critic yalnız --source ile devam eğitiminde kullanılır; "
            "sıfırdan modelin critic'i zaten yenidir."
        )

    # İlk eğitim daha fazla keşif ve daha büyük öğrenme oranıyla başlar. Daha
    # önce eğitilmiş modelden devam ederken politika bozulmasın diye değerler
    # otomatik olarak küçültülür. Komut satırında verilen değerler önceliklidir.
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else (1e-4 if continuing_training else 3e-4)
    )
    action_std = (
        args.action_std
        if args.action_std is not None
        else (0.20 if continuing_training else 0.50)
    )

    # Her stage kendi klasörlerini kullanır. Böylece bir sonraki aşama önceki
    # best modeli veya checkpoint'leri ezmez.
    model_directory = PROJECT_ROOT / "models" / args.stage
    checkpoint_directory = PROJECT_ROOT / "checkpoints" / args.stage
    evaluation_directory = PROJECT_ROOT / "logs" / "evaluation" / args.stage
    tensorboard_directory = PROJECT_ROOT / "logs" / "tensorboard"

    for directory in (
        model_directory,
        checkpoint_directory,
        evaluation_directory,
        tensorboard_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # Reward V3 ayarları ana simülasyon içindeki dataclass'tan gelir. Burada
    # yalnızca curriculum senaryosunu seçiyoruz: düz hedef ve başlangıçta sabit
    # başlangıç koşulları.
    randomization_enabled = (
        args.randomize or args.geometry_mode != PPO_GEOMETRY_LEGACY
    )
    environment_config = PPOEnvironmentConfig(
        target_maneuver_kind="straight",
        randomize_initial_conditions=randomization_enabled,
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
        target_sphere_priority_azimuth_windows_deg=(
            priority_azimuth_windows
        ),
        target_sphere_priority_probability=(
            args.priority_azimuth_probability
        ),
        target_sphere_aim_azimuth_error_deg=args.aim_azimuth_error_deg,
        target_sphere_aim_elevation_error_deg=args.aim_elevation_error_deg,
        target_sphere_ground_clearance_m=args.sphere_ground_clearance_m,
        target_sphere_auto_raise_target=args.sphere_auto_raise_target,
        legacy_replay_probability=args.legacy_replay_probability,
        passed_target_arm_progress_m=args.passed_target_arm_progress_m,
        closest_approach_training=args.closest_approach_training,
        predictive_cpa_reward_enabled=args.predictive_cpa_reward,
        predictive_cpa_progress_weight=args.predictive_cpa_weight,
        predictive_cpa_progress_scale_m=args.predictive_cpa_scale_m,
        predictive_cpa_progress_clip_m=args.predictive_cpa_clip_m,
        predictive_cpa_horizon_s=args.predictive_cpa_horizon_s,
    )
    reward_profile = apply_ppo_reward_profile(
        config=environment_config,
        profile=args.reward_profile,
    )
    if (
        environment_config.predictive_cpa_reward_enabled
        and not environment_config.closest_approach_training
    ):
        raise RuntimeError(
            "Etkin reward ayarı predictive CPA için closest-approach modunu "
            "açmalıdır."
        )
    apply_randomization_scale(
        config=environment_config,
        scale=args.randomization_scale,
    )
    evaluation_environment_config = (
        make_uniform_evaluation_config(environment_config)
        if priority_azimuth_windows
        else environment_config
    )

    # Yanlışlıkla eski ana kod çalıştırılırsa uzun bir eğitime başlamadan hata ver.
    if not hasattr(environment_config, "reward_scale"):
        raise RuntimeError(
            "Reward V3 bulunamadı. Güncel air_to_air_3dof_ppo.py dosyasını kullanın."
        )

    # Tek ortamda DummyVecEnv, birden fazla ortamda ayrı işlem kullanan
    # SubprocVecEnv oluşturulur. "spawn" Windows için güvenli başlangıç
    # yöntemidir. Her worker aynı fizik modelini bağımsız olarak çalıştırır.
    vector_environment_class = SubprocVecEnv if args.n_envs > 1 else None
    vector_environment_arguments = (
        {"start_method": "spawn"} if args.n_envs > 1 else None
    )

    training_environment = make_vec_env(
        env_id=build_curriculum_environment,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={
            "ppo_environment_config": environment_config,
            "hit_radius_m": args.radius,
            "observation_version": args.observation_version,
        },
        vec_env_cls=vector_environment_class,
        vec_env_kwargs=vector_environment_arguments,
    )

    evaluation_environment = build_curriculum_environment(
        ppo_environment_config=evaluation_environment_config,
        hit_radius_m=args.radius,
        observation_version=args.observation_version,
    )
    evaluation_seeds = [
        args.eval_seed + episode_index
        for episode_index in range(args.eval_episodes)
    ]

    if continuing_training:
        # Önce modeli ortam bağlamadan yalnızca observation uyumluluğunu kontrol
        # etmek için açıyoruz. Kaydedilmiş model tek ortamla oluşturulmuş olsa bile
        # asıl yükleme aşağıda doğrudan yeni vektör ortam verilerek yapılacaktır.
        # Bu iki aşamalı yükleme, SB3'ün set_env sırasında uyguladığı
        # "eski n_envs == yeni n_envs" zorunluluğuna takılmamızı engeller.
        model_for_compatibility_check = PPO.load(
            str(source_path),
            device="cpu",
        )
        source_observation_version = infer_policy_observation_version(
            model_for_compatibility_check
        )
        if source_observation_version != args.observation_version:
            raise RuntimeError(
                "Kaynak model ile eğitim ortamının observation sürümü farklı.\n"
                f"Kaynak: {source_observation_version} "
                f"({rl_observation_size(source_observation_version)} girdi)\n"
                f"Ortam : {args.observation_version} "
                f"({rl_observation_size(args.observation_version)} girdi)\n"
                "legacy10 modeli v2'ye geçirmek için önce şu aracı kullanın:\n"
                "  python training/migrate_observation_v2.py "
                "--source <eski_model.zip> --stage obs_v2_start"
            )
        del model_for_compatibility_check

        # Kaydedilmiş actor ve critic ağırlıklarını bu kez doğrudan istenen
        # sayıdaki eğitim ortamına bağlı olarak yükle. PPO.load(..., env=...)
        # bir ortamlı checkpoint'i 4/8 gibi farklı sayıda paralel ortama güvenli
        # biçimde bağlamanın Stable-Baselines3 tarafından desteklenen yoludur.
        model = PPO.load(
            str(source_path),
            env=training_environment,
            device="cpu",
            tensorboard_log=str(tensorboard_directory),
            learning_rate=learning_rate,
            force_reset=True,
        )
        model.learning_rate = learning_rate
        model.lr_schedule = FloatSchedule(learning_rate)
        for parameter_group in model.policy.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(action_std)))

        model.ent_coef = 0.0

        # Reward sürümü veya reward ölçeği değiştiğinde eski critic, yeni return
        # büyüklüklerine göre yanlış kalibrasyonda kalır. Bu seçenek yalnızca
        # critic dalını yeniden başlatır; füze komutunu üreten actor ağırlıkları
        # aynen korunur. Adam'ın eski momentlerini de temizleyerek yeni critic'in
        # geçmiş reward istatistiklerinden etkilenmesini önlüyoruz.
        if args.reset_critic:
            for module in model.policy.mlp_extractor.value_net.modules():
                reset_parameters = getattr(module, "reset_parameters", None)
                if callable(reset_parameters):
                    reset_parameters()
            model.policy.value_net.reset_parameters()
            model.policy.optimizer.state.clear()

        # Konsoldaki hareketli ortalamaların önceki stage ile karışmasını engelle.
        if getattr(model, "ep_info_buffer", None) is not None:
            model.ep_info_buffer.clear()
        if getattr(model, "ep_success_buffer", None) is not None:
            model.ep_success_buffer.clear()
    else:
        source_observation_version = None
        # Sıfırdan eğitimde iki gizli katmanlı küçük bir MLP kullanılır. Aksiyon
        # dağılımının ilk standart sapması log_std_init ile belirlenir.
        model = PPO(
            policy="MlpPolicy",
            env=training_environment,
            learning_rate=learning_rate,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.20,
            ent_coef=0.0,
            policy_kwargs={
                "net_arch": [128, 128],
                "activation_fn": torch.nn.Tanh,
                "log_std_init": float(np.log(action_std)),
            },
            seed=args.seed,
            device="cpu",
            verbose=1,
            tensorboard_log=str(tensorboard_directory),
        )

    # Özel doğrulama callback'i doğrudan model.num_timesteps değerini izlediği
    # için değerlendirme aralığı gerçek PPO örneği cinsindendir.
    # CheckpointCallback ise callback çağrısı saydığı için n_envs'e bölünür.
    evaluation_frequency_timesteps = args.eval_frequency
    checkpoint_frequency = max(10_000 // args.n_envs, 1)

    # Devam eğitiminde kaynak model, yeni stage başlamadan önce aynı evaluation
    # senaryosunda ölçülür ve ilk best_model olarak kaydedilir. Böylece eğitim
    # sırasında politika kötüleşirse doğrulama callback'i daha kötü bir modeli
    # "bu stage'deki ilk değerlendirme" olduğu için best_model yapamaz.
    baseline_metrics = None
    initial_metrics_label = "source_baseline"
    if continuing_training:
        baseline_metrics = evaluate_on_fixed_seeds(
            model=model,
            environment=evaluation_environment,
            evaluation_seeds=evaluation_seeds,
            azimuth_sector_count=args.eval_sector_count,
        )
        model.save(str(model_directory / "best_model"))
    elif args.record_initial_evaluation:
        baseline_metrics = evaluate_on_fixed_seeds(
            model=model,
            environment=evaluation_environment,
            evaluation_seeds=evaluation_seeds,
            azimuth_sector_count=args.eval_sector_count,
        )
        initial_metrics_label = "initial_random_policy"
        model.save(str(model_directory / "best_model"))

    evaluation_callback = FixedSeedEvalCallback(
        evaluation_environment=evaluation_environment,
        evaluation_seeds=evaluation_seeds,
        azimuth_sector_count=args.eval_sector_count,
        evaluation_frequency_timesteps=evaluation_frequency_timesteps,
        best_model_path=model_directory / "best_model",
        log_directory=evaluation_directory,
        initial_best_metrics=baseline_metrics,
        initial_metrics_label=initial_metrics_label,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_frequency,
        save_path=str(checkpoint_directory),
        name_prefix=f"ppo_{args.stage}",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    callbacks = CallbackList([evaluation_callback, checkpoint_callback])

    starting_timesteps = int(model.num_timesteps)
    starting_std = float(model.policy.log_std.exp().mean().item())

    print()
    print("=" * 72)
    print("PPO EĞİTİMİ")
    print("=" * 72)
    print("Proje klasörü       :", PROJECT_ROOT)
    print("Stage                :", args.stage)
    print("Reward profili       :", reward_profile)
    print("Reward açıklaması    :", ppo_reward_profile_description(reward_profile))
    print(
        "Başlangıç modu       :",
        "SIFIRDAN — yeni actor ve critic"
        if not continuing_training
        else "checkpoint'ten devam",
    )
    print("Vurma yarıçapı       :", args.radius, "m")
    print("Kaynak model         :", source_path or "YOK — sıfırdan eğitim")
    print(
        "Observation          :",
        args.observation_version,
        f"({rl_observation_size(args.observation_version)} girdi)",
    )
    print("Başlangıç PPO adımı  :", starting_timesteps)
    print("Eklenecek adım       :", args.steps)
    print("Öğrenme oranı        :", learning_rate)
    print("Başlangıç action std :", starting_std)
    print("Critic sıfırlandı    :", args.reset_critic)
    print("Randomizasyon        :", randomization_enabled)
    print("Randomizasyon ölçeği :", args.randomization_scale)
    print("Geometri modu        :", args.geometry_mode)
    if args.geometry_mode == PPO_GEOMETRY_ROTATED_NOMINAL:
        print("Global dönüş sınırı  :", args.global_rotation_limit_deg, "deg")
        print("Passed-target arm    :", args.passed_target_arm_progress_m, "m")
    elif args.geometry_mode == PPO_GEOMETRY_POLAR:
        print(
            "Polar menzil aralığı :",
            f"{args.range_min_m:g}-{args.range_max_m:g} m",
        )
        print(
            "Bağıl bearing aralığı:",
            f"{args.bearing_min_deg:g}..{args.bearing_max_deg:g} deg",
        )
        print(
            "Elevation aralığı    :",
            f"{args.elevation_min_deg:g}..{args.elevation_max_deg:g} deg",
        )
        print(
            "Hedef aspect aralığı :",
            f"{args.target_aspect_min_deg:g}..{args.target_aspect_max_deg:g} deg",
        )
        print("Legacy tekrar oranı  :", args.legacy_replay_probability)
        print("Passed-target arm     :", args.passed_target_arm_progress_m, "m")
    elif args.geometry_mode == PPO_GEOMETRY_TARGET_SPHERE:
        print(
            "Küre menzil aralığı  :",
            f"{args.range_min_m:g}-{args.range_max_m:g} m",
        )
        print(
            "Launch azimut aralığı:",
            f"{args.launch_azimuth_min_deg:g}.."
            f"{args.launch_azimuth_max_deg:g} deg",
        )
        print(
            "Launch elev. aralığı :",
            f"{args.launch_elevation_min_deg:g}.."
            f"{args.launch_elevation_max_deg:g} deg",
        )
        print(
            "Başlangıç nişan hata :",
            f"az ±{args.aim_azimuth_error_deg:g} / "
            f"el ±{args.aim_elevation_error_deg:g} deg",
        )
        print("Küre yer payı        :", args.sphere_ground_clearance_m, "m")
        print("Hedef irtifa düzeltme:", args.sphere_auto_raise_target)
        print("Legacy tekrar oranı  :", args.legacy_replay_probability)
        print("Passed-target arm     :", args.passed_target_arm_progress_m, "m")
    print("Priority Replay V5   :", bool(priority_azimuth_windows))
    if priority_azimuth_windows:
        priority_window_text = ", ".join(
            f"[{minimum_deg:g}, {maximum_deg:g})"
            for minimum_deg, maximum_deg in priority_azimuth_windows
        )
        full_width_deg = (
            args.launch_azimuth_max_deg - args.launch_azimuth_min_deg
        )
        priority_width_deg = sum(
            maximum_deg - minimum_deg
            for minimum_deg, maximum_deg in priority_azimuth_windows
        )
        expected_priority_region_fraction = (
            args.priority_azimuth_probability
            + (1.0 - args.priority_azimuth_probability)
            * priority_width_deg
            / full_width_deg
        )
        print("Priority pencereler  :", priority_window_text)
        print(
            "Priority dal olasılığı:",
            f"%{100.0 * args.priority_azimuth_probability:.2f}",
        )
        print(
            "Uniform replay oranı :",
            f"%{100.0 * (1.0 - args.priority_azimuth_probability):.2f}",
        )
        print(
            "Zor bölgede beklenen :",
            f"%{100.0 * expected_priority_region_fraction:.2f}",
        )
        print("Doğrulama dağılımı   : uniform ana launch aralığı")
    print("Paralel ortam sayısı :", args.n_envs)
    print("Rollout örnek sayısı :", int(model.n_steps) * args.n_envs)
    print("Doğrulama aralığı    :", evaluation_frequency_timesteps, "adım")
    print("Doğrulama episode'u  :", args.eval_episodes)
    print("Dengeli azimut sektörü:", args.eval_sector_count)
    print(
        "Closest-approach V4  :",
        environment_config.closest_approach_training,
    )
    print(
        "Predictive CPA V6    :",
        environment_config.predictive_cpa_reward_enabled,
    )
    if environment_config.predictive_cpa_reward_enabled:
        print(
            "CPA progress ağırlığı:",
            environment_config.predictive_cpa_progress_weight,
        )
        print(
            "CPA reward ölçeği    :",
            environment_config.predictive_cpa_progress_scale_m,
            "m",
        )
        print(
            "CPA adım kırpması    :",
            environment_config.predictive_cpa_progress_clip_m,
            "m",
        )
        print(
            "CPA tahmin ufku      :",
            environment_config.predictive_cpa_horizon_s,
            "s",
        )
    print("Reward ölçeği        :", environment_config.reward_scale)
    print("Hit reward (ham)      :", environment_config.hit_reward)
    print("Anlık menzil ağırlığı:", environment_config.range_progress_weight)
    print(
        "Best menzil ağırlığı :",
        environment_config.minimum_range_progress_weight,
    )
    print("Kontrol cezası        :", environment_config.control_effort_penalty)
    print("Zaman cezası          :", environment_config.time_penalty)
    print(
        "Passed-target cezası:",
        environment_config.passed_target_penalty,
    )
    print(
        "Miss-distance ağırlığı:",
        environment_config.miss_distance_penalty_weight,
    )
    print(
        "Doğrulama seedleri   :",
        f"{evaluation_seeds[0]}-{evaluation_seeds[-1]}",
    )
    if baseline_metrics is not None:
        print(
            "Başlangıç başarı oranı:",
            f"%{100.0 * baseline_metrics['success_rate']:.2f}",
        )
        print(
            "Başlangıç ort. menzil:",
            f"{baseline_metrics['mean_minimum_range_m']:.3f} m",
        )
    print("=" * 72)

    model.learn(
        total_timesteps=args.steps,
        reset_num_timesteps=not continuing_training,
        callback=callbacks,
        tb_log_name=f"PPO_{args.stage}",
    )

    final_model_path = model_directory / "final_model"
    model.save(str(final_model_path))

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": args.stage,
        "hit_radius_m": args.radius,
        "requested_additional_timesteps": args.steps,
        "starting_timesteps": starting_timesteps,
        "ending_timesteps": int(model.num_timesteps),
        "source_model": str(source_path) if source_path else None,
        "fresh_start_requested": args.fresh_start,
        "record_initial_evaluation": args.record_initial_evaluation,
        "initial_metrics_label": (
            initial_metrics_label if baseline_metrics is not None else None
        ),
        "source_observation_version": source_observation_version,
        "observation_version": args.observation_version,
        "observation_size": rl_observation_size(args.observation_version),
        "learning_rate": learning_rate,
        "starting_action_std": starting_std,
        "ending_action_std": float(model.policy.log_std.exp().mean().item()),
        "critic_reset": args.reset_critic,
        "randomize_initial_conditions": randomization_enabled,
        "randomization_scale": args.randomization_scale,
        "initial_geometry_mode": args.geometry_mode,
        "global_rotation_limit_deg": args.global_rotation_limit_deg,
        "polar_range_min_m": args.range_min_m,
        "polar_range_max_m": args.range_max_m,
        "polar_bearing_min_deg": args.bearing_min_deg,
        "polar_bearing_max_deg": args.bearing_max_deg,
        "polar_elevation_min_deg": args.elevation_min_deg,
        "polar_elevation_max_deg": args.elevation_max_deg,
        "polar_target_aspect_min_deg": args.target_aspect_min_deg,
        "polar_target_aspect_max_deg": args.target_aspect_max_deg,
        "target_sphere_range_min_m": args.range_min_m,
        "target_sphere_range_max_m": args.range_max_m,
        "target_sphere_launch_azimuth_min_deg": args.launch_azimuth_min_deg,
        "target_sphere_launch_azimuth_max_deg": args.launch_azimuth_max_deg,
        "target_sphere_launch_elevation_min_deg": args.launch_elevation_min_deg,
        "target_sphere_launch_elevation_max_deg": args.launch_elevation_max_deg,
        "target_sphere_priority_azimuth_windows_deg": (
            priority_azimuth_windows
        ),
        "target_sphere_priority_probability": (
            args.priority_azimuth_probability
        ),
        "evaluation_target_sphere_priority_probability": 0.0,
        "evaluation_geometry_sampling": (
            "uniform_main_launch_range"
            if priority_azimuth_windows
            else "same_as_training"
        ),
        "target_sphere_aim_azimuth_error_deg": args.aim_azimuth_error_deg,
        "target_sphere_aim_elevation_error_deg": args.aim_elevation_error_deg,
        "target_sphere_ground_clearance_m": args.sphere_ground_clearance_m,
        "target_sphere_auto_raise_target": args.sphere_auto_raise_target,
        "legacy_replay_probability": args.legacy_replay_probability,
        "passed_target_arm_progress_m": args.passed_target_arm_progress_m,
        "reward_profile": reward_profile,
        "reward_profile_description": ppo_reward_profile_description(
            reward_profile
        ),
        "range_progress_weight": environment_config.range_progress_weight,
        "range_progress_scale_m": environment_config.range_progress_scale_m,
        "minimum_range_progress_weight": (
            environment_config.minimum_range_progress_weight
        ),
        "minimum_range_progress_scale_m": (
            environment_config.minimum_range_progress_scale_m
        ),
        "control_effort_penalty": environment_config.control_effort_penalty,
        "time_penalty": environment_config.time_penalty,
        "closest_approach_training": (
            environment_config.closest_approach_training
        ),
        "predictive_cpa_reward_enabled": (
            environment_config.predictive_cpa_reward_enabled
        ),
        "predictive_cpa_progress_weight": (
            environment_config.predictive_cpa_progress_weight
        ),
        "predictive_cpa_progress_scale_m": (
            environment_config.predictive_cpa_progress_scale_m
        ),
        "predictive_cpa_progress_clip_m": (
            environment_config.predictive_cpa_progress_clip_m
        ),
        "predictive_cpa_horizon_s": (
            environment_config.predictive_cpa_horizon_s
        ),
        "parallel_environments": args.n_envs,
        "rollout_samples": int(model.n_steps) * args.n_envs,
        "reward_version": (
            "v6_predictive_cpa"
            if environment_config.predictive_cpa_reward_enabled
            else (
                "v4_closest_approach"
                if environment_config.closest_approach_training
                else "v3"
            )
        ),
        "geometry_sampling_version": (
            "v5_priority_replay"
            if priority_azimuth_windows
            else "uniform"
        ),
        "reward_scale": environment_config.reward_scale,
        "hit_reward_unscaled": environment_config.hit_reward,
        "passed_target_penalty_unscaled": (
            environment_config.passed_target_penalty
        ),
        "miss_distance_penalty_weight_unscaled": (
            environment_config.miss_distance_penalty_weight
        ),
        "miss_distance_penalty_cap_unscaled": (
            environment_config.miss_distance_penalty_cap
        ),
        "evaluation_episode_count": args.eval_episodes,
        "evaluation_azimuth_sector_count": args.eval_sector_count,
        "evaluation_frequency_timesteps": evaluation_frequency_timesteps,
        "evaluation_first_seed": evaluation_seeds[0],
        "evaluation_last_seed": evaluation_seeds[-1],
        "source_validation_metrics": (
            baseline_metrics if continuing_training else None
        ),
        "initial_random_policy_metrics": (
            baseline_metrics
            if args.record_initial_evaluation and not continuing_training
            else None
        ),
        "seed": args.seed,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
    }
    metadata_path = model_directory / "training_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("EĞİTİM TAMAMLANDI")
    print("=" * 72)
    print("Test edilecek best model:")
    print(model_directory / "best_model.zip")
    print()
    print("Son adım modeli:")
    print(final_model_path.with_suffix(".zip"))
    print()
    print("Eğitim bilgileri:")
    print(metadata_path)
    print("=" * 72)

    training_environment.close()
    evaluation_environment.close()


if __name__ == "__main__":
    main()
