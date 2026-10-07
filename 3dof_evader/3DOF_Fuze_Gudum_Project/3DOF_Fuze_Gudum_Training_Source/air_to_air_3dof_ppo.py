from __future__ import annotations

import sys
from pathlib import Path

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from air_to_air_3dof_core import (
G0,DEG,NORTH,EAST,DOWN,SPEED,CHI,GAMMA,MASS,POSITION,
RL_OBSERVATION_LEGACY10, RL_OBSERVATION_MARKOV13, RL_OBSERVATION_V2, RL_OBSERVATION_SIZES,
EnvironmentConfig,InitialState,AeroConfig,PropulsionConfig,ManeuverConfig,VehicleSpec,SimulationResult,
wrap_angle_deg,make_initial_state,flight_path_basis,velocity_from_state,altitude_from_down,
air_density_from_down,drag_force_n,thrust_and_mass_flow,vehicle_derivative,rk4_step,
GuidanceObservation,GuidanceOutput,GuidanceLaw,
PNConfig,VectorPNGuidance,
RLInterfaceConfig,canonical_rl_observation_version,rl_observation_size,infer_rl_observation_version,infer_policy_observation_version,
default_rl_observation,build_normalized_rl_observation,rl_action_to_normal_acceleration,RLGuidanceAdapter,
smooth_random_wave,aircraft_maneuver_command,
closest_approach_on_step,validate_scenario,run_simulation,
SampleAndHoldGuidance
)

PPO_REWARD_PROFILE_CURRENT = "current"
PPO_REWARD_PROFILE_WEAK_BINARY = "weak_binary"
PPO_REWARD_PROFILE_SPARSE = "sparse"
PPO_REWARD_PROFILE_RANGE_PROGRESS = "range_progress"
PPO_REWARD_PROFILE_BEST_RANGE = "best_range"
PPO_REWARD_PROFILE_TERMINAL_DISTANCE = "terminal_distance"
PPO_REWARD_PROFILE_PREDICTIVE_CPA = "predictive_cpa"
PPO_REWARD_PROFILES = (
    PPO_REWARD_PROFILE_CURRENT,
    PPO_REWARD_PROFILE_WEAK_BINARY,
    PPO_REWARD_PROFILE_SPARSE,
    PPO_REWARD_PROFILE_RANGE_PROGRESS,
    PPO_REWARD_PROFILE_BEST_RANGE,
    PPO_REWARD_PROFILE_TERMINAL_DISTANCE,
    PPO_REWARD_PROFILE_PREDICTIVE_CPA,
)
PPO_REWARD_TERM_NAMES = (
    "range_progress",
    "minimum_range_progress",
    "predictive_cpa_progress",
    "control_effort",
    "time",
    "terminal",
    "terminal_accuracy",
    "miss_distance",
)
PPO_GEOMETRY_LEGACY = "legacy"
PPO_GEOMETRY_ROTATED_NOMINAL = "rotated_nominal"
PPO_GEOMETRY_POLAR = "polar"
PPO_GEOMETRY_TARGET_SPHERE = "target_sphere"
PPO_GEOMETRY_SOURCE_TARGET_SPHERE_PRIORITY = "target_sphere_priority"
PPO_GEOMETRY_MODES = (
    PPO_GEOMETRY_LEGACY,
    PPO_GEOMETRY_ROTATED_NOMINAL,
    PPO_GEOMETRY_POLAR,
    PPO_GEOMETRY_TARGET_SPHERE,
)

def sample_target_sphere_launch_azimuth_deg(
    rng: np.random.Generator,
    uniform_min_deg: float,
    uniform_max_deg: float,
    priority_windows_deg: Tuple[Tuple[float, float], ...] = (),
    priority_probability: float = 0.0,
) -> Tuple[float, bool]:
    
    if not 0.0 <= priority_probability <= 1.0:
        raise ValueError("priority_probability [0, 1] içinde olmalıdır.")

    use_priority = bool(
        priority_windows_deg
        and priority_probability > 0.0
        and rng.random() < priority_probability
    )
    if not use_priority:
        return (
            float(rng.uniform(uniform_min_deg, uniform_max_deg)),
            False,
        )

    widths_deg = np.asarray(
        [maximum - minimum for minimum, maximum in priority_windows_deg],
        dtype=float,
    )
    cumulative_widths_deg = np.cumsum(widths_deg)
    window_draw_deg = float(rng.uniform(0.0, cumulative_widths_deg[-1]))
    window_index = min(
        int(np.searchsorted(cumulative_widths_deg, window_draw_deg, side="right")),
        len(priority_windows_deg) - 1,
    )
    minimum_deg, maximum_deg = priority_windows_deg[window_index]
    return float(rng.uniform(minimum_deg, maximum_deg)), True

# =============================================================================
# 4. HEDEF UCAK KONTROLU
# =============================================================================

@lru_cache(maxsize=256)
def _random_wave_parameters(
    random_seed: int,
    channel_id: int,
    harmonic_count: int,
    base_period_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tekrarlanabilir, düzgün rastgele dalga için Fourier parametreleri üretir.

    Gerçek beyaz gürültü her zaman adımında ani ve fiziksel olmayan yön değişimleri
    meydana getirirdi. Bunun yerine birkaç sinüs bileşeni toplanır. Frekanslar,
    fazlar ve ağırlıklar seed kullanılarak rastgele seçilir; sonuç zamanın sürekli
    ve düzgün bir fonksiyonudur.

    channel_id yatay ve düşey kanalların aynı rastgele sayıları kullanmasını
    engeller. Böylece heading-rate ve flight-path-rate birbirinden bağımsız fakat
    ikisi de tekrar üretilebilir sinyaller olur.
    """
    count = max(1, int(harmonic_count))
    period_s = max(float(base_period_s), 1e-3)

    # Büyük asal çarpan, aynı random_seed için iki kanala farklı RNG dizileri verir.
    rng = np.random.default_rng(int(random_seed) + 1009 * int(channel_id))

    # Her bileşenin frekansı ana frekansın rastgele bir katıdır. Bileşenlerin aynı
    # frekansta olmaması sinyalin tek bir düzenli sinüse benzemesini engeller.
    base_angular_frequency = 2.0 * np.pi / period_s
    angular_frequencies = base_angular_frequency * rng.uniform(
        0.55,
        1.65,
        size=count,
    )

    # Her sinüs farklı noktadan başlar.
    random_phases = rng.uniform(0.0, 2.0 * np.pi, size=count)

    # Pozitif ve negatif ağırlıklar kullanılır. Öklid normunun bire getirilmesi,
    # çok sayıdaki bileşenin birbirini tamamen bastırmasını önler ve tek sinüse
    # kıyasla yeterince belirgin rastgele kıvrımlar oluşturur. Nihai birleşim
    # smooth_random_wave içinde tanh ile düzgün biçimde [-1,+1] aralığına alınır.
    weights = rng.uniform(-1.0, 1.0, size=count)
    weight_norm = np.linalg.norm(weights)
    if weight_norm < 1e-12:
        weights[0] = 1.0
        weight_norm = 1.0
    weights = weights / weight_norm

    # Cache içinde yanlışlıkla değiştirilmemeleri için diziler salt-okunur yapılır.
    angular_frequencies.setflags(write=False)
    random_phases.setflags(write=False)
    weights.setflags(write=False)
    return angular_frequencies, random_phases, weights

#grafikler

def set_axes_equal_3d(axis) -> None:
    """
    Matplotlib 3B grafiğinde üç ekseni aynı geometrik ölçekte gösterir.

    Eksenler eşitlenmezse küçük bir düşey hareket, otomatik ölçekleme nedeniyle
    büyük bir manevra gibi görünebilir. Yörünge geometrisini yorumlarken bu durum
    yanıltıcı olacağından aynı kilometre her eksende aynı uzunlukta çizilir.
    """
    x_limits = axis.get_xlim3d()
    y_limits = axis.get_ylim3d()
    z_limits = axis.get_zlim3d()
    ranges = np.array(
        [np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits)]
    )
    centers = np.array(
        [np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)]
    )
    radius = 0.5 * max(ranges.max(), 1.0)

    axis.set_xlim3d(centers[0] - radius, centers[0] + radius)
    axis.set_ylim3d(centers[1] - radius, centers[1] + radius)
    axis.set_zlim3d(centers[2] - radius, centers[2] + radius)


def plot_results(
    result: SimulationResult,
    vehicles: List[VehicleSpec],
) -> None:
    specs = {vehicle.name: vehicle for vehicle in vehicles}
    figure = plt.figure(figsize=(15, 11))

    # 3B grafikte modelin Down koordinatı yerine -Down = irtifa çizilir. Model
    # hâlâ NED'dir; yalnızca fiziksel yörüngeyi daha kolay okumak için üçüncü
    # görsel eksen yukarı-pozitif irtifa olarak gösterilir.
    axis_3d = figure.add_subplot(2, 2, 1, projection="3d")
    for name, history in result.states.items():
        role = specs[name].role
        line_style = "-" if role == "missile" else "--"
        axis_3d.plot(
            history[:, NORTH] / 1000.0,
            history[:, EAST] / 1000.0,
            -history[:, DOWN] / 1000.0,
            line_style,
            linewidth=2.0,
            label=f"{name} ({role})",
        )
        axis_3d.scatter(
            history[0, NORTH] / 1000.0,
            history[0, EAST] / 1000.0,
            -history[0, DOWN] / 1000.0,
            s=40,
        )

    axis_3d.set_xlabel("North [km]")
    axis_3d.set_ylabel("East [km]")
    axis_3d.set_zlabel("irtifa = -Down [km]")
    axis_3d.set_title("NED modelinden 3B fiziksel yörüngeler")
    axis_3d.legend()
    set_axes_equal_3d(axis_3d)

    axis_range = figure.add_subplot(2, 2, 2)
    for name, values in result.range_m.items():
        end_index = result.terminal_index[name]
        axis_range.plot(
            result.time_s[: end_index + 1],
            values[: end_index + 1],
            label=name,
        )
    axis_range.axhline(
        result.hit_radius_m,
        color="tab:red",
        linestyle=":",
        label="vuruş yarıçapı",
    )
    axis_range.set_xlabel("zaman [s]")
    axis_range.set_ylabel("füze-hedef menzili [m]")
    axis_range.set_title("Bağıl menzil")
    axis_range.grid(True)
    axis_range.legend()

    axis_speed = figure.add_subplot(2, 2, 3)
    for name, history in result.states.items():
        axis_speed.plot(result.time_s, history[:, SPEED], label=name)
    axis_speed.set_xlabel("zaman [s]")
    axis_speed.set_ylabel("hız [m/s]")
    axis_speed.set_title("Araç hızları")
    axis_speed.grid(True)
    axis_speed.legend()

    axis_command = figure.add_subplot(2, 2, 4)
    for name, values in result.command_g.items():
        end_index = result.terminal_index[name]
        axis_command.plot(
            result.time_s[: end_index + 1],
            values[: end_index + 1],
            label=name,
        )
    axis_command.set_xlabel("zaman [s]")
    axis_command.set_ylabel("normal ivme komutu [g]")
    axis_command.set_title("Güdüm komutu")
    axis_command.grid(True)
    axis_command.legend()

    figure.tight_layout()
    plt.show()


def plot_vehicle_state_history(
    result: SimulationResult,
    vehicle_name: str,
) -> None:
    """
    Seçilen aracın bütün 3DOF durum değişkenlerini zamana göre çizer.

    Füzenin state'i yedi elemanlıdır: [N,E,D,V,chi,gamma,m]. Down işaretini
    zihinde çevirmek zorunda kalmamak için sekizinci panelde türetilmiş irtifa
    h=-D de ayrıca gösterilir. Açılar grafikte okunabilirlik için dereceye çevrilir;
    simülasyon içinde her zaman radyan tutulur.

    Bu fonksiyon özellikle sinüzoidal hedef manevrası sırasında füzenin nasıl
    tepki verdiğini gözlemek için eklenmiştir. Vuruş gerçekleşmese bile terminal
    zamana kadar bütün durum geçmişi çizilir.
    """
    if vehicle_name not in result.states:
        raise KeyError(
            f"{vehicle_name!r} sonuçlarda yok. Mevcut araçlar: "
            f"{sorted(result.states)}"
        )

    time_s = result.time_s
    state = result.states[vehicle_name]

    # Baş açısı -180/180 sınırından geçtiğinde grafikte yapay sıçrama oluşmaması
    # için önce unwrap yapılır, sonra dereceye dönüştürülür.
    heading_deg = np.rad2deg(np.unwrap(state[:, CHI]))
    flight_path_deg = np.rad2deg(state[:, GAMMA])
    altitude_m = -state[:, DOWN]

    series = [
        (state[:, NORTH], "North N [m]", "tab:blue"),
        (state[:, EAST], "East E [m]", "tab:orange"),
        (state[:, DOWN], "Down D [m]", "tab:green"),
        (altitude_m, "irtifa h=-D [m]", "tab:olive"),
        (state[:, SPEED], "hız V [m/s]", "tab:red"),
        (heading_deg, "baş açısı chi [deg]", "tab:purple"),
        (flight_path_deg, "uçuş-yolu gamma [deg]", "tab:brown"),
        (state[:, MASS], "kütle m [kg]", "tab:gray"),
    ]

    figure, axes = plt.subplots(4, 2, figsize=(15, 14), sharex=True)
    for axis, (values, ylabel, color) in zip(axes.flat, series):
        axis.plot(time_s, values, color=color, linewidth=1.8)
        axis.set_ylabel(ylabel)
        axis.grid(True)

    for axis in axes[-1, :]:
        axis.set_xlabel("zaman [s]")

    figure.suptitle(
        f"{vehicle_name}: NED 3DOF durum geçmişi",
        fontsize=15,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    plt.show()


def plot_ned_position_history(
    result: SimulationResult,
    vehicle_name: str,
) -> None:
    """
    Seçilen aracın North, East ve Down konumlarını üç ayrı grafikte gösterir.

    Bu grafik özellikle hedef uçağın sinüzoidal manevrasını eksen eksen görmek
    için kullanılır. Değerler simülasyondaki gerçek NED işaretleriyle çizilir:

        North : kuzeye doğru pozitif
        East  : doğuya doğru pozitif
        Down  : aşağı doğru pozitif

    Bu nedenle yüksek irtifadaki bir uçak için Down değerinin negatif görünmesi
    normaldir. Örneğin D=-3300 m, h=3300 m irtifa anlamına gelir.
    """
    if vehicle_name not in result.states:
        raise KeyError(
            f"{vehicle_name!r} sonuçlarda yok. Mevcut araçlar: "
            f"{sorted(result.states)}"
        )

    time_s = result.time_s
    state = result.states[vehicle_name]

    # Her NED konumu ayrı eksende çizilir. Aynı figürde fakat ayrı panellerde
    # olmaları, farklı büyüklüklerdeki değişimlerin birbirini ezmesini engeller.
    components = [
        (state[:, NORTH], "North N [m]", "tab:blue"),
        (state[:, EAST], "East E [m]", "tab:orange"),
        (state[:, DOWN], "Down D [m]", "tab:green"),
    ]

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, (values, ylabel, color) in zip(axes, components):
        axis.plot(time_s, values, color=color, linewidth=2.0)
        axis.set_ylabel(ylabel)
        axis.grid(True)

    axes[-1].set_xlabel("zaman [s]")
    figure.suptitle(
        f"{vehicle_name}: ayrı NED konum grafikleri (irtifa h=-D)",
        fontsize=15,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.show()


def plot_aircraft_path_projections(
    result: SimulationResult,
    aircraft_name: str,
) -> None:
    """
    Rastgele nonlinear uçak yolunu üç farklı 2B görünüşte çizer.

    N-E grafiği yörüngeyi üstten, N-irtifa grafiği kuzey doğrultusundan düşey
    hareketi, E-irtifa grafiği ise doğu doğrultusundan düşey hareketi gösterir.
    Üç izdüşüm birlikte incelendiğinde yolun yalnızca yatay değil gerçekten 3B
    ve nonlinear olup olmadığı daha kolay anlaşılır.
    """
    if aircraft_name not in result.states:
        raise KeyError(
            f"{aircraft_name!r} sonuçlarda yok. Mevcut araçlar: "
            f"{sorted(result.states)}"
        )

    state = result.states[aircraft_name]
    north_m = state[:, NORTH]
    east_m = state[:, EAST]
    altitude_m = -state[:, DOWN]

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    projections = [
        (north_m, east_m, "North N [m]", "East E [m]", "Üstten N-E görünüşü"),
        (
            north_m,
            altitude_m,
            "North N [m]",
            "irtifa h=-D [m]",
            "N-irtifa görünüşü",
        ),
        (
            east_m,
            altitude_m,
            "East E [m]",
            "irtifa h=-D [m]",
            "E-irtifa görünüşü",
        ),
    ]

    for axis, (x_values, y_values, xlabel, ylabel, title) in zip(
        axes,
        projections,
    ):
        axis.plot(x_values, y_values, linewidth=2.0)

        # Yeşil başlangıç ve kırmızı bitiş işaretleri yolun izlenme yönünü açıklar.
        axis.scatter(
            x_values[0],
            y_values[0],
            color="tab:green",
            s=55,
            label="başlangıç",
        )
        axis.scatter(
            x_values[-1],
            y_values[-1],
            color="tab:red",
            s=55,
            label="bitiş",
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True)
        axis.legend()

    figure.suptitle(
        f"{aircraft_name}: rastgele düzgün nonlinear 3B yol izdüşümleri",
        fontsize=15,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    plt.show()


# =============================================================================
# 7. KOLAY DEGISTIRILEBILEN ORNEK SENARYO
# =============================================================================


def build_default_scenario():
    """
    Başlangıç ve çevre parametrelerinin düzenleneceği ana bölüm.

    Yeni bir deneme yapmak için öncelikle bu fonksiyondaki sayıları değiştirmeniz
    yeterlidir. Simülasyon, dinamik veya PN sınıflarına dokunmak gerekmez.

    ÖNEMLİ NED HATIRLATMASI:
        position_m = (North, East, Down)
        3000 m irtifa -> Down = -3000 m
    """
    env = EnvironmentConfig(
        dt_s=0.02,
        t_final_s=35.0,
        gravity_mps2=G0,
        rho0_kgpm3=1.225,
        atmosphere_scale_height_m=8500.0,
        ground_altitude_m=0.0,
        hit_radius_m=10.0,
        minimum_flying_speed_mps=60.0,
    )

    aircraft_1 = VehicleSpec(
        name="aircraft_1",
        role="aircraft",
        initial=InitialState(
            # N=5000 m, E=900 m, irtifa=3300 m olduğundan D=-3300 m.
            position_m=(5000.0, 900.0, -3300.0),
            speed_mps=220.0,
            # NED heading: 0 North, 90 East, 180 South.
            azimuth_deg=195.0,
            # Pozitif gamma tırmanıştır; başlangıçta yatay uçuş.
            flight_path_deg=0.0,
            mass_kg=8000.0,
        ),
        aero=AeroConfig(
            reference_area_m2=25.0,
            drag_coefficient=0.030,
            max_normal_accel_g=8.0,
        ),
        propulsion=PropulsionConfig(
            dry_mass_kg=8000.0,
            max_thrust_n=120_000.0,
        ),
        speed_hold_mps=220.0,
        maneuver=ManeuverConfig(
            # "straight" düz uçuş, "sinusoidal" düzenli sinüs, "random_smooth"
            # ise seed ile tekrar üretilebilen rastgele nonlinear yol oluşturur.
            kind="random_smooth",

            # Bunlar doğrudan heading/gamma açıları değil, rastgele dalgaların
            # çarpıldığı chi_dot ve gamma_dot maksimum ölçekleridir.
            heading_rate_deg_s=16.0,
            flight_path_rate_deg_s=6.0,

            # random_smooth modunda tek bir kesin periyot yerine, rastgele Fourier
            # bileşenlerinin üretileceği karakteristik zaman ölçeğidir.
            weave_period_s=5.0,

            # Kullanıcı fazları rastgele üretilen fazların üzerine eklenir.
            heading_phase_deg=0.0,
            flight_path_phase_deg=90.0,

            # Seed değiştirilirse farklı nonlinear yol; aynı bırakılırsa her
            # çalıştırmada birebir aynı yol üretilir.
            random_seed=2026,
            random_harmonics=5,
        ),
    )

    missile_1 = VehicleSpec(
        name="missile_1",
        role="missile",
        target_name="aircraft_1",
        initial=InitialState(
            # Füze N=0, E=0 ve 3000 m irtifadan fırlatılıyor: D=-3000 m.
            position_m=(0.0, 0.0, -3000.0),
            speed_mps=320.0,
            azimuth_deg=5.0,
            # +2 derece, NED sisteminde negatif Down yönüne yani tırmanışa gider.
            flight_path_deg=2.0,
            mass_kg=90.0,
        ),
        aero=AeroConfig(
            reference_area_m2=0.035,
            drag_coefficient=0.35,
            max_normal_accel_g=35.0,
        ),
        propulsion=PropulsionConfig(
            thrust_n=6500.0,
            burn_time_s=5.0,
            mass_flow_kgps=7.0,
            dry_mass_kg=55.0,
            max_thrust_n=6500.0,
        ),
    )

    vehicles = [missile_1, aircraft_1]

    # -------------------------------------------------------------------------
    # GÜDÜM SEÇİMİ SADECE BURADA YAPILIR
    # -------------------------------------------------------------------------
    # Ana simülasyon döngüsü aşağıdaki nesnenin PN mi RL mi olduğunu bilmez.
    # Her füzenin adına karşılık bir GuidanceLaw verilmesi çoklu füze desteğini
    # sağlar. Örneğin iki füze aynı hedefe farklı algoritmalarla gönderilebilir.
    guidance_laws: Dict[str, GuidanceLaw] = {
        "missile_1": VectorPNGuidance(
            PNConfig(
                navigation_constant=4.0,
                max_command_g=30.0,
            )
        )
    }

    # RL modeli eğitildikten sonra yukarıdaki sözlük şu şekilde değiştirilebilir:
    #
    # trained_model = PPO.load("trained_guidance_model")
    # guidance_laws = {
    #     "missile_1": RLGuidanceAdapter(
    #         policy=trained_model,
    #         config=RLInterfaceConfig(max_command_g=30.0),
    #     )
    # }
    #
    # run_simulation, araç dinamiği ve senaryo kodunda başka değişiklik gerekmez.

    return env, vehicles, guidance_laws


# =============================================================================
# 8. GYMNASIUM ORTAMI VE PPO EGITIM ALTYAPISI
# =============================================================================


@dataclass
class PPOEnvironmentConfig:

    control_period_s: float = 0.10
    target_maneuver_kind: Optional[str] = "straight"
    randomize_initial_conditions: bool = True
    missile_position_jitter_m: Tuple[float, float, float] = (40.0, 40.0, 20.0)
    target_position_jitter_m: Tuple[float, float, float] = (200.0, 200.0, 80.0)
    missile_speed_jitter_mps: float = 10.0
    target_speed_jitter_mps: float = 10.0
    missile_heading_jitter_deg: float = 2.0
    target_heading_jitter_deg: float = 5.0
    missile_flight_path_jitter_deg: float = 1.0
    target_flight_path_jitter_deg: float = 2.0

    # target_sphere:
    #   Hedef jitter'lı nominal konumunda ve nominal düz uçuş yönünde kalır.
    #   Füze hedef-merkezli küresel kabukta örneklenir ve burnu hedefe çevrilir.
    #   Tam küre için launch azimuth=-180..180 ve elevation=-90..90 kullanılır.
    initial_geometry_mode: str = PPO_GEOMETRY_LEGACY

    # rotated_nominal modunda uygulanacak yaw açısı uniform olarak
    # [-limit, +limit] aralığından seçilir. 180 derece bütün çemberi kapsar.
    global_rotation_limit_deg: float = 180.0

    # polar modundaki 3B başlangıç kabuğu. Menzil Öklidyen 3B menzildir.
    polar_range_min_m: float = 4_800.0
    polar_range_max_m: float = 5_200.0

    # Hedef LOS azimutunun füze ilk azimutuna göre açısı. [-180, 180] tam yatay
    # çevreyi; küçük simetrik aralıklar aşamalı curriculum'u ifade eder.
    polar_bearing_min_deg: float = -15.0
    polar_bearing_max_deg: float = 15.0

    # Pozitif LOS elevation hedefin füzenin üzerinde olduğunu ifade eder.
    polar_elevation_min_deg: float = 1.0
    polar_elevation_max_deg: float = 6.0

    # Hedef heading'i LOS'un tam tersine çevrildiğinde hedef doğrudan füzeye
    # gelir. Bu değerlere eklenen offset hedefin head-on aspect sapmasıdır.
    polar_target_aspect_min_deg: float = -10.0
    polar_target_aspect_max_deg: float = 10.0

    # target_sphere modunda hedef merkezinden füze fırlatma noktasına olan 3B
    # Öklidyen menzil. İki değer de 5000 ise tam 5 km yarıçaplı yüzey oluşur.
    target_sphere_range_min_m: float = 5_000.0
    target_sphere_range_max_m: float = 5_000.0

    # Hedef -> füze doğrultusunun global N/E azimutu ve yükselişi. Azimut
    # [-180, 180], elevation [-90, 90] olduğunda kürenin bütün yüzeyi kapsanır.
    target_sphere_launch_azimuth_min_deg: float = -180.0
    target_sphere_launch_azimuth_max_deg: float = 180.0
    target_sphere_launch_elevation_min_deg: float = -90.0
    target_sphere_launch_elevation_max_deg: float = 90.0

    # Priority Replay V5: Eğitim episode'larının bir bölümü, tam çemberde zayıf
    # kalan ayrık azimut pencerelerinden seçilebilir. Kalan episode'lar yukarıdaki
    # normal target_sphere aralığından uniform gelir. Boş tuple ve probability=0
    # eski V4/V3 dağılımını birebir korur. Doğrulama tarafında bu alanlar kapatılıp
    # politika her zaman tarafsız uniform tam çemberde ölçülür.
    target_sphere_priority_azimuth_windows_deg: Tuple[
        Tuple[float, float], ...
    ] = ()
    target_sphere_priority_probability: float = 0.0

    # Füze normalde tam hedefe bakar. Bu iki simetrik hata sıfır bırakılırsa ilk
    # hız vektörü LOS ile çakışıktır; curriculum'un ilerleyen aşamasında küçük
    # değerler verilerek fırlatma yönü hatası eklenebilir.
    target_sphere_aim_azimuth_error_deg: float = 0.0
    target_sphere_aim_elevation_error_deg: float = 0.0

    # 5 km'lik tam küre, 3.3 km irtifadaki hedefin altında yer düzlemini keser.
    # auto_raise açıkken hedef merkezinin yalnız irtifası, bütün istenen kabuk
    # yer üstünde kalacak kadar yükseltilir; N/E merkezi ve bütün bağıl geometri
    # değişmez. Böylece küreden hiçbir sektör sessizce atılmaz.
    target_sphere_ground_clearance_m: float = 100.0
    target_sphere_auto_raise_target: bool = True

    # Geniş polar eğitim sırasında eski dar senaryonun unutulmaması için bazı
    # reset'lerde legacy-jitter örneği tutulabilir. Testte 0 kullanılır.
    legacy_replay_probability: float = 0.0

    # random_smooth hedef kullanıldığında her episode'a yeni fakat seed ile
    # tekrar üretilebilir bir Fourier yolu verilir.
    randomize_target_maneuver_seed: bool = True

    # ---------------------------------------------------------------------
    # REWARD KATSAYILARI
    # ---------------------------------------------------------------------
    # Anlık menzil değişimi ajana hangi yönde ilerlediğine dair hızlı bir işaret
    # verir. Bu terimin ağırlığı özellikle küçük tutulur; aksi halde füze hedefe
    # yaklaşıp daha sonra uzaklaşırken önce kazandığı bütün reward'u geri verir.
    range_progress_weight: float = 0.10
    range_progress_scale_m: float = 100.0

    # Episode boyunca ulaşılan EN İYİ (en küçük) menzilin iyileşmesine verilen
    # dense reward. Bu ödül geri alınmaz. Örneğin minimum menzil 80 m'den 70 m'ye
    # düşerse 10 m'lik gerçek bir yakınlaşma başarısı kaydedilmiş olur.
    minimum_range_progress_weight: float = 1.0
    minimum_range_progress_scale_m: float = 100.0

    # Sürekli maksimum komut verilmesini ve gereksiz zikzakları azaltır. İlk
    # aşamada küçük tutulur; fazla büyürse ajan manevra yapmamayı öğrenebilir.
    control_effort_penalty: float = 0.02

    # Her kontrol adımındaki küçük ceza daha kısa önleme sürelerini teşvik eder.
    time_penalty: float = 0.01

    # Reward V3 terminal ödül/cezaları.
    #
    # Önceki sürümde hit_reward=200 ve passed_target_penalty=50 idi. Böylece
    # sınırın yalnızca birkaç metre içinde kalan iki yörüngenin reward'ları
    # arasında yaklaşık 250 puanlık ani bir sıçrama oluşuyordu. Randomize
    # eğitimde critic bu iki tepeli return dağılımını öğrenemediği için
    # explained_variance sıfır/negatif, value_loss ise binler seviyesinde kaldı.
    #
    # V3'te hit hâlâ açık biçimde daha değerlidir fakat sıçrama küçültülmüştür.
    # Iskalama kalitesi aşağıdaki miss_distance cezasıyla sürekli ayrıştırılır.
    hit_reward: float = 50.0
    ground_penalty: float = 200.0
    low_speed_penalty: float = 200.0
    timeout_penalty: float = 50.0

    # Füze hedefe en yakın noktasını geçtikten sonra menzil yeniden büyüyor ve
    # bağıl hareket açılmaya başlıyorsa episode erken bitirilir. Böylece ajan
    # faydasız biçimde 35. saniyeye kadar uçup son menzil üzerinden cezalandırılmaz.
    terminate_after_closest_approach: bool = True
    closest_approach_rebound_m: float = 25.0
    minimum_opening_speed_mps: float = 1.0
    # 0 eski davranışı aynen korur. Pozitif değer kullanıldığında füze başlangıç
    # menzilinden en az bu kadar yaklaşmadan ``passed target`` sonlandırması
    # devreye girmez. Bu, arkada başlayıp önce uzaklaşırken dönmesi gereken
    # senaryoların ilk saniyelerde yanlışlıkla bitmesini engeller.
    passed_target_arm_progress_m: float = 0.0
    passed_target_penalty: float = 0.0

    # Reward V4 closest-approach modu. False bırakıldığında eski V3 davranışı
    # birebir korunur: füze hit_radius içine girer girmez episode biter ve sabit
    # hit bonusu verilir. True olduğunda hit_radius yalnız curriculum başarı
    # eşiğidir; episode en yakın geçiş tamamlanana kadar sürer. Böylece politika
    # eşik içindeki daha hassas yörüngeleri de görür ve 100 -> 75 -> 50 m gibi
    # geçişlerde daha önce hiç görmediği bir terminal bölge oluşmaz.
    closest_approach_training: bool = False

    # Reward V6: mevcut bağıl konum ve hız sabit kalacakmış gibi öngörülen
    # en-yakın-geçiş mesafesindeki (constant-velocity CPA / zero-effort miss)
    # iyileşmeyi her kontrol adımında ödüllendirir. Eski minimum-range shaping
    # yalnız gerçekten ulaşılan yeni menzilleri gördüğü için çapraz geometrilerde
    # doğru/yanlış önleme manevrasını geç ayırır. Bu terim ise ilk saniyeden
    # itibaren yanal önleme kalitesine işaret verir.
    #
    # Varsayılan False olduğundan Reward V5/V4/V3 davranışı aynen korunur.
    predictive_cpa_reward_enabled: bool = False
    predictive_cpa_progress_weight: float = 1.0
    predictive_cpa_progress_scale_m: float = 100.0
    predictive_cpa_progress_clip_m: float = 250.0
    predictive_cpa_horizon_s: float = 35.0

    # Hit olmadan biten episode'da ceza son menzile değil, ulaşılan minimum
    # menzilin vurma yarıçapını ne kadar aştığına bağlanır:
    #
    #   normalized_miss = (minimum_range - hit_radius) / hit_radius
    #   penalty = min(weight * normalized_miss, cap)
    #
    # Böylece 10 m sınırında 20 m'lik yakın geçiş, yüzlerce metrelik bir
    # kaçırmadan belirgin biçimde daha iyi reward alır.
    # Örneğin 50 m vurma yarıçapında 10 m dışarıda kalmak -10 ham puan,
    # 50 m dışarıda kalmak -50 ham puan üretir. Böylece PPO yalnızca hit/miss
    # etiketini değil, ıskalamayı hangi yönde iyileştirdiğini de daha net görür.
    miss_distance_penalty_weight: float = 50.0
    miss_distance_penalty_cap: float = 100.0

    # Critic'in yaklaşık 100-250 büyüklüğündeki return'leri tahmin etmek yerine
    # daha iyi koşullandırılmış 10-25 mertebesinde değerler öğrenmesini sağlar.
    # Bütün reward terimleri aynı katsayıyla çarpıldığı için optimum politika
    # değişmez; yalnızca sayısal öğrenme ölçeği iyileşir.
    reward_scale: float = 0.10

    # Eski notebook hücreleriyle geriye dönük uyumluluk için tutulmaktadır.
    # Reward v2 son menzili artık kullanmaz; minimum menzili kullanır.
    terminal_range_scale_m: float = 1_000.0


def canonical_ppo_reward_profile(profile: str) -> str:
    """Reward çalışma profili adını doğrular ve küçük alias'ları eşler."""
    normalized = str(profile).strip().lower()
    aliases = {
        "current": PPO_REWARD_PROFILE_CURRENT,
        "weak": PPO_REWARD_PROFILE_WEAK_BINARY,
        "binary": PPO_REWARD_PROFILE_WEAK_BINARY,
        "weak_binary": PPO_REWARD_PROFILE_WEAK_BINARY,
        "sparse": PPO_REWARD_PROFILE_SPARSE,
        "range": PPO_REWARD_PROFILE_RANGE_PROGRESS,
        "range_progress": PPO_REWARD_PROFILE_RANGE_PROGRESS,
        "best": PPO_REWARD_PROFILE_BEST_RANGE,
        "best_range": PPO_REWARD_PROFILE_BEST_RANGE,
        "terminal": PPO_REWARD_PROFILE_TERMINAL_DISTANCE,
        "terminal_distance": PPO_REWARD_PROFILE_TERMINAL_DISTANCE,
        "cpa": PPO_REWARD_PROFILE_PREDICTIVE_CPA,
        "predictive_cpa": PPO_REWARD_PROFILE_PREDICTIVE_CPA,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Bilinmeyen PPO reward profili: {profile!r}. "
            f"Desteklenenler: {', '.join(PPO_REWARD_PROFILES)}."
        )
    return aliases[normalized]


def ppo_reward_profile_description(profile: str) -> str:
    """Raporlarda kullanılabilecek kısa, teknik profil açıklamasını döndürür."""
    canonical = canonical_ppo_reward_profile(profile)
    descriptions = {
        PPO_REWARD_PROFILE_CURRENT: (
            "Komut satırındaki mevcut V3/V4/V6 reward ayarlarını korur."
        ),
        PPO_REWARD_PROFILE_WEAK_BINARY: (
            "Naif başlangıç: episode boyunca dense sinyal yok; yalnız hit=+1 "
            "ve bütün terminal başarısızlıklar=-1."
        ),
        PPO_REWARD_PROFILE_SPARSE: (
            "Yalnız terminal hit/fizik olayları ile zaman ve kontrol maliyeti; "
            "yaklaşma veya ıskalama kalitesi için dense sinyal yok."
        ),
        PPO_REWARD_PROFILE_RANGE_PROGRESS: (
            "Sparse reward'a yalnız anlık menzil azalması eklenir; geri "
            "açılmada kazanılan reward geri verilebilir."
        ),
        PPO_REWARD_PROFILE_BEST_RANGE: (
            "Yalnız episode'un yeni minimum menzil rekoru ödüllendirilir; "
            "terminal ıskalama kalitesi henüz ayrıştırılmaz."
        ),
        PPO_REWARD_PROFILE_TERMINAL_DISTANCE: (
            "Anlık ve best-range shaping'e V3 minimum-menzil terminal cezası "
            "eklenir; hit eşiğinde kesikli terminal bonusu sürer."
        ),
        PPO_REWARD_PROFILE_PREDICTIVE_CPA: (
            "V4 sürekli terminal doğruluğu ve V6 predictive-CPA progress ile "
            "tam güncel reward mekanizması."
        ),
    }
    return descriptions[canonical]


def apply_ppo_reward_profile(
    config: PPOEnvironmentConfig,
    profile: str,
) -> str:
    """Seçilen kontrollü reward profilini ``config`` üzerine uygular.

    ``current`` hiçbir alanı değiştirmez ve eski eğitim komutlarının davranışını
    korur. Diğer profiller yalnız reward/terminal değerlendirme alanlarını
    değiştirir; fizik, başlangıç geometrisi, observation ve PPO hiperparametreleri
    aynı kalır. Dönüş değeri canonical profil adıdır.
    """
    canonical = canonical_ppo_reward_profile(profile)
    if canonical == PPO_REWARD_PROFILE_CURRENT:
        return canonical

    # R2-R6 boyunca aynı sayısal ölçek ve fiziksel başarısızlık cezaları
    # korunur. R1 weak_binary ayrı bir kalibrasyon basamağıdır: standart bir
    # acemi tasarımı olan yalnız +1/-1 terminal sinyalini ölçer. Bu fark,
    # manifestte ve raporda açıkça kaydedilir; tek-değişkenli shaping
    # karşılaştırması R2'den itibaren başlar.
    config.reward_scale = 0.10
    config.hit_reward = 50.0
    config.ground_penalty = 200.0
    config.low_speed_penalty = 200.0
    config.timeout_penalty = 50.0
    config.control_effort_penalty = 0.02
    config.time_penalty = 0.01
    config.passed_target_penalty = 0.0
    config.terminate_after_closest_approach = True
    config.closest_approach_training = False
    config.predictive_cpa_reward_enabled = False
    config.predictive_cpa_progress_weight = 1.0
    config.predictive_cpa_progress_scale_m = 100.0
    config.predictive_cpa_progress_clip_m = 250.0
    config.predictive_cpa_horizon_s = 35.0
    config.range_progress_weight = 0.0
    config.minimum_range_progress_weight = 0.0
    config.miss_distance_penalty_weight = 0.0
    config.miss_distance_penalty_cap = 0.0

    if canonical == PPO_REWARD_PROFILE_WEAK_BINARY:
        # Tamamen seyrek ve küçük ölçekli başlangıç: ara adımların tümü 0,
        # başarılı terminal +1, nedeni ne olursa olsun başarısız terminal -1.
        # Bu profil kasıtlı bir bug değil; yaygın bir ilk RL baseline'ıdır.
        config.reward_scale = 1.0
        config.hit_reward = 1.0
        config.ground_penalty = 1.0
        config.low_speed_penalty = 1.0
        config.timeout_penalty = 1.0
        config.passed_target_penalty = 1.0
        config.control_effort_penalty = 0.0
        config.time_penalty = 0.0
    elif canonical == PPO_REWARD_PROFILE_SPARSE:
        pass
    elif canonical == PPO_REWARD_PROFILE_RANGE_PROGRESS:
        # Mevcut V6'daki küçük anlık-menzil ağırlığı korunur. Böylece sonraki
        # profiller kümülatif bir ablation merdiveni oluşturur ve R2 -> R3
        # geçişinde değişen tek shaping alanı minimum-range progress olur.
        config.range_progress_weight = 0.10
        config.minimum_range_progress_weight = 0.0
        config.miss_distance_penalty_weight = 0.0
        config.miss_distance_penalty_cap = 0.0
    elif canonical == PPO_REWARD_PROFILE_BEST_RANGE:
        config.range_progress_weight = 0.10
        config.minimum_range_progress_weight = 1.0
        config.miss_distance_penalty_weight = 0.0
        config.miss_distance_penalty_cap = 0.0
    elif canonical == PPO_REWARD_PROFILE_TERMINAL_DISTANCE:
        config.range_progress_weight = 0.10
        config.minimum_range_progress_weight = 1.0
        config.miss_distance_penalty_weight = 50.0
        config.miss_distance_penalty_cap = 100.0
    elif canonical == PPO_REWARD_PROFILE_PREDICTIVE_CPA:
        config.range_progress_weight = 0.10
        config.minimum_range_progress_weight = 1.0
        config.miss_distance_penalty_weight = 50.0
        config.miss_distance_penalty_cap = 100.0
        config.closest_approach_training = True
        config.predictive_cpa_reward_enabled = True
    else:  # pragma: no cover - canonical doğrulaması savunmacı olarak yinelenir.
        raise RuntimeError(f"İşlenmeyen reward profili: {canonical!r}")

    return canonical


def smooth_closest_approach_terminal_term(
    minimum_range_m: float,
    success_radius_m: float,
    hit_reward: float,
    miss_distance_penalty_weight: float,
    miss_distance_penalty_cap: float,
) -> float:
    """Reward V4 için kesintisiz terminal doğruluk terimini döndürür.

    Eşik içinde ödül merkeze yaklaştıkça doğrusal biçimde artar; eşik dışında
    mevcut V3 miss-distance eğimi aynen sürer. Varsayılan katsayılarla:

        0 * radius -> +50
        1 * radius ->   0
        2 * radius -> -50
        3 * radius -> -100 (cap)

    Böylece örneğin 74.9 m ile 75.1 m arasında yaklaşık 50 ham puanlık bir
    sıçrama kalmaz; buna karşılık daha küçük minimum menzil her zaman daha iyi
    reward üretir.
    """
    if success_radius_m <= 0.0:
        raise ValueError("success_radius_m pozitif olmalıdır.")
    if hit_reward < 0.0:
        raise ValueError("hit_reward negatif olamaz.")
    if miss_distance_penalty_weight < 0.0:
        raise ValueError("miss_distance_penalty_weight negatif olamaz.")
    if miss_distance_penalty_cap < 0.0:
        raise ValueError("miss_distance_penalty_cap negatif olamaz.")

    normalized_range = max(float(minimum_range_m), 0.0) / success_radius_m
    if normalized_range <= 1.0:
        return float(hit_reward * (1.0 - normalized_range))
    return float(
        -min(
            miss_distance_penalty_weight * (normalized_range - 1.0),
            miss_distance_penalty_cap,
        )
    )


def constant_velocity_closest_approach(
    relative_position_m: np.ndarray,
    relative_velocity_mps: np.ndarray,
    horizon_s: float,
) -> Tuple[float, float]:
    """Sabit bağıl hız varsayımıyla öngörülen CPA mesafesi ve zamanını döndürür.

    ``relative_position_m`` hedef eksi füze konumu, ``relative_velocity_mps``
    hedef eksi füze hızıdır. Serbest en-yakın-geçiş zamanı

        t_cpa = -dot(r, v) / dot(v, v)

    olup geçmişe veya episode ufkunun dışına taşmaması için ``[0, horizon]``
    aralığına kırpılır. Bu bir güdüm komutu değildir; yalnız reward için erken
    önleme-kalitesi tahminidir.
    """
    position = np.asarray(relative_position_m, dtype=float).reshape(-1)
    velocity = np.asarray(relative_velocity_mps, dtype=float).reshape(-1)
    if position.size != 3 or velocity.size != 3:
        raise ValueError("CPA hesabı üç boyutlu konum ve hız gerektirir.")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("CPA konum ve hız değerleri sonlu olmalıdır.")
    if not np.isfinite(horizon_s) or horizon_s < 0.0:
        raise ValueError("CPA ufku sonlu ve negatif olmayan bir değer olmalıdır.")

    relative_speed_squared = float(np.dot(velocity, velocity))
    if relative_speed_squared <= 1e-12 or horizon_s <= 0.0:
        time_to_cpa_s = 0.0
    else:
        unconstrained_time_s = -float(np.dot(position, velocity)) / (
            relative_speed_squared
        )
        time_to_cpa_s = float(np.clip(unconstrained_time_s, 0.0, horizon_s))

    predicted_separation = position + velocity * time_to_cpa_s
    return float(np.linalg.norm(predicted_separation)), time_to_cpa_s


def predictive_cpa_progress_term(
    previous_distance_m: float,
    current_distance_m: float,
    weight: float,
    scale_m: float,
    clip_m: float,
) -> float:
    """Öngörülen CPA mesafesindeki adımlık iyileşmeyi boyutsuz reward'a çevirir."""
    if not np.isfinite(previous_distance_m) or not np.isfinite(current_distance_m):
        raise ValueError("CPA reward mesafeleri sonlu olmalıdır.")
    if weight < 0.0:
        raise ValueError("CPA reward ağırlığı negatif olamaz.")
    if scale_m <= 0.0:
        raise ValueError("CPA reward ölçeği pozitif olmalıdır.")
    if clip_m <= 0.0:
        raise ValueError("CPA reward kırpma mesafesi pozitif olmalıdır.")

    improvement_m = float(
        np.clip(previous_distance_m - current_distance_m, -clip_m, clip_m)
    )
    return float(weight * improvement_m / scale_m)


@dataclass
class PPOTrainingConfig:
    total_timesteps: int = 250_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    ent_coef: float = 0.0
    seed: int = 42
    device: str = "auto"
    verbose: int = 1
    tensorboard_log: Optional[str] = "./ppo_logs/"
    model_path: str = "ppo_missile_guidance"


class MissileGuidanceEnv(gym.Env):
    """
    Tek füze-tek hedef 3DOF simülasyonunu Gymnasium ortamına dönüştürür.

    Aksiyon:
        action[0] -> normalize yatay normal ivme, a_chi / a_max
        action[1] -> normalize düşey normal ivme, a_gamma / a_max

    Observation:
        ``legacy10`` -> özgün 10 elemanlı observation
        ``v2``       -> ilk 10 eleman + 13 fizik/terminal-geometri elemanı

    Bu ilk eğitim ortamı bilinçli olarak tek füze ve tek hedef kabul eder. Ana
    run_simulation fonksiyonu çoklu araç desteğini korur. İleride çoklu füze RL
    çalışması ayrı bir multi-agent problem olarak ele alınabilir.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        environment: EnvironmentConfig,
        vehicles: List[VehicleSpec],
        missile_name: str = "missile_1",
        rl_interface_config: Optional[RLInterfaceConfig] = None,
        ppo_environment_config: Optional[PPOEnvironmentConfig] = None,
    ):
        super().__init__()

        self.environment_template = deepcopy(environment)
        self.vehicle_templates = deepcopy(vehicles)
        self.missile_name = missile_name
        self.ppo_config = deepcopy(
            ppo_environment_config
            if ppo_environment_config is not None
            else PPOEnvironmentConfig()
        )
        self.rl_config = deepcopy(
            rl_interface_config
            if rl_interface_config is not None
            else RLInterfaceConfig()
        )
        self.rl_config.observation_version = canonical_rl_observation_version(
            self.rl_config.observation_version
        )

        template_specs = {
            vehicle.name: vehicle
            for vehicle in self.vehicle_templates
        }
        if missile_name not in template_specs:
            raise KeyError(f"PPO füze adı bulunamadı: {missile_name!r}")

        missile_template = template_specs[missile_name]
        if missile_template.role != "missile":
            raise ValueError(f"{missile_name!r} role='missile' olmalıdır.")
        self.target_name = missile_template.target_name
        if self.target_name not in template_specs:
            raise ValueError("PPO füzesi geçerli bir target_name gerektirir.")
        if template_specs[self.target_name].role != "aircraft":
            raise ValueError("İlk PPO ortamında hedef role='aircraft' olmalıdır.")

        expected_names = {self.missile_name, self.target_name}
        if set(template_specs) != expected_names:
            raise ValueError(
                "İlk MissileGuidanceEnv tam olarak bir füze ve onun hedef "
                "uçağını kabul eder. Çoklu araç ana simülatörde kullanılabilir."
            )

        if self.ppo_config.control_period_s <= 0.0:
            raise ValueError("control_period_s pozitif olmalıdır.")
        if self.environment_template.dt_s <= 0.0:
            raise ValueError("Fizik dt_s değeri pozitif olmalıdır.")
        if self.ppo_config.range_progress_scale_m <= 0.0:
            raise ValueError("range_progress_scale_m pozitif olmalıdır.")
        if self.ppo_config.minimum_range_progress_scale_m <= 0.0:
            raise ValueError("minimum_range_progress_scale_m pozitif olmalıdır.")
        if self.ppo_config.terminal_range_scale_m <= 0.0:
            raise ValueError("terminal_range_scale_m pozitif olmalıdır.")
        if self.ppo_config.closest_approach_rebound_m < 0.0:
            raise ValueError("closest_approach_rebound_m negatif olamaz.")
        if self.ppo_config.minimum_opening_speed_mps < 0.0:
            raise ValueError("minimum_opening_speed_mps negatif olamaz.")
        if self.ppo_config.passed_target_arm_progress_m < 0.0:
            raise ValueError("passed_target_arm_progress_m negatif olamaz.")
        if self.ppo_config.hit_reward < 0.0:
            raise ValueError("hit_reward negatif olamaz.")
        if self.ppo_config.miss_distance_penalty_weight < 0.0:
            raise ValueError("miss_distance_penalty_weight negatif olamaz.")
        if self.ppo_config.miss_distance_penalty_cap < 0.0:
            raise ValueError("miss_distance_penalty_cap negatif olamaz.")
        if self.ppo_config.predictive_cpa_progress_weight < 0.0:
            raise ValueError("predictive_cpa_progress_weight negatif olamaz.")
        if self.ppo_config.predictive_cpa_progress_scale_m <= 0.0:
            raise ValueError("predictive_cpa_progress_scale_m pozitif olmalıdır.")
        if self.ppo_config.predictive_cpa_progress_clip_m <= 0.0:
            raise ValueError("predictive_cpa_progress_clip_m pozitif olmalıdır.")
        if self.ppo_config.predictive_cpa_horizon_s <= 0.0:
            raise ValueError("predictive_cpa_horizon_s pozitif olmalıdır.")
        if (
            self.ppo_config.predictive_cpa_reward_enabled
            and not self.ppo_config.closest_approach_training
        ):
            raise ValueError(
                "Predictive CPA reward, closest_approach_training gerektirir."
            )
        if self.ppo_config.reward_scale <= 0.0:
            raise ValueError("reward_scale pozitif olmalıdır.")

        geometry_mode = self.ppo_config.initial_geometry_mode
        if geometry_mode not in PPO_GEOMETRY_MODES:
            raise ValueError(
                "initial_geometry_mode şu değerlerden biri olmalıdır: "
                + ", ".join(PPO_GEOMETRY_MODES)
            )
        if self.ppo_config.global_rotation_limit_deg < 0.0:
            raise ValueError("global_rotation_limit_deg negatif olamaz.")
        if self.ppo_config.polar_range_min_m <= 0.0:
            raise ValueError("polar_range_min_m pozitif olmalıdır.")
        if self.ppo_config.polar_range_max_m < self.ppo_config.polar_range_min_m:
            raise ValueError("polar_range_max_m, polar_range_min_m'den küçük olamaz.")
        if (
            self.ppo_config.polar_bearing_max_deg
            < self.ppo_config.polar_bearing_min_deg
        ):
            raise ValueError(
                "polar_bearing_max_deg, polar_bearing_min_deg'den küçük olamaz."
            )
        if (
            self.ppo_config.polar_elevation_max_deg
            < self.ppo_config.polar_elevation_min_deg
        ):
            raise ValueError(
                "polar_elevation_max_deg, polar_elevation_min_deg'den küçük olamaz."
            )
        if not (
            -89.0 <= self.ppo_config.polar_elevation_min_deg <= 89.0
            and -89.0 <= self.ppo_config.polar_elevation_max_deg <= 89.0
        ):
            raise ValueError("Polar elevation sınırları [-89, 89] içinde olmalıdır.")
        if (
            self.ppo_config.polar_target_aspect_max_deg
            < self.ppo_config.polar_target_aspect_min_deg
        ):
            raise ValueError(
                "polar_target_aspect_max_deg, minimum değerden küçük olamaz."
            )
        if self.ppo_config.target_sphere_range_min_m <= 0.0:
            raise ValueError("target_sphere_range_min_m pozitif olmalıdır.")
        if (
            self.ppo_config.target_sphere_range_max_m
            < self.ppo_config.target_sphere_range_min_m
        ):
            raise ValueError(
                "target_sphere_range_max_m, minimum değerden küçük olamaz."
            )
        if (
            self.ppo_config.target_sphere_launch_azimuth_max_deg
            < self.ppo_config.target_sphere_launch_azimuth_min_deg
        ):
            raise ValueError(
                "target_sphere launch azimuth maksimumu minimumdan küçük olamaz."
            )
        if (
            self.ppo_config.target_sphere_launch_elevation_max_deg
            < self.ppo_config.target_sphere_launch_elevation_min_deg
        ):
            raise ValueError(
                "target_sphere launch elevation maksimumu minimumdan küçük olamaz."
            )
        if not (
            -90.0
            <= self.ppo_config.target_sphere_launch_elevation_min_deg
            <= 90.0
            and -90.0
            <= self.ppo_config.target_sphere_launch_elevation_max_deg
            <= 90.0
        ):
            raise ValueError(
                "Target-sphere launch elevation sınırları [-90, 90] içinde olmalıdır."
            )
        if self.ppo_config.target_sphere_aim_azimuth_error_deg < 0.0:
            raise ValueError("target_sphere aim azimuth hatası negatif olamaz.")
        if self.ppo_config.target_sphere_aim_elevation_error_deg < 0.0:
            raise ValueError("target_sphere aim elevation hatası negatif olamaz.")
        if self.ppo_config.target_sphere_ground_clearance_m < 0.0:
            raise ValueError("target_sphere ground clearance negatif olamaz.")
        priority_probability = (
            self.ppo_config.target_sphere_priority_probability
        )
        if not 0.0 <= priority_probability <= 1.0:
            raise ValueError(
                "target_sphere_priority_probability [0, 1] içinde olmalıdır."
            )
        priority_windows = (
            self.ppo_config.target_sphere_priority_azimuth_windows_deg
        )
        if priority_probability > 0.0 and not priority_windows:
            raise ValueError(
                "Priority olasılığı pozitifse en az bir azimut penceresi gerekir."
            )
        for window in priority_windows:
            if len(window) != 2:
                raise ValueError(
                    "Her priority azimut penceresi (minimum, maksimum) olmalıdır."
                )
            minimum_deg, maximum_deg = (float(value) for value in window)
            if not np.isfinite(minimum_deg) or not np.isfinite(maximum_deg):
                raise ValueError("Priority azimut sınırları sonlu olmalıdır.")
            if maximum_deg <= minimum_deg:
                raise ValueError(
                    "Priority azimut maksimumu minimumdan büyük olmalıdır."
                )
            if (
                minimum_deg
                < self.ppo_config.target_sphere_launch_azimuth_min_deg
                or maximum_deg
                > self.ppo_config.target_sphere_launch_azimuth_max_deg
            ):
                raise ValueError(
                    "Priority azimut pencereleri ana launch azimut aralığının "
                    "içinde kalmalıdır."
                )
        if not 0.0 <= self.ppo_config.legacy_replay_probability <= 1.0:
            raise ValueError("legacy_replay_probability [0, 1] içinde olmalıdır.")

        # Eğitim ve değerlendirme aynı kontrol periyodu ile aynı motor-zaman
        # özelliğini kullanmalıdır. Bu iki alan senaryo şablonundan eşitlenir.
        self.rl_config.control_period_s = self.ppo_config.control_period_s
        self.rl_config.burn_time_s = missile_template.propulsion.burn_time_s

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                rl_observation_size(self.rl_config.observation_version),
            ),
            dtype=np.float32,
        )

        self.render_mode = None
        self.environment: EnvironmentConfig
        self.vehicles: List[VehicleSpec]
        self.specs: Dict[str, VehicleSpec]
        self.states: Dict[str, np.ndarray]
        self.time_s = 0.0
        self.previous_range_m = np.nan
        self.minimum_range_m = np.inf
        self.initial_range_m = np.nan
        self.initial_closing_speed_mps = np.nan
        self.initial_predictive_cpa_distance_m = np.nan
        self.initial_predictive_cpa_time_s = np.nan
        self.previous_predictive_cpa_distance_m = np.nan
        self.initial_missile_position_m = (np.nan, np.nan, np.nan)
        self.initial_target_position_m = (np.nan, np.nan, np.nan)
        self.initial_missile_speed_mps = np.nan
        self.initial_target_speed_mps = np.nan
        self.initial_missile_heading_deg = np.nan
        self.initial_target_heading_deg = np.nan
        self.initial_missile_flight_path_deg = np.nan
        self.initial_los_azimuth_deg = np.nan
        self.initial_relative_bearing_deg = np.nan
        self.initial_los_elevation_deg = np.nan
        self.initial_target_aspect_deg = np.nan
        self.initial_launch_azimuth_deg = np.nan
        self.initial_launch_elevation_deg = np.nan
        self.target_sphere_altitude_shift_m = 0.0
        self._geometry_source = "nominal"
        self._closest_approach_armed = False
        self._success_achieved = False
        self._first_success_time_s = np.nan
        self.status = "not reset"
        self.terminal_time_s = np.nan
        self._episode_done = True
        self._last_reward_terms: Dict[str, float] = {}
        self._episode_reward_terms: Dict[str, float] = {
            name: 0.0 for name in PPO_REWARD_TERM_NAMES
        }

    def _apply_initial_randomization(self) -> None:
        """Yalnızca reset sırasında, Gymnasium'un seed'li RNG'sini kullanır."""
        self._geometry_source = "nominal"
        self.target_sphere_altitude_shift_m = 0.0
        if not self.ppo_config.randomize_initial_conditions:
            return

        missile = self.specs[self.missile_name]
        target = self.specs[self.target_name]

        for vehicle, jitter_m in (
            (missile, self.ppo_config.missile_position_jitter_m),
            (target, self.ppo_config.target_position_jitter_m),
        ):
            jitter = np.asarray(jitter_m, dtype=float)
            if jitter.shape != (3,) or np.any(jitter < 0.0):
                raise ValueError("Konum jitter değerleri üç elemanlı ve >=0 olmalıdır.")
            nominal_position = np.asarray(vehicle.initial.position_m, dtype=float)
            randomized_position = nominal_position + self.np_random.uniform(
                low=-jitter,
                high=jitter,
            )
            vehicle.initial.position_m = tuple(randomized_position)

        missile.initial.speed_mps = max(
            1.0,
            missile.initial.speed_mps
            + self.np_random.uniform(
                -self.ppo_config.missile_speed_jitter_mps,
                self.ppo_config.missile_speed_jitter_mps,
            ),
        )
        target.initial.speed_mps = max(
            1.0,
            target.initial.speed_mps
            + self.np_random.uniform(
                -self.ppo_config.target_speed_jitter_mps,
                self.ppo_config.target_speed_jitter_mps,
            ),
        )

        missile.initial.azimuth_deg += self.np_random.uniform(
            -self.ppo_config.missile_heading_jitter_deg,
            self.ppo_config.missile_heading_jitter_deg,
        )
        target.initial.azimuth_deg += self.np_random.uniform(
            -self.ppo_config.target_heading_jitter_deg,
            self.ppo_config.target_heading_jitter_deg,
        )
        missile.initial.flight_path_deg += self.np_random.uniform(
            -self.ppo_config.missile_flight_path_jitter_deg,
            self.ppo_config.missile_flight_path_jitter_deg,
        )
        target.initial.flight_path_deg += self.np_random.uniform(
            -self.ppo_config.target_flight_path_jitter_deg,
            self.ppo_config.target_flight_path_jitter_deg,
        )

        geometry_mode = self.ppo_config.initial_geometry_mode
        if geometry_mode == PPO_GEOMETRY_LEGACY:
            self._geometry_source = PPO_GEOMETRY_LEGACY
        elif geometry_mode == PPO_GEOMETRY_ROTATED_NOMINAL:
            self._geometry_source = PPO_GEOMETRY_ROTATED_NOMINAL
            yaw_deg = self.np_random.uniform(
                -self.ppo_config.global_rotation_limit_deg,
                self.ppo_config.global_rotation_limit_deg,
            )
            yaw_rad = yaw_deg * DEG
            cosine = float(np.cos(yaw_rad))
            sine = float(np.sin(yaw_rad))

            missile_position = np.asarray(missile.initial.position_m, dtype=float)
            target_position = np.asarray(target.initial.position_m, dtype=float)
            relative_position = target_position - missile_position
            rotated_relative_position = relative_position.copy()
            rotated_relative_position[NORTH] = (
                cosine * relative_position[NORTH]
                - sine * relative_position[EAST]
            )
            rotated_relative_position[EAST] = (
                sine * relative_position[NORTH]
                + cosine * relative_position[EAST]
            )
            target.initial.position_m = tuple(
                missile_position + rotated_relative_position
            )
            missile.initial.azimuth_deg = wrap_angle_deg(
                missile.initial.azimuth_deg + yaw_deg
            )
            target.initial.azimuth_deg = wrap_angle_deg(
                target.initial.azimuth_deg + yaw_deg
            )
        elif geometry_mode == PPO_GEOMETRY_POLAR:
            use_legacy_replay = (
                self.np_random.random()
                < self.ppo_config.legacy_replay_probability
            )
            if use_legacy_replay:
                self._geometry_source = "legacy_replay"
            else:
                self._geometry_source = PPO_GEOMETRY_POLAR
                range_m = self.np_random.uniform(
                    self.ppo_config.polar_range_min_m,
                    self.ppo_config.polar_range_max_m,
                )
                relative_bearing_deg = self.np_random.uniform(
                    self.ppo_config.polar_bearing_min_deg,
                    self.ppo_config.polar_bearing_max_deg,
                )
                elevation_deg = self.np_random.uniform(
                    self.ppo_config.polar_elevation_min_deg,
                    self.ppo_config.polar_elevation_max_deg,
                )
                target_aspect_deg = self.np_random.uniform(
                    self.ppo_config.polar_target_aspect_min_deg,
                    self.ppo_config.polar_target_aspect_max_deg,
                )

                los_azimuth_deg = (
                    missile.initial.azimuth_deg + relative_bearing_deg
                )
                los_azimuth_rad = los_azimuth_deg * DEG
                elevation_rad = elevation_deg * DEG
                horizontal_range_m = range_m * float(np.cos(elevation_rad))
                relative_position = np.array(
                    [
                        horizontal_range_m * float(np.cos(los_azimuth_rad)),
                        horizontal_range_m * float(np.sin(los_azimuth_rad)),
                        -range_m * float(np.sin(elevation_rad)),
                    ],
                    dtype=float,
                )
                missile_position = np.asarray(
                    missile.initial.position_m,
                    dtype=float,
                )
                target_position = missile_position + relative_position
                target_altitude_m = -float(target_position[DOWN])
                if target_altitude_m <= self.environment_template.ground_altitude_m:
                    raise ValueError(
                        "Polar geometri hedefi yer altında üretti. "
                        "Elevation/range sınırlarını veya füze irtifasını düzeltin."
                    )

                target.initial.position_m = tuple(target_position)
                target.initial.azimuth_deg = wrap_angle_deg(
                    los_azimuth_deg + 180.0 + target_aspect_deg
                )
        elif geometry_mode == PPO_GEOMETRY_TARGET_SPHERE:
            use_legacy_replay = (
                self.np_random.random()
                < self.ppo_config.legacy_replay_probability
            )
            if use_legacy_replay:
                self._geometry_source = "legacy_replay"
            else:
                range_m = self.np_random.uniform(
                    self.ppo_config.target_sphere_range_min_m,
                    self.ppo_config.target_sphere_range_max_m,
                )
                launch_azimuth_deg, used_priority_window = (
                    sample_target_sphere_launch_azimuth_deg(
                        rng=self.np_random,
                        uniform_min_deg=(
                            self.ppo_config.target_sphere_launch_azimuth_min_deg
                        ),
                        uniform_max_deg=(
                            self.ppo_config.target_sphere_launch_azimuth_max_deg
                        ),
                        priority_windows_deg=(
                            self.ppo_config
                            .target_sphere_priority_azimuth_windows_deg
                        ),
                        priority_probability=(
                            self.ppo_config
                            .target_sphere_priority_probability
                        ),
                    )
                )
                self._geometry_source = (
                    PPO_GEOMETRY_SOURCE_TARGET_SPHERE_PRIORITY
                    if used_priority_window
                    else PPO_GEOMETRY_TARGET_SPHERE
                )

                # Elevation'ı doğrudan uniform seçmek kutuplarda aşırı örnekleme
                # yapar. sin(elevation) uniform seçildiğinde küre yüzeyindeki alan
                # yoğunluğu eşit olur; her eşit-alan bölgesi eşit olasılıktadır.
                elevation_min_rad = (
                    self.ppo_config.target_sphere_launch_elevation_min_deg * DEG
                )
                elevation_max_rad = (
                    self.ppo_config.target_sphere_launch_elevation_max_deg * DEG
                )
                sin_elevation = self.np_random.uniform(
                    np.sin(elevation_min_rad),
                    np.sin(elevation_max_rad),
                )
                launch_elevation_rad = float(
                    np.arcsin(np.clip(sin_elevation, -1.0, 1.0))
                )
                # Tam alt yarımküre istenirse nominal 3.3 km hedef irtifası
                # yetersizdir. Hedef merkezini bütün olası kabuk ve target Down
                # jitter'ı yer üstünde kalacak kadar bir kez yukarı taşırız.
                target_position = np.asarray(
                    target.initial.position_m,
                    dtype=float,
                )
                minimum_sine = float(np.sin(elevation_min_rad))
                if minimum_sine < 0.0:
                    minimum_altitude_offset_m = (
                        self.ppo_config.target_sphere_range_max_m * minimum_sine
                    )
                else:
                    minimum_altitude_offset_m = (
                        self.ppo_config.target_sphere_range_min_m * minimum_sine
                    )
                required_center_altitude_m = (
                    self.environment_template.ground_altitude_m
                    + self.ppo_config.target_sphere_ground_clearance_m
                    - minimum_altitude_offset_m
                )
                template_target = next(
                    vehicle
                    for vehicle in self.vehicle_templates
                    if vehicle.name == self.target_name
                )
                nominal_target_altitude_m = -float(
                    template_target.initial.position_m[DOWN]
                )
                vertical_jitter_bound_m = float(
                    self.ppo_config.target_position_jitter_m[DOWN]
                )
                if self.ppo_config.target_sphere_auto_raise_target:
                    self.target_sphere_altitude_shift_m = max(
                        0.0,
                        required_center_altitude_m
                        + vertical_jitter_bound_m
                        - nominal_target_altitude_m,
                    )
                    target_position[DOWN] -= self.target_sphere_altitude_shift_m
                    target.initial.position_m = tuple(target_position)

                launch_azimuth_rad = launch_azimuth_deg * DEG
                horizontal_range_m = range_m * float(
                    np.cos(launch_elevation_rad)
                )
                launch_offset = np.array(
                    [
                        horizontal_range_m * float(np.cos(launch_azimuth_rad)),
                        horizontal_range_m * float(np.sin(launch_azimuth_rad)),
                        -range_m * float(np.sin(launch_elevation_rad)),
                    ],
                    dtype=float,
                )
                missile_position = target_position + launch_offset
                missile_altitude_m = -float(missile_position[DOWN])
                minimum_allowed_altitude_m = (
                    self.environment_template.ground_altitude_m
                    + self.ppo_config.target_sphere_ground_clearance_m
                )
                if missile_altitude_m < minimum_allowed_altitude_m - 1e-7:
                    raise ValueError(
                        "Target-sphere fırlatma noktası yer sınırının altında. "
                        "--sphere-auto-raise-target kullanın, hedef irtifasını "
                        "artırın veya elevation sınırını daraltın."
                    )

                # Füze -> hedef LOS'u, hedef -> füze launch offset'inin tersidir.
                los_to_target = -launch_offset
                los_horizontal_m = float(
                    np.hypot(los_to_target[NORTH], los_to_target[EAST])
                )
                los_azimuth_deg = float(
                    np.degrees(
                        np.arctan2(
                            los_to_target[EAST],
                            los_to_target[NORTH],
                        )
                    )
                )
                los_elevation_deg = float(
                    np.degrees(
                        np.arctan2(-los_to_target[DOWN], los_horizontal_m)
                    )
                )
                azimuth_error_deg = self.np_random.uniform(
                    -self.ppo_config.target_sphere_aim_azimuth_error_deg,
                    self.ppo_config.target_sphere_aim_azimuth_error_deg,
                )
                elevation_error_deg = self.np_random.uniform(
                    -self.ppo_config.target_sphere_aim_elevation_error_deg,
                    self.ppo_config.target_sphere_aim_elevation_error_deg,
                )

                missile.initial.position_m = tuple(missile_position)
                missile.initial.azimuth_deg = wrap_angle_deg(
                    los_azimuth_deg + azimuth_error_deg
                )
                missile.initial.flight_path_deg = float(
                    np.clip(
                        los_elevation_deg + elevation_error_deg,
                        -89.999,
                        89.999,
                    )
                )

        # Hedef hız tutucu varsa episode başındaki rastgele hedef hızını korusun.
        if target.speed_hold_mps is not None:
            target.speed_hold_mps = target.initial.speed_mps

    def _get_guidance_observation(self) -> GuidanceObservation:
        return GuidanceObservation(
            time_s=self.time_s,
            dt_s=self.ppo_config.control_period_s,
            missile_state=self.states[self.missile_name].copy(),
            target_state=self.states[self.target_name].copy(),
            environment=self.environment,
        )

    def _get_observation(self) -> np.ndarray:
        return build_normalized_rl_observation(
            observation=self._get_guidance_observation(),
            config=self.rl_config,
        )

    def _engagement_metrics(self) -> Tuple[float, float, float]:
        missile_state = self.states[self.missile_name]
        target_state = self.states[self.target_name]
        relative_position = target_state[POSITION] - missile_state[POSITION]
        relative_velocity = (
            velocity_from_state(target_state)
            - velocity_from_state(missile_state)
        )
        range_m = float(np.linalg.norm(relative_position))
        if range_m < 1e-9:
            return range_m, 0.0, 0.0

        los_unit = relative_position / range_m
        closing_speed_mps = -float(np.dot(relative_velocity, los_unit))
        los_rate_radps = float(
            np.linalg.norm(
                np.cross(relative_position, relative_velocity) / range_m**2
            )
        )
        return range_m, closing_speed_mps, los_rate_radps

    def _predictive_cpa_metrics(self) -> Tuple[float, float]:
        """Mevcut state için sabit-bağıl-hız CPA tahminini döndürür."""
        missile_state = self.states[self.missile_name]
        target_state = self.states[self.target_name]
        relative_position = target_state[POSITION] - missile_state[POSITION]
        relative_velocity = (
            velocity_from_state(target_state)
            - velocity_from_state(missile_state)
        )
        remaining_episode_s = max(
            self.environment.t_final_s - self.time_s,
            0.0,
        )
        horizon_s = min(
            self.ppo_config.predictive_cpa_horizon_s,
            remaining_episode_s,
        )
        return constant_velocity_closest_approach(
            relative_position_m=relative_position,
            relative_velocity_mps=relative_velocity,
            horizon_s=horizon_s,
        )

    def _get_info(self, command_g: float = 0.0) -> Dict[str, object]:
        range_m, closing_speed_mps, los_rate_radps = self._engagement_metrics()
        predictive_cpa_distance_m, predictive_cpa_time_s = (
            self._predictive_cpa_metrics()
        )
        miss_distance_m = max(
            self.minimum_range_m - self.environment.hit_radius_m,
            0.0,
        )
        if self.ppo_config.closest_approach_training:
            is_success = self.minimum_range_m <= self.environment.hit_radius_m
        else:
            is_success = self.status == "hit"
        return {
            "status": self.status,
            "is_success": bool(is_success),
            "closest_approach_training": bool(
                self.ppo_config.closest_approach_training
            ),
            "first_success_time_s": float(self._first_success_time_s),
            "time_s": float(self.time_s),
            "terminal_time_s": float(self.terminal_time_s),
            "range_m": range_m,
            "minimum_range_m": float(self.minimum_range_m),
            "initial_range_m": float(self.initial_range_m),
            "initial_closing_speed_mps": float(self.initial_closing_speed_mps),
            "initial_predictive_cpa_distance_m": float(
                self.initial_predictive_cpa_distance_m
            ),
            "initial_predictive_cpa_time_s": float(
                self.initial_predictive_cpa_time_s
            ),
            "initial_missile_position_m": self.initial_missile_position_m,
            "initial_target_position_m": self.initial_target_position_m,
            "initial_missile_speed_mps": float(self.initial_missile_speed_mps),
            "initial_target_speed_mps": float(self.initial_target_speed_mps),
            "initial_missile_heading_deg": float(
                self.initial_missile_heading_deg
            ),
            "initial_target_heading_deg": float(self.initial_target_heading_deg),
            "initial_missile_flight_path_deg": float(
                self.initial_missile_flight_path_deg
            ),
            "initial_los_azimuth_deg": float(self.initial_los_azimuth_deg),
            "initial_relative_bearing_deg": float(
                self.initial_relative_bearing_deg
            ),
            "initial_los_elevation_deg": float(
                self.initial_los_elevation_deg
            ),
            "initial_target_aspect_deg": float(self.initial_target_aspect_deg),
            "initial_launch_azimuth_deg": float(
                self.initial_launch_azimuth_deg
            ),
            "initial_launch_elevation_deg": float(
                self.initial_launch_elevation_deg
            ),
            "target_sphere_altitude_shift_m": float(
                self.target_sphere_altitude_shift_m
            ),
            "geometry_source": self._geometry_source,
            "closest_approach_armed": bool(self._closest_approach_armed),
            "miss_distance_m": float(miss_distance_m),
            "closing_speed_mps": closing_speed_mps,
            "los_rate_radps": los_rate_radps,
            "predictive_cpa_distance_m": float(predictive_cpa_distance_m),
            "predictive_cpa_time_s": float(predictive_cpa_time_s),
            "command_g": float(command_g),
            "reward_terms": self._last_reward_terms.copy(),
            "episode_reward_terms": self._episode_reward_terms.copy(),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Yeni bir füze-hedef episode'u başlatır."""
        super().reset(seed=seed)
        del options  # İlk sürümde özel reset option kullanılmıyor.

        self.environment = deepcopy(self.environment_template)
        self.vehicles = deepcopy(self.vehicle_templates)
        self.specs = {vehicle.name: vehicle for vehicle in self.vehicles}

        target = self.specs[self.target_name]
        if self.ppo_config.target_maneuver_kind is not None:
            target.maneuver.kind = self.ppo_config.target_maneuver_kind
        if (
            target.maneuver.kind in {"random_smooth", "random_nonlinear", "random"}
            and self.ppo_config.randomize_target_maneuver_seed
        ):
            target.maneuver.random_seed = int(
                self.np_random.integers(0, np.iinfo(np.int32).max)
            )

        self._apply_initial_randomization()
        self.states = {
            name: make_initial_state(vehicle.initial)
            for name, vehicle in self.specs.items()
        }
        self.time_s = 0.0
        self.status = "flying"
        self.terminal_time_s = np.nan
        self._episode_done = False
        self._last_reward_terms = {}
        self._episode_reward_terms = {
            name: 0.0 for name in PPO_REWARD_TERM_NAMES
        }
        self._success_achieved = False
        self._first_success_time_s = np.nan

        initial_range_m, initial_closing_speed_mps, _ = self._engagement_metrics()
        missile_state = self.states[self.missile_name]
        target_state = self.states[self.target_name]
        relative_position = target_state[POSITION] - missile_state[POSITION]
        horizontal_range_m = float(
            np.hypot(relative_position[NORTH], relative_position[EAST])
        )
        los_azimuth_deg = float(
            np.degrees(
                np.arctan2(
                    relative_position[EAST],
                    relative_position[NORTH],
                )
            )
        )
        los_elevation_deg = float(
            np.degrees(
                np.arctan2(-relative_position[DOWN], horizontal_range_m)
            )
        )
        missile_heading_deg = float(np.degrees(missile_state[CHI]))
        missile_flight_path_deg = float(np.degrees(missile_state[GAMMA]))
        target_heading_deg = float(np.degrees(target_state[CHI]))
        launch_position = missile_state[POSITION] - target_state[POSITION]
        launch_horizontal_m = float(
            np.hypot(launch_position[NORTH], launch_position[EAST])
        )
        launch_azimuth_deg = float(
            np.degrees(
                np.arctan2(
                    launch_position[EAST],
                    launch_position[NORTH],
                )
            )
        )
        launch_elevation_deg = float(
            np.degrees(
                np.arctan2(-launch_position[DOWN], launch_horizontal_m)
            )
        )

        self.initial_range_m = initial_range_m
        self.initial_closing_speed_mps = initial_closing_speed_mps
        (
            self.initial_predictive_cpa_distance_m,
            self.initial_predictive_cpa_time_s,
        ) = self._predictive_cpa_metrics()
        self.previous_predictive_cpa_distance_m = (
            self.initial_predictive_cpa_distance_m
        )
        self.initial_missile_position_m = tuple(
            float(value) for value in missile_state[POSITION]
        )
        self.initial_target_position_m = tuple(
            float(value) for value in target_state[POSITION]
        )
        self.initial_missile_speed_mps = float(missile_state[SPEED])
        self.initial_target_speed_mps = float(target_state[SPEED])
        self.initial_missile_heading_deg = missile_heading_deg
        self.initial_missile_flight_path_deg = missile_flight_path_deg
        self.initial_target_heading_deg = target_heading_deg
        self.initial_los_azimuth_deg = wrap_angle_deg(los_azimuth_deg)
        self.initial_relative_bearing_deg = wrap_angle_deg(
            los_azimuth_deg - missile_heading_deg
        )
        self.initial_los_elevation_deg = los_elevation_deg
        self.initial_target_aspect_deg = wrap_angle_deg(
            target_heading_deg - (los_azimuth_deg + 180.0)
        )
        self.initial_launch_azimuth_deg = wrap_angle_deg(launch_azimuth_deg)
        self.initial_launch_elevation_deg = launch_elevation_deg
        self._closest_approach_armed = (
            self.ppo_config.passed_target_arm_progress_m <= 0.0
        )
        self.previous_range_m = initial_range_m
        self.minimum_range_m = initial_range_m
        return self._get_observation(), self._get_info(command_g=0.0)

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, object]]:
        """
        Bir RL kontrol periyodu boyunca aksiyonu uygular ve fiziği ilerletir.

        Reward V3/V4'ün temel farkları:

        1. Anlık menzil azalması küçük bir yönlendirme sinyali olarak korunur.
        2. Episode minimum menzilindeki her iyileşme ayrıca ödüllendirilir ve
           füze daha sonra uzaklaşsa bile bu ödül geri alınmaz.
        3. Hedef geçilip menzil yeniden büyümeye başladığında episode erken biter.
        4. Kaçırma cezası 35. saniyedeki son menzile değil minimum menzile bağlıdır.
        """
        if self._episode_done:
            raise RuntimeError("Episode bitti; step öncesinde reset çağrılmalıdır.")

        normalized_action = np.asarray(action, dtype=float).reshape(-1)
        if normalized_action.size != 2:
            raise ValueError("PPO aksiyonu shape=(2,) olmalıdır.")
        normalized_action = np.clip(normalized_action, -1.0, 1.0)

        controlled_duration_s = min(
            self.ppo_config.control_period_s,
            self.environment.t_final_s - self.time_s,
        )
        remaining_control_time_s = controlled_duration_s
        terminated = False
        truncated = False
        last_command = np.zeros(3)

        # Bu kontrol adımı başlamadan önce episode'un ulaştığı en iyi menzil.
        # Kontrol adımı sonunda bundan daha küçük bir menzil görülürse ajan gerçek
        # bir closest-approach iyileşmesi yapmış demektir.
        previous_minimum_range_m = self.minimum_range_m

        missile = self.specs[self.missile_name]
        target = self.specs[self.target_name]

        while remaining_control_time_s > 1e-12:
            physics_dt_s = min(
                self.environment.dt_s,
                remaining_control_time_s,
                self.environment.t_final_s - self.time_s,
            )
            if physics_dt_s <= 1e-12:
                break

            old_missile_state = self.states[self.missile_name].copy()
            old_target_state = self.states[self.target_name].copy()

            # Normalize aksiyon kontrol periyodu boyunca sabittir. Füze yönü her
            # fizik alt adımında değişebildiği için NED komutu anlık e_chi/e_gamma
            # bazında yeniden hesaplanır.
            last_command = rl_action_to_normal_acceleration(
                action=normalized_action,
                missile_state=old_missile_state,
                environment=self.environment,
                config=self.rl_config,
            )
            target_command = aircraft_maneuver_command(
                time_s=self.time_s,
                state=old_target_state,
                maneuver=target.maneuver,
            )

            new_missile_state = rk4_step(
                time_s=self.time_s,
                state=old_missile_state,
                dt_s=physics_dt_s,
                vehicle=missile,
                env=self.environment,
                normal_accel_command_mps2=last_command,
            )
            new_target_state = rk4_step(
                time_s=self.time_s,
                state=old_target_state,
                dt_s=physics_dt_s,
                vehicle=target,
                env=self.environment,
                normal_accel_command_mps2=target_command,
            )

            relative_start = (
                old_target_state[POSITION] - old_missile_state[POSITION]
            )
            relative_end = (
                new_target_state[POSITION] - new_missile_state[POSITION]
            )
            minimum_step_distance_m, tau = closest_approach_on_step(
                relative_start,
                relative_end,
            )

            self.states[self.missile_name] = new_missile_state
            self.states[self.target_name] = new_target_state
            self.minimum_range_m = min(
                self.minimum_range_m,
                minimum_step_distance_m,
            )
            if (
                self.initial_range_m - self.minimum_range_m
                >= self.ppo_config.passed_target_arm_progress_m
            ):
                self._closest_approach_armed = True

            step_start_time_s = self.time_s
            self.time_s += physics_dt_s
            remaining_control_time_s -= physics_dt_s

            if minimum_step_distance_m <= self.environment.hit_radius_m:
                if not self._success_achieved:
                    self._success_achieved = True
                    self._first_success_time_s = (
                        step_start_time_s + tau * physics_dt_s
                    )

                # V4'te curriculum eşiğine girildiğinde simülasyonu kesmeyiz.
                # Politika gerçek en yakın geçişe kadar uçmaya devam eder ve
                # eşik içindeki 74 m / 10 m / 1 m yörüngeleri ayrıştırılabilir.
                if not self.ppo_config.closest_approach_training:
                    self.status = "hit"
                    self.terminal_time_s = self._first_success_time_s
                    terminated = True

            # V3 hit durumunda yukarıda episode zaten biter. V4 ise eşiğin
            # içinde uçmaya devam ettiği için fiziksel güvenlik sonlandırmaları
            # aynı alt adımda ayrıca denetlenmelidir.
            if (
                not terminated
                and new_missile_state[DOWN]
                >= -self.environment.ground_altitude_m
            ):
                self.status = "miss: ground"
                self.terminal_time_s = self.time_s
                terminated = True
            elif (
                not terminated
                and new_missile_state[SPEED]
                <= self.environment.minimum_flying_speed_mps
            ):
                self.status = "miss: low speed"
                self.terminal_time_s = self.time_s
                terminated = True

            # Hedef geçildiğinde bağıl menzil artar ve closing speed negatif olur.
            # Minimum menzilden küçük bir rebound kadar uzaklaşıldığında episode'u
            # erken bitirmek, yakın geçiş bilgisinin son menzil tarafından
            # bozulmasını ve yüzlerce faydasız RL adımını önler.
            if (
                not terminated
                and self.ppo_config.terminate_after_closest_approach
                and self._closest_approach_armed
            ):
                step_range_m, step_closing_speed_mps, _ = self._engagement_metrics()
                range_rebound_m = step_range_m - self.minimum_range_m
                opening_fast_enough = (
                    step_closing_speed_mps
                    <= -self.ppo_config.minimum_opening_speed_mps
                )
                rebounded_after_minimum = (
                    range_rebound_m
                    >= self.ppo_config.closest_approach_rebound_m
                )

                if opening_fast_enough and rebounded_after_minimum:
                    if (
                        self.ppo_config.closest_approach_training
                        and self._success_achieved
                    ):
                        self.status = "hit"
                    else:
                        self.status = "miss: passed target"
                    self.terminal_time_s = self.time_s
                    terminated = True

            if terminated:
                break

        current_range_m, _, _ = self._engagement_metrics()

        # Anlık menzil değişimi: hedefe yönelme hakkında hızlı fakat düşük
        # ağırlıklı bir işaret verir. Uzaklaşırken negatif olabilir.
        instantaneous_range_progress = (
            self.previous_range_m - current_range_m
        ) / self.ppo_config.range_progress_scale_m

        # Minimum menzil iyileşmesi yalnızca yeni bir episode rekoru kırıldığında
        # pozitiftir. max(..., 0) sayesinde bu kazanım daha sonra geri alınmaz.
        minimum_range_progress = max(
            previous_minimum_range_m - self.minimum_range_m,
            0.0,
        ) / self.ppo_config.minimum_range_progress_scale_m

        acceleration_limit = (
            self.rl_config.max_command_g * self.environment.gravity_mps2
        )
        if acceleration_limit > 1e-12:
            command_fraction = float(
                np.linalg.norm(last_command) / acceleration_limit
            )
        else:
            command_fraction = 0.0

        progress_term = (
            self.ppo_config.range_progress_weight
            * instantaneous_range_progress
        )
        minimum_range_term = (
            self.ppo_config.minimum_range_progress_weight
            * minimum_range_progress
        )

        # Reward V6: Füze henüz hedefe kilometrelerce uzaktayken bile mevcut
        # bağıl hızla öngörülen yakın geçişin iyileşip iyileşmediğini gösterir.
        # Bu değer gerçek terminal minimum menzilin yerine geçmez; V4 terminal
        # doğruluk ödülüne erken bir credit-assignment köprüsü ekler.
        current_predictive_cpa_distance_m, _ = self._predictive_cpa_metrics()
        predictive_cpa_term = 0.0
        if self.ppo_config.predictive_cpa_reward_enabled:
            predictive_cpa_term = predictive_cpa_progress_term(
                previous_distance_m=self.previous_predictive_cpa_distance_m,
                current_distance_m=current_predictive_cpa_distance_m,
                weight=self.ppo_config.predictive_cpa_progress_weight,
                scale_m=self.ppo_config.predictive_cpa_progress_scale_m,
                clip_m=self.ppo_config.predictive_cpa_progress_clip_m,
            )
        control_term = -self.ppo_config.control_effort_penalty * command_fraction**2

        # Episode fizik alt adımlarından birinde erken bittiyse yalnızca gerçekten
        # geçen kontrol süresi kadar zaman cezası uygula.
        elapsed_control_time_s = (
            controlled_duration_s - remaining_control_time_s
        )
        time_fraction = elapsed_control_time_s / self.ppo_config.control_period_s
        time_term = -self.ppo_config.time_penalty * time_fraction
        terminal_term = 0.0
        terminal_accuracy_term = 0.0
        miss_distance_term = 0.0

        if (
            not terminated
            and self.time_s >= self.environment.t_final_s - 1e-12
        ):
            self.status = "timeout"
            self.terminal_time_s = self.time_s
            truncated = True

        if self.ppo_config.closest_approach_training:
            # V4: Başarı eşiğinde sabit +hit bonusuna sıçramak yerine bütün
            # normal terminal durumlarını aynı sürekli minimum-menzil fonksiyonu
            # ile puanla. Ground/low-speed fiziksel başarısızlık cezaları ayrıca
            # korunur; timeout ise doğruluk terimine ek olarak süre aşımı alır.
            if self.status == "miss: ground":
                terminal_term -= self.ppo_config.ground_penalty
            elif self.status == "miss: low speed":
                terminal_term -= self.ppo_config.low_speed_penalty
            elif self.status == "timeout":
                terminal_term -= self.ppo_config.timeout_penalty
            elif self.status == "miss: passed target":
                terminal_term -= self.ppo_config.passed_target_penalty

            if (
                (terminated or truncated)
                and self.status in {"hit", "miss: passed target", "timeout"}
            ):
                terminal_accuracy_term = smooth_closest_approach_terminal_term(
                    minimum_range_m=self.minimum_range_m,
                    success_radius_m=self.environment.hit_radius_m,
                    hit_reward=self.ppo_config.hit_reward,
                    miss_distance_penalty_weight=(
                        self.ppo_config.miss_distance_penalty_weight
                    ),
                    miss_distance_penalty_cap=(
                        self.ppo_config.miss_distance_penalty_cap
                    ),
                )
        else:
            # Reward V3 geriye dönük uyumluluğu.
            if self.status == "hit":
                terminal_term += self.ppo_config.hit_reward
            elif self.status == "miss: ground":
                terminal_term -= self.ppo_config.ground_penalty
            elif self.status == "miss: low speed":
                terminal_term -= self.ppo_config.low_speed_penalty
            elif self.status == "miss: passed target":
                terminal_term -= self.ppo_config.passed_target_penalty
            elif self.status == "timeout":
                terminal_term -= self.ppo_config.timeout_penalty

            # Normal bir yakın geçiş/timeout kaçırmasında ceza, son menzilden
            # değil ulaşılan minimum menzilden hesaplanır.
            if self.status in {"miss: passed target", "timeout"}:
                hit_radius_m = max(self.environment.hit_radius_m, 1e-6)
                miss_distance_m = max(
                    self.minimum_range_m - hit_radius_m,
                    0.0,
                )
                normalized_miss_distance = miss_distance_m / hit_radius_m
                miss_distance_term = -min(
                    self.ppo_config.miss_distance_penalty_weight
                    * normalized_miss_distance,
                    self.ppo_config.miss_distance_penalty_cap,
                )

        unscaled_reward = (
            progress_term
            + minimum_range_term
            + predictive_cpa_term
            + control_term
            + time_term
            + terminal_term
            + terminal_accuracy_term
            + miss_distance_term
        )
        reward_scale = self.ppo_config.reward_scale
        reward = reward_scale * unscaled_reward
        self._last_reward_terms = {
            "range_progress": float(reward_scale * progress_term),
            "minimum_range_progress": float(reward_scale * minimum_range_term),
            "predictive_cpa_progress": float(
                reward_scale * predictive_cpa_term
            ),
            "control_effort": float(reward_scale * control_term),
            "time": float(reward_scale * time_term),
            "terminal": float(reward_scale * terminal_term),
            "terminal_accuracy": float(
                reward_scale * terminal_accuracy_term
            ),
            "miss_distance": float(reward_scale * miss_distance_term),
            "total": float(reward),
            "unscaled_total": float(unscaled_reward),
            "reward_scale": float(reward_scale),
        }
        for term_name in PPO_REWARD_TERM_NAMES:
            self._episode_reward_terms[term_name] += float(
                self._last_reward_terms[term_name]
            )

        self.previous_range_m = current_range_m
        self.previous_predictive_cpa_distance_m = current_predictive_cpa_distance_m
        self._episode_done = terminated or truncated
        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info(command_g=command_fraction * self.rl_config.max_command_g),
        )

    def render(self):
        """Eğitim sırasında çizim yapılmaz; sonuçlar ana plot fonksiyonlarındadır."""
        return None

    def close(self) -> None:
        return None


def build_default_ppo_environment(
    ppo_environment_config: Optional[PPOEnvironmentConfig] = None,
    rl_interface_config: Optional[RLInterfaceConfig] = None,
) -> MissileGuidanceEnv:
    """Mevcut örnek senaryodan başlangıç PPO ortamını kurar."""
    environment, vehicles, _ = build_default_scenario()
    missile = next(vehicle for vehicle in vehicles if vehicle.name == "missile_1")

    interface_config = deepcopy(
        rl_interface_config
        if rl_interface_config is not None
        else RLInterfaceConfig(
            max_command_g=30.0,
            burn_time_s=missile.propulsion.burn_time_s,
        )
    )
    return MissileGuidanceEnv(
        environment=environment,
        vehicles=vehicles,
        missile_name="missile_1",
        rl_interface_config=interface_config,
        ppo_environment_config=ppo_environment_config,
    )


def evaluate_ppo_in_default_simulation(
    model_or_path,
    deterministic: bool = True,
) -> SimulationResult:
    """Eğitilmiş PPO modelini ana run_simulation döngüsünde bir kez çalıştırır."""
    if isinstance(model_or_path, str):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "PPO modelini yüklemek için stable-baselines3 gerekir."
            ) from exc
        policy = PPO.load(model_or_path)
    else:
        policy = model_or_path

    observation_version = infer_policy_observation_version(policy)

    environment, vehicles, _ = build_default_scenario()
    missile = next(vehicle for vehicle in vehicles if vehicle.name == "missile_1")
    interface_config = RLInterfaceConfig(
        max_command_g=30.0,
        observation_version=observation_version,
        burn_time_s=missile.propulsion.burn_time_s,
        control_period_s=0.10,
        deterministic=deterministic,
    )
    guidance_laws: Dict[str, GuidanceLaw] = {
        "missile_1": RLGuidanceAdapter(
            policy=policy,
            config=interface_config,
        )
    }
    return run_simulation(vehicles, environment, guidance_laws)


def print_summary(result: SimulationResult) -> None:
    """Her füzenin terminal durumunu ve kolay okunabilir son NED state'ini yazdırır."""
    for missile_name, missile_status in result.status.items():
        end_index = result.terminal_index[missile_name]
        minimum_recorded_range_m = float(
            np.min(result.range_m[missile_name][: end_index + 1])
        )
        print(
            f"{missile_name:12s} -> {missile_status:16s} | "
            f"terminal t = {result.terminal_time_s[missile_name]:6.3f} s | "
            f"kayıtlı min. menzil = {minimum_recorded_range_m:7.3f} m"
        )

        final_state = result.states[missile_name][end_index]
        print(
            " " * 15
            + "son state [N,E,D,V,chi,gamma,m] = "
            + np.array2string(
                np.array(
                    [
                        final_state[NORTH],
                        final_state[EAST],
                        final_state[DOWN],
                        final_state[SPEED],
                        np.rad2deg(final_state[CHI]),
                        np.rad2deg(final_state[GAMMA]),
                        final_state[MASS],
                    ]
                ),
                precision=3,
                suppress_small=True,
            )
        )
        print(
            " " * 15
            + "Not: yazdırılan chi ve gamma değerleri okunabilirlik için derecedir."
        )


def run_basic_checks(
    result: SimulationResult,
    vehicles: List[VehicleSpec],
) -> None:
    assert np.all(np.diff(result.time_s) > 0.0)
    assert all(
        len(history) == len(result.time_s)
        for history in result.states.values()
    )
    assert all(
        np.all(np.isfinite(history))
        for history in result.states.values()
    )

    for vehicle in vehicles:
        mass_history = result.states[vehicle.name][:, MASS]
        assert np.all(
            mass_history >= vehicle.propulsion.dry_mass_kg - 1e-8
        ), f"{vehicle.name}: kuru kütle ihlali"


def main() -> None:
    env, vehicles, guidance_laws = build_default_scenario()
    result = run_simulation(vehicles, env, guidance_laws)

    print_summary(result)
    run_basic_checks(result, vehicles)

    # Rastgele nonlinear hedef senaryosu gözlem amaçlıdır. Manevra, seed veya
    # başlangıç koşulları değiştirildiğinde füze vurabilir, kaçırabilir veya
    # zaman aşımına gidebilir.
    # Bu nedenle burada özellikle "hit" assertion'ı kullanılmaz; yalnızca sayısal
    # ve fiziksel temel kontroller uygulanır.
    print("Temel kontroller başarılı.")

    plot_results(result, vehicles)
    plot_vehicle_state_history(result, "missile_1")

    # Nonlinear manevra yapan hedef uçağın N, E ve D konumları ayrıca üç ayrı
    # panelde gösterilir. Başka bir uçak eklenirse yalnızca adı değiştirilerek aynı
    # fonksiyon tekrar çağrılabilir.
    plot_ned_position_history(result, "aircraft_1")

    # Yolun rastgele yatay/düşey kıvrımlarını daha açık görmek için üç farklı 2B
    # izdüşüm de çizilir: N-E, N-irtifa ve E-irtifa.
    plot_aircraft_path_projections(result, "aircraft_1")


if __name__ == "__main__":
    main()
