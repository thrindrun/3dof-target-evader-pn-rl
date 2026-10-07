"""Final 3DOF füze güdüm uygulaması: PPO, PN veya eşzamanlı karşılaştırma.

Normal kullanımda yalnızca aşağıdaki ``KULLANICI AYARLARI`` bölümü değiştirilir.
Komut satırı seçenekleri hızlı denemeler için bu ayarların bir kısmını geçersiz
kılabilir. Koordinatlar kullanıcı tarafında (North, East, irtifa) biçimindedir;
çekirdek içinde otomatik olarak NED (North, East, Down) biçimine çevrilir.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from air_to_air_3dof_core import (
    CHI,
    DOWN,
    EAST,
    GAMMA,
    MASS,
    NORTH,
    SPEED,
    AeroConfig,
    EnvironmentConfig,
    InitialState,
    ManeuverConfig,
    PNConfig,
    PropulsionConfig,
    RLGuidanceAdapter,
    RLInterfaceConfig,
    SampleAndHoldGuidance,
    SimulationResult,
    VectorPNGuidance,
    VehicleSpec,
    infer_policy_observation_version,
    run_simulation,
)

# =============================================================================
# KULLANICI AYARLARI
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# "ppo", "pn" veya "both". "both" aynı hedefe aynı başlangıç durumundaki iki
# füze gönderir; yalnızca güdüm yasaları farklıdır.
GUIDANCE_MODE = "both"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fresh_cpa_v6_hardpocket_7p5m_300k"
    / "best_model.zip"
)

SCENARIO_NAME = "final_default_comparison"

# Kullanıcı koordinat sırası: North [m], East [m], irtifa [m].
TARGET_N_E_ALT_M = (9000.000, 1000.000, 3300.000)
MISSILE_N_E_ALT_M = (14312.593, 2423.505, 3300.000)

TARGET_SPEED_MPS = 300.0
TARGET_HEADING_DEG = 195.0
TARGET_FLIGHT_PATH_DEG = 0.0

# straight | constant_turn | sinusoidal | random_smooth
TARGET_MANEUVER_KIND = "constant_turn"
TARGET_HEADING_RATE_DEG_S = 12.0
TARGET_FLIGHT_PATH_RATE_DEG_S = 0.0
TARGET_WEAVE_PERIOD_S = 5.0
TARGET_HEADING_PHASE_DEG = 0.0
TARGET_FLIGHT_PATH_PHASE_DEG = 90.0
TARGET_RANDOM_SEED = 2026
TARGET_RANDOM_HARMONICS = 5

MISSILE_SPEED_MPS = 310.0

# "explicit": Aşağıdaki iki açı doğrudan füzenin ilk hız yönünü belirler.
# "toward_target": LOS açıları hesaplanır; altındaki hata açıları eklenir.
# Böylece füzenin otomatik olarak hedefe bakıp bakmadığı gizli kalmaz.
MISSILE_ATTITUDE_MODE = "explicit"
MISSILE_HEADING_DEG = -165.0
MISSILE_FLIGHT_PATH_DEG = 0.0
MISSILE_AIM_AZIMUTH_ERROR_DEG = 0.0
MISSILE_AIM_ELEVATION_ERROR_DEG = 0.0

# Ortak fizik ve karar hızı. PPO bu kontrol periyoduyla eğitilmiştir; PN de
# SampleAndHoldGuidance üzerinden aynı 10 Hz karar hızında güncellenir.
CONTROL_PERIOD_S = 0.10
PHYSICS_DT_S = 0.02
FINAL_TIME_S = 35.0
# 1 m, rapordaki en sıkı ölçüttür ve 2.5/5/10 m eşiklerinin de aynı koşuda
# güvenilir biçimde değerlendirilmesini sağlar. Proje ana kriteri yalnızca 10 m
# ise bu değer 10.0 yapılabilir; o durumda 1/2.5/5 m sonuçları erken terminal
# nedeniyle ancak gerçekten aşılmışlarsa kesin kabul edilir.
HIT_RADIUS_M = 1.0
REPORT_THRESHOLDS_M = (1.0, 2.5, 5.0, 10.0)

PPO_MAX_COMMAND_G = 30.0
PPO_DETERMINISTIC = True
PN_NAVIGATION_CONSTANT = 4.0
PN_MAX_COMMAND_G = 30.0

GROUND_ALTITUDE_M = 0.0
MINIMUM_FLYING_SPEED_MPS = 60.0

SAVE_OUTPUTS = True
SHOW_FIGURE = True
OUTPUT_DIR = PROJECT_ROOT / "results" / SCENARIO_NAME


# =============================================================================
# DÖNÜŞÜM VE DOĞRULAMA YARDIMCILARI
# =============================================================================


def to_ned(north_east_altitude: tuple[float, float, float]) -> np.ndarray:
    """(North, East, irtifa) değerini (North, East, Down) biçimine çevir."""
    north, east, altitude = north_east_altitude
    return np.asarray([north, east, -altitude], dtype=float)


def aim_angles_deg(
    missile_position_ned: np.ndarray,
    target_position_ned: np.ndarray,
) -> tuple[float, float]:
    """Füzeden hedefe bakan LOS heading ve tırmanış açısını dereceyle bul."""
    relative = target_position_ned - missile_position_ned
    horizontal_range_m = float(np.hypot(relative[NORTH], relative[EAST]))
    heading_deg = float(np.degrees(np.arctan2(relative[EAST], relative[NORTH])))
    flight_path_deg = float(
        np.degrees(np.arctan2(-relative[DOWN], horizontal_range_m))
    )
    return heading_deg, flight_path_deg


def resolve_model_path(raw_path: str | Path) -> Path:
    """Model yolunu çözer ve yanlışlıkla oluşan ``.zip.zip`` ekini düzeltir."""
    path_text = str(raw_path).strip().strip('"')
    while path_text.lower().endswith(".zip.zip"):
        path_text = path_text[:-4]

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    candidates = [path]
    if path.suffix.lower() != ".zip":
        candidates.append(path.with_suffix(".zip"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"PPO modeli bulunamadı. Kontrol edilen yollar:\n  - {checked}")


def selected_launch_angles(
    missile_position_ned: np.ndarray,
    target_position_ned: np.ndarray,
) -> tuple[float, float, float, float]:
    """Uygulanan açıları ve salt LOS açılarını birlikte döndür."""
    los_heading_deg, los_gamma_deg = aim_angles_deg(
        missile_position_ned,
        target_position_ned,
    )
    mode = MISSILE_ATTITUDE_MODE.strip().lower()
    if mode == "explicit":
        heading_deg = float(MISSILE_HEADING_DEG)
        gamma_deg = float(MISSILE_FLIGHT_PATH_DEG)
    elif mode in {"toward_target", "los", "los_with_error"}:
        heading_deg = los_heading_deg + MISSILE_AIM_AZIMUTH_ERROR_DEG
        gamma_deg = los_gamma_deg + MISSILE_AIM_ELEVATION_ERROR_DEG
    else:
        raise ValueError(
            "MISSILE_ATTITUDE_MODE 'explicit' veya 'toward_target' olmalıdır."
        )
    return heading_deg, gamma_deg, los_heading_deg, los_gamma_deg


def validate_user_settings(mode: str) -> None:
    if mode not in {"ppo", "pn", "both"}:
        raise ValueError("Güdüm modu 'ppo', 'pn' veya 'both' olmalıdır.")
    if PHYSICS_DT_S <= 0.0 or CONTROL_PERIOD_S <= 0.0 or FINAL_TIME_S <= 0.0:
        raise ValueError("Fizik adımı, kontrol periyodu ve süre pozitif olmalıdır.")
    if HIT_RADIUS_M <= 0.0:
        raise ValueError("Vurma yarıçapı pozitif olmalıdır.")
    if not REPORT_THRESHOLDS_M or min(REPORT_THRESHOLDS_M) <= 0.0:
        raise ValueError("Rapor eşiklerinin tamamı pozitif olmalıdır.")
    if MISSILE_SPEED_MPS <= 0.0 or TARGET_SPEED_MPS <= 0.0:
        raise ValueError("Füze ve hedef başlangıç hızları pozitif olmalıdır.")


# =============================================================================
# SENARYO VE GÜDÜM NESNELERİ
# =============================================================================


def make_environment(hit_radius_m: float) -> EnvironmentConfig:
    return EnvironmentConfig(
        dt_s=PHYSICS_DT_S,
        t_final_s=FINAL_TIME_S,
        gravity_mps2=9.80665,
        rho0_kgpm3=1.225,
        atmosphere_scale_height_m=8500.0,
        ground_altitude_m=GROUND_ALTITUDE_M,
        hit_radius_m=hit_radius_m,
        minimum_flying_speed_mps=MINIMUM_FLYING_SPEED_MPS,
    )


def make_target(target_position_ned: np.ndarray) -> VehicleSpec:
    return VehicleSpec(
        name="aircraft_1",
        role="aircraft",
        initial=InitialState(
            position_m=tuple(float(value) for value in target_position_ned),
            speed_mps=TARGET_SPEED_MPS,
            azimuth_deg=TARGET_HEADING_DEG,
            flight_path_deg=TARGET_FLIGHT_PATH_DEG,
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
        speed_hold_mps=TARGET_SPEED_MPS,
        maneuver=ManeuverConfig(
            kind=TARGET_MANEUVER_KIND,
            heading_rate_deg_s=TARGET_HEADING_RATE_DEG_S,
            flight_path_rate_deg_s=TARGET_FLIGHT_PATH_RATE_DEG_S,
            weave_period_s=TARGET_WEAVE_PERIOD_S,
            heading_phase_deg=TARGET_HEADING_PHASE_DEG,
            flight_path_phase_deg=TARGET_FLIGHT_PATH_PHASE_DEG,
            random_seed=TARGET_RANDOM_SEED,
            random_harmonics=TARGET_RANDOM_HARMONICS,
        ),
    )


def make_missile_template(
    missile_position_ned: np.ndarray,
    heading_deg: float,
    gamma_deg: float,
) -> VehicleSpec:
    return VehicleSpec(
        name="missile_template",
        role="missile",
        target_name="aircraft_1",
        initial=InitialState(
            position_m=tuple(float(value) for value in missile_position_ned),
            speed_mps=MISSILE_SPEED_MPS,
            azimuth_deg=heading_deg,
            flight_path_deg=gamma_deg,
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


def load_ppo_model(model_path: Path):
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PPO modu için stable-baselines3 bulunamadı. "
            "Önce 'python -m pip install -r requirements.txt' çalıştırın."
        ) from exc
    return PPO.load(str(model_path), device="cpu")


def build_simulation(
    mode: str,
    hit_radius_m: float,
    model_path: Path | None,
) -> tuple[
    EnvironmentConfig,
    list[VehicleSpec],
    dict[str, Any],
    dict[str, Any],
]:
    target_position_ned = to_ned(TARGET_N_E_ALT_M)
    missile_position_ned = to_ned(MISSILE_N_E_ALT_M)
    heading_deg, gamma_deg, los_heading_deg, los_gamma_deg = selected_launch_angles(
        missile_position_ned,
        target_position_ned,
    )

    environment = make_environment(hit_radius_m)
    target = make_target(target_position_ned)
    template = make_missile_template(missile_position_ned, heading_deg, gamma_deg)

    vehicles: list[VehicleSpec] = []
    guidance_laws: dict[str, Any] = {}
    observation_version: str | None = None

    if mode in {"ppo", "both"}:
        if model_path is None:
            raise ValueError("PPO modu için model yolu gereklidir.")
        model = load_ppo_model(model_path)
        observation_version = infer_policy_observation_version(model)
        if observation_version != "v2":
            raise ValueError(
                "Final model için 23 girdili observation v2 bekleniyordu; "
                f"yüklenen model {observation_version!r} kullanıyor."
            )

        missile_ppo = deepcopy(template)
        missile_ppo.name = "missile_ppo"
        vehicles.append(missile_ppo)
        guidance_laws[missile_ppo.name] = RLGuidanceAdapter(
            policy=model,
            config=RLInterfaceConfig(
                max_command_g=PPO_MAX_COMMAND_G,
                observation_version=observation_version,
                burn_time_s=missile_ppo.propulsion.burn_time_s,
                control_period_s=CONTROL_PERIOD_S,
                deterministic=PPO_DETERMINISTIC,
            ),
        )

    if mode in {"pn", "both"}:
        missile_pn = deepcopy(template)
        missile_pn.name = "missile_pn"
        vehicles.append(missile_pn)
        guidance_laws[missile_pn.name] = SampleAndHoldGuidance(
            VectorPNGuidance(
                PNConfig(
                    navigation_constant=PN_NAVIGATION_CONSTANT,
                    max_command_g=PN_MAX_COMMAND_G,
                )
            ),
            period_s=CONTROL_PERIOD_S,
        )

    vehicles.append(target)
    context = {
        "mode": mode,
        "model_path": str(model_path) if model_path is not None else None,
        "observation_version": observation_version,
        "initial_range_m": float(
            np.linalg.norm(target_position_ned - missile_position_ned)
        ),
        "missile_heading_deg": heading_deg,
        "missile_flight_path_deg": gamma_deg,
        "los_heading_deg": los_heading_deg,
        "los_flight_path_deg": los_gamma_deg,
    }
    return environment, vehicles, guidance_laws, context


# =============================================================================
# SONUÇ METRİKLERİ, DOSYALAR VE GRAFİKLER
# =============================================================================


def method_label(missile_name: str) -> str:
    if missile_name.endswith("ppo"):
        return "PPO"
    if missile_name.endswith("pn"):
        return "PN"
    return missile_name


def compute_metrics(result: SimulationResult) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for missile_name in result.status:
        end_index = result.terminal_index[missile_name]
        command_g = result.command_g[missile_name][: end_index + 1]
        recorded_range_m = result.range_m[missile_name][: end_index + 1]
        terminal_state = result.states[missile_name][end_index]
        cpa_m = float(result.closest_approach_m[missile_name])

        metrics[missile_name] = {
            "method": method_label(missile_name),
            "status": result.status[missile_name],
            "terminal_time_s": float(result.terminal_time_s[missile_name]),
            "closest_approach_m": cpa_m,
            "closest_approach_time_s": float(
                result.closest_approach_time_s[missile_name]
            ),
            "minimum_recorded_range_m": float(np.min(recorded_range_m)),
            "maximum_command_g": float(np.max(command_g)),
            "rms_command_g": float(np.sqrt(np.mean(np.square(command_g)))),
            "terminal_speed_mps": float(terminal_state[SPEED]),
            "terminal_mass_kg": float(terminal_state[MASS]),
            "threshold_success": {
                f"{threshold:g}_m": (
                    bool(cpa_m <= threshold)
                    if threshold >= result.hit_radius_m
                    or result.status[missile_name] != "hit"
                    or cpa_m <= threshold
                    else None
                )
                for threshold in sorted(set(REPORT_THRESHOLDS_M))
            },
        }
    return metrics


def print_run_header(context: dict[str, Any], hit_radius_m: float) -> None:
    approximate_turn_load_g = (
        TARGET_SPEED_MPS
        * np.deg2rad(TARGET_HEADING_RATE_DEG_S)
        / 9.80665
        if TARGET_MANEUVER_KIND == "constant_turn"
        else float("nan")
    )
    print("=" * 94)
    print("3DOF FÜZE GÜDÜMÜ — FINAL ANA UYGULAMA")
    print("=" * 94)
    print(f"Senaryo                  : {SCENARIO_NAME}")
    print(f"Kontrolcü                : {context['mode']}")
    if context["model_path"]:
        print(f"PPO modeli               : {context['model_path']}")
        print(f"PPO observation          : {context['observation_version']} (23 girdi)")
    print(f"Başlangıç menzili        : {context['initial_range_m']:.3f} m")
    print(
        "Füze heading / gamma    : "
        f"{context['missile_heading_deg']:.3f} / "
        f"{context['missile_flight_path_deg']:.3f} deg"
    )
    print(
        "Hedef LOS heading/gamma : "
        f"{context['los_heading_deg']:.3f} / "
        f"{context['los_flight_path_deg']:.3f} deg"
    )
    print(f"Başlangıç yön modu       : {MISSILE_ATTITUDE_MODE}")
    print(f"Füze / hedef hızı        : {MISSILE_SPEED_MPS:.1f} / {TARGET_SPEED_MPS:.1f} m/s")
    print(f"Hedef heading / gamma    : {TARGET_HEADING_DEG:.3f} / {TARGET_FLIGHT_PATH_DEG:.3f} deg")
    print(
        "Hedef manevrası          : "
        f"{TARGET_MANEUVER_KIND}, chi_dot={TARGET_HEADING_RATE_DEG_S:g} deg/s, "
        f"gamma_dot={TARGET_FLIGHT_PATH_RATE_DEG_S:g} deg/s"
    )
    if np.isfinite(approximate_turn_load_g):
        print(f"Yaklaşık hedef dönüş yükü: {approximate_turn_load_g:.3f} g")
    print(f"Kontrol / fizik adımı    : {CONTROL_PERIOD_S:.3f} / {PHYSICS_DT_S:.3f} s")
    print(f"Vurma yarıçapı           : {hit_radius_m:.3f} m")
    print(f"Rapor eşikleri           : {', '.join(f'{x:g} m' for x in REPORT_THRESHOLDS_M)}")
    print("=" * 94)


def print_metrics(metrics: dict[str, dict[str, Any]]) -> None:
    print()
    print("-" * 94)
    print(
        f"{'Yöntem':<10}{'Durum':<19}{'Terminal [s]':>14}"
        f"{'CPA [m]':>12}{'Maks [g]':>12}{'RMS [g]':>12}{'Vson [m/s]':>15}"
    )
    print("-" * 94)
    for item in metrics.values():
        print(
            f"{item['method']:<10}{item['status']:<19}"
            f"{item['terminal_time_s']:>14.3f}"
            f"{item['closest_approach_m']:>12.3f}"
            f"{item['maximum_command_g']:>12.3f}"
            f"{item['rms_command_g']:>12.3f}"
            f"{item['terminal_speed_mps']:>15.3f}"
        )
    print("-" * 94)
    for item in metrics.values():
        def threshold_label(passed: bool | None) -> str:
            if passed is None:
                return "erken terminal—ölçülmedi"
            return "EVET" if passed else "hayır"

        threshold_text = ", ".join(
            f"{name.replace('_m', ' m')}={threshold_label(passed)}"
            for name, passed in item["threshold_success"].items()
        )
        print(f"{item['method']}: {threshold_text}")
    print(
        "Not: CPA adım-içi doğrusal bağıl hareketten hesaplanır. 'Kayıtlı menzil' "
        "ise yalnızca 20 ms örnek noktalarını içerdiği için CPA'dan büyük olabilir."
    )


def _set_axes_equal_3d(axis) -> None:
    limits = np.asarray(
        [axis.get_xlim3d(), axis.get_ylim3d(), axis.get_zlim3d()],
        dtype=float,
    )
    centers = np.mean(limits, axis=1)
    radius = max(float(np.max(np.ptp(limits, axis=1))) / 2.0, 0.001)
    axis.set_xlim3d(centers[0] - radius, centers[0] + radius)
    axis.set_ylim3d(centers[1] - radius, centers[1] + radius)
    axis.set_zlim3d(centers[2] - radius, centers[2] + radius)


def create_result_figure(
    result: SimulationResult,
    vehicles: list[VehicleSpec],
    metrics: dict[str, dict[str, Any]],
) -> plt.Figure:
    specs = {vehicle.name: vehicle for vehicle in vehicles}
    figure = plt.figure(figsize=(16, 11))
    axis_3d = figure.add_subplot(2, 2, 1, projection="3d")

    for name, history in result.states.items():
        role = specs[name].role
        if role == "missile":
            end_index = result.terminal_index[name]
            history = history[: end_index + 1]
            line_style = "-"
            label = method_label(name)
        else:
            line_style = "--"
            label = "Hedef"
        axis_3d.plot(
            history[:, NORTH] / 1000.0,
            history[:, EAST] / 1000.0,
            -history[:, DOWN] / 1000.0,
            line_style,
            linewidth=2.0,
            label=label,
        )
        axis_3d.scatter(
            history[0, NORTH] / 1000.0,
            history[0, EAST] / 1000.0,
            -history[0, DOWN] / 1000.0,
            s=36,
        )
        axis_3d.scatter(
            history[-1, NORTH] / 1000.0,
            history[-1, EAST] / 1000.0,
            -history[-1, DOWN] / 1000.0,
            marker="x",
            s=44,
        )

    axis_3d.set_xlabel("North [km]")
    axis_3d.set_ylabel("East [km]")
    axis_3d.set_zlabel("İrtifa [km]")
    axis_3d.set_title("3B fiziksel yörüngeler")
    axis_3d.legend()
    _set_axes_equal_3d(axis_3d)

    axis_range = figure.add_subplot(2, 2, 2)
    for name, values in result.range_m.items():
        end_index = result.terminal_index[name]
        axis_range.plot(
            result.time_s[: end_index + 1],
            values[: end_index + 1],
            linewidth=1.8,
            label=f"{method_label(name)} (CPA={metrics[name]['closest_approach_m']:.2f} m)",
        )
    axis_range.axhline(
        result.hit_radius_m,
        color="tab:red",
        linestyle=":",
        label=f"Vurma yarıçapı ({result.hit_radius_m:g} m)",
    )
    axis_range.set_xlabel("Zaman [s]")
    axis_range.set_ylabel("Füze-hedef menzili [m]")
    axis_range.set_title("Bağıl menzil")
    axis_range.grid(True, alpha=0.35)
    axis_range.legend()

    axis_speed = figure.add_subplot(2, 2, 3)
    for name, history in result.states.items():
        role = specs[name].role
        end_index = result.terminal_index.get(name, len(result.time_s) - 1)
        axis_speed.plot(
            result.time_s[: end_index + 1],
            history[: end_index + 1, SPEED],
            linewidth=1.8,
            linestyle="-" if role == "missile" else "--",
            label=method_label(name) if role == "missile" else "Hedef",
        )
    axis_speed.set_xlabel("Zaman [s]")
    axis_speed.set_ylabel("Hız [m/s]")
    axis_speed.set_title("Araç hızları")
    axis_speed.grid(True, alpha=0.35)
    axis_speed.legend()

    axis_command = figure.add_subplot(2, 2, 4)
    for name, values in result.command_g.items():
        end_index = result.terminal_index[name]
        axis_command.plot(
            result.time_s[: end_index + 1],
            values[: end_index + 1],
            linewidth=1.8,
            label=method_label(name),
        )
    axis_command.set_xlabel("Zaman [s]")
    axis_command.set_ylabel("Normal ivme komutu [g]")
    axis_command.set_title("Güdüm komut büyüklüğü")
    axis_command.grid(True, alpha=0.35)
    axis_command.legend()

    figure.suptitle(
        f"{SCENARIO_NAME} — {TARGET_MANEUVER_KIND} hedef manevrası",
        fontsize=15,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return figure


def scenario_snapshot(
    context: dict[str, Any],
    environment: EnvironmentConfig,
) -> dict[str, Any]:
    return {
        "scenario_name": SCENARIO_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "guidance_mode": context["mode"],
        "model_path": context["model_path"],
        "observation_version": context["observation_version"],
        "target_n_e_alt_m": list(TARGET_N_E_ALT_M),
        "missile_n_e_alt_m": list(MISSILE_N_E_ALT_M),
        "initial_range_m": context["initial_range_m"],
        "missile_attitude_mode": MISSILE_ATTITUDE_MODE,
        "missile_heading_deg": context["missile_heading_deg"],
        "missile_flight_path_deg": context["missile_flight_path_deg"],
        "los_heading_deg": context["los_heading_deg"],
        "los_flight_path_deg": context["los_flight_path_deg"],
        "missile_speed_mps": MISSILE_SPEED_MPS,
        "target_speed_mps": TARGET_SPEED_MPS,
        "target_heading_deg": TARGET_HEADING_DEG,
        "target_flight_path_deg": TARGET_FLIGHT_PATH_DEG,
        "target_maneuver": asdict(
            ManeuverConfig(
                kind=TARGET_MANEUVER_KIND,
                heading_rate_deg_s=TARGET_HEADING_RATE_DEG_S,
                flight_path_rate_deg_s=TARGET_FLIGHT_PATH_RATE_DEG_S,
                weave_period_s=TARGET_WEAVE_PERIOD_S,
                heading_phase_deg=TARGET_HEADING_PHASE_DEG,
                flight_path_phase_deg=TARGET_FLIGHT_PATH_PHASE_DEG,
                random_seed=TARGET_RANDOM_SEED,
                random_harmonics=TARGET_RANDOM_HARMONICS,
            )
        ),
        "environment": asdict(environment),
        "control_period_s": CONTROL_PERIOD_S,
        "pn_navigation_constant": PN_NAVIGATION_CONSTANT,
        "report_thresholds_m": list(REPORT_THRESHOLDS_M),
    }


def save_history_csv(
    result: SimulationResult,
    vehicles: list[VehicleSpec],
    destination: Path,
) -> None:
    vehicle_names = [vehicle.name for vehicle in vehicles]
    missile_names = list(result.status)
    fieldnames = ["time_s"]
    for name in vehicle_names:
        fieldnames.extend(
            [
                f"{name}_north_m",
                f"{name}_east_m",
                f"{name}_altitude_m",
                f"{name}_speed_mps",
                f"{name}_heading_deg",
                f"{name}_flight_path_deg",
                f"{name}_mass_kg",
            ]
        )
    for name in missile_names:
        fieldnames.extend([f"{name}_range_m", f"{name}_command_g"])

    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, time_s in enumerate(result.time_s):
            row: dict[str, Any] = {"time_s": float(time_s)}
            for name in vehicle_names:
                state = result.states[name][index]
                row.update(
                    {
                        f"{name}_north_m": float(state[NORTH]),
                        f"{name}_east_m": float(state[EAST]),
                        f"{name}_altitude_m": float(-state[DOWN]),
                        f"{name}_speed_mps": float(state[SPEED]),
                        f"{name}_heading_deg": float(np.degrees(state[CHI])),
                        f"{name}_flight_path_deg": float(np.degrees(state[GAMMA])),
                        f"{name}_mass_kg": float(state[MASS]),
                    }
                )
            for name in missile_names:
                row[f"{name}_range_m"] = float(result.range_m[name][index])
                row[f"{name}_command_g"] = float(result.command_g[name][index])
            writer.writerow(row)


def save_outputs(
    result: SimulationResult,
    vehicles: list[VehicleSpec],
    environment: EnvironmentConfig,
    context: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    figure: plt.Figure,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "trajectory_comparison.png"
    summary_path = output_dir / "summary.json"
    history_path = output_dir / "history.csv"

    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    save_history_csv(result, vehicles, history_path)
    payload = {
        "scenario": scenario_snapshot(context, environment),
        "results": metrics,
        "files": {
            "figure": str(figure_path.resolve()),
            "history_csv": str(history_path.resolve()),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print()
    print(f"Grafik : {figure_path.resolve()}")
    print(f"Geçmiş: {history_path.resolve()}")
    print(f"Özet   : {summary_path.resolve()}")


# =============================================================================
# KOMUT SATIRI VE PROGRAM GİRİŞİ
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3DOF füze güdümünü PPO, PN veya ikisiyle çalıştırır."
    )
    parser.add_argument(
        "--controller",
        choices=("ppo", "pn", "both"),
        default=GUIDANCE_MODE,
        help=f"Güdüm modu (varsayılan: {GUIDANCE_MODE}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="PPO model yolu; .zip eki isteğe bağlıdır.",
    )
    parser.add_argument(
        "--hit-radius",
        type=float,
        default=HIT_RADIUS_M,
        help=f"Vurma yarıçapı [m] (varsayılan: {HIT_RADIUS_M:g}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="PNG, CSV ve JSON çıktı klasörü.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="PNG/CSV/JSON dosyalarını yazma.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Grafik penceresini açma (sunucu/otomasyon için).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = args.controller.lower()
    validate_user_settings(mode)

    model_path = resolve_model_path(args.model) if mode in {"ppo", "both"} else None
    environment, vehicles, guidance_laws, context = build_simulation(
        mode=mode,
        hit_radius_m=float(args.hit_radius),
        model_path=model_path,
    )
    print_run_header(context, environment.hit_radius_m)

    result = run_simulation(vehicles, environment, guidance_laws)
    metrics = compute_metrics(result)
    print_metrics(metrics)
    figure = create_result_figure(result, vehicles, metrics)

    should_save = SAVE_OUTPUTS and not args.no_save
    should_show = SHOW_FIGURE and not args.no_show
    if should_save:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        save_outputs(
            result,
            vehicles,
            environment,
            context,
            metrics,
            figure,
            output_dir,
        )

    if should_show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
