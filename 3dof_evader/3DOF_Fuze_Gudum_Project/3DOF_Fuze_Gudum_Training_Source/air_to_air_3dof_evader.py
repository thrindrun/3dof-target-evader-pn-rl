from __future__ import annotations

"""Gymnasium environment for training an RL aircraft to evade a fixed PN missile.

The environment deliberately reuses the project's current 3DOF physics and
``VectorPNGuidance`` implementation.  PPO controls only the target aircraft.
The target action is interpreted as desired heading-rate and flight-path-rate,
then passed through a first-order actuator response before it becomes a normal
acceleration command for the same RK4 point-mass dynamics used by the missile.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np


MODULE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = MODULE_DIRECTORY.parent
for import_directory in (MODULE_DIRECTORY, PROJECT_DIRECTORY):
    if str(import_directory) not in sys.path:
        sys.path.insert(0, str(import_directory))

from air_to_air_3dof_core import (
    CHI,
    DEG,
    DOWN,
    GAMMA,
    G0,
    MASS,
    POSITION,
    SPEED,
    AeroConfig,
    EnvironmentConfig,
    GuidanceObservation,
    InitialState,
    ManeuverConfig,
    PNConfig,
    PropulsionConfig,
    VehicleSpec,
    VectorPNGuidance,
    altitude_from_down,
    closest_approach_on_step,
    flight_path_basis,
    make_initial_state,
    rk4_step,
    velocity_from_state,
)


EVADER_OBSERVATION_VERSION = "evader_v1"
EVADER_OBSERVATION_SIZE = 27
EVADER_ACTION_SIZE = 2


@dataclass
class EvaderEnvironmentConfig:
    """Configuration for the single-missile/single-target evasion problem."""

    control_period_s: float = 0.10

    # Fixed missile controller used during evader training.
    pn_navigation_constant: float = 4.0
    pn_max_command_g: float = 30.0

    # First-order missile acceleration/autopilot response.
    # 0.0 preserves the original instantaneous-command behavior.
    missile_actuator_time_constant_s: float = 0.0

    # action[0] and action[1] are mapped to these rate limits.
    max_heading_rate_deg_s: float = 16.0
    max_flight_path_rate_deg_s: float = 6.0
    actuator_time_constant_s: float = 0.25

    # E1 defaults to one fixed geometry.  E2 can enable these jitters.
    randomize_initial_conditions: bool = False
    missile_position_jitter_m: tuple[float, float, float] = (40.0, 40.0, 20.0)
    target_position_jitter_m: tuple[float, float, float] = (200.0, 200.0, 80.0)
    missile_speed_jitter_mps: float = 10.0
    target_speed_jitter_mps: float = 10.0
    missile_heading_jitter_deg: float = 2.0
    target_heading_jitter_deg: float = 5.0
    missile_flight_path_jitter_deg: float = 1.0
    target_flight_path_jitter_deg: float = 2.0
    minimum_initial_altitude_m: float = 250.0

    # Episode termination after a completed closest approach.
    passed_missile_arm_progress_m: float = 100.0
    closest_approach_rebound_m: float = 25.0
    minimum_opening_speed_mps: float = 1.0

    # As in the missile project's closest-approach training, crossing the hit
    # radius is recorded but does not immediately stop the virtual trajectory.
    # The episode continues until the true closest approach is completed, which
    # gives PPO a continuous terminal-distance learning signal.
    closest_approach_training: bool = True

    # Observation normalization.
    position_scale_m: float = 10_000.0
    velocity_scale_mps: float = 1_000.0
    altitude_scale_m: float = 10_000.0
    missile_mass_scale_kg: float = 100.0
    los_rate_scale_radps: float = 0.20
    observation_clip: float = 10.0

    # Reward.  Predictive-CPA progress is signed to avoid oscillation exploits.
    predictive_cpa_horizon_s: float = 35.0
    predictive_cpa_progress_weight: float = 1.0
    predictive_cpa_progress_scale_m: float = 100.0
    predictive_cpa_progress_clip: float = 2.5
    survival_time_reward: float = 0.02
    control_effort_penalty: float = 0.02
    action_smoothness_penalty: float = 0.01
    low_altitude_penalty: float = 0.02
    soft_minimum_altitude_m: float = 300.0
    terminal_survival_reward: float = 30.0
    terminal_intercept_penalty: float = 30.0
    terminal_target_failure_penalty: float = 30.0
    terminal_distance_weight: float = 10.0
    terminal_distance_scale_m: float = 25.0

    # Histories are disabled in vectorized training and enabled for plotting.
    record_history: bool = False


@dataclass
class EvaderEpisodeResult:
    time_s: np.ndarray
    missile_state: np.ndarray
    target_state: np.ndarray
    range_m: np.ndarray
    predictive_cpa_m: np.ndarray
    missile_command_g: np.ndarray
    target_command_g: np.ndarray
    action: np.ndarray
    actual_rates_deg_s: np.ndarray
    status: str
    survived: bool
    minimum_range_m: float
    terminal_time_s: float
    hit_radius_m: float


def build_default_evader_scenario() -> tuple[
    EnvironmentConfig,
    list[VehicleSpec],
]:
    """Return the current nominal NED missile/aircraft scenario.

    This mirrors the physical defaults used by the current project, but the
    target's scripted maneuver is disabled because PPO supplies its command.
    """

    environment = EnvironmentConfig(
        dt_s=0.02,
        t_final_s=35.0,
        gravity_mps2=G0,
        rho0_kgpm3=1.225,
        atmosphere_scale_height_m=8500.0,
        ground_altitude_m=0.0,
        hit_radius_m=10.0,
        minimum_flying_speed_mps=60.0,
    )

    target = VehicleSpec(
        name="aircraft_1",
        role="aircraft",
        initial=InitialState(
            position_m=(5000.0, 900.0, -3300.0),
            speed_mps=220.0,
            azimuth_deg=195.0,
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
        maneuver=ManeuverConfig(kind="straight"),
    )

    missile = VehicleSpec(
        name="missile_1",
        role="missile",
        target_name="aircraft_1",
        initial=InitialState(
            position_m=(0.0, 0.0, -3000.0),
            speed_mps=320.0,
            azimuth_deg=5.0,
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
    return environment, [missile, target]


def constant_velocity_closest_approach(
    relative_position_m: np.ndarray,
    relative_velocity_mps: np.ndarray,
    horizon_s: float,
) -> tuple[float, float]:
    """Return predicted CPA distance and time under constant relative velocity."""

    relative_position_m = np.asarray(relative_position_m, dtype=float)
    relative_velocity_mps = np.asarray(relative_velocity_mps, dtype=float)
    velocity_squared = float(np.dot(relative_velocity_mps, relative_velocity_mps))
    if velocity_squared < 1e-12:
        return float(np.linalg.norm(relative_position_m)), 0.0

    time_to_cpa_s = -float(
        np.dot(relative_position_m, relative_velocity_mps)
    ) / velocity_squared
    time_to_cpa_s = float(np.clip(time_to_cpa_s, 0.0, max(horizon_s, 0.0)))
    miss_vector_m = relative_position_m + time_to_cpa_s * relative_velocity_mps
    return float(np.linalg.norm(miss_vector_m)), time_to_cpa_s


def evader_rates_to_normal_acceleration(
    target_state: np.ndarray,
    heading_rate_radps: float,
    flight_path_rate_radps: float,
) -> np.ndarray:
    """Convert target chi/gamma rates to a NED net-normal acceleration."""

    _, e_chi, e_gamma = flight_path_basis(target_state)
    speed_mps = float(target_state[SPEED])
    gamma_rad = float(target_state[GAMMA])
    return (
        speed_mps * np.cos(gamma_rad) * heading_rate_radps * e_chi
        + speed_mps * flight_path_rate_radps * e_gamma
    )


class MissileEvaderEnv(gym.Env):
    """PPO target aircraft versus a fixed Vector-PN missile."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        environment: EnvironmentConfig,
        vehicles: list[VehicleSpec],
        config: EvaderEnvironmentConfig | None = None,
        missile_name: str = "missile_1",
    ) -> None:
        super().__init__()
        self.environment_template = deepcopy(environment)
        self.vehicle_templates = deepcopy(vehicles)
        self.config = deepcopy(config or EvaderEnvironmentConfig())
        self.missile_name = missile_name

        template_specs = {vehicle.name: vehicle for vehicle in self.vehicle_templates}
        if missile_name not in template_specs:
            raise KeyError(f"Missile not found: {missile_name!r}")
        missile = template_specs[missile_name]
        if missile.role != "missile" or missile.target_name not in template_specs:
            raise ValueError("The controlled scenario needs one missile and its target.")
        self.target_name = str(missile.target_name)
        if template_specs[self.target_name].role != "aircraft":
            raise ValueError("The missile target must have role='aircraft'.")
        if set(template_specs) != {self.missile_name, self.target_name}:
            raise ValueError("MissileEvaderEnv currently accepts exactly two vehicles.")

        if self.environment_template.dt_s <= 0.0:
            raise ValueError("environment.dt_s must be positive.")
        if self.config.control_period_s <= 0.0:
            raise ValueError("control_period_s must be positive.")
        if self.config.max_heading_rate_deg_s <= 0.0:
            raise ValueError("max_heading_rate_deg_s must be positive.")
        if self.config.max_flight_path_rate_deg_s <= 0.0:
            raise ValueError("max_flight_path_rate_deg_s must be positive.")
        if self.config.actuator_time_constant_s < 0.0:
            raise ValueError("actuator_time_constant_s cannot be negative.")
        if self.config.missile_actuator_time_constant_s < 0.0:
            raise ValueError(
                "missile_actuator_time_constant_s cannot be negative."
            )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(EVADER_ACTION_SIZE,),
            dtype=np.float32,
        )
        clip = float(self.config.observation_clip)
        self.observation_space = spaces.Box(
            low=-clip,
            high=clip,
            shape=(EVADER_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.environment = deepcopy(self.environment_template)
        self.specs: dict[str, VehicleSpec] = {}
        self.states: dict[str, np.ndarray] = {}
        self.pn_guidance = VectorPNGuidance(
            PNConfig(
                navigation_constant=self.config.pn_navigation_constant,
                max_command_g=self.config.pn_max_command_g,
            )
        )
        self.time_s = 0.0
        self.status = "not_reset"
        self.minimum_range_m = np.inf
        self.terminal_time_s = np.nan
        self.previous_predictive_cpa_m = 0.0
        self.previous_action = np.zeros(EVADER_ACTION_SIZE, dtype=float)
        self.actual_heading_rate_radps = 0.0
        self.actual_flight_path_rate_radps = 0.0
        self.applied_missile_normal_accel_mps2 = np.zeros(
            3,
            dtype=float,
        )
        self._closest_approach_armed = False
        self._intercept_occurred = False
        self._first_intercept_time_s = np.nan
        self._episode_done = False
        self._last_reward_terms: dict[str, float] = {}
        self._reset_histories()

    def _reset_histories(self) -> None:
        self._time_history: list[float] = []
        self._missile_state_history: list[np.ndarray] = []
        self._target_state_history: list[np.ndarray] = []
        self._range_history: list[float] = []
        self._predictive_cpa_history: list[float] = []
        self._missile_command_g_history: list[float] = []
        self._target_command_g_history: list[float] = []
        self._action_history: list[np.ndarray] = []
        self._actual_rate_history: list[np.ndarray] = []

    def _apply_initial_randomization(self) -> None:
        if not self.config.randomize_initial_conditions:
            return

        missile = self.specs[self.missile_name]
        target = self.specs[self.target_name]
        for vehicle, jitter in (
            (missile, self.config.missile_position_jitter_m),
            (target, self.config.target_position_jitter_m),
        ):
            jitter_array = np.asarray(jitter, dtype=float)
            if jitter_array.shape != (3,) or np.any(jitter_array < 0.0):
                raise ValueError("Position jitter must be a nonnegative 3-vector.")
            position = np.asarray(vehicle.initial.position_m, dtype=float)
            position += self.np_random.uniform(-jitter_array, jitter_array)
            minimum_altitude = max(
                self.environment.ground_altitude_m
                + self.config.minimum_initial_altitude_m,
                1.0,
            )
            position[DOWN] = min(position[DOWN], -minimum_altitude)
            vehicle.initial.position_m = tuple(float(value) for value in position)

        missile.initial.speed_mps = max(
            self.environment.minimum_flying_speed_mps + 1.0,
            missile.initial.speed_mps
            + self.np_random.uniform(
                -self.config.missile_speed_jitter_mps,
                self.config.missile_speed_jitter_mps,
            ),
        )
        target.initial.speed_mps = max(
            self.environment.minimum_flying_speed_mps + 1.0,
            target.initial.speed_mps
            + self.np_random.uniform(
                -self.config.target_speed_jitter_mps,
                self.config.target_speed_jitter_mps,
            ),
        )
        missile.initial.azimuth_deg += self.np_random.uniform(
            -self.config.missile_heading_jitter_deg,
            self.config.missile_heading_jitter_deg,
        )
        target.initial.azimuth_deg += self.np_random.uniform(
            -self.config.target_heading_jitter_deg,
            self.config.target_heading_jitter_deg,
        )
        missile.initial.flight_path_deg += self.np_random.uniform(
            -self.config.missile_flight_path_jitter_deg,
            self.config.missile_flight_path_jitter_deg,
        )
        target.initial.flight_path_deg += self.np_random.uniform(
            -self.config.target_flight_path_jitter_deg,
            self.config.target_flight_path_jitter_deg,
        )
        if target.speed_hold_mps is not None:
            target.speed_hold_mps = target.initial.speed_mps

    def _relative_kinematics(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        missile_state = self.states[self.missile_name]
        target_state = self.states[self.target_name]
        relative_position = missile_state[POSITION] - target_state[POSITION]
        relative_velocity = (
            velocity_from_state(missile_state) - velocity_from_state(target_state)
        )
        range_m = float(np.linalg.norm(relative_position))
        if range_m < 1e-12:
            closing_speed_mps = 0.0
        else:
            closing_speed_mps = -float(
                np.dot(relative_velocity, relative_position / range_m)
            )
        return relative_position, relative_velocity, range_m, closing_speed_mps

    def _predictive_cpa(self) -> tuple[float, float]:
        relative_position, relative_velocity, _, _ = self._relative_kinematics()
        remaining_time_s = max(self.environment.t_final_s - self.time_s, 0.0)
        horizon_s = min(self.config.predictive_cpa_horizon_s, remaining_time_s)
        return constant_velocity_closest_approach(
            relative_position,
            relative_velocity,
            horizon_s,
        )

    def _get_observation(self) -> np.ndarray:
        missile_state = self.states[self.missile_name]
        target_state = self.states[self.target_name]
        relative_position, relative_velocity, range_m, closing_speed_mps = (
            self._relative_kinematics()
        )

        e_velocity, e_chi, e_gamma = flight_path_basis(target_state)
        local_basis = np.vstack((e_velocity, e_chi, e_gamma))
        position_local = local_basis @ relative_position
        velocity_local = local_basis @ relative_velocity

        if range_m < 1e-12:
            los_rate = np.zeros(3)
        else:
            los_rate = np.cross(relative_position, relative_velocity) / range_m**2
        los_rate_local = local_basis @ los_rate

        relative_heading = float(missile_state[CHI] - target_state[CHI])
        max_heading_rate = self.config.max_heading_rate_deg_s * DEG
        max_flight_path_rate = self.config.max_flight_path_rate_deg_s * DEG

        observation = np.concatenate(
            (
                position_local / self.config.position_scale_m,
                velocity_local / self.config.velocity_scale_mps,
                los_rate_local / self.config.los_rate_scale_radps,
                np.array(
                    [
                        range_m / self.config.position_scale_m,
                        closing_speed_mps / self.config.velocity_scale_mps,
                        target_state[SPEED] / self.config.velocity_scale_mps,
                        missile_state[SPEED] / self.config.velocity_scale_mps,
                        altitude_from_down(target_state[DOWN])
                        / self.config.altitude_scale_m,
                        altitude_from_down(missile_state[DOWN])
                        / self.config.altitude_scale_m,
                        np.sin(target_state[GAMMA]),
                        np.cos(target_state[GAMMA]),
                        np.sin(missile_state[GAMMA]),
                        np.cos(missile_state[GAMMA]),
                        np.sin(relative_heading),
                        np.cos(relative_heading),
                        self.time_s / self.environment.t_final_s,
                        self.actual_heading_rate_radps / max_heading_rate,
                        self.actual_flight_path_rate_radps / max_flight_path_rate,
                        missile_state[MASS] / self.config.missile_mass_scale_kg,
                    ],
                    dtype=float,
                ),
                self.previous_action,
            )
        )
        if observation.shape != (EVADER_OBSERVATION_SIZE,):
            raise RuntimeError(
                f"Evader observation has shape {observation.shape}; "
                f"expected {(EVADER_OBSERVATION_SIZE,)}."
            )
        observation = np.nan_to_num(
            observation,
            nan=0.0,
            posinf=self.config.observation_clip,
            neginf=-self.config.observation_clip,
        )
        observation = np.clip(
            observation,
            -self.config.observation_clip,
            self.config.observation_clip,
        )
        return observation.astype(np.float32)

    def _is_survival_status(self) -> bool:
        return self.status.startswith("survived:")

    def _get_info(
        self,
        missile_command_g: float = 0.0,
        target_command_g: float = 0.0,
    ) -> dict[str, Any]:
        _, _, range_m, closing_speed_mps = self._relative_kinematics()
        predictive_cpa_m, predictive_cpa_time_s = self._predictive_cpa()
        return {
            "observation_version": EVADER_OBSERVATION_VERSION,
            "status": self.status,
            "survived": self._is_survival_status(),
            "intercept_occurred": bool(self._intercept_occurred),
            "first_intercept_time_s": float(self._first_intercept_time_s),
            "time_s": float(self.time_s),
            "range_m": float(range_m),
            "minimum_range_m": float(self.minimum_range_m),
            "closing_speed_mps": float(closing_speed_mps),
            "predictive_cpa_m": float(predictive_cpa_m),
            "predictive_cpa_time_s": float(predictive_cpa_time_s),
            "missile_command_g": float(missile_command_g),
            "target_command_g": float(target_command_g),
            "actual_heading_rate_deg_s": float(
                self.actual_heading_rate_radps / DEG
            ),
            "actual_flight_path_rate_deg_s": float(
                self.actual_flight_path_rate_radps / DEG
            ),
            "reward_terms": dict(self._last_reward_terms),
        }

    def _record_sample(
        self,
        missile_command_g: float,
        target_command_g: float,
        action: np.ndarray,
    ) -> None:
        if not self.config.record_history:
            return
        _, _, range_m, _ = self._relative_kinematics()
        predictive_cpa_m, _ = self._predictive_cpa()
        self._time_history.append(float(self.time_s))
        self._missile_state_history.append(
            self.states[self.missile_name].copy()
        )
        self._target_state_history.append(self.states[self.target_name].copy())
        self._range_history.append(float(range_m))
        self._predictive_cpa_history.append(float(predictive_cpa_m))
        self._missile_command_g_history.append(float(missile_command_g))
        self._target_command_g_history.append(float(target_command_g))
        self._action_history.append(np.asarray(action, dtype=float).copy())
        self._actual_rate_history.append(
            np.array(
                [
                    self.actual_heading_rate_radps / DEG,
                    self.actual_flight_path_rate_radps / DEG,
                ],
                dtype=float,
            )
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self.environment = deepcopy(self.environment_template)
        self.specs = {
            vehicle.name: vehicle
            for vehicle in deepcopy(self.vehicle_templates)
        }
        self._apply_initial_randomization()
        self.states = {
            name: make_initial_state(vehicle.initial)
            for name, vehicle in self.specs.items()
        }
        self.pn_guidance = VectorPNGuidance(
            PNConfig(
                navigation_constant=self.config.pn_navigation_constant,
                max_command_g=self.config.pn_max_command_g,
            )
        )
        self.pn_guidance.reset()

        self.time_s = 0.0
        self.status = "flying"
        self.terminal_time_s = np.nan
        self.previous_action = np.zeros(EVADER_ACTION_SIZE, dtype=float)
        self.actual_heading_rate_radps = 0.0
        self.actual_flight_path_rate_radps = 0.0
        self.applied_missile_normal_accel_mps2 = np.zeros(
            3,
            dtype=float,
        )
        self._closest_approach_armed = (
            self.config.passed_missile_arm_progress_m <= 0.0
        )
        self._intercept_occurred = False
        self._first_intercept_time_s = np.nan
        self._episode_done = False
        self._last_reward_terms = {}

        _, _, initial_range_m, _ = self._relative_kinematics()
        self.initial_range_m = initial_range_m
        self.minimum_range_m = initial_range_m
        self.previous_predictive_cpa_m, _ = self._predictive_cpa()

        self._reset_histories()
        self._record_sample(0.0, 0.0, np.zeros(EVADER_ACTION_SIZE))
        return self._get_observation(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("Episode is done; call reset() before step().")

        normalized_action = np.asarray(action, dtype=float).reshape(-1)
        if normalized_action.shape != (EVADER_ACTION_SIZE,):
            raise ValueError("Evader action must have shape=(2,).")
        normalized_action = np.clip(normalized_action, -1.0, 1.0)

        desired_heading_rate = (
            normalized_action[0] * self.config.max_heading_rate_deg_s * DEG
        )
        desired_flight_path_rate = (
            normalized_action[1] * self.config.max_flight_path_rate_deg_s * DEG
        )

        remaining_control_time_s = min(
            self.config.control_period_s,
            self.environment.t_final_s - self.time_s,
        )
        controlled_duration_s = remaining_control_time_s
        terminated = False
        truncated = False
        last_missile_command_g = 0.0
        last_target_command_g = 0.0

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

            if self.config.actuator_time_constant_s <= 1e-12:
                response_fraction = 1.0
            else:
                response_fraction = 1.0 - np.exp(
                    -physics_dt_s / self.config.actuator_time_constant_s
                )
            self.actual_heading_rate_radps += response_fraction * (
                desired_heading_rate - self.actual_heading_rate_radps
            )
            self.actual_flight_path_rate_radps += response_fraction * (
                desired_flight_path_rate - self.actual_flight_path_rate_radps
            )

            guidance_observation = GuidanceObservation(
                time_s=self.time_s,
                dt_s=physics_dt_s,
                missile_state=old_missile_state,
                target_state=old_target_state,
                environment=self.environment,
            )
            raw_missile_command = np.asarray(
                self.pn_guidance.command(
                    guidance_observation
                ).normal_accel_mps2,
                dtype=float,
            )

            missile_tau_s = float(
                self.config.missile_actuator_time_constant_s
            )

            if missile_tau_s <= 1e-12:
                self.applied_missile_normal_accel_mps2 = (
                    raw_missile_command.copy()
                )
            else:
                missile_response_fraction = 1.0 - np.exp(
                    -physics_dt_s / missile_tau_s
                )

                self.applied_missile_normal_accel_mps2 += (
                    missile_response_fraction
                    * (
                        raw_missile_command
                        - self.applied_missile_normal_accel_mps2
                    )
                )

                # Keep the filtered acceleration perpendicular to the
                # current missile velocity, as required by the 3DOF
                # normal-acceleration input.
                missile_velocity = velocity_from_state(
                    old_missile_state
                )
                missile_speed = float(
                    np.linalg.norm(missile_velocity)
                )

                if missile_speed > 1e-12:
                    missile_velocity_unit = (
                        missile_velocity / missile_speed
                    )
                    self.applied_missile_normal_accel_mps2 -= (
                        np.dot(
                            self.applied_missile_normal_accel_mps2,
                            missile_velocity_unit,
                        )
                        * missile_velocity_unit
                    )

            missile_command = (
                self.applied_missile_normal_accel_mps2.copy()
            )

            target_command = evader_rates_to_normal_acceleration(
                old_target_state,
                self.actual_heading_rate_radps,
                self.actual_flight_path_rate_radps,
            )

            new_missile_state = rk4_step(
                time_s=self.time_s,
                state=old_missile_state,
                dt_s=physics_dt_s,
                vehicle=self.specs[self.missile_name],
                env=self.environment,
                normal_accel_command_mps2=missile_command,
            )
            new_target_state = rk4_step(
                time_s=self.time_s,
                state=old_target_state,
                dt_s=physics_dt_s,
                vehicle=self.specs[self.target_name],
                env=self.environment,
                normal_accel_command_mps2=target_command,
            )

            relative_start = old_target_state[POSITION] - old_missile_state[POSITION]
            relative_end = new_target_state[POSITION] - new_missile_state[POSITION]
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
                >= self.config.passed_missile_arm_progress_m
            ):
                self._closest_approach_armed = True

            step_start_time_s = self.time_s
            self.time_s += physics_dt_s
            remaining_control_time_s -= physics_dt_s

            last_missile_command_g = float(
                np.linalg.norm(missile_command) / self.environment.gravity_mps2
            )
            last_target_command_g = float(
                np.linalg.norm(target_command) / self.environment.gravity_mps2
            )
            self._record_sample(
                last_missile_command_g,
                last_target_command_g,
                normalized_action,
            )

            if minimum_step_distance_m <= self.environment.hit_radius_m:
                if not self._intercept_occurred:
                    self._intercept_occurred = True
                    self._first_intercept_time_s = (
                        step_start_time_s + tau * physics_dt_s
                    )
                if not self.config.closest_approach_training:
                    self.status = "intercepted"
                    self.terminal_time_s = self._first_intercept_time_s
                    terminated = True
            elif new_target_state[DOWN] >= -self.environment.ground_altitude_m:
                self.status = (
                    "intercepted"
                    if self._intercept_occurred
                    else "failed: target ground"
                )
                self.terminal_time_s = self.time_s
                terminated = True
            elif (
                new_target_state[SPEED]
                <= self.environment.minimum_flying_speed_mps
            ):
                self.status = (
                    "intercepted"
                    if self._intercept_occurred
                    else "failed: target low speed"
                )
                self.terminal_time_s = self.time_s
                terminated = True
            elif new_missile_state[DOWN] >= -self.environment.ground_altitude_m:
                self.status = (
                    "intercepted"
                    if self._intercept_occurred
                    else "survived: missile ground"
                )
                self.terminal_time_s = self.time_s
                terminated = True
            elif (
                new_missile_state[SPEED]
                <= self.environment.minimum_flying_speed_mps
            ):
                self.status = (
                    "intercepted"
                    if self._intercept_occurred
                    else "survived: missile low speed"
                )
                self.terminal_time_s = self.time_s
                terminated = True

            if not terminated and self._closest_approach_armed:
                _, _, current_range_m, closing_speed_mps = (
                    self._relative_kinematics()
                )
                if (
                    current_range_m - self.minimum_range_m
                    >= self.config.closest_approach_rebound_m
                    and closing_speed_mps
                    <= -self.config.minimum_opening_speed_mps
                ):
                    self.status = (
                        "intercepted"
                        if self._intercept_occurred
                        else "survived: passed missile"
                    )
                    self.terminal_time_s = self.time_s
                    terminated = True

            if terminated:
                break

        if (
            not terminated
            and self.time_s >= self.environment.t_final_s - 1e-12
        ):
            self.status = (
                "intercepted"
                if self._intercept_occurred
                else "survived: timeout"
            )
            self.terminal_time_s = self.time_s
            truncated = True

        current_predictive_cpa_m, _ = self._predictive_cpa()
        cpa_progress = (
            current_predictive_cpa_m - self.previous_predictive_cpa_m
        ) / self.config.predictive_cpa_progress_scale_m
        cpa_progress = float(
            np.clip(
                cpa_progress,
                -self.config.predictive_cpa_progress_clip,
                self.config.predictive_cpa_progress_clip,
            )
        )
        predictive_cpa_term = (
            self.config.predictive_cpa_progress_weight * cpa_progress
        )

        elapsed_control_time_s = controlled_duration_s - remaining_control_time_s
        elapsed_fraction = elapsed_control_time_s / self.config.control_period_s
        survival_time_term = self.config.survival_time_reward * elapsed_fraction
        control_term = -self.config.control_effort_penalty * float(
            np.mean(normalized_action**2)
        )
        smoothness_term = -self.config.action_smoothness_penalty * float(
            np.mean((normalized_action - self.previous_action) ** 2)
        )

        target_altitude_m = altitude_from_down(
            self.states[self.target_name][DOWN]
        )
        altitude_deficit_fraction = max(
            self.config.soft_minimum_altitude_m - target_altitude_m,
            0.0,
        ) / max(self.config.soft_minimum_altitude_m, 1.0)
        altitude_term = -self.config.low_altitude_penalty * (
            altitude_deficit_fraction**2
        )

        terminal_term = 0.0
        terminal_distance_term = 0.0
        if terminated or truncated:
            signed_safety_margin_m = (
                self.minimum_range_m - self.environment.hit_radius_m
            )
            terminal_distance_term = (
                self.config.terminal_distance_weight
                * np.tanh(
                    signed_safety_margin_m
                    / max(self.config.terminal_distance_scale_m, 1e-6)
                )
            )
            if self._is_survival_status():
                terminal_term += self.config.terminal_survival_reward
            elif self.status == "intercepted":
                terminal_term -= self.config.terminal_intercept_penalty
            else:
                terminal_term -= self.config.terminal_target_failure_penalty

        reward = float(
            predictive_cpa_term
            + survival_time_term
            + control_term
            + smoothness_term
            + altitude_term
            + terminal_term
            + terminal_distance_term
        )
        self._last_reward_terms = {
            "predictive_cpa_progress": float(predictive_cpa_term),
            "survival_time": float(survival_time_term),
            "control_effort": float(control_term),
            "action_smoothness": float(smoothness_term),
            "low_altitude": float(altitude_term),
            "terminal": float(terminal_term),
            "terminal_distance": float(terminal_distance_term),
            "total": reward,
        }

        self.previous_predictive_cpa_m = current_predictive_cpa_m
        self.previous_action = normalized_action.copy()
        self._episode_done = terminated or truncated
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(last_missile_command_g, last_target_command_g),
        )

    def episode_result(self) -> EvaderEpisodeResult:
        if not self.config.record_history:
            raise RuntimeError(
                "Set EvaderEnvironmentConfig(record_history=True) before reset()."
            )
        if not self._time_history:
            raise RuntimeError("No episode history is available.")
        return EvaderEpisodeResult(
            time_s=np.asarray(self._time_history, dtype=float),
            missile_state=np.asarray(self._missile_state_history, dtype=float),
            target_state=np.asarray(self._target_state_history, dtype=float),
            range_m=np.asarray(self._range_history, dtype=float),
            predictive_cpa_m=np.asarray(self._predictive_cpa_history, dtype=float),
            missile_command_g=np.asarray(
                self._missile_command_g_history,
                dtype=float,
            ),
            target_command_g=np.asarray(
                self._target_command_g_history,
                dtype=float,
            ),
            action=np.asarray(self._action_history, dtype=float),
            actual_rates_deg_s=np.asarray(self._actual_rate_history, dtype=float),
            status=self.status,
            survived=self._is_survival_status(),
            minimum_range_m=float(self.minimum_range_m),
            terminal_time_s=float(self.terminal_time_s),
            hit_radius_m=float(self.environment.hit_radius_m),
        )

    def render(self):
        return None

    def close(self) -> None:
        return None


def build_default_evader_environment(
    config: EvaderEnvironmentConfig | None = None,
    hit_radius_m: float = 10.0,
) -> MissileEvaderEnv:
    environment, vehicles = build_default_evader_scenario()
    environment.hit_radius_m = float(hit_radius_m)
    return MissileEvaderEnv(
        environment=environment,
        vehicles=vehicles,
        config=config,
    )


def plot_evader_episode(
    result: EvaderEpisodeResult,
    *,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """Create one compact six-panel diagnostic dashboard."""

    import matplotlib.pyplot as plt

    time_s = result.time_s
    missile = result.missile_state
    target = result.target_state
    figure = plt.figure(figsize=(18, 10))

    axis_3d = figure.add_subplot(2, 3, 1, projection="3d")
    axis_3d.plot(
        missile[:, 0], missile[:, 1], -missile[:, 2], label="PN missile"
    )
    axis_3d.plot(
        target[:, 0], target[:, 1], -target[:, 2], label="RL target"
    )
    closest_index = int(np.argmin(result.range_m))
    axis_3d.scatter(
        target[closest_index, 0],
        target[closest_index, 1],
        -target[closest_index, 2],
        color="black",
        s=35,
        label="closest approach",
    )
    axis_3d.set_xlabel("North [m]")
    axis_3d.set_ylabel("East [m]")
    axis_3d.set_zlabel("Altitude [m]")
    axis_3d.set_title("3D trajectories")
    axis_3d.legend()

    axis = figure.add_subplot(2, 3, 2)
    axis.plot(time_s, result.range_m, label="range")
    axis.axhline(result.hit_radius_m, color="red", linestyle="--", label="hit radius")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Distance [m]")
    axis.set_title(f"Range | min={result.minimum_range_m:.2f} m")
    axis.grid(True)
    axis.legend()

    axis = figure.add_subplot(2, 3, 3)
    axis.plot(time_s, result.predictive_cpa_m, label="predicted CPA")
    axis.axhline(result.hit_radius_m, color="red", linestyle="--")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Distance [m]")
    axis.set_title("Constant-velocity predicted CPA")
    axis.grid(True)
    axis.legend()

    axis = figure.add_subplot(2, 3, 4)
    axis.plot(time_s, result.action[:, 0], label="heading action")
    axis.plot(time_s, result.action[:, 1], label="flight-path action")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Normalized action")
    axis.set_ylim(-1.1, 1.1)
    axis.set_title("PPO target action")
    axis.grid(True)
    axis.legend()

    axis = figure.add_subplot(2, 3, 5)
    axis.plot(time_s, result.actual_rates_deg_s[:, 0], label="heading rate")
    axis.plot(time_s, result.actual_rates_deg_s[:, 1], label="flight-path rate")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Rate [deg/s]")
    axis.set_title("Realized target rates")
    axis.grid(True)
    axis.legend()

    axis = figure.add_subplot(2, 3, 6)
    axis.plot(time_s, result.missile_command_g, label="missile PN")
    axis.plot(time_s, result.target_command_g, label="target")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Normal acceleration [g]")
    axis.set_title("Command magnitude")
    axis.grid(True)
    axis.legend()

    if title is None:
        title = (
            f"Evader episode | {result.status} | "
            f"minimum range={result.minimum_range_m:.2f} m"
        )
    figure.suptitle(title, fontsize=15)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    if save_path is not None:
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure
