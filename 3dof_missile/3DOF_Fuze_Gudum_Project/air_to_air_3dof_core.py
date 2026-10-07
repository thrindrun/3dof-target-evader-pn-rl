from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

G0 = 9.80665
DEG = np.pi / 180.0

NORTH, EAST, DOWN, SPEED, CHI, GAMMA, MASS = range(7)

POSITION = slice(NORTH, DOWN + 1)

RL_OBSERVATION_LEGACY10 = "legacy10"
RL_OBSERVATION_MARKOV13 = "markov13"
RL_OBSERVATION_V2 = "v2"
RL_OBSERVATION_SIZES = {
    RL_OBSERVATION_LEGACY10: 10,
    RL_OBSERVATION_MARKOV13: 13,
    RL_OBSERVATION_V2: 23,
}

def wrap_angle_deg(angle_deg: float) -> float:
    """Bir açıyı [-180, 180) derece aralığına sarar."""
    return float((angle_deg + 180.0) % 360.0 - 180.0)


# =============================================================================
# 1. KONFIGURASYON SINIFLARI
# =============================================================================


@dataclass
class EnvironmentConfig:
    """Simülasyon boyunca ortak kullanılan çevre ve çözücü ayarları."""

    # Sabit integrasyon adımı. Daha küçük değer daha hassas fakat daha yavaştır.
    dt_s: float = 0.02

    # Bir füze daha önce terminal duruma geçmezse simülasyon bu zamanda biter.
    t_final_s: float = 35.0

    # NED'de yerçekimi vektörü +Down yönündedir: g_NED = [0, 0, +g].
    gravity_mps2: float = G0

    # Deniz seviyesindeki yoğunluk ve basit üstel atmosfer ölçek yüksekliği.
    rho0_kgpm3: float = 1.225
    atmosphere_scale_height_m: float = 8500.0

    # Arazinin deniz seviyesine göre irtifasıdır. NED Down koordinatı değildir.
    # Örneğin ground_altitude_m=500 ise arazi düzlemi D=-500 m olur.
    ground_altitude_m: float = 0.0

    # Füze-hedef adım içi minimum mesafesi bu değerin altına inerse "hit" olur.
    hit_radius_m: float = 10.0

    # Füze bu hızın altına düşerse kontrol edilebilir uçuş bitti kabul edilir.
    minimum_flying_speed_mps: float = 60.0


@dataclass
class InitialState:
    """Bir aracın t=0 anındaki kullanıcı dostu başlangıç koşulları."""

    # Sıra kesinlikle (North, East, Down) şeklindedir; birim metredir.
    position_m: tuple[float, float, float]
    speed_mps: float

    # Heading: kuzeyden doğuya pozitif, derece cinsinden.
    azimuth_deg: float

    # Pozitif değer tırmanıştır. NED Down hızı bu durumda negatif olur.
    flight_path_deg: float
    mass_kg: float


@dataclass
class AeroConfig:
    reference_area_m2: float
    drag_coefficient: float
    max_normal_accel_g: float


@dataclass
class PropulsionConfig:
    # İlk sürüm: sabit itki + sabit kütle debisi.
    thrust_n: float = 0.0
    burn_time_s: float = 0.0
    mass_flow_kgps: float = 0.0
    dry_mass_kg: float = 0.0
    max_thrust_n: float = np.inf


@dataclass
class ManeuverConfig:
    """
    Hedef uçağın ideal net-normal ivme komutunu tanımlar.

    kind:
        "straight"      -> baş ve uçuş-yolu açısı sabit
        "constant_turn" -> sabit chi_dot ve gamma_dot
        "sinusoidal"    -> sinüzoidal chi_dot ve gamma_dot
        "random_smooth" -> rastgele fakat düzgün chi_dot ve gamma_dot

    Sinüzoidal model konumun doğrudan sinüsü değildir. Dönüş ve tırmanış oranı
    sinüsle değiştiği için hız yönü sürekli değişir ve ortaya düzgün, doğrusal
    olmayan bir 3B yörünge çıkar.
    """

    kind: str = "straight"

    # constant_turn için sabit oran; sinusoidal için tepe/genlik değeridir.
    heading_rate_deg_s: float = 0.0
    flight_path_rate_deg_s: float = 0.0

    # Sinüzoidal manevranın periyodu ve iki eksendeki başlangıç fazları.
    weave_period_s: float = 8.0
    heading_phase_deg: float = 0.0
    flight_path_phase_deg: float = 90.0

    # random_smooth manevrası için kullanılan seed aynı kaldığı sürece aynı
    # rastgele yol tekrar üretilir. Seed değiştirilirse farklı bir yol elde edilir.
    random_seed: int = 42

    # Rastgele düzgün sinyal, farklı frekans ve fazlara sahip birkaç sinüsün
    # toplamıyla oluşturulur. Bileşen sayısı arttıkça yol daha karmaşık hale gelir.
    random_harmonics: int = 4


@dataclass
class VehicleSpec:
    name: str
    role: str  # "missile" veya "aircraft"
    initial: InitialState
    aero: AeroConfig
    propulsion: PropulsionConfig = field(default_factory=PropulsionConfig)

    # Yalnızca füze için kullanılır.
    target_name: str | None = None

    # Uçak için basit boylamsal hız tutucu ve manevra ayarları.
    speed_hold_mps: float | None = None
    speed_hold_gain: float = 0.8
    max_longitudinal_accel_mps2: float = 3.0
    maneuver: ManeuverConfig = field(default_factory=ManeuverConfig)


@dataclass
class SimulationResult:
    time_s: np.ndarray
    states: dict[str, np.ndarray]
    range_m: dict[str, np.ndarray]
    command_g: dict[str, np.ndarray]
    status: dict[str, str]
    terminal_time_s: dict[str, float]
    terminal_index: dict[str, int]
    target_of: dict[str, str]
    hit_radius_m: float
    closest_approach_m: dict[str, float]
    closest_approach_time_s: dict[str, float]


# =============================================================================
# 2. DURUM VE 3DOF ARAÇ DİNAMİĞİ
# =============================================================================


def make_initial_state(initial: InitialState) -> np.ndarray:
    """
    InitialState nesnesini sayısal NED durum dizisine çevirir.

    Kullanıcı açıları dereceyle girer; dinamik denklemleri ise radyan kullanır.
    Dönüşüm yalnızca burada yapıldığı için simülasyonun geri kalanında açı birimi
    karışıklığı oluşmaz.
    """
    return np.array(
        [
            *initial.position_m,
            initial.speed_mps,
            initial.azimuth_deg * DEG,
            initial.flight_path_deg * DEG,
            initial.mass_kg,
        ],
        dtype=float,
    )


def flight_path_basis(state: np.ndarray):
    """
    Füzenin yerel uçuş-yolu eksenlerini NED bileşenleriyle döndürür.

    Dönen üç vektör birbirine diktir:

        e_velocity : hız doğrultusu
        e_chi      : baş açısını artıran yatay-normal doğrultu
        e_gamma    : uçuş-yolu açısını artıran, yani tırmanış doğrultusu

    Gamma pozitif tırmanış olduğundan e_velocity'nin Down bileşeni
    -sin(gamma)'dır. Benzer biçimde e_gamma'nın Down bileşeni -cos(gamma)'dır.
    Bu iki eksi işareti NED dönüşümünün en önemli kısmıdır.
    """
    chi = state[CHI]
    gamma = state[GAMMA]

    e_velocity = np.array(
        [
            np.cos(gamma) * np.cos(chi),
            np.cos(gamma) * np.sin(chi),
            -np.sin(gamma),
        ]
    )

    # chi arttığında hız vektörünün yatay düzlemde yöneldiği doğrultu.
    e_chi = np.array([-np.sin(chi), np.cos(chi), 0.0])

    # gamma arttığında hız vektörünün yöneldiği doğrultu. Son bileşen negatiftir,
    # çünkü pozitif gamma tırmanış, tırmanış ise negatif Down hareketidir.
    e_gamma = np.array(
        [
            -np.sin(gamma) * np.cos(chi),
            -np.sin(gamma) * np.sin(chi),
            -np.cos(gamma),
        ]
    )
    return e_velocity, e_chi, e_gamma


def velocity_from_state(state: np.ndarray) -> np.ndarray:
    """Skaler hız ve uçuş açılarını [V_N, V_E, V_D] vektörüne çevirir."""
    e_velocity, _, _ = flight_path_basis(state)
    return state[SPEED] * e_velocity


def altitude_from_down(down_m: float) -> float:
    """Deniz-seviyesi referanslı NED Down konumunu irtifaya çevirir: h=-D."""
    return -float(down_m)


def air_density_from_down(down_m: float, env: EnvironmentConfig) -> float:
    """
    NED Down koordinatından basit üstel atmosfer yoğunluğunu hesaplar.

    Yoğunluk modeli deniz seviyesini referans alır. Negatif irtifalarda üstel
    modelin gereksiz büyümesini engellemek için irtifa en az sıfır alınmıştır.
    Daha gerçekçi ISA atmosferi eklenirse yalnızca bu fonksiyon değiştirilebilir.
    """
    altitude_m = max(0.0, altitude_from_down(down_m))
    return env.rho0_kgpm3 * np.exp(
        -altitude_m / env.atmosphere_scale_height_m
    )


def drag_force_n(
    state: np.ndarray,
    vehicle: VehicleSpec,
    env: EnvironmentConfig,
) -> float:
    """D = 0.5*rho*V^2*Cd*S bağıntısıyla sürükleme kuvveti büyüklüğü."""

    # state[DOWN] doğrudan irtifa değildir. Önce air_density_from_down içinde
    # h=-D dönüşümü yapılır.
    rho = air_density_from_down(state[DOWN], env)
    speed = max(state[SPEED], 0.0)
    return (
        0.5
        * rho
        * speed**2
        * vehicle.aero.drag_coefficient
        * vehicle.aero.reference_area_m2
    )


def thrust_and_mass_flow(
    time_s: float,
    state: np.ndarray,
    vehicle: VehicleSpec,
) -> tuple[float, float]:
    """
    Mevcut sabit itki modeli.

    Daha sonra tablo tabanlı/değişken itki istenirse yalnızca bu fonksiyon veya
    bunun yerine bir propulsion strategy sınıfı değiştirilmelidir.
    """
    propulsion = vehicle.propulsion
    burning = (
        time_s < propulsion.burn_time_s
        and state[MASS] > propulsion.dry_mass_kg + 1e-9
    )

    if not burning:
        return 0.0, 0.0

    thrust_n = min(propulsion.thrust_n, propulsion.max_thrust_n)
    return thrust_n, propulsion.mass_flow_kgps


def vehicle_derivative(
    time_s: float,
    state: np.ndarray,
    vehicle: VehicleSpec,
    env: EnvironmentConfig,
    commanded_normal_accel_mps2: np.ndarray,
) -> np.ndarray:
    """
    state = [N,E,D,V,chi,gamma,m] için 3DOF türevi.

    commanded_normal_accel_mps2 güdüm/otopilot katmanından gelir. Bu fonksiyon
    komutun PN, RL veya başka bir algoritma tarafından üretildiğini bilmez.
    """
    safe_state = state.copy()
    safe_state[SPEED] = max(safe_state[SPEED], 1.0)
    safe_state[MASS] = max(
        safe_state[MASS],
        max(vehicle.propulsion.dry_mass_kg, 1e-3),
    )

    speed = safe_state[SPEED]
    gamma = safe_state[GAMMA]
    mass = safe_state[MASS]
    e_velocity, e_chi, e_gamma = flight_path_basis(safe_state)

    drag_n = drag_force_n(safe_state, vehicle, env)
    thrust_n, mass_flow_kgps = thrust_and_mass_flow(time_s, safe_state, vehicle)

    # -------------------------------------------------------------------------
    # BOYLAMSAL HIZ TUTUCU
    # -------------------------------------------------------------------------
    # Hedef uçağın motor/itki dinamiğini bu ilk sürümde ayrıntılı modellemiyoruz.
    # speed_hold_mps tanımlıysa oransal bir hız kontrolcüsü, sürüklemeyi ve uçuş
    # yolu boyunca yerçekimi bileşenini dengeleyecek itkiyi hesaplar. Füze için
    # speed_hold_mps=None bırakıldığı için füze tamamen itki-sürükleme-kütle
    # denklemleriyle hızlanır/yavaşlar.
    if vehicle.speed_hold_mps is not None:
        speed_error = vehicle.speed_hold_mps - speed
        desired_longitudinal_accel = np.clip(
            vehicle.speed_hold_gain * speed_error,
            -vehicle.max_longitudinal_accel_mps2,
            vehicle.max_longitudinal_accel_mps2,
        )
        required_thrust = drag_n + mass * (
            env.gravity_mps2 * np.sin(gamma) + desired_longitudinal_accel
        )
        thrust_n = np.clip(
            required_thrust,
            0.0,
            vehicle.propulsion.max_thrust_n,
        )
        mass_flow_kgps = 0.0

    # -------------------------------------------------------------------------
    # NORMAL İVME KOMUTUNUN UYGULANMASI
    # -------------------------------------------------------------------------
    # PN veya RL ideal olarak hıza dik bir komut vermelidir. Yuvarlama hatası ya
    # da farklı bir RL aksiyon dönüşümü hıza paralel küçük bir bileşen üretebilir.
    # Bu bileşen burada çıkarılır; boylamsal hız yalnızca itki, sürükleme ve
    # yerçekimi tarafından belirlenir.
    normal_accel = np.asarray(commanded_normal_accel_mps2, dtype=float)
    normal_accel = normal_accel - np.dot(normal_accel, e_velocity) * e_velocity

    # Güdüm algoritmasının kendi sınırı olsa bile aracın fiziksel sınırı ikinci ve
    # bağımsız bir emniyet katmanı olarak burada uygulanır.
    physical_limit = vehicle.aero.max_normal_accel_g * env.gravity_mps2
    accel_norm = np.linalg.norm(normal_accel)
    if accel_norm > physical_limit > 0.0:
        normal_accel *= physical_limit / accel_norm

    a_chi = float(np.dot(normal_accel, e_chi))
    a_gamma = float(np.dot(normal_accel, e_gamma))

    cos_gamma_safe = np.sign(np.cos(gamma)) * max(abs(np.cos(gamma)), 1e-4)

    # -------------------------------------------------------------------------
    # 3DOF NOKTASAL KÜTLE DENKLEMLERİ
    # -------------------------------------------------------------------------
    # Konum türevi NED hız vektörüdür:
    #   N_dot = V*cos(gamma)*cos(chi)
    #   E_dot = V*cos(gamma)*sin(chi)
    #   D_dot = -V*sin(gamma)
    #
    # g_NED=[0,0,+g] olduğundan hız doğrultusundaki yerçekimi bileşeni
    # -g*sin(gamma)'dır. Pozitif tırmanışta yerçekimi aracı yavaşlatır.
    #
    # a_chi ve a_gamma burada net kinematik normal ivmelerdir. İç otopilotun
    # normal yöndeki yerçekimi telafisini ideal ve gecikmesiz yaptığı varsayılır.
    derivative = np.zeros(7)
    derivative[POSITION] = speed * e_velocity
    derivative[SPEED] = (
        (thrust_n - drag_n) / mass
        - env.gravity_mps2 * np.sin(gamma)
    )
    derivative[CHI] = a_chi / (speed * cos_gamma_safe)
    derivative[GAMMA] = a_gamma / speed
    derivative[MASS] = -mass_flow_kgps
    return derivative


def rk4_step(
    time_s: float,
    state: np.ndarray,
    dt_s: float,
    vehicle: VehicleSpec,
    env: EnvironmentConfig,
    normal_accel_command_mps2: np.ndarray,
) -> np.ndarray:
    """Komutu dt boyunca sabit kabul eden RK4 integrasyon adımı."""
    f = vehicle_derivative

    k1 = f(time_s, state, vehicle, env, normal_accel_command_mps2)
    k2 = f(
        time_s + dt_s / 2.0,
        state + dt_s * k1 / 2.0,
        vehicle,
        env,
        normal_accel_command_mps2,
    )
    k3 = f(
        time_s + dt_s / 2.0,
        state + dt_s * k2 / 2.0,
        vehicle,
        env,
        normal_accel_command_mps2,
    )
    k4 = f(
        time_s + dt_s,
        state + dt_s * k3,
        vehicle,
        env,
        normal_accel_command_mps2,
    )

    new_state = state + dt_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    new_state[SPEED] = max(new_state[SPEED], 0.0)
    new_state[MASS] = max(new_state[MASS], vehicle.propulsion.dry_mass_kg)
    new_state[CHI] = (new_state[CHI] + np.pi) % (2.0 * np.pi) - np.pi
    new_state[GAMMA] = np.clip(new_state[GAMMA], -89.0 * DEG, 89.0 * DEG)
    return new_state


# =============================================================================
# 3. GUDUM ARAYUZU: PN VE RL BURADAN AYRILIR
# =============================================================================


@dataclass(frozen=True)
class GuidanceObservation:
    time_s: float
    dt_s: float
    missile_state: np.ndarray
    target_state: np.ndarray
    environment: EnvironmentConfig


@dataclass
class GuidanceOutput:
    normal_accel_mps2: np.ndarray
    info: dict[str, float] = field(default_factory=dict)


class GuidanceLaw(ABC):
    """
    PN, RL veya başka bir güdüm algoritmasının ortak arayüzü.

    Araç dinamiği bu sınıfın alt sınıflarını doğrudan tanımaz. Simülasyon döngüsü
    her adımda bir GuidanceObservation hazırlar, command metodunu çağırır ve
    dönen NED ivme vektörünü araç dinamiğine verir. Böylece PN kodu daha sonra
    ayrı bir pn.py dosyasına taşınabilir veya tamamen kaldırılıp RL kullanılabilir.
    """

    def reset(self) -> None:
        """Yeni episode/simülasyon başında iç durumları temizler."""

    @abstractmethod
    def command(self, observation: GuidanceObservation) -> GuidanceOutput:
        """Füze için inertiyal eksenlerde 3B normal ivme komutu üretir."""
        raise NotImplementedError


@dataclass
class PNConfig:
    navigation_constant: float = 4.0
    max_command_g: float = 30.0


class VectorPNGuidance(GuidanceLaw):
    """
    NED vektörleriyle çalışan true/vector proportional navigation uygulaması.

    Vektörel form koordinat sisteminden bağımsızdır; bütün konum ve hız vektörleri
    aynı sağ-elli NED koordinatında verildiği sürece çapraz çarpımlar doğrudur.
    Kullanılan temel bağıntılar:

        omega_LOS = cross(r, v_relative) / |r|^2
        V_closing = -dot(v_relative, r_hat)
        a_command = N * V_closing * cross(omega_LOS, r_hat)

    Buradaki N, navigation_constant değeridir; North koordinatıyla karıştırmayın.
    """

    def __init__(self, config: PNConfig):
        self.config = config

    def command(self, observation: GuidanceObservation) -> GuidanceOutput:
        missile = observation.missile_state
        target = observation.target_state
        env = observation.environment

        # Bütün vektörler [North, East, Down] sırasındadır.
        relative_position = target[POSITION] - missile[POSITION]
        relative_velocity = velocity_from_state(target) - velocity_from_state(missile)
        range_m = float(np.linalg.norm(relative_position))

        if range_m < 1e-9:
            return GuidanceOutput(np.zeros(3), {"range_m": range_m})

        # Füze konumundan hedef konumuna bakan görüş-hattı birim vektörü.
        los_unit = relative_position / range_m

        # Pozitif closing speed menzilin azaldığını ifade eder. Hedef açılıyorsa
        # değer negatiftir ve mevcut basit PN uygulaması ters komut vermemesi için
        # max(Vc,0) kullanır.
        closing_speed_mps = -float(np.dot(relative_velocity, los_unit))

        # LOS açısal hızının NED bileşenleri. Birimi rad/s'dir.
        los_rate_radps = np.cross(relative_position, relative_velocity) / range_m**2

        command = (
            self.config.navigation_constant
            * max(closing_speed_mps, 0.0)
            * np.cross(los_rate_radps, los_unit)
        )

        # İdeal PN komutu normal ivmedir. Sayısal hatayla oluşabilecek hız-paralel
        # bileşeni bir kez de güdüm katmanında kaldırılır.
        missile_velocity = velocity_from_state(missile)
        missile_speed = max(np.linalg.norm(missile_velocity), 1e-9)
        velocity_unit = missile_velocity / missile_speed
        command -= np.dot(command, velocity_unit) * velocity_unit

        # PN'nin istediği komut algoritmik sınırı aşarsa yön korunarak büyüklük
        # kırpılır. Araç fiziksel limiti ayrıca vehicle_derivative içinde vardır.
        command_limit = self.config.max_command_g * env.gravity_mps2
        command_norm = np.linalg.norm(command)
        if command_norm > command_limit > 0.0:
            command *= command_limit / command_norm

        return GuidanceOutput(
            normal_accel_mps2=command,
            info={
                "range_m": range_m,
                "closing_speed_mps": closing_speed_mps,
                "los_rate_radps": float(np.linalg.norm(los_rate_radps)),
            },
        )


class SampleAndHoldGuidance(GuidanceLaw):
    """Bir güdüm yasasını sabit karar periyoduyla örnekle ve arada tut.

    Bu sarmalayıcı özellikle PN'yi PPO ile aynı 10 Hz karar hızında çalıştırmak
    için kullanılır. Fizik integratörü daha küçük bir adımla ilerlemeye devam
    ederken son PN ivme vektörü sıfırıncı-derece tutucu ile korunur.
    """

    def __init__(self, inner_law: GuidanceLaw, period_s: float):
        if period_s <= 0.0:
            raise ValueError("Kontrol periyodu pozitif olmalıdır.")
        self.inner_law = inner_law
        self.period_s = float(period_s)
        self._held_output: GuidanceOutput | None = None
        self._next_update_s = 0.0

    def reset(self) -> None:
        self.inner_law.reset()
        self._held_output = None
        self._next_update_s = 0.0

    def command(self, observation: GuidanceObservation) -> GuidanceOutput:
        if (
            self._held_output is None
            or observation.time_s + 1e-12 >= self._next_update_s
        ):
            output = self.inner_law.command(observation)
            self._held_output = GuidanceOutput(
                normal_accel_mps2=np.asarray(
                    output.normal_accel_mps2,
                    dtype=float,
                ).copy(),
                info=dict(output.info),
            )
            self._next_update_s = observation.time_s + self.period_s

        return GuidanceOutput(
            normal_accel_mps2=self._held_output.normal_accel_mps2.copy(),
            info=dict(self._held_output.info),
        )


@dataclass
class RLInterfaceConfig:
    max_command_g: float = 30.0
    observation_version: str = RL_OBSERVATION_V2
    position_scale_m: float = 10_000.0
    velocity_scale_mps: float = 1_000.0
    mass_scale_kg: float = 100.0
    altitude_scale_m: float = 10_000.0
    fine_cross_track_scale_m: float = 100.0
    los_rate_scale_radps: float = 0.10
    time_to_go_scale_s: float = 10.0
    burn_time_s: float = 5.0
    control_period_s: float = 0.10
    deterministic: bool = True


def canonical_rl_observation_version(version: str) -> str:
    """Observation sürüm adını doğrular ve kabul edilen kısa adları eşler."""
    normalized = str(version).strip().lower()
    aliases = {
        "10": RL_OBSERVATION_LEGACY10,
        "legacy": RL_OBSERVATION_LEGACY10,
        "legacy10": RL_OBSERVATION_LEGACY10,
        "v1": RL_OBSERVATION_LEGACY10,
        "13": RL_OBSERVATION_MARKOV13,
        "markov": RL_OBSERVATION_MARKOV13,
        "markov13": RL_OBSERVATION_MARKOV13,
        "23": RL_OBSERVATION_V2,
        "v2": RL_OBSERVATION_V2,
    }
    if normalized not in aliases:
        supported = ", ".join(sorted(RL_OBSERVATION_SIZES))
        raise ValueError(
            f"Bilinmeyen RL observation sürümü: {version!r}. "
            f"Desteklenen sürümler: {supported}."
        )
    return aliases[normalized]


def rl_observation_size(version: str) -> int:
    """Observation sürümünün ağ giriş boyutunu döndürür."""
    return RL_OBSERVATION_SIZES[canonical_rl_observation_version(version)]


def infer_rl_observation_version(observation_size: int) -> str:
    """Kaydedilmiş modelin giriş boyutundan observation sürümünü belirler."""
    matches = [
        version
        for version, size in RL_OBSERVATION_SIZES.items()
        if size == int(observation_size)
    ]
    if len(matches) != 1:
        expected = ", ".join(
            f"{version}={size}"
            for version, size in RL_OBSERVATION_SIZES.items()
        )
        raise ValueError(
            f"Observation boyutu {observation_size} tanınmıyor. "
            f"Beklenen boyutlar: {expected}."
        )
    return matches[0]


def infer_policy_observation_version(policy) -> str:
    """SB3 benzeri bir politikanın observation space'inden sürümü bulur."""
    observation_space = getattr(policy, "observation_space", None)
    shape = getattr(observation_space, "shape", None)
    if shape is None or len(shape) != 1:
        raise ValueError(
            "Politikanın tek boyutlu observation_space.shape bilgisi bulunamadı."
        )
    return infer_rl_observation_version(int(shape[0]))


def default_rl_observation(observation: GuidanceObservation) -> np.ndarray:

     #   [r_v, r_chi, r_gamma, vr_v, vr_chi, vr_gamma, V_m, m_m]

    missile = observation.missile_state
    target = observation.target_state
    e_velocity, e_chi, e_gamma = flight_path_basis(missile)
    rotation = np.vstack((e_velocity, e_chi, e_gamma))

    relative_position = target[POSITION] - missile[POSITION]
    relative_velocity = velocity_from_state(target) - velocity_from_state(missile)

    local_position = rotation @ relative_position
    local_velocity = rotation @ relative_velocity
    return np.concatenate(
        (
            local_position,
            local_velocity,
            [missile[SPEED], missile[MASS]],
        )
    ).astype(np.float32)


def build_normalized_rl_observation(
    observation: GuidanceObservation,
    config: RLInterfaceConfig,
    observation_builder: Callable[
        [GuidanceObservation],
        np.ndarray,
    ] = default_rl_observation,
) -> np.ndarray:
    """
        0:3  -> yerel bağıl konum / position_scale_m
        3:6  -> yerel bağıl hız / velocity_scale_mps
        6    -> füze hızı / velocity_scale_mps
        7    -> füze kütlesi / mass_scale_kg
        8    -> geçen zaman / episode süresi
        9    -> kalan motor çalışma oranı
        10    -> füze irtifası / altitude_scale_m
        11:13 -> sin(gamma), cos(gamma)
        13:15 -> ince yanal konum: tanh(r_chi/scale), tanh(r_gamma/scale)
        15:18 -> yerel LOS birim vektörü
        18    -> yaklaşma hızı / velocity_scale_mps
        19:22 -> yerel LOS açısal hızı / los_rate_scale_radps
        22    -> tahmini time-to-go / time_to_go_scale_s
    """
    raw_observation = np.asarray(
        observation_builder(observation),
        dtype=float,
    ).reshape(-1)
    if raw_observation.size != 8:
        raise ValueError(
            "RL observation_builder tam olarak 8 ham eleman üretmelidir: "
            "[r_local(3), vr_local(3), V_m, m_m]."
        )

    if (
        config.position_scale_m <= 0.0
        or config.velocity_scale_mps <= 0.0
        or config.mass_scale_kg <= 0.0
    ):
        raise ValueError("RL observation ölçekleri pozitif olmalıdır.")

    normalized = raw_observation.copy()
    normalized[0:3] /= config.position_scale_m
    normalized[3:7] /= config.velocity_scale_mps
    normalized[7] /= config.mass_scale_kg

    episode_duration_s = max(observation.environment.t_final_s, 1e-9)
    normalized_time = np.clip(
        observation.time_s / episode_duration_s,
        0.0,
        1.0,
    )

    if config.burn_time_s > 1e-9:
        remaining_burn_fraction = np.clip(
            (config.burn_time_s - observation.time_s) / config.burn_time_s,
            0.0,
            1.0,
        )
    else:
        remaining_burn_fraction = 0.0

    legacy_observation = np.concatenate(
        (
            normalized,
            [normalized_time, remaining_burn_fraction],
        )
    ).astype(np.float32)

    observation_version = canonical_rl_observation_version(
        config.observation_version
    )
    if observation_version == RL_OBSERVATION_LEGACY10:
        return legacy_observation

    if (
        config.altitude_scale_m <= 0.0
    ):
        raise ValueError("Observation irtifa ölçeği pozitif olmalıdır.")

    missile = observation.missile_state
    target = observation.target_state
    missile_altitude_m = -float(missile[DOWN])
    gamma_rad = float(missile[GAMMA])
    markov_extension = np.asarray(
        [
            missile_altitude_m / config.altitude_scale_m,
            np.sin(gamma_rad),
            np.cos(gamma_rad),
        ],
        dtype=np.float32,
    )
    markov_observation = np.concatenate(
        (legacy_observation, markov_extension)
    ).astype(np.float32)
    if observation_version == RL_OBSERVATION_MARKOV13:
        return markov_observation

    if (
        config.fine_cross_track_scale_m <= 0.0
        or config.los_rate_scale_radps <= 0.0
        or config.time_to_go_scale_s <= 0.0
    ):
        raise ValueError("Observation V2 ölçeklerinin tamamı pozitif olmalıdır.")

    e_velocity, e_chi, e_gamma = flight_path_basis(missile)
    rotation = np.vstack((e_velocity, e_chi, e_gamma))

    relative_position = target[POSITION] - missile[POSITION]
    relative_velocity = (
        velocity_from_state(target) - velocity_from_state(missile)
    )
    range_m = float(np.linalg.norm(relative_position))

    if range_m > 1e-9:
        los_unit_ned = relative_position / range_m
        los_unit_local = rotation @ los_unit_ned
        closing_speed_mps = -float(
            np.dot(relative_velocity, los_unit_ned)
        )
        los_rate_ned_radps = (
            np.cross(relative_position, relative_velocity) / range_m**2
        )
        los_rate_local_radps = rotation @ los_rate_ned_radps
    else:
        los_unit_local = np.zeros(3, dtype=float)
        closing_speed_mps = 0.0
        los_rate_local_radps = np.zeros(3, dtype=float)

    local_position = raw_observation[0:3]
    fine_cross_track = np.tanh(
        local_position[1:3] / config.fine_cross_track_scale_m
    )

    if closing_speed_mps > 1e-6:
        time_to_go_s = range_m / closing_speed_mps
    else:
        # Hedef açılıyorsa sonlu ve anlamlı bir üst değer kullanılır.
        time_to_go_s = observation.environment.t_final_s
    time_to_go_s = float(
        np.clip(time_to_go_s, 0.0, observation.environment.t_final_s)
    )

    scaled_los_rate = np.clip(
        los_rate_local_radps / config.los_rate_scale_radps,
        -10.0,
        10.0,
    )

    v2_extra_features = np.concatenate(
        (
            fine_cross_track,
            los_unit_local,
            [closing_speed_mps / config.velocity_scale_mps],
            scaled_los_rate,
            [time_to_go_s / config.time_to_go_scale_s],
        )
    ).astype(np.float32)

    result = np.concatenate(
        (markov_observation, v2_extra_features)
    ).astype(np.float32)
    expected_size = rl_observation_size(observation_version)
    if result.size != expected_size:
        raise RuntimeError(
            f"Observation V2 boyutu {result.size}, beklenen {expected_size}."
        )
    return result


def rl_action_to_normal_acceleration(
    action: np.ndarray,
    missile_state: np.ndarray,
    environment: EnvironmentConfig,
    config: RLInterfaceConfig,
) -> np.ndarray:

    normalized_action = np.asarray(action, dtype=float).reshape(-1)
    if normalized_action.size != 2:
        raise ValueError(
            "RL aksiyonu iki elemanlı olmalıdır: [a_chi, a_gamma]."
        )
    normalized_action = np.clip(normalized_action, -1.0, 1.0)

    _, e_chi, e_gamma = flight_path_basis(missile_state)
    acceleration_limit = config.max_command_g * environment.gravity_mps2
    if acceleration_limit < 0.0:
        raise ValueError("RL max_command_g negatif olamaz.")

    command = acceleration_limit * (
        normalized_action[0] * e_chi
        + normalized_action[1] * e_gamma
    )

    command_norm = np.linalg.norm(command)
    if command_norm > acceleration_limit > 0.0:
        command *= acceleration_limit / command_norm
    if acceleration_limit == 0.0:
        command[:] = 0.0
    return command


class RLGuidanceAdapter(GuidanceLaw):
 
    def __init__(
        self,
        policy,
        config: RLInterfaceConfig,
        observation_builder: Callable[
            [GuidanceObservation], np.ndarray
        ] = default_rl_observation,
    ):
        self.policy = policy
        self.config = config
        self.observation_builder = observation_builder
        self._held_action: np.ndarray | None = None
        self._next_action_time_s = 0.0

    def reset(self) -> None:
        self._held_action = None
        self._next_action_time_s = 0.0
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _predict(self, normalized_observation: np.ndarray) -> np.ndarray:
        if hasattr(self.policy, "predict"):
            prediction = self.policy.predict(
                normalized_observation,
                deterministic=self.config.deterministic,
            )
            action = prediction[0] if isinstance(prediction, tuple) else prediction
        elif callable(self.policy):
            action = self.policy(normalized_observation)
        else:
            raise TypeError("RL policy callable olmalı veya predict metodu taşımalı.")

        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size != 2:
            raise ValueError(
                "RL policy iki aksiyon üretmeli: "
                "[a_chi_normalized, a_gamma_normalized]."
            )
        return np.clip(action, -1.0, 1.0)

    def command(self, observation: GuidanceObservation) -> GuidanceOutput:
        action_was_updated = (
            self._held_action is None
            or observation.time_s + 1e-12 >= self._next_action_time_s
        )
        if action_was_updated:
            normalized_observation = build_normalized_rl_observation(
                observation=observation,
                config=self.config,
                observation_builder=self.observation_builder,
            )
            self._held_action = self._predict(normalized_observation)
            self._next_action_time_s = (
                observation.time_s + max(self.config.control_period_s, 1e-9)
            )

        action = self._held_action
        command = rl_action_to_normal_acceleration(
            action=action,
            missile_state=observation.missile_state,
            environment=observation.environment,
            config=self.config,
        )

        return GuidanceOutput(
            normal_accel_mps2=command,
            info={
                "action_chi": float(action[0]),
                "action_gamma": float(action[1]),
                "action_was_updated": float(action_was_updated),
            },
        )


# =============================================================================
# 4. HEDEF UCAK KONTROLU
# =============================================================================


@lru_cache(maxsize=256)
def _random_wave_parameters(
    random_seed: int,
    channel_id: int,
    harmonic_count: int,
    base_period_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def smooth_random_wave(
    time_s: float,
    maneuver: ManeuverConfig,
    channel_id: int,
    user_phase_deg: float,
) -> float:
    """
    Verilen zamanda [-1,+1] aralığında düzgün ve tekrarlanabilir rastgele değer.

    Fonksiyon state tutmaz. Aynı zaman, seed ve konfigürasyon verildiğinde aynı
    sonucu üretmesi Monte Carlo karşılaştırmaları ve hata ayıklama için önemlidir.
    """
    frequencies, random_phases, weights = _random_wave_parameters(
        random_seed=maneuver.random_seed,
        channel_id=channel_id,
        harmonic_count=maneuver.random_harmonics,
        base_period_s=maneuver.weave_period_s,
    )
    user_phase_rad = user_phase_deg * DEG
    components = np.sin(
        frequencies * time_s + random_phases + user_phase_rad
    )
    # tanh keskin bir clip işlemi gibi köşe oluşturmaz; giriş ne kadar büyürse
    # büyüsün çıktı düzgün biçimde -1 ile +1 arasında kalır.
    return float(np.tanh(np.dot(weights, components)))


def aircraft_maneuver_command(
    time_s: float,
    state: np.ndarray,
    maneuver: ManeuverConfig,
) -> np.ndarray:
    """
    Hedef uçağın ideal net-normal ivme komutunu üretir.

    Bu fonksiyon hedef yörüngesini doğrudan konum denklemiyle zorlamaz. Bunun
    yerine istenen chi_dot ve gamma_dot oranlarından fiziksel normal ivme üretir:

        a_chi   = V*cos(gamma)*chi_dot
        a_gamma = V*gamma_dot

    Böylece hedef de füze ile aynı 3DOF dinamik ve RK4 integratöründen geçer.

    "sinusoidal" seçeneğinde:

        chi_dot(t)   = A_chi   * sin(2*pi*t/T + phase_chi)
        gamma_dot(t) = A_gamma * sin(2*pi*t/T + phase_gamma)

    "random_smooth" seçeneğinde tek bir sinüs yerine seed ile seçilmiş farklı
    frekans, faz ve ağırlıklardaki sinüsler toplanır. Bu yöntem rastgele görünen,
    tekrar üretilebilir ve türevi süreksiz olmayan bir manevra üretir.

    Dönüş oranları sinüzoidal olduğu için hedefin hız yönü ve NED konumu doğrusal
    olmayan, yumuşak bir yörünge izler. Bu basit manevra yalnızca gözlem/test
    amaçlıdır; gerçek bir uçak otopilot veya aerodinamik modeli değildir.
    """
    if maneuver.kind == "straight":
        heading_rate_radps = 0.0
        flight_path_rate_radps = 0.0
    elif maneuver.kind == "constant_turn":
        heading_rate_radps = maneuver.heading_rate_deg_s * DEG
        flight_path_rate_radps = maneuver.flight_path_rate_deg_s * DEG
    elif maneuver.kind in {"sinusoidal", "sine", "weave"}:
        period_s = max(maneuver.weave_period_s, 1e-3)
        angular_frequency = 2.0 * np.pi / period_s

        heading_wave = np.sin(
            angular_frequency * time_s + maneuver.heading_phase_deg * DEG
        )
        flight_path_wave = np.sin(
            angular_frequency * time_s + maneuver.flight_path_phase_deg * DEG
        )

        heading_rate_radps = (
            maneuver.heading_rate_deg_s * DEG * heading_wave
        )
        flight_path_rate_radps = (
            maneuver.flight_path_rate_deg_s * DEG * flight_path_wave
        )
    elif maneuver.kind in {"random_smooth", "random_nonlinear", "random"}:
        # channel_id=0 yatay, channel_id=1 düşey kanal içindir. Aynı seed verilse
        # bile kanallar farklı rastgele Fourier katsayıları kullanır.
        heading_wave = smooth_random_wave(
            time_s=time_s,
            maneuver=maneuver,
            channel_id=0,
            user_phase_deg=maneuver.heading_phase_deg,
        )
        flight_path_wave = smooth_random_wave(
            time_s=time_s,
            maneuver=maneuver,
            channel_id=1,
            user_phase_deg=maneuver.flight_path_phase_deg,
        )

        # Rastgele dalgalar [-1,+1] aralığında olduğundan class içindeki rate
        # değerleri komutların maksimum ölçeğini belirlemeye devam eder.
        heading_rate_radps = (
            maneuver.heading_rate_deg_s * DEG * heading_wave
        )
        flight_path_rate_radps = (
            maneuver.flight_path_rate_deg_s * DEG * flight_path_wave
        )
    else:
        raise ValueError(f"Bilinmeyen manevra türü: {maneuver.kind!r}")

    # e_chi ve e_gamma NED koordinatlarındadır. Dolayısıyla dönen komut da
    # [a_N, a_E, a_D] sırasındadır ve füze komutuyla aynı arabirimi kullanır.
    _, e_chi, e_gamma = flight_path_basis(state)
    speed = state[SPEED]
    gamma = state[GAMMA]
    return (
        speed * np.cos(gamma) * heading_rate_radps * e_chi
        + speed * flight_path_rate_radps * e_gamma
    )


# =============================================================================
# 5. SIMULASYON YONETICISI
# =============================================================================


def closest_approach_on_step(
    relative_position_start: np.ndarray,
    relative_position_end: np.ndarray,
) -> tuple[float, float]:
    """Adım içindeki doğrusal bağıl hareket için (minimum mesafe, adım oranı)."""
    delta = relative_position_end - relative_position_start
    denominator = float(np.dot(delta, delta))

    if denominator < 1e-15:
        tau = 0.0
    else:
        tau = float(
            np.clip(
                -np.dot(relative_position_start, delta) / denominator,
                0.0,
                1.0,
            )
        )

    distance_m = float(
        np.linalg.norm(relative_position_start + tau * delta)
    )
    return distance_m, tau


def validate_scenario(
    vehicles: list[VehicleSpec],
    env: EnvironmentConfig,
    guidance_laws: dict[str, GuidanceLaw],
) -> None:
    names = [vehicle.name for vehicle in vehicles]
    known_names = set(names)

    if len(names) != len(known_names):
        raise ValueError("Her aracın adı benzersiz olmalıdır.")
    if env.dt_s <= 0.0 or env.t_final_s <= 0.0 or env.hit_radius_m <= 0.0:
        raise ValueError("dt_s, t_final_s ve hit_radius_m pozitif olmalıdır.")

    missile_names = set()
    for vehicle in vehicles:
        if vehicle.role not in {"missile", "aircraft"}:
            raise ValueError(
                f"{vehicle.name}: role 'missile' veya 'aircraft' olmalıdır."
            )
        if vehicle.initial.speed_mps <= 0.0 or vehicle.initial.mass_kg <= 0.0:
            raise ValueError(
                f"{vehicle.name}: başlangıç hızı ve kütlesi pozitif olmalıdır."
            )
        if vehicle.propulsion.dry_mass_kg > vehicle.initial.mass_kg:
            raise ValueError(
                f"{vehicle.name}: kuru kütle başlangıç kütlesini aşamaz."
            )

        if vehicle.role == "missile":
            missile_names.add(vehicle.name)
            if vehicle.target_name not in known_names:
                raise ValueError(
                    f"{vehicle.name}: geçerli bir target_name gereklidir."
                )
            if vehicle.target_name == vehicle.name:
                raise ValueError(f"{vehicle.name}: kendisini hedefleyemez.")

    missing_guidance = missile_names - set(guidance_laws)
    extra_guidance = set(guidance_laws) - missile_names
    if missing_guidance:
        raise ValueError(
            f"Şu füzeler için GuidanceLaw eksik: {sorted(missing_guidance)}"
        )
    if extra_guidance:
        raise ValueError(
            "Füze olmayan/bilinmeyen isimler için güdüm verilmiş: "
            f"{sorted(extra_guidance)}"
        )
    if not all(isinstance(law, GuidanceLaw) for law in guidance_laws.values()):
        raise TypeError("Bütün guidance_laws değerleri GuidanceLaw olmalıdır.")


def run_simulation(
    vehicles: list[VehicleSpec],
    env: EnvironmentConfig,
    guidance_laws: dict[str, GuidanceLaw],
) -> SimulationResult:
    validate_scenario(vehicles, env, guidance_laws)

    specs = {vehicle.name: vehicle for vehicle in vehicles}
    states = {
        name: make_initial_state(vehicle.initial)
        for name, vehicle in specs.items()
    }
    missiles = [vehicle for vehicle in vehicles if vehicle.role == "missile"]

    for guidance_law in guidance_laws.values():
        guidance_law.reset()

    status = {missile.name: "flying" for missile in missiles}
    terminal_time_s = {missile.name: np.nan for missile in missiles}
    terminal_index = {missile.name: -1 for missile in missiles}
    target_of = {missile.name: missile.target_name for missile in missiles}

    time_history = [0.0]
    state_history = {
        name: [state.copy()]
        for name, state in states.items()
    }
    range_history = {missile.name: [] for missile in missiles}
    command_history = {missile.name: [] for missile in missiles}
    closest_approach_m: dict[str, float] = {}
    closest_approach_time_s: dict[str, float] = {}

    # İlk zaman örneği için kayıt.
    for missile in missiles:
        relative_position = (
            states[missile.target_name][POSITION]
            - states[missile.name][POSITION]
        )
        initial_range_m = float(np.linalg.norm(relative_position))
        range_history[missile.name].append(initial_range_m)
        command_history[missile.name].append(0.0)
        closest_approach_m[missile.name] = initial_range_m
        closest_approach_time_s[missile.name] = 0.0

    maximum_steps = int(np.ceil(env.t_final_s / env.dt_s))

    for _ in range(maximum_steps):
        time_s = time_history[-1]
        dt_s = min(env.dt_s, env.t_final_s - time_s)
        if dt_s <= 1e-12:
            break

        old_states = {
            name: state.copy()
            for name, state in states.items()
        }
        commands = {}

        for name, vehicle in specs.items():
            if vehicle.role == "missile":
                if status[name] == "flying":
                    observation = GuidanceObservation(
                        time_s=time_s,
                        dt_s=dt_s,
                        missile_state=old_states[name].copy(),
                        target_state=old_states[vehicle.target_name].copy(),
                        environment=env,
                    )
                    output = guidance_laws[name].command(observation)
                    commands[name] = np.asarray(
                        output.normal_accel_mps2,
                        dtype=float,
                    )
                    if commands[name].shape != (3,):
                        raise ValueError(
                            f"{name}: GuidanceLaw çıktısı (3,) biçiminde olmalı."
                        )
                else:
                    commands[name] = np.zeros(3)
            else:
                commands[name] = aircraft_maneuver_command(
                    time_s,
                    old_states[name],
                    vehicle.maneuver,
                )

        # Bütün araçlar aynı eski zaman örneğinden ilerletilir.
        for name, vehicle in specs.items():
            if vehicle.role == "missile" and status[name] != "flying":
                states[name] = old_states[name]
            else:
                states[name] = rk4_step(
                    time_s,
                    old_states[name],
                    dt_s,
                    vehicle,
                    env,
                    commands[name],
                )

        new_time_s = time_s + dt_s
        time_history.append(new_time_s)
        for name in specs:
            state_history[name].append(states[name].copy())

        for missile in missiles:
            name = missile.name
            target_name = missile.target_name

            if status[name] == "flying":
                # Çarpışma kontrolü yalnızca örnekleme anlarındaki menzile bakmaz.
                # Adımın başı ve sonu arasındaki doğrusal bağıl doğru parçasının
                # minimum mesafesini hesaplar. Bu, yüksek hızlı bir füzenin iki
                # 20 ms örnek arasında hedefin yanından geçmesini kaçırmaz.
                relative_start = (
                    old_states[target_name][POSITION]
                    - old_states[name][POSITION]
                )
                relative_end = (
                    states[target_name][POSITION]
                    - states[name][POSITION]
                )
                minimum_step_distance_m, tau = closest_approach_on_step(
                    relative_start,
                    relative_end,
                )
                if minimum_step_distance_m < closest_approach_m[name]:
                    closest_approach_m[name] = minimum_step_distance_m
                    closest_approach_time_s[name] = time_s + tau * dt_s

                if minimum_step_distance_m <= env.hit_radius_m:
                    status[name] = "hit"
                    terminal_time_s[name] = time_s + tau * dt_s
                    terminal_index[name] = len(time_history) - 1
                # NED'de Down aşağı yönlü pozitiftir. Deniz seviyesindeki düz
                # zeminin D değeri 0, 500 m rakımlı zeminin D değeri -500'dür.
                # Araç Down koordinatı zemin düzlemine ulaştığında çarpışmıştır.
                elif states[name][DOWN] >= -env.ground_altitude_m:
                    status[name] = "miss: ground"
                    terminal_time_s[name] = new_time_s
                    terminal_index[name] = len(time_history) - 1
                elif states[name][SPEED] <= env.minimum_flying_speed_mps:
                    status[name] = "miss: low speed"
                    terminal_time_s[name] = new_time_s
                    terminal_index[name] = len(time_history) - 1

            current_range_m = float(
                np.linalg.norm(
                    states[target_name][POSITION]
                    - states[name][POSITION]
                )
            )
            range_history[name].append(current_range_m)
            command_history[name].append(
                float(np.linalg.norm(commands[name]) / env.gravity_mps2)
            )

        if missiles and all(
            status[missile.name] != "flying"
            for missile in missiles
        ):
            break

    for missile in missiles:
        if status[missile.name] == "flying":
            status[missile.name] = "timeout"
            terminal_time_s[missile.name] = time_history[-1]
            terminal_index[missile.name] = len(time_history) - 1

    return SimulationResult(
        time_s=np.asarray(time_history),
        states={
            name: np.asarray(history)
            for name, history in state_history.items()
        },
        range_m={
            name: np.asarray(history)
            for name, history in range_history.items()
        },
        command_g={
            name: np.asarray(history)
            for name, history in command_history.items()
        },
        status=status,
        terminal_time_s=terminal_time_s,
        terminal_index=terminal_index,
        target_of=target_of,
        hit_radius_m=env.hit_radius_m,
        closest_approach_m=closest_approach_m,
        closest_approach_time_s=closest_approach_time_s,
    )
