"""
Eski 10-girdili PPO checkpoint'ini 23-girdili Observation V2 modeline taşır.

Bu betik eğitim yapmaz. Eski actor ve critic ağırlıklarını yeni ağa kopyalar.
İlk katmanların eski 10 sütunu aynen korunur; yeni 13 sütun sıfır ağırlıkla
başlatılır. Bu nedenle migrasyondan hemen sonra V2 modelin deterministik
aksiyonu eski modelinkiyle aynı olur. Yeni özellikler sonraki PPO eğitiminde
öğrenilmeye başlar.

Örnek:

    python training/migrate_observation_v2.py ^
        --source models/random25_rewardv3_50m/best_model.zip ^
        --stage obs_v2_start
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from air_to_air_3dof_ppo import (  # noqa: E402
    PPOEnvironmentConfig,
    RLInterfaceConfig,
    RL_OBSERVATION_LEGACY10,
    RL_OBSERVATION_V2,
    build_default_ppo_environment,
    infer_policy_observation_version,
    rl_observation_size,
)


def safe_stage_name(value: str) -> str:
    """Komut satırı değerini güvenli bir klasör adına dönüştürür."""
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value.strip()
    )
    if not cleaned:
        raise argparse.ArgumentTypeError("Stage adı boş olamaz.")
    return cleaned


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO checkpoint'ini 10 girdiden Observation V2'ye taşı",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Taşınacak eski 10-girdili PPO .zip modelinin yolu",
    )
    parser.add_argument(
        "--stage",
        type=safe_stage_name,
        default="obs_v2_start",
        help="Yeni modelin models altında kaydedileceği klasör adı",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Migrasyon doğrulama seed'i. Varsayılan: 42",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Hedef starting_model.zip zaten varsa üzerine yaz",
    )
    return parser.parse_args()


def resolve_source_path(value: str) -> Path:
    source = Path(value)
    if not source.is_absolute():
        source = PROJECT_ROOT / source

    candidates = [source]
    if source.suffix.lower() != ".zip":
        candidates.append(source.with_suffix(".zip"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Migrasyon kaynak modeli bulunamadı. Kontrol edilen yollar:\n" + checked
    )


def file_sha256(path: Path) -> str:
    """Kaynak checkpoint kimliğini metadata'ya yazmak için SHA-256 üretir."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_schedule_value(schedule, fallback: float) -> float:
    """SB3 float/schedule alanını güvenli biçimde tek sayıya dönüştürür."""
    if callable(schedule):
        return float(schedule(1.0))
    try:
        return float(schedule)
    except (TypeError, ValueError):
        return float(fallback)


def transfer_policy_weights(old_model: PPO, new_model: PPO) -> list[str]:
    """Eşleşen tensorleri kopyalar, yalnızca ilk giriş katmanlarını genişletir."""
    old_state = old_model.policy.state_dict()
    new_state = new_model.policy.state_dict()
    transferred: dict[str, object] = {}
    expanded_names: list[str] = []

    old_size = rl_observation_size(RL_OBSERVATION_LEGACY10)
    new_size = rl_observation_size(RL_OBSERVATION_V2)

    for name, new_tensor in new_state.items():
        if name not in old_state:
            raise RuntimeError(f"Eski policy state içinde {name!r} bulunamadı.")

        old_tensor = old_state[name]
        if old_tensor.shape == new_tensor.shape:
            transferred[name] = old_tensor.detach().clone()
            continue

        is_expandable_input_layer = (
            old_tensor.ndim == 2
            and new_tensor.ndim == 2
            and old_tensor.shape[0] == new_tensor.shape[0]
            and old_tensor.shape[1] == old_size
            and new_tensor.shape[1] == new_size
        )
        if not is_expandable_input_layer:
            raise RuntimeError(
                "Beklenmeyen policy tensor şekli farkı: "
                f"{name}: eski={tuple(old_tensor.shape)}, "
                f"yeni={tuple(new_tensor.shape)}"
            )

        expanded = new_tensor.detach().clone()
        expanded.zero_()
        expanded[:, :old_size] = old_tensor
        transferred[name] = expanded
        expanded_names.append(name)

    if len(expanded_names) != 2:
        raise RuntimeError(
            "Tam olarak actor ve critic ilk katmanlarının genişletilmesi "
            f"bekleniyordu; bulunan tensorler: {expanded_names}"
        )

    new_model.policy.load_state_dict(transferred, strict=True)
    return expanded_names


def build_environment(observation_version: str):
    """Aynı fizik senaryosunu seçilen observation sürümüyle oluşturur."""
    environment_config = PPOEnvironmentConfig(
        target_maneuver_kind="straight",
        randomize_initial_conditions=False,
        control_period_s=0.10,
    )
    interface_config = RLInterfaceConfig(
        observation_version=observation_version,
    )
    return build_default_ppo_environment(
        ppo_environment_config=environment_config,
        rl_interface_config=interface_config,
    )


def verify_equivalence(
    old_model: PPO,
    new_model: PPO,
    seed: int,
    maximum_steps: int = 25,
) -> dict[str, float]:
    """Migrasyonun ilk davranışı değiştirmediğini kısa rollout ile doğrular."""
    legacy_environment = build_environment(RL_OBSERVATION_LEGACY10)
    v2_environment = build_environment(RL_OBSERVATION_V2)

    old_observation, _ = legacy_environment.reset(seed=seed)
    new_observation, _ = v2_environment.reset(seed=seed)

    maximum_prefix_difference = 0.0
    maximum_action_difference = 0.0
    checked_steps = 0

    try:
        for _ in range(maximum_steps):
            prefix_difference = float(
                np.max(
                    np.abs(
                        new_observation[: old_observation.size]
                        - old_observation
                    )
                )
            )
            maximum_prefix_difference = max(
                maximum_prefix_difference,
                prefix_difference,
            )

            old_action, _ = old_model.predict(
                old_observation,
                deterministic=True,
            )
            new_action, _ = new_model.predict(
                new_observation,
                deterministic=True,
            )
            action_difference = float(
                np.max(
                    np.abs(
                        np.asarray(new_action) - np.asarray(old_action)
                    )
                )
            )
            maximum_action_difference = max(
                maximum_action_difference,
                action_difference,
            )

            old_observation, _, old_terminated, old_truncated, _ = (
                legacy_environment.step(old_action)
            )
            new_observation, _, new_terminated, new_truncated, _ = (
                v2_environment.step(new_action)
            )
            checked_steps += 1

            if (old_terminated or old_truncated) != (
                new_terminated or new_truncated
            ):
                raise RuntimeError(
                    "Legacy ve V2 doğrulama episode'ları farklı zamanda bitti."
                )
            if old_terminated or old_truncated:
                break
    finally:
        legacy_environment.close()
        v2_environment.close()

    tolerance = 1e-6
    if maximum_prefix_difference > tolerance:
        raise RuntimeError(
            "V2'nin ilk 10 girdisi legacy observation ile aynı değil: "
            f"maksimum fark={maximum_prefix_difference:.3e}"
        )
    if maximum_action_difference > tolerance:
        raise RuntimeError(
            "Migrasyon başlangıç aksiyonunu değiştirdi: "
            f"maksimum fark={maximum_action_difference:.3e}"
        )

    return {
        "checked_steps": float(checked_steps),
        "maximum_legacy_prefix_difference": maximum_prefix_difference,
        "maximum_deterministic_action_difference": maximum_action_difference,
    }


def main() -> None:
    args = parse_arguments()
    source_path = resolve_source_path(args.source)
    output_directory = PROJECT_ROOT / "models" / args.stage
    output_model_path = output_directory / "starting_model.zip"
    metadata_path = output_directory / "migration_metadata.json"

    if output_model_path.exists() and not args.force:
        raise FileExistsError(
            f"Hedef zaten var: {output_model_path}\n"
            "Üzerine yazmak istiyorsanız --force kullanın veya başka --stage seçin."
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    old_model = PPO.load(str(source_path), device="cpu")
    old_version = infer_policy_observation_version(old_model)
    if old_version != RL_OBSERVATION_LEGACY10:
        raise ValueError(
            "Kaynak model legacy10 olmalıdır; bulunan observation sürümü: "
            f"{old_version}."
        )

    v2_environment = build_environment(RL_OBSERVATION_V2)
    try:
        policy_kwargs = dict(old_model.policy_kwargs)
        new_model = PPO(
            policy=old_model.policy_class,
            env=v2_environment,
            learning_rate=current_schedule_value(
                old_model.lr_schedule,
                fallback=1e-5,
            ),
            n_steps=int(old_model.n_steps),
            batch_size=int(old_model.batch_size),
            n_epochs=int(old_model.n_epochs),
            gamma=float(old_model.gamma),
            gae_lambda=float(old_model.gae_lambda),
            clip_range=current_schedule_value(old_model.clip_range, 0.20),
            clip_range_vf=(
                None
                if old_model.clip_range_vf is None
                else current_schedule_value(old_model.clip_range_vf, 0.20)
            ),
            normalize_advantage=bool(old_model.normalize_advantage),
            ent_coef=float(old_model.ent_coef),
            vf_coef=float(old_model.vf_coef),
            max_grad_norm=float(old_model.max_grad_norm),
            use_sde=bool(old_model.use_sde),
            sde_sample_freq=int(old_model.sde_sample_freq),
            target_kl=old_model.target_kl,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            device="cpu",
            verbose=0,
        )

        expanded_names = transfer_policy_weights(old_model, new_model)
        new_model.num_timesteps = int(old_model.num_timesteps)
        new_model._n_updates = int(old_model._n_updates)

        verification = verify_equivalence(
            old_model=old_model,
            new_model=new_model,
            seed=args.seed,
        )
        new_model.save(str(output_model_path.with_suffix("")))
    finally:
        v2_environment.close()

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "operation": "legacy10_to_observation_v2_migration",
        "source_model": str(source_path),
        "source_sha256": file_sha256(source_path),
        "source_observation_version": old_version,
        "source_observation_size": rl_observation_size(old_version),
        "target_observation_version": RL_OBSERVATION_V2,
        "target_observation_size": rl_observation_size(RL_OBSERVATION_V2),
        "expanded_policy_tensors": expanded_names,
        "source_num_timesteps": int(old_model.num_timesteps),
        "verification": verification,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("OBSERVATION V2 MİGRASYONU TAMAMLANDI")
    print("=" * 72)
    print("Kaynak model          :", source_path)
    print("Eski observation      :", RL_OBSERVATION_LEGACY10, "(10)")
    print("Yeni observation      :", RL_OBSERVATION_V2, "(23)")
    print("Genişletilen tensorler:", expanded_names)
    print(
        "Maks. aksiyon farkı  :",
        f"{verification['maximum_deterministic_action_difference']:.3e}",
    )
    print("Yeni başlangıç modeli :", output_model_path)
    print("Migrasyon bilgileri   :", metadata_path)
    print("=" * 72)


if __name__ == "__main__":
    main()
