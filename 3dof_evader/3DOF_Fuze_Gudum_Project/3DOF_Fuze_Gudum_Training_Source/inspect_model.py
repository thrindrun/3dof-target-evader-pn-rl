"""Kaydedilmiş Stable-Baselines3 PPO modelinin temel eğitim bilgisini yazdır."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "models"
    / "fresh_cpa_v6_hardpocket_7p5m_300k"
    / "best_model.zip"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO checkpoint adımını ve ağ/algoritma ayarlarını gösterir."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    model_path = args.model
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")

    model = PPO.load(str(model_path), device="cpu")
    observation_shape = tuple(model.observation_space.shape)
    action_shape = tuple(model.action_space.shape)

    print("=" * 72)
    print("PPO CHECKPOINT İNCELEMESİ")
    print("=" * 72)
    print("Model                  :", model_path.resolve())
    print("Kaydedilmiş PPO adımı  :", int(model.num_timesteps))
    print("Observation şekli      :", observation_shape)
    print("Action şekli           :", action_shape)
    print("n_steps                :", int(model.n_steps))
    print("batch_size             :", int(model.batch_size))
    print("n_epochs               :", int(model.n_epochs))
    print("gamma                  :", float(model.gamma))
    print("GAE lambda             :", float(model.gae_lambda))
    print("clip_range(1.0)        :", float(model.clip_range(1.0)))
    print("entropy katsayısı      :", float(model.ent_coef))
    print("value katsayısı        :", float(model.vf_coef))
    print("maks. gradient normu   :", float(model.max_grad_norm))
    print("Ortalama action std    :", end=" ")
    print(float(model.policy.log_std.exp().mean().item()))
    print("Policy                  :", model.policy)
    print("=" * 72)


if __name__ == "__main__":
    main()

