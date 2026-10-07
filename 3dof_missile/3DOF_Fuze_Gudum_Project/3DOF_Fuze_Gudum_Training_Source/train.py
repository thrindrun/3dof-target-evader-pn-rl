"""PPO eğitim aracının proje-kökü giriş noktası.

Asıl uygulama ``training/train_ppo.py`` içindedir. Bu küçük sarmalayıcı sayesinde
öğretmen incelemesinde ve komut satırında ``python train.py ...`` kullanılabilir.
"""

from training.train_ppo import main


if __name__ == "__main__":
    main()

