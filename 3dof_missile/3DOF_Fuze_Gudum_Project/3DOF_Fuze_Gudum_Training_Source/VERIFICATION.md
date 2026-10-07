# Paket Doğrulama Kaydı

Doğrulama tarihi: 16 Eylül 2026

Teslim arşivi oluşturulmadan önce aşağıdaki kontroller temiz bir geçici Python
ortamında çalıştırılmıştır. Smoke-test sırasında üretilen model/log dosyaları
teslim paketine eklenmemiştir.

## Kaynak ve CLI

- Bütün Python dosyaları `py_compile` kontrolünden geçti.
- Kritik sözdizimi/isim hataları için Ruff `E9,F63,F7,F82` kontrolü geçti.
- `python train.py --help` bütün eğitim seçeneklerini başarıyla yükledi.
- Reward + observation çalışması `--phase all --dry-run` ile dokuz bağımsız
  sıfırdan eğitim komutunu ve manifesti üretti.

## Gerçek eğitim smoke-test'i

Tek ortam, V6 predictive-CPA reward ve 23 girdili observation ile yeni actor ve
critic oluşturuldu. İstenen `1` adım, PPO'nun tam rollout kuralı nedeniyle 1024
çevre adımı olarak tamamlandı. Başlangıç ve final sabit-seed değerlendirmeleri,
TensorBoard kaydı, `best_model.zip`, `final_model.zip` ve
`training_metadata.json` başarıyla üretildi.

Bu testin amacı performans öğrenmek değil, uçtan uca eğitim zincirinin çalıştığını
kanıtlamaktır.

## Sürüm kontrolleri

Aşağıdaki betiklerin tamamı geçti:

- `tests/check_geometry_v1.py`
- `tests/check_target_sphere_v1.py`
- `tests/check_closest_approach_v4.py`
- `tests/check_priority_replay_v5.py`
- `tests/check_fresh_start_cpa_v6.py`

Doğrulanan başlıklar arasında global dönüş tutarlılığı, target-sphere örnekleme,
eşit-alan elevation bantları, V4 reward sürekliliği, V5 priority/uniform ayrımı,
CPA matematiği ve 12 sektörlü sabit-seed değerlendirme bulunmaktadır.

## Paketlenmiş checkpoint

`inspect_model.py` ile doğrudan model arşivinden okunan değerler:

| Alan | Değer |
|---|---:|
| Kaydedilmiş PPO adımı | 700008 |
| Observation | 23 |
| Action | 2 |
| Gizli katmanlar | 128–128, Tanh |
| `n_steps` | 1024 |
| `batch_size` | 128 |
| `n_epochs` | 10 |
| `gamma` | 0.995 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.20 |

Paketlenmiş model ayrıca `tests/test_ppo.py` ile deterministik nominal senaryoda
çalıştırıldı: 10 m yarıçapta `hit`, minimum kayıtlı menzil yaklaşık `4.100 m`,
terminal zamanı yaklaşık `6.780 s` olarak elde edildi. Bu tek senaryo yalnız
yükleme/çıkarım smoke-test'idir; geniş-geometri başarı oranı yerine geçmez.

