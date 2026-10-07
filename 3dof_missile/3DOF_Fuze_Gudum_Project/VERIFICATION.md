# Final Paket Doğrulaması

Paket, CPU üzerinde gerçek `best_model.zip` yüklenerek baştan sona
çalıştırılmıştır.

Komut:

```text
python main.py --controller both --no-show --no-save
```

Doğrulama senaryosu:

- başlangıç menzili: 5500.001 m;
- füze/hedef hızı: 310/300 m/s;
- füze heading/gamma: -165°/0° (explicit);
- hedef: 195° heading, 12°/s sabit dönüş;
- fizik/kontrol adımı: 0.02/0.10 s;
- vuruş yarıçapı: 1 m;
- model observation/action boyutu: 23/2.

Sonuç:

| Yöntem | Durum | Terminal [s] | Adım-içi CPA [m] | Maks. komut [g] | RMS komut [g] |
|---|---:|---:|---:|---:|---:|
| PPO | hit | 12.999 | 0.495 | 12.061 | 4.451 |
| PN | hit | 13.294 | 0.007 | 12.372 | 5.371 |

Ek kontroller:

- Python sözdizimi derlemesi geçti.
- Ruff statik kontrolü geçti.
- PPO modeli Stable-Baselines3 ile CPU üzerinde yüklendi.
- Model observation sürümü `v2`, şekli `(23,)` olarak doğrulandı.
- Model action şekli `(2,)` olarak doğrulandı.
- PPO ve PN aynı 10 Hz karar periyoduyla eşzamanlı çalıştırıldı.
- PNG, CSV ve JSON çıktı üretimi doğrulandı.

Buradaki sayılar yalnızca paketin entegrasyon doğrulamasıdır; istatistiksel bir
başarı oranı iddiası değildir. Geniş-geometri/Monte Carlo sonuçları ayrı deney
verileridir.
