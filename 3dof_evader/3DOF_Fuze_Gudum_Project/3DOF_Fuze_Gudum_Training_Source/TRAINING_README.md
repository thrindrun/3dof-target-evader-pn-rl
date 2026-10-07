# 3DOF Füze Güdümü — PPO Eğitim Kaynak Paketi

Bu paket, final çalışma-zamanı uygulamasından ayrı olarak PPO politikasının
nasıl oluşturulduğunu, reward ve observation deneylerinin nasıl yürütüldüğünü
ve checkpoint'lerin nasıl seçildiğini gösterir. Paket yalnız bir komut örneği
değil; kullanılan 3DOF ortamını, Gymnasium arayüzünü, eğitim döngüsünü, sabit
seed doğrulamasını, checkpoint yönetimini ve öğrenme grafiği üretimini içerir.

## 1. En kısa kullanım

Windows PowerShell'de zip'i açtıktan sonra:

```powershell
cd C:\Projects\3DOF_Fuze_Gudum_Training_Source
.\setup.bat
.\verify_source.bat
.\.venv\Scripts\python.exe .\train.py --help
```

`setup.bat`, `.venv` sanal ortamını oluşturur ve `requirements.txt` içindeki
bağımlılıkları kurar. Dosyaya çift tıklamak da mümkündür; hata metnini daha
rahat görmek için PowerShell'den çalıştırmak önerilir.

Paketlenmiş başarılı checkpoint'in adımını ve PPO hiperparametrelerini görmek
için:

```powershell
.\.venv\Scripts\python.exe .\inspect_model.py
```

## 2. Dosya yapısı

| Yol | Amaç |
|---|---|
| `train.py` | Kök dizinden kullanılan sade eğitim giriş noktası. |
| `training/train_ppo.py` | Asıl PPO eğitim, devam, doğrulama ve checkpoint kodu. |
| `training/run_learning_study.py` | Altı reward ve üç observation deneyini kontrollü biçimde sıfırdan yürütür. |
| `training/migrate_observation_v2.py` | Eski 10 girdili ağı 23 girdili yapıya geçirirken kullanılan yardımcı araç. |
| `air_to_air_3dof_ppo.py` | 3DOF fizik, hedef manevraları, PN, RL arayüzü, Gymnasium ortamı ve reward fonksiyonu. |
| `tests/plot_learning_study.py` | Sabit-seed değerlendirme kayıtlarından öğrenme eğrilerini üretir. |
| `tests/evaluate_ppo_geometry.py` | Eğitilmiş modeli bağımsız geniş-geometri seed'lerinde ölçer. |
| `tests/check_*.py` | Geometri V1, target-sphere, closest-approach V4, priority replay V5 ve CPA V6 kontrolleri. |
| `models/.../best_model.zip` | Teslim edilen başarılı 23 girdili PPO checkpoint'i. |
| `models/.../training_metadata.json` | Son eğitim aşamasının gerçek, makine tarafından yazılmış ayar kaydı. |
| `examples/` | Sıfırdan eğitim, dry-run ve son aşama komut örnekleri. |
| `docs/` | Geometri/reward sürümlerinin tarihsel teknik notları. |

`models`, `checkpoints`, `logs` ve `results` klasörleri yeni bir eğitim
başlatıldığında otomatik oluşturulur. Her stage ayrı klasöre yazıldığı için bir
deney diğerinin modelini ezmez.

## 3. Sıfırdan eğitim ile devam eğitimi aynı şey değildir

İki çalışma biçimi kod içinde açıkça ayrılmıştır:

- `--fresh-start`: Yeni actor ve critic oluşturur; `num_timesteps=0` ile başlar.
  Eski bir model yüklenemez.
- `--source <checkpoint>`: Actor/critic ağırlıklarını ve PPO sayacını yükleyip
  curriculum'un sonraki aşamasına devam eder.
- `--reset-critic`: Yalnız devam eğitiminde kullanılabilir. Actor korunurken
  value ağı ve optimizer momentleri yeniden başlatılır. Reward ölçeği veya
  hedef görev anlamlı biçimde değiştiğinde kullanılır.

Kod, `--fresh-start` ile `--source` birlikte verilirse eğitimi başlamadan
durdurur. Böylece sıfırdan diye raporlanan bir deneyin gizlice eski politika
ağırlığı kullanması engellenir.

## 4. Temiz bir V6 eğitimi başlatma

Hazır PowerShell örneği:

```powershell
.\examples\train_fresh_v6.ps1
```

Bu örnek:

- yeni actor ve critic ile sıfırdan başlar;
- güncelleme öncesi rastgele politikayı aynı seed'lerde ölçer;
- 23 girdili `v2` observation kullanır;
- predictive-CPA reward profilini seçer;
- 5 km target-sphere geometrisinde azimutun tamamını ve elevation
  `[-30°, 30°]` aralığını örnekler;
- 12 paralel ortam ve 12 dengeli azimut doğrulama sektörü kullanır;
- 20 bin adımda bir aynı 120 doğrulama episode'unu çalıştırır.

Komut yaklaşık 200 bin PPO örneği ister. Stable-Baselines3 güncellemeyi tam
rollout bloklarıyla yaptığı için gerçek son sayaç istenen değeri biraz aşabilir.
Bu durum hata değildir.

## 5. Reward gelişim deneyleri

Reward karşılaştırmalarında fizik, seed, PPO hiperparametreleri, başlangıç
geometrisi ve observation sabit tutulur; yalnız reward profili değiştirilir.
Her deney ayrı actor/critic ile sıfırdan başlar.

| Sıra | Profil | Eklenen bilgi |
|---:|---|---|
| R1 | `weak_binary` | Yalnız terminalde hit `+1`, başarısızlık `-1`; dense bilgi yok. |
| R2 | `sparse` | Kalibre terminal ödül/cezaları ile küçük zaman ve kontrol maliyetleri. |
| R3 | `range_progress` | Anlık menzil azalması ile erken yön bilgisi. |
| R4 | `best_range` | Episode içinde ulaşılan yeni minimum menzil rekoru. |
| R5 | `terminal_distance` | Kaçırmaların minimum mesafeye göre sürekli ayrıştırılması. |
| R6 | `predictive_cpa` | Sürekli closest-approach terminal doğruluğu ve tahminî CPA ilerlemesi. |

Reward fazını çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe .\training\run_learning_study.py `
  --phase reward `
  --tag reward_study `
  --reward-steps 200000 `
  --n-envs 12 `
  --eval-episodes 120 `
  --eval-frequency 20000
```

Önce komutları incelemek, hiçbir eğitim başlatmamak için:

```powershell
.\examples\run_learning_study_dry.ps1
```

Tamamlanan deneyler tekrar çağrıldığında atlanır. Aynı stage için yarım çıktı
bulunursa kod otomatik silme yapmaz; yeni bir `--tag` ister. Bu koruma, yarım
deney ile temiz deney sonuçlarının karışmasını önler.

## 6. Observation karşılaştırması

Observation fazında reward da dahil diğer kontrol değişkenleri sabittir. Yalnız
ağın giriş vektörü değişir:

| Sürüm | Boyut | İçerik özeti |
|---|---:|---|
| `legacy10` | 10 | Yerel bağıl konum/hız, füze hız/kütlesi, normalize zaman ve kalan motor oranı. |
| `markov13` | 13 | İlk 10 girdiye füze irtifası ile `sin(gamma)`, `cos(gamma)` eklenir. |
| `v2` | 23 | İlk 13 girdiye ince yanal hata, yerel LOS birim vektörü, closing speed, LOS-rate ve time-to-go eklenir. |

Üç observation deneyini çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe .\training\run_learning_study.py `
  --phase observation `
  --tag observation_study `
  --observation-steps 150000 `
  --n-envs 12 `
  --eval-episodes 120 `
  --eval-frequency 20000
```

Daha büyük observation teorik olarak daha fazla bilgi taşır; fakat aynı adım
bütçesinde mutlaka daha hızlı öğreneceği söylenemez. Sonuçlar başarı oranı,
ortalama/P90 minimum menzil ve eğitim eğrisi birlikte değerlendirilmelidir.

## 7. Policy ağı, action ve PPO ayarları

Sıfırdan eğitimde ağ `MlpPolicy` biçimindedir:

- actor/critic ortak yapı tanımı: iki gizli katman, `[128, 128]`;
- aktivasyon: `tanh`;
- action boyutu: 2;
- `action[0]`: normalize yatay-normal ivme komutu;
- `action[1]`: normalize düşey-normal ivme komutu;
- iki action `[-1,1]` aralığından fiziksel `max_command_g` sınırına ölçeklenir.

PPO hiperparametreleri:

| Parametre | Değer |
|---|---:|
| `n_steps` | 1024 / ortam |
| `batch_size` | 128 |
| `n_epochs` | 10 |
| `gamma` | 0.995 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.20 |
| `ent_coef` | 0.0 |
| ağ | 128–128, Tanh |
| cihaz | CPU |

Örneğin `n_envs=12` ise bir rollout `1024 × 12 = 12288` çevre adımıdır. Bu
yüzden 300000 ek adım istenen bir koşu, rollout sınırında 307200 gerçek adım
ekleyebilir. `training_metadata.json` hem istenen hem gerçek başlangıç/bitiş
sayaçlarını kaydeder.

## 8. V6 reward denkleminin yapısı

Her kontrol adımındaki toplam reward, aşağıdaki terimlerin toplamının ortak
`reward_scale` ile çarpılmasıdır:

```text
r = scale * (
      range_progress
    + minimum_range_progress
    + predictive_cpa_progress
    - control_effort
    - time_cost
    + terminal
    + terminal_accuracy
    + miss_distance
)
```

- `range_progress`: Bir önceki adıma göre menzil azalması; uzaklaşmada negatiftir.
- `minimum_range_progress`: Yalnız yeni minimum menzil rekorunda pozitiftir.
- `predictive_cpa_progress`: Mevcut bağıl hız sabit kalsaydı oluşacak CPA
  mesafesindeki iyileşmedir; erken credit assignment sağlar.
- `control_effort`: Gereksiz büyük normal ivmeyi azaltır.
- `time_cost`: Daha kısa önlemeyi tercih ettiren küçük adım maliyetidir.
- `terminal_accuracy`: Closest-approach modunda eşik çevresindeki kesikli
  hit/miss sıçramasını sürekli bir minimum-menzil fonksiyonuyla değiştirir.
- `miss_distance`: Eski terminal profillerinde ıskalamaları minimum menzile göre
  ayırır.

V6'da predictive CPA, bir PN komutu değildir. Policy hâlâ iki ivme action'ını
üretir; CPA yalnız reward içinde erken kalite sinyali sağlar.

## 9. Sabit-seed doğrulama ve best model seçimi

Her değerlendirmede aynı seed listesi kullanılır. Target-sphere testinde episode
sayısı sektör sayısının katıysa her azimut sektörü eşit örnek alır. Kaydedilen
başlıca metrikler:

- başarı oranı;
- ortalama, medyan ve P90 minimum menzil;
- ortalama terminal süresi ve reward;
- reward terimlerinin ayrı ortalamaları;
- sektör başarıları ve en kötü sektör ortalama minimum menzili.

`best_model.zip`, yalnız son eğitim adımı değildir. Callback önce başarıyı,
yakın başarılar arasında en kötü sektörü, sonra P90/ortalama minimum menzili
karşılaştırır. `final_model.zip` son güncellemeyi; `best_model.zip` doğrulamada
seçilen checkpoint'i temsil eder. Nihai testte best model kullanılmalıdır.

Kayıt yerleri:

```text
models/<stage>/best_model.zip
models/<stage>/final_model.zip
models/<stage>/training_metadata.json
checkpoints/<stage>/ppo_<stage>_*.zip
logs/evaluation/<stage>/fixed_seed_evaluations.jsonl
logs/evaluation/<stage>/best_validation_metrics.json
logs/tensorboard/
```

TensorBoard:

```powershell
.\.venv\Scripts\tensorboard.exe --logdir .\logs\tensorboard
```

## 10. Paketlenmiş son modelin gerçek kökeni

Paket içindeki makine kaydı, son hard-pocket aşaması için şunları bildirir:

- stage: `fresh_cpa_v6_hardpocket_7p5m_300k`;
- observation: `v2`, 23 girdi;
- reward: V6 predictive CPA;
- kaynak sayaç: `400008`;
- istenen ek adım: `300000`;
- gerçek stage bitişi: `707208`;
- doğrulamayla seçilen paketli `best_model.zip` sayacı: `700008`;
- öğrenme oranı: `1e-5`;
- başlangıç action std: yaklaşık `0.05`;
- geometri: 5 km target-sphere;
- ana elevation cebi: `[-60°, -40°]`;
- priority azimut cebi: `[-180°, -150°)`;
- priority dal olasılığı: `0.7`;
- doğrulama: 600 episode, 12 dengeli sektör, seed `810000–810599`.

Bu nedenle son aşama **sıfırdan 300 bin adımlık bir eğitim değildir**; önceki
curriculum checkpoint'inden devamdır. `examples/final_stage_command.ps1`, metadata
ile aynı parametreleri açıkça gösterir. Ancak o aşamanın kaynak checkpoint'i
`fresh_cpa_v6_el30_100m_1m/best_model.zip` bu teslimde bulunmadığı için birebir
yeniden çalıştırma için ayrıca temin edilmelidir. Teslim edilen son best model
incelenebilir ve yeni bir stage'e kaynak olarak kullanılabilir.

Bu ayrım bilimsel izlenebilirlik açısından bilinçlidir: var olmayan bir başlangıç
checkpoint'i pakette varmış gibi gösterilmemiştir.

## 11. Bağımsız geometri testi

Paketlenmiş modeli yeni seed'lerde ölçmek için örnek:

```powershell
.\.venv\Scripts\python.exe .\tests\evaluate_ppo_geometry.py `
  --model .\models\fresh_cpa_v6_hardpocket_7p5m_300k\best_model.zip `
  --episodes 1000 `
  --seed 990000 `
  --geometry-mode target_sphere `
  --range-min-m 5000 `
  --range-max-m 5000 `
  --launch-azimuth-min-deg -180 `
  --launch-azimuth-max-deg 180 `
  --launch-elevation-min-deg -85 `
  --launch-elevation-max-deg 85 `
  --randomization-scale 0.5 `
  --passed-target-arm-progress-m 100 `
  --sector-count 12 `
  --sector-threshold-m 10 `
  --thresholds 100 50 25 10 5 2.5 1
```

Eğitim seed'leri ile nihai test seed'leri ayrı tutulmalıdır. Test sonucunda tek
başına toplam başarı değil; elevation bantları, azimut sektörleri, P90 ve en kötü
örnek de raporlanmalıdır.

## 12. Yeniden üretilebilirlik sınırları

- Seed'ler, sabit doğrulama listeleri, kütüphane sürümleri ve eğitim metadata'sı
  kaydedilir.
- Çok işlemli ortam sıralaması, işletim sistemi ve PyTorch/SB3 sürümü nedeniyle
  bit düzeyinde aynı ağırlık dosyası garanti edilmez.
- `requirements-lock.txt`, geliştirilen makinenin ayrıntılı sürüm kaydıdır.
  Bazı Windows kurulumlarında `torch==...+cpu` için PyTorch'un uygun wheel
  kaynağı gerekebilir; normal kurulum bu nedenle `requirements.txt` kullanır.
- Eğitim çıktıları büyük ve uzun süreli olabilir. Önce `--dry-run`, sonra kısa
  bir stage, en son tam bütçe önerilir.

## 13. Sunumda gösterilebilecek kısa akış

1. `inspect_model.py` ile checkpoint'in 23 girişli, iki action'lı ve kaç PPO
   adımında kaydedildiğini gösterin.
2. `run_learning_study.py --dry-run` ile bütün reward/observation koşullarının
   bağımsız sıfırdan başlatıldığını gösterin.
3. `apply_ppo_reward_profile` ve `build_normalized_rl_observation` fonksiyonlarını
   açarak değişkenlerin gerçekten kodda ayrıldığını gösterin.
4. `fixed_seed_evaluations.jsonl` ve öğrenme eğrilerinde başarı ile P90 minimum
   menzilin birlikte izlendiğini açıklayın.
5. Final modelin curriculum devam aşaması olduğunu metadata üzerinden dürüstçe
   belirtin; bağımsız geniş-geometri testini ayrı kanıt olarak sunun.
