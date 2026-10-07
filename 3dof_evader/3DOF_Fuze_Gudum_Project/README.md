# 3DOF Füze Güdümü — Final Ana Uygulama

Bu paket, eğitim ve toplu test betiklerinden ayrılmış son çalışma-zamanı
uygulamasıdır. Aynı 3DOF fizik ortamında eğitilmiş PPO politikası, vektörel
oransal seyrüsefer (PN) veya ikisi eşzamanlı çalıştırılabilir.

## Paket içeriği

- `main.py`: Senaryo ayarları, PPO/PN seçimi, özet, grafik ve sonuç dosyaları.
- `air_to_air_3dof_core.py`: 3DOF fizik, hedef manevraları, PN, PPO çıkarım
  adaptörü, adım-içi CPA ve simülasyon yöneticisi.
- `models/fresh_cpa_v6_hardpocket_7p5m_300k/best_model.zip`: 23 girdili son
  PPO politikası.
- `models/.../training_metadata.json`: Modelin eğitim konfigürasyonu ve kayıtlı
  doğrulama bilgileri.
- `requirements.txt`: Tekrar üretilebilir Python bağımlılıkları.
- `setup.bat`: Windows üzerinde sanal ortam oluşturup bağımlılıkları kurar.
- `run.bat`: Varsayılan senaryoyu çalıştırır.

Bu final pakette Gym reward fonksiyonu, PPO eğitim ortamı, checkpoint yönetimi,
curriculum ve araştırma/test betikleri bulunmaz. Model yalnızca çıkarım için
yüklenir.

## Hızlı başlangıç

Windows PowerShell veya Komut İstemi:

```powershell
cd C:\Projects\missile_guidance_final
setup.bat
run.bat
```

Projede çalışan bir `.venv` zaten varsa kurulumu tekrar yapmadan:

```powershell
.\.venv\Scripts\python.exe .\main.py
```

Grafik penceresi açılmadan PPO ve PN karşılaştırması:

```powershell
.\.venv\Scripts\python.exe .\main.py --controller both --no-show
```

Yalnız PN:

```powershell
.\.venv\Scripts\python.exe .\main.py --controller pn
```

Yalnız PPO ve 10 m vuruş yarıçapı:

```powershell
.\.venv\Scripts\python.exe .\main.py --controller ppo --hit-radius 10
```

Tüm seçenekler:

```powershell
.\.venv\Scripts\python.exe .\main.py --help
```

## Senaryoyu değiştirme

`main.py` dosyasının başındaki **KULLANICI AYARLARI** bölümü tek düzenleme
noktasıdır. En sık değiştirilecek alanlar:

- `GUIDANCE_MODE`: `"ppo"`, `"pn"` veya `"both"`.
- `TARGET_N_E_ALT_M`, `MISSILE_N_E_ALT_M`: North, East ve **irtifa** [m].
- hedef ve füze hızları, heading ve uçuş-yolu açıları.
- `TARGET_MANEUVER_KIND`: `straight`, `constant_turn`, `sinusoidal` veya
  `random_smooth`.
- manevra oranları, sinüs periyodu ve rastgele manevra seed'i.
- `HIT_RADIUS_M`, `FINAL_TIME_S`, `PHYSICS_DT_S` ve `CONTROL_PERIOD_S`.
- `SAVE_OUTPUTS`, `SHOW_FIGURE` ve `OUTPUT_DIR`.

### Füzenin ilk hız yönü

Başlangıç hız vektörü füzenin heading ve uçuş-yolu açısıyla belirlenir.
`MISSILE_ATTITUDE_MODE = "explicit"` iken füze otomatik olarak hedefe bakmaz;
`MISSILE_HEADING_DEG` ve `MISSILE_FLIGHT_PATH_DEG` doğrudan kullanılır.

`MISSILE_ATTITUDE_MODE = "toward_target"` seçilirse LOS açıları hesaplanır ve
şu iki hata eklenir:

- `MISSILE_AIM_AZIMUTH_ERROR_DEG`
- `MISSILE_AIM_ELEVATION_ERROR_DEG`

Program hem uygulanan açıları hem de salt LOS açılarını başlangıçta yazdırır.
Bu sayede ilk hızın hedefe dönük olup olmadığı sonuçlardan gizlenmez.

## Koordinat ve açı tanımları

Kullanıcı ayarlarında konum `(North, East, irtifa)` sırasındadır. Çekirdek bunu
NED sistemine `(North, East, Down)` çevirir:

```text
Down = -irtifa
```

- Heading 0°: kuzey
- Heading 90°: doğu
- Heading ±180°: güney
- Pozitif uçuş-yolu açısı: tırmanış

Durum vektörü çekirdekte şöyledir:

```text
[N, E, D, V, chi, gamma, mass]
```

## PPO ve PN'nin adil çalıştırılması

`both` modunda iki füzenin başlangıç konumu, hızı, yönü, aerodinamiği,
propulsiyonu ve hedefi aynıdır. PPO ile PN arasındaki tek kasıtlı fark güdüm
yasasıdır.

- Fizik integrasyonu: varsayılan 0.02 s (50 Hz).
- Güdüm karar periyodu: varsayılan 0.10 s (10 Hz).
- PPO, eğitildiği gibi aksiyonunu 10 Hz günceller.
- PN, `SampleAndHoldGuidance` ile aynı 10 Hz'de güncellenir.
- Her iki yöntem aynı 30 g algoritmik sınıra ve aynı füze fiziksel sınırına
  tabidir.

PN sabiti varsayılan `N=4` değeridir ve `PN_NAVIGATION_CONSTANT` ile
değiştirilebilir.

## Vuruş ve CPA yorumu

Fizik 20 ms örneklerle kaydedilse de çarpışma kararı yalnızca örnek noktalarına
bakmaz. Her fizik adımının başı ve sonu arasındaki bağıl doğru parçasında en
yakın yaklaşma (CPA) analitik olarak hesaplanır. Bu nedenle:

- `CPA [m]` gerçek karar metriğidir.
- CSV'deki/konsoldaki “minimum kayıtlı menzil” örneklenmiş değerdir.
- Kayıtlı minimumun vuruş yarıçapından büyük, adım-içi CPA'nın küçük olması
  hata değildir.

Varsayılan vuruş yarıçapı 1 m seçilmiştir. Böylece 1, 2.5, 5 ve 10 m eşikleri
aynı koşuda güvenilir biçimde raporlanır. Vuruş yarıçapı örneğin 10 m yapılırsa
simülasyon ilk 10 m olayında biter; o anda henüz aşılmamış daha küçük eşikler
“erken terminal—ölçülmedi” olarak gösterilir.

## Üretilen sonuçlar

Varsayılan klasör `results/final_default_comparison`:

- `trajectory_comparison.png`: 3B yol, bağıl menzil, hız ve komut grafikleri.
- `history.csv`: Her fizik örneğindeki araç durumları, menzil ve komutlar.
- `summary.json`: Senaryo ayarları ile PPO/PN performans metrikleri.

Konsol tablosunda durum, terminal zamanı, adım-içi CPA, maksimum/RMS komut ve
terminal hızı birlikte verilir.

## Model bilgisi ve geçerlilik sınırı

Paketlenen PPO modeli:

- observation: `v2`, 23 girdi;
- action: iki normalize komut (`a_chi`, `a_gamma`);
- eğitim kontrol periyodu: 0.10 s;
- kaydedilmiş SB3 adımı: yaklaşık 700 bin;
- eğitim geometrisi: 5 km hedef-merkezli küre örneklemesi;
- eğitimde yoğunlaştırılan bölge: azimut `[-180°, -150°)`, elevation yaklaşık
  `[-60°, -40°]`;
- reward: predictive-CPA tabanlı V6;
- model seçimi: sabit-seed doğrulama.

Model farklı azimut/elevation testlerinde güçlü genelleme göstermiş olsa da bu,
her hız, menzil, sensör hatası ve manevrada matematiksel garanti anlamına gelmez.
Özellikle eğitim dağılımından çok uzak hızlar, çok uzun menziller, büyük ilk yön
hataları veya gerçekçi gecikme/gürültü eklenmiş durumlar ayrıca doğrulanmalıdır.

Bu benzetim noktasal-kütle 3DOF modelidir. Arayıcı görüş alanı, ölçüm gürültüsü,
otopilot/aktüatör gecikmesi, aerodinamik açı dinamiği ve gerçek harp başlığı
modeli final uygulamada yer almaz. Sonuçlar bu model kabulleri içinde
yorumlanmalıdır.

## Sorun giderme

### Model yolu `.zip.zip` oluyor

`main.py` tekrarlanan `.zip.zip` ekini otomatik düzeltir ve uzantısız yolu da
dener. Yine de önerilen yol:

```python
MODEL_PATH = PROJECT_ROOT / "models" / "fresh_cpa_v6_hardpocket_7p5m_300k" / "best_model.zip"
```

### `stable_baselines3` bulunamadı

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PN-only modunda model yüklenmez; bu yüzden fizik ve PN tarafı PPO paketi
yüklenmeden de kod incelemesi amacıyla çalıştırılabilir. PPO ve `both` modları
Stable-Baselines3 ile PyTorch gerektirir.
