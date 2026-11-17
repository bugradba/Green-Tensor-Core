"""
Karbon ve trafik modeli: saatlik şebeke karbon yoğunluğu, baz istasyonu
trafik profili ve PIM donanımının gömülü (üretim) karbonu.

ÖNEMLİ: Buradaki saatlik profiller *stilize* profillerdir; gerçek ölçüm değildir.
Yıllık ortalama ve güneşin öğle payı yayımlanmış değerlere göre ayarlanmıştır
(kaynaklar README'de). Gerçek saatlik veriyle (ör. EPİAŞ Şeffaflık Platformu
üretim verisinden hesaplanan gCO2/kWh) çalışmak için `load_profile_csv` kullanın.
"""
import csv
from dataclasses import dataclass

import numpy as np

HOURS = np.arange(24)


# ---------------------------------------------------------------------------
# Şebeke karbon yoğunluğu
# ---------------------------------------------------------------------------

def _solar_shape(sunrise, sunset):
    """Gün içi güneş üretim şekli (0..1), sinüs yaklaşımı, saat ortası örneklenir."""
    t = HOURS + 0.5
    s = np.sin(np.pi * (t - sunrise) / (sunset - sunrise))
    return np.where((t > sunrise) & (t < sunset), s, 0.0)


def grid_carbon_intensity(season='summer', non_solar_gco2=460.0, solar_peak_share=None):
    """
    Saatlik şebeke karbon yoğunluğu (gCO2/kWh), 24 değer.

    Model: CI(h) = non_solar_gco2 × (1 − güneş_payı(h)).
    Güneş dışındaki karışım (kömür, gaz, hidro, rüzgâr...) gün içinde sabit
    kabul edilir; değişkenliği güneş sağlar. Bu basitleştirme, akşam gaz
    santrallerinin devreye girmesi gibi etkileri ihmal eder.

    Varsayılan değerler:
        non_solar_gco2 = 460  → yıllık ortalama ≈ 425 gCO2/kWh
                               (yayımlanmış yıllık değerler: 411–428)
        yaz: öğle güneş payı %30 (saatlik rekor %46), gün doğumu 5:30, batımı 20:00
        kış: öğle güneş payı %15, gün doğumu 7:30, batımı 17:00
    """
    if season == 'summer':
        peak = 0.30 if solar_peak_share is None else solar_peak_share
        shape = _solar_shape(5.5, 20.0)
    elif season == 'winter':
        peak = 0.15 if solar_peak_share is None else solar_peak_share
        shape = _solar_shape(7.5, 17.0)
    else:
        raise ValueError("season: 'summer' veya 'winter'")
    return non_solar_gco2 * (1.0 - peak * shape)


# ---------------------------------------------------------------------------
# Baz istasyonu trafiği
# ---------------------------------------------------------------------------

# Normalize edilmiş (tepe = 1) stilize mobil şebeke günlük yük profili:
# gece 04-05'te en düşük, öğleden sonra yükselen, 20-22 arası tepe.
_TRAFFIC = np.array([
    0.45, 0.32, 0.23, 0.18, 0.15, 0.16, 0.22, 0.38,   # 00-07
    0.55, 0.66, 0.72, 0.76, 0.80, 0.80, 0.78, 0.78,   # 08-15
    0.80, 0.84, 0.88, 0.93, 0.98, 1.00, 0.90, 0.68,   # 16-23
])


def traffic_profile():
    """Normalize günlük trafik (tepe = 1), 24 değer."""
    return _TRAFFIC.copy()


def load_profile_csv(path, column):
    """
    24 satırlık CSV'den saatlik profil okur (başlık satırı gerekli).
    Örn. column='gco2_per_kwh' ile gerçek karbon yoğunluğu.
    """
    with open(path, newline='') as f:
        vals = [float(row[column]) for row in csv.DictReader(f)]
    if len(vals) != 24:
        raise ValueError(f"24 saatlik değer bekleniyordu, {len(vals)} bulundu")
    return np.array(vals)


# ---------------------------------------------------------------------------
# Gömülü karbon
# ---------------------------------------------------------------------------

@dataclass
class EmbodiedCarbon:
    """
    PIM donanımının üretim karbonu.

    pPIM, DRAM içine eklenen mantık devreleriyle çalışır. Gömülü karbon,
    DRAM kapasitesi × DRAM'in GB başına karbonu × PIM mantık ek yükü
    olarak tahmin edilir.

        dram_gco2_per_gb = 48  (LPDDR4 için yayımlanmış değer, ~48 g/GB)
        pim_overhead     = 2.0 (PIM mantığı, test, paketleme için varsayım)

    Yayımlanmış gömülü karbon değerlerinde 1,6–4 kat belirsizlik olduğu
    bilinir; deney bu yüzden geniş bir aralık tarar.
    """
    capacity_gb: float = 8.0
    dram_gco2_per_gb: float = 48.0
    pim_overhead: float = 2.0

    @property
    def kgco2(self):
        return self.capacity_gb * self.dram_gco2_per_gb * self.pim_overhead / 1000.0


def payback_days(embodied_kg, daily_saving_kg):
    """Gömülü karbonun operasyonel tasarrufla geri ödenme süresi (gün)."""
    if daily_saving_kg <= 0:
        return float('inf')
    return embodied_kg / daily_saving_kg
