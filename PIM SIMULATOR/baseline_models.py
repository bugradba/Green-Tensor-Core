"""
GPU / CPU baz modelleri.

Basit bir roofline modeli kullanılır:
    gecikme = max(hesaplama süresi, DRAM transfer süresi)
    enerji  = hesaplama gücü × hesaplama süresi
            + boşta güç × (bellek beklerken geçen süre)
            + DRAM erişim enerjisi (bit başına)

Hesaplama gücü, TDP'den DRAM'in tepe gücü düşülerek bulunur; böylece
ortalama güç hiçbir zaman cihazın TDP'sini aşmaz.

Veri yeniden kullanımı (data reuse): Katmanın input, weight ve output
tensörlerinin DRAM'den bir kez okunup/yazıldığı varsayılır (`data_bits`).
Bu, GPU lehine iyimser bir varsayımdır; yani PIM karşılaştırması
"en iyi durumdaki GPU"ya karşı yapılır. `data_bits` verilmezse
MAC başına `default_bits_per_mac` bit taşındığı varsayılır.
"""


class _RooflineDevice:
    name = "Generic"
    tdp_watts = 10.0
    idle_watts = 2.0
    peak_mac_per_second = 1e9
    utilization_rate = 0.5
    dram_energy_per_bit = 20e-12      # J/bit
    dram_bandwidth_bits = 25.6e9 * 8  # bit/s
    # Hiç veri boyutu verilmezse, on-chip cache'in her byte'ı ~64 MAC'te
    # yeniden kullandığı varsayılır: 8 bit / 64 MAC.
    default_bits_per_mac = 8 / 64

    @property
    def dram_peak_watts(self):
        return self.dram_energy_per_bit * self.dram_bandwidth_bits

    @property
    def compute_watts(self):
        return max(self.tdp_watts - self.dram_peak_watts, self.idle_watts)

    def model_inference(self, total_macs, model_size_bits=0):
        """
        Args:
            total_macs: MAC sayısı
            model_size_bits: DRAM'den taşınan toplam veri (bit). 0 ise
                             `default_bits_per_mac` varsayımı kullanılır.
        """
        effective_macs_sec = self.peak_mac_per_second * self.utilization_rate
        compute_s = total_macs / effective_macs_sec

        bits = model_size_bits if model_size_bits > 0 else total_macs * self.default_bits_per_mac
        memory_s = bits / self.dram_bandwidth_bits

        latency_s = max(compute_s, memory_s)
        stall_s = latency_s - compute_s

        compute_energy_j = self.compute_watts * compute_s + self.idle_watts * stall_s
        memory_energy_j = bits * self.dram_energy_per_bit
        total_energy_mj = (compute_energy_j + memory_energy_j) * 1000
        latency_ms = latency_s * 1000

        return {
            'total_energy_mj': total_energy_mj,
            'total_latency_ms': latency_ms,
            'energy_compute_mj': compute_energy_j * 1000,
            'energy_memory_mj': memory_energy_j * 1000,
            'bound': 'memory' if memory_s > compute_s else 'compute',
            # mJ / s = mW
            'avg_power_mw': total_energy_mj / (latency_ms / 1000) if latency_ms > 0 else 0,
            'platform': self.name,
            'details': f"Utilization: %{self.utilization_rate*100}",
        }


class GPUBaseline(_RooflineDevice):
    """
    NVIDIA Jetson Nano baseline (Turkcell edge cihazlarında yaygın).
    """

    def __init__(self):
        self.name = "NVIDIA Jetson Nano"
        self.tdp_watts = 10   # Max güç tüketimi (modül, DRAM dahil)
        self.idle_watts = 2   # Boşta güç tüketimi

        # Teorik tepe: 472 GFLOPS (FP16). Karşılaştırmayı basit tutmak için
        # MAC/s olarak kullanılıyor.
        self.peak_mac_per_second = 472e9

        # Gerçekte "Memory Wall" yüzünden GPU %100 çalışamaz (%40-60).
        self.utilization_rate = 0.5

        # LPDDR4: ~20 pJ/bit, 25.6 GB/s
        self.dram_energy_per_bit = 20e-12
        self.dram_bandwidth_bits = 25.6e9 * 8


class CPUBaseline(_RooflineDevice):
    """
    Intel Core i5 baseline (Standart Laptop/Edge Gateway).
    """

    def __init__(self):
        self.name = "Intel Core i5 (Baseline)"
        self.tdp_watts = 15   # U-serisi işlemciler
        self.idle_watts = 3
        self.peak_mac_per_second = 200e9

        # CPU'lar paralel matris çarpımında (AI) daha düşük verimlidir.
        self.utilization_rate = 0.7

        # DDR4: ~25 pJ/bit, ~34 GB/s
        self.dram_energy_per_bit = 25e-12
        self.dram_bandwidth_bits = 34e9 * 8
