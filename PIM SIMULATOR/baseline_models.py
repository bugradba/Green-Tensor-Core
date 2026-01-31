class GPUBaseline:
    """
    NVIDIA Jetson Nano baseline (Turkcell edge cihazlarında yaygın).
    """

    def __init__(self):
        self.name = "NVIDIA Jetson Nano"
        self.tdp_watts = 10  # Max güç tüketimi
        self.idle_watts = 2  # Boşta güç tüketimi

        # Teorik Tepe Hız: 472 GFLOPS (FP16)
        # Ancak PIM projemiz INT8/INT4 odaklı olduğu için karşılaştırma adil olsun diye
        # Jetson'ın INT8 performansını değil, genel işlem kapasitesini baz alıyoruz.
        self.peak_mac_per_second = 472e9

        # KRİTİK EKLEME: Donanım Kullanım Oranı (Utilization)
        # Gerçek hayatta "Memory Wall" yüzünden GPU %100 çalışamaz.
        # Genellikle %40-%60 arası verim alınır.
        self.utilization_rate = 0.5

        # DRAM Erişim Maliyeti (Joule/bit) - LPDDR4 tahmini
        self.dram_energy_per_bit = 20e-12

    def model_inference(self, total_macs, model_size_bits=0):
        """
        Args:
            total_macs: İşlem sayısı
            model_size_bits: Modelin bellekten okunması gereken veri boyutu (opsiyonel)
        """
        # 1. Efektif Hız (Gerçek dünya hızı)
        effective_macs_sec = self.peak_mac_per_second * self.utilization_rate
        
        # 2. Latency Hesabı
        latency_seconds = total_macs / effective_macs_sec
        
        # 3. İşlemci Enerjisi (Compute Energy)
        # İşlem sırasında TDP kadar, bekleme durumunda idle kadar yakar varsayımı basitleştirilmiştir.
        # Burada worst-case (tam yük) varsayıyoruz.
        compute_energy_joules = self.tdp_watts * latency_seconds
        
        # 4. Bellek Enerjisi (Memory Energy) - PIM'in en büyük avantajı burası!
        # Eğer model_size verilmezse, her MAC işlemi için en az 2 bit veri (weight+input) okunduğunu varsayalım.
        total_bits_accessed = model_size_bits if model_size_bits > 0 else total_macs * 8 
        memory_energy_joules = total_bits_accessed * self.dram_energy_per_bit
        
        total_energy_mj = (compute_energy_joules + memory_energy_joules) * 1000

        return {
            'total_energy_mj': total_energy_mj,
            'total_latency_ms': latency_seconds * 1000,
            'avg_power_mw': (total_energy_mj / (latency_seconds * 1000)) * 1000 if latency_seconds > 0 else 0,
            'platform': self.name,
            'details': f"Utilization: %{self.utilization_rate*100}"
        }


class CPUBaseline:
    """
    Intel Core i5 baseline (Standart Laptop/Edge Gateway).
    """
    def __init__(self):
        self.name = "Intel Core i5 (Baseline)"
        self.tdp_watts = 15  # U-serisi işlemciler
        self.peak_mac_per_second = 200e9 
        
        # CPU'lar seri işlemlerde iyidir ama paralel matris çarpımında (AI) verimi düşüktür.
        self.utilization_rate = 0.7 
        
        # DDR4 Enerjisi (LPDDR'dan daha yüksektir)
        self.dram_energy_per_bit = 25e-12

    def model_inference(self, total_macs, model_size_bits=0):
        effective_macs_sec = self.peak_mac_per_second * self.utilization_rate
        latency_seconds = total_macs / effective_macs_sec
        
        compute_energy_joules = self.tdp_watts * latency_seconds
        
        total_bits_accessed = model_size_bits if model_size_bits > 0 else total_macs * 8
        memory_energy_joules = total_bits_accessed * self.dram_energy_per_bit
        
        total_energy_mj = (compute_energy_joules + memory_energy_joules) * 1000
        
        return {
            'total_energy_mj': total_energy_mj,
            'total_latency_ms': latency_seconds * 1000,
            'avg_power_mw': self.tdp_watts * 1000, # CPU genelde max TDP'de çalışır AI yükünde
            'platform': self.name
        }






































