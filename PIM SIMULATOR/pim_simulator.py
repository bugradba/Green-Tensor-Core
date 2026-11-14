import numpy as np

class PIM_Core:
    """
    Tek bir pPIM core'u simüle eder.
    4-bit girişler alır, LUT-based işlem yapar.
    """

    def __init__(self, technology_node=28):
        # pPIM makalesinden alınan değerler
        self.delay_ns = 0.8  # Nano Saniye
        self.power_mw = 2.7  # Miliwat
        self.lut_size = 256  # 2**8
        self.tech_node = technology_node

        # LUT initialize
        self.lut = self._build_multiplication_lut()

    def _build_multiplication_lut(self):
        """
        4-bit x 4-bit için lookup table.
        """
        lut = np.zeros((16, 16), dtype=np.uint8)
        for i in range(16):
            for j in range(16):
                lut[i, j] = i * j
        return lut
    
    # DÜZELTME: Bu fonksiyon sola çekildi (artık sınıfın bir parçası)
    def multiply_4bit(self, a, b):
        """
        LUT kullanarak 4-bit çarpma.
        Args:
            a, b: 0-15 arası integer
        Returns:
            result: Çarpım sonucu
            energy_pj: Harcanan enerji
            latency_ns: İşlem gecikmesi
        """
        # LUT'tan oku
        result = self.lut[a, b]

        # Enerji hesabı: P(mW) × t(ns) = pJ
        energy_pj = self.power_mw * self.delay_ns

        return result, energy_pj, self.delay_ns

    def add_4bit(self, a, b):
        """
        4-bit toplama.
        """
        result = (a + b) & 0xFF  # 8 bit sonuç
        
        # DÜZELTME: 'energy.pj' yazım hatası 'energy_pj' yapıldı
        energy_pj = self.power_mw * self.delay_ns

        return result, energy_pj, self.delay_ns


class PIMCluster():
    """
    
    9 core(Çekirdek) dan oluşturuyoruz
    
    ÇÜNKÜ:8-bitlik bir çarpma işlemini, 4-bitlik donanımla simüle etmek için işlemi parçalara ayırdıklarında, 
    bu parçaları paralel işleyip birleştirmek için en uygun geometrik ve sayısal yapı 3 x 3 = 9 çekirdekli bir kümedir.

    """
    def __init__(self, technology_node=28):
        self.cores = [PIM_Core(technology_node) for _ in range(9)]

        #pPIM: A Programmable Processor-in-Memory Architecture With Precision-Scaling for Deep Learning 
        # makalesindeki alınan Cluster-level özellikler 

        self.mac_delay_ns = 6.4 # 8 stage işlme
        self.mac_power = 5.2
        self.router_overhead = 0.5 #Corelar arası iletişim

    def mac_8bit(self, a_8bit, b_8bit, accumulator=0, precision=8):
         """
        8-bit Multiply-Accumulate işlemi.
        
        İşlem adımları (pPIM paper Fig. 2):
        1. 8-bit sayıları 4-bit'lere böl: aH, aL, bH, bL
        2. Partial products: V0=aLxbL, V1=aLxbH, V2=aHxbL, V3=aHxbH
        3. 8 stage'de topla
        
        Args:
            a_8bit, b_8bit: 0-255 arası integer
            accumulator: Önceki sonuç (MAC'te kullanılır)
            precision: 8 veya 4 (precision scaling için)
        
        Returns:
            result: MAC sonucu
            total_energy_pj: Toplam enerji
            total_latency_ns: Toplam gecikme
        """
         
         total_energy = 0
         total_latency = 0

         if precision == 8:
            # 8-bit tam hassasiyet
            aH, aL = (a_8bit >> 4) & 0xF, a_8bit & 0xF
            bH, bL = (b_8bit >> 4) & 0xF, b_8bit & 0xF
            
            # Stage 1-4: Partial products (4 core çarpma yapar)
            V0, e0, l0 = self.cores[0].multiply_4bit(aL, bL)
            V1, e1, l1 = self.cores[1].multiply_4bit(aL, bH)
            V2, e2, l2 = self.cores[2].multiply_4bit(aH, bL)
            V3, e3, l3 = self.cores[3].multiply_4bit(aH, bH)
            
            # Stage 5-8: Accumulation (5 core toplama yapar)
            # Basitleştirilmiş: gerçekte daha karmaşık bit-shifting var
            result = int(V0) + (int(V1) << 4) + (int(V2) << 4) + (int(V3) << 8)
            result += accumulator
            
            # Enerji: 4 multiply + 5 add + router overhead
            total_energy = (e0 + e1 + e2 + e3) + (5 * self.cores[0].power_mw * 0.8)
            total_latency = self.mac_delay_ns  # Pipeline olarak 8 stage
            
        
         elif precision == 4:
            # 4-bit precision scaling (pPIM: A Programmable Processor-in-Memory Architecture 
            # With Precision-Scaling for Deep Learning  Fig. 2 - approximate)
            
            # Sadece en önemli 4 biti kullan
             aH = (a_8bit >> 4) & 0xF
             bH = (b_8bit >> 4) & 0xF
             
             # Tek çarpma + tek toplama yeter
             V3, e3, l3 = self.cores[0].multiply_4bit(aH, bH)
             result = (int(V3) << 8) + accumulator

            
            # Enerji: 1 multiply + 1 add (papPIM: A Programmable Processor-in-Memory Architecture 
            # With Precision-Scaling for Deep Learning per'da 1.35x tasarruf)
             
             total_energy = e3 + (self.cores[0].power_mw * 0.8)
             total_latency = 3.2  # Yarı süre (4 stage yerine 2)
         else:
            raise ValueError("Precision 4 veya 8 olmalı")
        
         return int(result) & 0xFFFF, total_energy, total_latency



# ---------------------------------------------------------------------------
# Katman boyut / iş yükü yardımcıları
# Tüm modüller (scheduler, testler, görselleştirme) MAC sayısını buradan alır,
# böylece stride/padding her yerde aynı şekilde hesaba katılır.
# ---------------------------------------------------------------------------

def conv_output_hw(H, W, Kh, Kw, stride=1, padding=0):
    """Conv çıkış yüksekliği/genişliği: floor((H + 2P - K) / S) + 1"""
    if stride < 1:
        raise ValueError("stride >= 1 olmalı")
    out_H = (H + 2 * padding - Kh) // stride + 1
    out_W = (W + 2 * padding - Kw) // stride + 1
    if out_H <= 0 or out_W <= 0:
        raise ValueError(f"Geçersiz conv boyutu: giriş {H}x{W}, kernel {Kh}x{Kw}")
    return out_H, out_W


def conv_stats(input_shape, kernel_shape, stride=1, padding=0):
    """
    Bir Conv2D katmanının MAC sayısı ve taşınan veri miktarı (eleman cinsinden).

    Returns:
        dict: total_macs, out_shape, input_elems, weight_elems, output_elems
    """
    C_out, C_in_k, Kh, Kw = kernel_shape
    C_in, H, W = input_shape
    if C_in != C_in_k:
        raise ValueError(f"Kanal uyuşmazlığı: input C={C_in}, kernel C_in={C_in_k}")
    out_H, out_W = conv_output_hw(H, W, Kh, Kw, stride, padding)
    total_outputs = C_out * out_H * out_W
    return {
        'total_macs': total_outputs * C_in * Kh * Kw,
        'out_shape': (C_out, out_H, out_W),
        'input_elems': C_in * H * W,
        'weight_elems': C_out * C_in * Kh * Kw,
        'output_elems': total_outputs,
    }


def layer_workload(layer):
    """
    Bir katman sözlüğünden (cnn_models formatı) iş yükü çıkarır.

    Returns:
        (total_macs, data_elems): MAC/işlem sayısı ve bellekten taşınan eleman
        sayısı. Şekil bilgisi olmayan katmanlarda (None, None) döner.
    """
    t = layer.get('type')
    if t in ('Conv2D', 'Conv') and 'input' in layer and 'kernel' in layer:
        st = conv_stats(layer['input'], layer['kernel'],
                        layer.get('stride', 1), layer.get('padding', 0))
        return st['total_macs'], st['input_elems'] + st['weight_elems'] + st['output_elems']
    if t in ('Linear', 'FC') and 'in_features' in layer and 'out_features' in layer:
        i, o = layer['in_features'], layer['out_features']
        return i * o, i * o + i + o
    if 'input' in layer:
        # Eleman bazlı katmanlar (ReLU, Pool, Dropout, BatchNorm): eleman başına 1 işlem
        n = int(np.prod(layer['input']))
        return n, 2 * n
    return None, None


class PIMArray:
    """
    256 cluster'dan oluşan tam PIM chip simülasyonu.
    """
    # DRAM satır boyutu (byte). RowClone/LISA bir satırı bütün olarak kopyalar.
    ROW_SIZE_BYTES = 8192

    def __init__(self, num_clusters=256, technology_node=28):
        self.clusters = [PIMCluster(technology_node) for _ in range(num_clusters)]
        self.num_clusters = num_clusters

        # Memory communication (pPIM: A Programmable Processor-in-Memory Architecture
        # With Precision-Scaling for Deep Learning, Table 1) — satır kopyası başına maliyet
        self.intra_subarray_energy_uj = 0.028  # RowClone
        self.intra_subarray_latency_ns = 63.0

        self.inter_subarray_energy_uj = 0.09  # LISA (1 hop)
        self.inter_subarray_latency_ns = 148.5

    def mac_workload(self, total_macs, data_elems=0, precision=8):
        """
        Genel bir MAC iş yükünün PIM maliyeti.

        Args:
            total_macs: MAC sayısı
            data_elems: Subarray'ler arasında taşınan eleman sayısı
                        (input + weight + output). Her eleman `precision` bit.
            precision: 8 veya 4

        Varsayımlar:
            - MAC'ler cluster'lara eşit dağılır, cluster'lar paralel çalışır.
            - Veri hareketi RowClone ile satır (8 KB) bazında yapılır; maliyet
              MAC başına değil, kopyalanan satır başına ödenir.
            - Satır kopyaları cluster sayısı kadar paralel yürür.
        """
        _, mac_energy_pj, mac_latency_ns = self.clusters[0].mac_8bit(128, 128, 0, precision)

        macs_per_cluster = -(-total_macs // self.num_clusters)  # ceil

        compute_energy_mj = (total_macs * mac_energy_pj) / 1e9  # pJ -> mJ
        compute_latency_ms = (macs_per_cluster * mac_latency_ns) / 1e6  # ns -> ms

        data_bytes = data_elems * precision / 8
        rows = int(-(-data_bytes // self.ROW_SIZE_BYTES))
        comm_energy_mj = (rows * self.intra_subarray_energy_uj) / 1e3  # uJ -> mJ
        comm_latency_ms = (-(-rows // self.num_clusters) * self.intra_subarray_latency_ns) / 1e6

        latency_ms = compute_latency_ms + comm_latency_ms
        energy_mj = compute_energy_mj + comm_energy_mj
        return {
            'precision': precision,
            'total_macs': total_macs,
            'rows_moved': rows,
            'latency_ms': latency_ms,
            'energy_compute_mj': compute_energy_mj,
            'energy_communication_mj': comm_energy_mj,
            'energy_total_mj': energy_mj,
            # mJ / s = mW
            'power_mw': energy_mj / (latency_ms / 1000) if latency_ms > 0 else 0,
        }

    def convolution_layer(self, input_shape, kernel_shape, precision=8, stride=1, padding=0):
        """
        Bir CNN katmanını simüle eder.

        Args:
            input_shape: (C_in, H, W)
            kernel_shape: (C_out, C_in, Kh, Kw)
            precision: 8 veya 4 bit
            stride, padding: Conv parametreleri

        Returns:
            stats: Enerji ve gecikme istatistikleri
        """
        st = conv_stats(input_shape, kernel_shape, stride, padding)
        data_elems = st['input_elems'] + st['weight_elems'] + st['output_elems']
        stats = self.mac_workload(st['total_macs'], data_elems, precision)
        stats['out_shape'] = st['out_shape']
        return stats

    def model_inference(self, layer_configs, precision=8):
        """
        Tüm CNN modelini simüle eder.

        Args:
            layer_configs: (input_shape, kernel_shape) ya da
                           (input_shape, kernel_shape, stride, padding) tuple'ları
            precision: 8 veya 4 bit
        """
        total_energy = 0
        total_latency = 0
        total_macs = 0

        for cfg in layer_configs:
            input_shape, kernel_shape = cfg[0], cfg[1]
            stride = cfg[2] if len(cfg) > 2 else 1
            padding = cfg[3] if len(cfg) > 3 else 0
            stats = self.convolution_layer(input_shape, kernel_shape, precision, stride, padding)
            total_energy += stats['energy_total_mj']
            total_latency += stats['latency_ms']
            total_macs += stats['total_macs']

        return {
            'total_energy_mj': total_energy,
            'total_latency_ms': total_latency,
            'total_macs': total_macs,
            'avg_power_mw': total_energy / (total_latency / 1000) if total_latency > 0 else 0,
            'precision': precision
        }
