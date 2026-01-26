import numpy as np

class HybridSystem:
    """
    PIM ve GPU arasında hem iş yükü tabanlı hem de katman tabanlı (Layer-wise)
    dağılım yapan Gelişmiş Hibrit Zamanlayıcı.
    """
    def __init__(self, pim_array, gpu_baseline):
        self.pim = pim_array
        self.gpu = gpu_baseline
        
        # Basit iş yükü eşiği (Eski testlerin çalışması için)
        self.workload_threshold = 50000 

        # Veri Transfer Maliyeti (PIM <-> GPU)
        # PCIe üzerinden veri aktarımı maliyetlidir.
        # Varsayım: 0.05 mJ/MB enerji ve 1.0 ms/MB gecikme
        self.transfer_energy_per_mb = 0.05 
        self.transfer_latency_per_mb = 1.0 

    # --- 1. ESKİ TESTLER İÇİN BASİT MANTIK ---
    def adaptive_processing(self, total_macs, model=None):
        """Basit iş yükü boyutuna göre karar verir."""
        # GPU Tahmini
        gpu_stats = self.gpu.model_inference(total_macs)
        
        # PIM Tahmini (Basit yaklaşım)
        _, e_pj, l_ns = self.pim.clusters[0].mac_8bit(128, 128, precision=8)
        pim_energy = (total_macs * e_pj) / 1e9
        pim_latency = (total_macs / 256 * l_ns) / 1e6 # 256 cluster paralel

        if total_macs < self.workload_threshold:
            return pim_energy, pim_latency, "PIM"
        else:
            return gpu_stats['total_energy_mj'], gpu_stats['total_latency_ms'], "GPU"

    def benchmark_comparison(self):
        """Basit benchmark raporu."""
        workloads = [1000, 50000, 10000000]
        results = []
        for w in workloads:
            pe, pl, _ = self.adaptive_processing(w) # PIM varsayımı
            gres = self.gpu.model_inference(w)
            ge, gl = gres['total_energy_mj'], gres['total_latency_ms']
            
            # Karar
            if w < self.workload_threshold:
                he, hl, d = pe, pl, "PIM"
            else:
                he, hl, d = ge, gl, "GPU"
                
            results.append({
                'workload': w, 'pim_energy': pe, 'pim_latency': pl,
                'gpu_energy': ge, 'gpu_latency': gl,
                'hybrid_energy': he, 'hybrid_latency': hl, 'decision': d
            })
        return results

    # --- 2. YENİ KATMAN BAZLI (LAYER-WISE) MANTIK ---
    
    def analyze_model_layers(self, model_layers, input_data_mb):
        """
        Modelin katmanlarını analiz eder ve her biri için en uygun cihazı seçer.
        Veri transfer maliyetlerini de hesaba katar.
        """
        execution_plan = []
        current_device = 'PIM' # Veri başlangıçta PIM'de (Sensör/Memory) varsayalım
        
        total_energy = 0
        total_latency = 0

        print(f"\n🔍 Model Analizi Başlıyor ({len(model_layers)} katman)...")

        for layer in model_layers:
            layer_name = layer['name']
            layer_type = layer['type']
            
            # Karar Mantığı
            decision = "GPU" # Varsayılan
            reason = ""

            # 1. Convolution Katmanları -> PIM (Memory Bound)
            if layer_type == 'Conv2D':
                decision = 'PIM'
                reason = "Memory-intensive MAC operations"
            
            # 2. Linear (Fully Connected) -> GPU (Compute Bound)
            elif layer_type == 'Linear':
                decision = 'GPU'
                reason = "Large Matrix Multiplication"
            
            # 3. Aktivasyonlar -> PIM (LUT Friendly)
            elif layer_type in ['ReLU', 'Sigmoid']:
                decision = 'PIM'
                reason = "Simple LUT Operation"
            
            # --- Maliyet Hesabı ---
            
            # Eğer cihaz değişirse Transfer Maliyeti ekle
            transfer_cost_e = 0
            transfer_cost_l = 0
            
            if decision != current_device:
                transfer_cost_e = input_data_mb * self.transfer_energy_per_mb
                transfer_cost_l = input_data_mb * self.transfer_latency_per_mb
                reason += f" + Data Transfer ({current_device}->{decision})"
                current_device = decision # Cihaz değişti
            
            # İşlem Maliyeti (Simülasyon)
            if decision == 'PIM':
                # PIM Maliyeti (Conv layer fonksiyonunu kullanarak)
                if layer_type == 'Conv2D':
                    stats = self.pim.convolution_layer(layer['input'], layer['kernel'], precision=8)
                    layer_energy = stats['energy_total_mj']
                    layer_latency = stats['latency_ms']
                else:
                    # Basit işlemler için çok düşük maliyet
                    layer_energy = 0.01
                    layer_latency = 0.005
            else:
                # GPU Maliyeti
                # İşlem sayısını tahmin et (Conv ise kernel, Linear ise matrix boyutu)
                if layer_type == 'Conv2D':
                    macs = layer['kernel'][0] * layer['kernel'][1] * layer['kernel'][2] * layer['kernel'][3] * layer['input'][1] * layer['input'][2]
                elif layer_type == 'Linear':
                     macs = layer['in_features'] * layer['out_features']
                else:
                    macs = 1000 # Basit işlem
                
                stats = self.gpu.model_inference(macs)
                layer_energy = stats['total_energy_mj']
                layer_latency = stats['total_latency_ms']

            # Toplamları Güncelle
            total_energy += layer_energy + transfer_cost_e
            total_latency += layer_latency + transfer_cost_l

            execution_plan.append({
                'layer': layer_name,
                'type': layer_type,
                'device': decision,
                'energy': layer_energy + transfer_cost_e,
                'latency': layer_latency + transfer_cost_l,
                'reason': reason
            })

        return execution_plan, total_energy, total_latency