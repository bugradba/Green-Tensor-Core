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
    

"""
Q-Learning Entegreli Gelişmiş Hibrit Scheduler

Bu versiyon, adaptive_scheduler.py'deki Q-Learning'i
gerçek hibrit sistemle entegre eder.
"""

from Q_Learning.adaptive_scheduler import AdaptiveQLearningScheduler

class AdvancedHybridScheduler:
    """
    Q-Learning ile güçlendirilmiş hibrit scheduler
    
    Özellikler:
    - Gerçek PIM/GPU simülasyonu kullanır
    - Q-Learning ile optimal kararlar verir
    - Online learning (çalışırken öğrenir)
    - Performans metrikleri toplar
    """
    
    def __init__(self, pim_array, gpu_baseline, use_qlearning=True, train_mode=False):
        """
        Args:
            pim_array: PIMArray instance
            gpu_baseline: GPUBaseline instance
            use_qlearning: True ise Q-Learning kullanır, False ise rule-based
            train_mode: True ise eğitim modu (exploration), False ise inference
        """
        self.pim = pim_array
        self.gpu = gpu_baseline
        self.use_qlearning = use_qlearning
        self.train_mode = train_mode
        
        # Q-Learning scheduler
        if use_qlearning:
            self.q_scheduler = AdaptiveQLearningScheduler(
                learning_rate=0.1,
                discount_factor=0.9,
                epsilon=0.3 if train_mode else 0.05,  # Eğitimde exploration, inference'da yok
                energy_weight=0.7,
                latency_weight=0.3
            )
            
            # Eğitilmiş model varsa yükle
            try:
                self.q_scheduler.load_model('adaptive_scheduler_trained.json')
                print("✅ Q-Learning modeli yüklendi (inference mode)")
            except:
                if not train_mode:
                    print("⚠️  Eğitilmiş model bulunamadı, varsayılan Q-table kullanılıyor")
        
        # Metrikler
        self.decisions = []
        self.total_energy = 0
        self.total_latency = 0
    
    def schedule_layer(self, layer_config, deadline_ms=None):
        """
        Tek bir katmanı schedule et (Q-Learning veya rule-based)
        
        Args:
            layer_config: {
                'name': 'conv1',
                'type': 'Conv2D',
                'input': (3, 227, 227),
                'kernel': (96, 3, 11, 11)
            }
            deadline_ms: Gecikme kısıtı
        
        Returns:
            decision: {
                'device': 'PIM' veya 'GPU',
                'energy': float (mJ),
                'latency': float (ms),
                'reason': str
            }
        """
        layer_type = layer_config['type']
        
        # Workload size tahmin et
        if 'input' in layer_config and 'kernel' in layer_config:
            # CNN layer
            input_shape = layer_config['input']
            kernel_shape = layer_config['kernel']
            
            C_in = input_shape[0] if len(input_shape) == 3 else input_shape[1]
            H, W = input_shape[-2:]
            C_out = kernel_shape[0]
            Kh, Kw = kernel_shape[-2:]
            
            workload_size = C_out * C_in * H * W * Kh * Kw
        else:
            # Basit tahmin
            workload_size = 1000000  # Default
        
        # Karar ver
        if self.use_qlearning:
            # Q-Learning ile karar
            device, confidence = self.q_scheduler.predict(
                workload_size, 
                layer_type, 
                deadline_ms
            )
            reason = f"Q-Learning (confidence: {confidence:.2f})"
        else:
            # Rule-based karar
            device, reason = self._rule_based_decision(layer_type, workload_size, deadline_ms)
        
        # Gerçek simülasyon ile enerji/latency hesapla
        energy, latency = self._simulate_layer(layer_config, device)
        
        # Online learning (eğitim modundaysa)
        if self.use_qlearning and self.train_mode:
            # Gerçek sonuçlarla Q-table güncelle
            self._update_qlearning(workload_size, layer_type, deadline_ms, 
                                  device, energy, latency)
        
        # Kaydet
        decision = {
            'layer': layer_config['name'],
            'type': layer_type,
            'device': device,
            'energy': energy,
            'latency': latency,
            'reason': reason
        }
        self.decisions.append(decision)
        self.total_energy += energy
        self.total_latency += latency
        
        return decision
    
    def _rule_based_decision(self, layer_type, workload_size, deadline_ms):
        """Statik kurallarla karar (baseline)"""
        if layer_type == 'ReLU' or layer_type == 'Pool':
            return 'PIM', 'Simple LUT Operation'
        
        if deadline_ms and deadline_ms < 10:
            return 'GPU', 'Strict Deadline Requirement'
        
        if workload_size < 100000:
            return 'PIM', 'Small Workload (Energy Priority)'
        elif workload_size > 10000000:
            return 'GPU', 'Large Workload (Speed Priority)'
        else:
            return 'PIM', 'Medium Workload (Memory-intensive)'
    
    def _simulate_layer(self, layer_config, device):
        """Gerçek PIM/GPU simülasyonu ile enerji/latency hesapla"""
        layer_type = layer_config['type']
        
        if device == 'PIM':
            if layer_type in ['Conv2D', 'Conv']:
                # PIM CNN simülasyonu
                if 'input' in layer_config and 'kernel' in layer_config:
                    stats = self.pim.convolution_layer(
                        layer_config['input'],
                        layer_config['kernel'],
                        precision=8
                    )
                    return stats['energy_total_mj'], stats['latency_ms']
                else:
                    # Basit tahmin
                    return 10.0, 5.0
            
            elif layer_type in ['ReLU', 'Pool']:
                # Çok düşük maliyet
                return 0.01, 0.01
            
            elif layer_type in ['Linear', 'FC']:
                # PIM'de FC daha pahalı
                in_f = layer_config.get('in_features', 1000)
                out_f = layer_config.get('out_features', 1000)
                ops = in_f * out_f
                return ops * 0.00001, ops * 0.000005
            
            else:
                return 5.0, 2.0
        
        else:  # GPU
            if layer_type in ['Conv2D', 'Conv']:
                # GPU CNN
                if 'input' in layer_config and 'kernel' in layer_config:
                    input_shape = layer_config['input']
                    kernel_shape = layer_config['kernel']
                    
                    C_in = input_shape[0]
                    H, W = input_shape[-2:]
                    C_out = kernel_shape[0]
                    Kh, Kw = kernel_shape[-2:]
                    
                    macs = C_out * C_in * H * W * Kh * Kw
                    
                    stats = self.gpu.model_inference(macs)
                    return stats['total_energy_mj'], stats['total_latency_ms']
                else:
                    return 15.0, 1.0
            
            elif layer_type in ['Linear', 'FC']:
                # GPU'da FC hızlı
                in_f = layer_config.get('in_features', 1000)
                out_f = layer_config.get('out_features', 1000)
                ops = in_f * out_f
                
                stats = self.gpu.model_inference(ops)
                return stats['total_energy_mj'], stats['total_latency_ms']
            
            else:
                return 5.0, 0.5
    
    def _update_qlearning(self, workload_size, layer_type, deadline_ms, 
                         chosen_device, actual_energy, actual_latency):
        """Online learning: Gerçek sonuçlarla Q-table güncelle"""
        
        # Alternatif cihazı da simüle et (karşılaştırma için)
        alt_device = 'GPU' if chosen_device == 'PIM' else 'PIM'
        
        # Basit tahmin (tam simülasyon pahalı)
        if alt_device == 'GPU':
            alt_energy = actual_energy * 1.5
            alt_latency = actual_latency * 0.2
        else:
            alt_energy = actual_energy * 0.6
            alt_latency = actual_latency * 5.0
        
        # State ve next_state
        state = self.q_scheduler._discretize_state(workload_size, layer_type, deadline_ms)
        next_state = state  # Basit: aynı state
        
        # Reward hesapla
        reward = self.q_scheduler.calculate_reward(
            alt_energy if alt_device == 'PIM' else actual_energy,  # PIM energy
            alt_energy if alt_device == 'GPU' else actual_energy,  # GPU energy
            alt_latency if alt_device == 'PIM' else actual_latency,  # PIM latency
            alt_latency if alt_device == 'GPU' else actual_latency,  # GPU latency
            chosen_device,
            deadline_ms
        )
        
        # Q-table güncelle
        self.q_scheduler.update_q_table(state, chosen_device, reward, next_state)
    
    def schedule_model(self, model_layers, deadlines=None):
        """
        Tam bir modeli schedule et
        
        Args:
            model_layers: Liste of layer configs
            deadlines: Her katman için deadline (opsiyonel)
        
        Returns:
            plan: Tüm kararlar listesi
            total_energy: Toplam enerji (mJ)
            total_latency: Toplam gecikme (ms)
        """
        self.decisions = []
        self.total_energy = 0
        self.total_latency = 0
        
        for i, layer in enumerate(model_layers):
            deadline = deadlines[i] if deadlines else None
            decision = self.schedule_layer(layer, deadline)
        
        return self.decisions, self.total_energy, self.total_latency
    
    def get_statistics(self):
        """Performans istatistikleri"""
        pim_count = sum(1 for d in self.decisions if d['device'] == 'PIM')
        gpu_count = sum(1 for d in self.decisions if d['device'] == 'GPU')
        
        pim_energy = sum(d['energy'] for d in self.decisions if d['device'] == 'PIM')
        gpu_energy = sum(d['energy'] for d in self.decisions if d['device'] == 'GPU')
        
        return {
            'total_layers': len(self.decisions),
            'pim_layers': pim_count,
            'gpu_layers': gpu_count,
            'total_energy': self.total_energy,
            'total_latency': self.total_latency,
            'pim_energy': pim_energy,
            'gpu_energy': gpu_energy,
            'avg_energy_per_layer': self.total_energy / max(1, len(self.decisions))
        }
    
    def save_model(self, filepath='hybrid_qlearning_trained.json'):
        """Q-Learning modelini kaydet"""
        if self.use_qlearning:
            self.q_scheduler.save_model(filepath)
    
    def compare_strategies(self, model_layers):
        """
        Q-Learning vs Rule-based karşılaştırması
        
        Returns:
            comparison: {
                'qlearning': {...},
                'rulebased': {...}
            }
        """
        # Q-Learning ile
        self.use_qlearning = True
        plan_q, energy_q, latency_q = self.schedule_model(model_layers)
        stats_q = self.get_statistics()
        
        # Rule-based ile
        self.use_qlearning = False
        plan_r, energy_r, latency_r = self.schedule_model(model_layers)
        stats_r = self.get_statistics()
        
        # Geri al
        self.use_qlearning = True
        
        return {
            'qlearning': {
                'energy': energy_q,
                'latency': latency_q,
                'stats': stats_q,
                'plan': plan_q
            },
            'rulebased': {
                'energy': energy_r,
                'latency': latency_r,
                'stats': stats_r,
                'plan': plan_r
            },
            'improvement': {
                'energy_saving': (energy_r - energy_q) / energy_r * 100,
                'latency_diff': (latency_q - latency_r) / latency_r * 100
            }
        }