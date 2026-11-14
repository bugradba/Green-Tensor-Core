import os

from pim_simulator import layer_workload
from Q_Learning.adaptive_scheduler import AdaptiveQLearningScheduler, normalize_layer_type

_HERE = os.path.dirname(os.path.abspath(__file__))

# HYBRID aksiyonunda iş yükünün PIM'e giden payı (geri kalanı GPU'ya).
HYBRID_PIM_SHARE = 0.7

# Şekil bilgisi olmayan katmanlar (örn. ReLU'ya input verilmemişse) için
# küçük sabit maliyetler (mJ, ms).
_FALLBACK_COST = {
    'PIM': (0.01, 0.01),
    'GPU': (0.05, 0.005),
}


def simulate_layer(layer, device, pim, gpu, precision=8):
    """
    Bir katmanı verilen cihazda simüle eder. Tüm scheduler'lar ve demolar
    maliyeti buradan alır, böylece aynı katman her yerde aynı maliyete sahiptir.

    Args:
        layer: cnn_models formatında katman sözlüğü
        device: 'PIM', 'GPU' veya 'HYBRID'
        pim: PIMArray
        gpu: GPUBaseline / CPUBaseline

    Returns:
        (energy_mj, latency_ms)
    """
    macs, data_elems = layer_workload(layer)

    if macs is None:
        if device == 'HYBRID':
            pe, pl = _FALLBACK_COST['PIM']
            ge, gl = _FALLBACK_COST['GPU']
            s = HYBRID_PIM_SHARE
            return s * pe + (1 - s) * ge, max(s * pl, (1 - s) * gl)
        return _FALLBACK_COST[device]

    if device == 'PIM':
        st = pim.mac_workload(macs, data_elems, precision)
        return st['energy_total_mj'], st['latency_ms']

    if device == 'GPU':
        st = gpu.model_inference(macs, data_elems * 8)
        return st['total_energy_mj'], st['total_latency_ms']

    if device == 'HYBRID':
        # İş yükü bölünür, iki cihaz paralel çalışır.
        s = HYBRID_PIM_SHARE
        p_macs = int(macs * s)
        g_macs = macs - p_macs
        p = pim.mac_workload(p_macs, int(data_elems * s), precision)
        g = gpu.model_inference(g_macs, int(data_elems * (1 - s)) * 8)
        return (p['energy_total_mj'] + g['total_energy_mj'],
                max(p['latency_ms'], g['total_latency_ms']))

    raise ValueError(f"Bilinmeyen cihaz: {device}")


class HybridSystem:
    """
    PIM ve GPU arasında hem iş yükü tabanlı hem de katman tabanlı (Layer-wise)
    dağılım yapan Gelişmiş Hibrit Zamanlayıcı.
    """
    def __init__(self, pim_array, gpu_baseline):
        self.pim = pim_array
        self.gpu = gpu_baseline

        # Basit iş yükü eşiği
        self.workload_threshold = 50000

        # Veri Transfer Maliyeti (PIM <-> GPU)
        # PCIe üzerinden veri aktarımı maliyetlidir.
        # Varsayım: 0.05 mJ/MB enerji ve 1.0 ms/MB gecikme
        self.transfer_energy_per_mb = 0.05
        self.transfer_latency_per_mb = 1.0

    # --- 1. BASİT İŞ YÜKÜ MANTIĞI ---
    def _costs(self, total_macs):
        p = self.pim.mac_workload(total_macs, max(1, total_macs // 64))
        g = self.gpu.model_inference(total_macs)
        return p['energy_total_mj'], p['latency_ms'], g['total_energy_mj'], g['total_latency_ms']

    def adaptive_processing(self, total_macs, model=None):
        """Basit iş yükü boyutuna göre karar verir."""
        pe, pl, ge, gl = self._costs(total_macs)
        if total_macs < self.workload_threshold:
            return pe, pl, "PIM"
        return ge, gl, "GPU"

    def benchmark_comparison(self, workloads=(1000, 50000, 10000000)):
        """Basit benchmark raporu."""
        results = []
        for w in workloads:
            pe, pl, ge, gl = self._costs(w)
            he, hl, d = self.adaptive_processing(w)
            results.append({
                'workload': w, 'pim_energy': pe, 'pim_latency': pl,
                'gpu_energy': ge, 'gpu_latency': gl,
                'hybrid_energy': he, 'hybrid_latency': hl, 'decision': d
            })
        return results

    # --- 2. KATMAN BAZLI (LAYER-WISE) MANTIK ---

    @staticmethod
    def _layer_rule(layer_type):
        t = normalize_layer_type(layer_type)
        if t == 'Conv':
            return 'PIM', "Memory-intensive MAC operations"
        if t == 'FC':
            return 'GPU', "Large Matrix Multiplication"
        if t in ('ReLU', 'Sigmoid', 'Pool'):
            return 'PIM', "Simple LUT Operation"
        return 'GPU', "Default"

    def analyze_model_layers(self, model_layers, input_data_mb):
        """
        Modelin katmanlarını analiz eder ve her biri için en uygun cihazı seçer.
        Veri transfer maliyetlerini de hesaba katar.
        """
        execution_plan = []
        current_device = 'PIM'  # Veri başlangıçta PIM'de (Sensör/Memory) varsayılır

        total_energy = 0
        total_latency = 0

        print(f"\n🔍 Model Analizi Başlıyor ({len(model_layers)} katman)...")

        for layer in model_layers:
            decision, reason = self._layer_rule(layer['type'])

            transfer_cost_e = 0
            transfer_cost_l = 0
            if decision != current_device:
                transfer_cost_e = input_data_mb * self.transfer_energy_per_mb
                transfer_cost_l = input_data_mb * self.transfer_latency_per_mb
                reason += f" + Data Transfer ({current_device}->{decision})"
                current_device = decision

            layer_energy, layer_latency = simulate_layer(layer, decision, self.pim, self.gpu)

            total_energy += layer_energy + transfer_cost_e
            total_latency += layer_latency + transfer_cost_l

            execution_plan.append({
                'layer': layer['name'],
                'type': layer['type'],
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


class AdvancedHybridScheduler:
    """
    Q-Learning ile güçlendirilmiş hibrit scheduler

    Özellikler:
    - Gerçek PIM/GPU simülasyonu kullanır
    - Q-Learning ile optimal kararlar verir
    - Online learning (çalışırken öğrenir)
    - Performans metrikleri toplar
    """

    def __init__(self, pim_array, gpu_baseline, use_qlearning=True, train_mode=False,
                 model_path=None):
        """
        Args:
            pim_array: PIMArray instance
            gpu_baseline: GPUBaseline instance
            use_qlearning: True ise Q-Learning kullanır, False ise rule-based
            train_mode: True ise eğitim modu (exploration), False ise inference
            model_path: Eğitilmiş Q-table yolu (varsayılan: bu klasördeki v2 modeli)
        """
        self.pim = pim_array
        self.gpu = gpu_baseline
        self.use_qlearning = use_qlearning
        self.train_mode = train_mode

        if use_qlearning:
            self.q_scheduler = AdaptiveQLearningScheduler(
                learning_rate=0.1,
                discount_factor=0.9,
                epsilon=0.3 if train_mode else 0.05,
                energy_weight=0.7,
                latency_weight=0.3
            )

            path = model_path or os.path.join(_HERE, 'adaptive_scheduler_trained_v2.json')
            try:
                self.q_scheduler.load_model(path)
                print("✅ Q-Learning modeli yüklendi")
            except FileNotFoundError:
                if not train_mode:
                    print("⚠️  Eğitilmiş model bulunamadı, varsayılan Q-table kullanılıyor")

        self.decisions = []
        self.total_energy = 0
        self.total_latency = 0

    @staticmethod
    def _workload_size(layer_config):
        macs, _ = layer_workload(layer_config)
        return macs if macs is not None else 1000000  # Şekil yoksa varsayılan

    def schedule_layer(self, layer_config, deadline_ms=None):
        """
        Tek bir katmanı schedule et (Q-Learning veya rule-based)

        Returns:
            decision: {'layer', 'type', 'device', 'energy' (mJ), 'latency' (ms), 'reason'}
        """
        layer_type = layer_config['type']
        workload_size = self._workload_size(layer_config)

        if self.use_qlearning:
            device, confidence = self.q_scheduler.predict(workload_size, layer_type, deadline_ms)
            reason = f"Q-Learning (Q: {confidence:.2f})"
        else:
            device, reason = self._rule_based_decision(layer_type, workload_size, deadline_ms)

        energy, latency = simulate_layer(layer_config, device, self.pim, self.gpu)

        if self.use_qlearning and self.train_mode:
            self._update_qlearning(layer_config, workload_size, layer_type, deadline_ms, device)

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
        if normalize_layer_type(layer_type) in ('ReLU', 'Pool'):
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
        """Geriye dönük uyumluluk için: simulate_layer'a yönlendirir."""
        return simulate_layer(layer_config, device, self.pim, self.gpu)

    def _update_qlearning(self, layer_config, workload_size, layer_type, deadline_ms,
                          chosen_device):
        """Online learning: PIM ve GPU gerçekten simüle edilerek Q-table güncellenir."""
        energy_pim, latency_pim = simulate_layer(layer_config, 'PIM', self.pim, self.gpu)
        energy_gpu, latency_gpu = simulate_layer(layer_config, 'GPU', self.pim, self.gpu)

        state = self.q_scheduler._discretize_state(workload_size, layer_type, deadline_ms)
        next_state = state  # Basit: aynı state

        reward = self.q_scheduler.calculate_reward(
            energy_pim, energy_gpu, latency_pim, latency_gpu, chosen_device, deadline_ms)

        self.q_scheduler.update_q_table(state, chosen_device, reward, next_state)

    def schedule_model(self, model_layers, deadlines=None):
        """
        Tam bir modeli schedule et

        Returns:
            plan, total_energy (mJ), total_latency (ms)
        """
        self.decisions = []
        self.total_energy = 0
        self.total_latency = 0

        for i, layer in enumerate(model_layers):
            deadline = deadlines[i] if deadlines else None
            self.schedule_layer(layer, deadline)

        return self.decisions, self.total_energy, self.total_latency

    def get_statistics(self):
        """Performans istatistikleri"""
        pim_count = sum(1 for d in self.decisions if d['device'] == 'PIM')
        gpu_count = sum(1 for d in self.decisions if d['device'] == 'GPU')
        hybrid_count = sum(1 for d in self.decisions if d['device'] == 'HYBRID')

        pim_energy = sum(d['energy'] for d in self.decisions if d['device'] == 'PIM')
        gpu_energy = sum(d['energy'] for d in self.decisions if d['device'] == 'GPU')

        return {
            'total_layers': len(self.decisions),
            'pim_layers': pim_count,
            'gpu_layers': gpu_count,
            'hybrid_layers': hybrid_count,
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
        """Q-Learning vs Rule-based karşılaştırması"""
        if not hasattr(self, 'q_scheduler'):
            raise RuntimeError("compare_strategies için use_qlearning=True ile başlatın")

        self.use_qlearning = True
        plan_q, energy_q, latency_q = self.schedule_model(model_layers)
        stats_q = self.get_statistics()

        self.use_qlearning = False
        plan_r, energy_r, latency_r = self.schedule_model(model_layers)
        stats_r = self.get_statistics()

        self.use_qlearning = True

        return {
            'qlearning': {'energy': energy_q, 'latency': latency_q, 'stats': stats_q, 'plan': plan_q},
            'rulebased': {'energy': energy_r, 'latency': latency_r, 'stats': stats_r, 'plan': plan_r},
            'improvement': {
                'energy_saving': (energy_r - energy_q) / energy_r * 100,
                'latency_diff': (latency_q - latency_r) / latency_r * 100
            }
        }
