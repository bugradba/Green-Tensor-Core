"""
Q-Learning Based Adaptive Scheduler for PIM/GPU Hybrid System

Bu modül, iş yükü özelliklerine göre PIM veya GPU seçimini öğrenir.
Geçmiş deneyimlerden reward alarak optimal kararlar verir.

State: (workload_size_category, layer_type, deadline_constraint)
Action: {PIM, GPU, HYBRID}
Reward: energy_saved - latency_penalty
"""

import ast
import numpy as np
import json
from collections import defaultdict

# Modellerde (cnn_models) ve eğitim verisinde farklı adlar kullanılıyor.
# Hepsi tek bir kanonik ada indirgenir; aksi halde çıkarımda durum Q-table'da
# hiç bulunmaz ve ajan her zaman varsayılan aksiyonu (PIM) seçer.
LAYER_TYPE_ALIASES = {
    'Conv2D': 'Conv', 'Conv': 'Conv',
    'Linear': 'FC', 'FC': 'FC',
    'MaxPool': 'Pool', 'AvgPool': 'Pool', 'Pool': 'Pool',
}


def normalize_layer_type(layer_type):
    return LAYER_TYPE_ALIASES.get(layer_type, layer_type)


def simulate_costs(pim_simulator, gpu_simulator, workload_size, data_elems=None):
    """
    Bir iş yükünün PIM ve GPU maliyetini gerçek simülatörlerle hesaplar.

    data_elems verilmezse, GPU baz modelindeki varsayımla tutarlı olarak
    her 64 MAC'te bir elemanın bellekten taşındığı varsayılır.

    Returns:
        energy_pim, energy_gpu (mJ), latency_pim, latency_gpu (ms)
    """
    if pim_simulator is None or gpu_simulator is None:
        raise ValueError(
            "train_episode gerçek simülatör ister: "
            "train_episode(PIMArray(), GPUBaseline(), workloads)"
        )
    if data_elems is None:
        data_elems = max(1, workload_size // 64)
    pim = pim_simulator.mac_workload(workload_size, data_elems, precision=8)
    gpu = gpu_simulator.model_inference(workload_size, data_elems * 8)
    return (pim['energy_total_mj'], gpu['total_energy_mj'],
            pim['latency_ms'], gpu['total_latency_ms'])

class AdaptiveQLearningScheduler:
    """
    Q-Learning tabanlı akıllı scheduler.

    Öğrenme süreci:
    1. Başlangıçta rastgele kararlar verir (exploration)
    2. Her karar sonrası reward alır (enerji tasarrufu - gecikme cezası)
    3. Q-table güncellenir
    4. Zaman içinde optimal stratejiyi öğrenir (exploitation)
    """

    def __init__(self,
                 learning_rate=0.1,
                 discount_factor=0.9,
                 epsilon=0.3,
                 energy_weight=0.7,
                 latency_weight=0.3):
        """
        Args:
            learning_rate (alpha): Q-value güncelleme hızı [0-1]
            discount_factor (gamma): Gelecek reward'ların önemi [0-1]
            epsilon: Exploration oranı (rastgele aksiyon alma) [0-1]
            energy_weight: Enerji tasarrufunun reward'daki ağırlığı
            latency_weight: Gecikme cezasının reward'daki ağırlığı
        """
        # Hyperparameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon  # Exploration rate
        
        self.energy_weight = energy_weight
        self.latency_weight = latency_weight
        
        # Q-Table: Q(state, action) -> expected reward
        # State: (workload_category, layer_type, deadline_category)
        # Action: 0=PIM, 1=GPU, 2=HYBRID
        self.q_table = defaultdict(lambda: np.zeros(3))
        
        # Action mapping
        self.actions = ['PIM', 'GPU', 'HYBRID']
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        
        # Learning statistics
        self.episode_count = 0
        self.total_reward = 0
        self.action_history = []
        self.reward_history = []

    def _discretize_state(self, workload_size, layer_type, deadline_ms=None):
        """
        Continuous state'i discrete kategorilere dönüştür.
        
        Args:
            workload_size: İşlem sayısı
            layer_type: 'Conv', 'FC', 'ReLU', vs.
            deadline_ms: Gecikme kısıtı (ms)
        
        Returns:
            state: (workload_cat, layer_type, deadline_cat) tuple
        """
        # Workload kategorisi
        if workload_size < 100000:
            workload_cat = 'small'
        elif workload_size < 10000000:
            workload_cat = 'medium'
        else:
            workload_cat = 'large'
        
        # Layer type (kanonik ada indir)
        layer_cat = normalize_layer_type(layer_type)
        
        # Deadline kategorisi
        if deadline_ms is None:
            deadline_cat = 'none'
        elif deadline_ms < 10:
            deadline_cat = 'strict'  # <10ms (real-time)
        elif deadline_ms < 50:
            deadline_cat = 'moderate'  # 10-50ms
        else:
            deadline_cat = 'relaxed'  # >50ms
        
        return (workload_cat, layer_cat, deadline_cat)
    
    def select_action(self, state, training=True):
        """
        Epsilon-greedy policy ile action seç.
        
        Args:
            state: Mevcut durum
            training: True ise exploration yapar
        
        Returns:
            action: 'PIM', 'GPU', veya 'HYBRID'
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: Rastgele action seç
            action_idx = np.random.choice(3)
        else:
            # Exploitation: En yüksek Q-value'lu action seç
            q_values = self.q_table[state]
            action_idx = np.argmax(q_values)
        
        return self.actions[action_idx]
    
    def calculate_reward(self, energy_pim, energy_gpu, 
                        latency_pim, latency_gpu, 
                        chosen_action, deadline_ms=None):
        """
        Reward fonksiyonu: enerji tasarrufu - gecikme cezası
        
        Reward yapısı:
        - Enerji tasarrufu: pozitif reward
        - Deadline ihlali: büyük negatif reward
        - Gereksiz yavaşlık: küçük negatif reward
        
        Args:
            energy_pim, energy_gpu: PIM ve GPU enerjisi (mJ)
            latency_pim, latency_gpu: PIM ve GPU gecikmesi (ms)
            chosen_action: Seçilen aksiyon
            deadline_ms: Gecikme kısıtı
        
        Returns:
            reward: Toplam reward değeri
        """
        # Seçilen aksiyonun maliyetleri
        if chosen_action == 'PIM':
            actual_energy = energy_pim
            actual_latency = latency_pim
        elif chosen_action == 'GPU':
            actual_energy = energy_gpu
            actual_latency = latency_gpu
        else:  # HYBRID
            actual_energy = (energy_pim * 0.7 + energy_gpu * 0.3)
            actual_latency = max(latency_pim * 0.7, latency_gpu * 0.3)
        
        # Enerji tasarrufu (GPU baseline'a göre)
        energy_saving = (energy_gpu - actual_energy) / energy_gpu
        energy_reward = energy_saving * self.energy_weight * 100
        
        # Gecikme cezası
        latency_penalty = 0
        
        if deadline_ms is not None:
            if actual_latency > deadline_ms:
                # Deadline ihlali: büyük ceza
                violation_ratio = (actual_latency - deadline_ms) / deadline_ms
                latency_penalty = violation_ratio * 200  # Büyük ceza
            else:
                # Deadline içinde ama ne kadar hızlı?
                # GPU'dan ne kadar yavaş?
                slowdown = (actual_latency - latency_gpu) / latency_gpu
                latency_penalty = slowdown * self.latency_weight * 50
        else:
            # Deadline yok, sadece göreceli yavaşlığa bak
            slowdown = (actual_latency - latency_gpu) / latency_gpu
            latency_penalty = slowdown * self.latency_weight * 30
        
        # Toplam reward
        reward = energy_reward - latency_penalty
        
        return reward
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Q-Learning güncelleme kuralı:
        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Önceki durum
            action: Yapılan aksiyon
            reward: Alınan reward
            next_state: Yeni durum
        """
        action_idx = self.action_to_idx[action]
        
        # Current Q-value
        current_q = self.q_table[state][action_idx]
        
        # Max Q-value for next state
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[state][action_idx] = new_q
        
        # Statistics
        self.reward_history.append(reward)
        self.total_reward += reward
    
    # ⬇ ÖNEMLİ: Bu fonksiyonlar update_q_table ile AYNI SEVİYEDE olmalı!
    def train_episode(self, pim_simulator, gpu_simulator, workload_data):
        """
        Bir eğitim episode'u çalıştır.
        
        Args:
            pim_simulator: PIMArray instance
            gpu_simulator: GPUBaseline instance
            workload_data: [(workload_size, layer_type, deadline), ...]
                           veya (workload_size, layer_type, deadline, data_elems)
        
        Returns:
            episode_reward: Episode toplam reward'u
        """
        episode_reward = 0
        
        for item in workload_data:
            workload_size, layer_type, deadline = item[:3]
            data_elems = item[3] if len(item) > 3 else None
            # State'i discretize et
            state = self._discretize_state(workload_size, layer_type, deadline)
            
            # Action seç (epsilon-greedy)
            action = self.select_action(state, training=True)
            
            # Gerçek PIM/GPU simülasyonu
            energy_pim, energy_gpu, latency_pim, latency_gpu = simulate_costs(
                pim_simulator, gpu_simulator, workload_size, data_elems)
            
            # Reward hesapla
            reward = self.calculate_reward(
                energy_pim, energy_gpu,
                latency_pim, latency_gpu,
                action, deadline
            )
            
            # Next state (basit: aynı state)
            next_state = state
            
            # Q-table güncelle
            self.update_q_table(state, action, reward, next_state)
            
            episode_reward += reward
            self.action_history.append((state, action))
        
        self.episode_count += 1
        
        # Epsilon decay (zamanla exploration azalt)
        self.epsilon = max(0.05, self.epsilon * 0.995)
        
        return episode_reward
    
    def predict(self, workload_size, layer_type, deadline_ms=None):
        """
        Öğrenilmiş policy ile en iyi aksiyonu seç (inference mode).
        
        Args:
            workload_size: İşlem sayısı
            layer_type: Katman tipi
            deadline_ms: Gecikme kısıtı
        
        Returns:
            action: 'PIM', 'GPU', veya 'HYBRID'
            confidence: Q-value (ne kadar emin?)
        """
        state = self._discretize_state(workload_size, layer_type, deadline_ms)
        q_values = self.q_table[state]
        
        best_action_idx = np.argmax(q_values)
        best_action = self.actions[best_action_idx]
        confidence = q_values[best_action_idx]
        
        return best_action, confidence
    
    def get_statistics(self):
        """Öğrenme istatistikleri"""
        return {
            'episodes': self.episode_count,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(1, self.episode_count),
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'recent_rewards': self.reward_history[-10:] if self.reward_history else []
        }
    
    def save_model(self, filepath='q_learning_model.json'):
        """Q-table'ı kaydet"""
        model_data = {
            'q_table': {str(k): v.tolist() for k, v in self.q_table.items()},
            'hyperparameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'energy_weight': self.energy_weight,
                'latency_weight': self.latency_weight
            },
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✅Model kaydedildi: {filepath}")
    
    def load_model(self, filepath='q_learning_model.json'):
        """Q-table'ı yükle"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Q-table'ı yükle
        self.q_table = defaultdict(lambda: np.zeros(3))
        for state_str, q_values in model_data['q_table'].items():
            state = ast.literal_eval(state_str)  # Güvenli: sadece literal tuple
            self.q_table[state] = np.array(q_values)
        
        # Hyperparameters
        hyper = model_data['hyperparameters']
        self.alpha = hyper['alpha']
        self.gamma = hyper['gamma']
        self.epsilon = hyper['epsilon']
        self.energy_weight = hyper['energy_weight']
        self.latency_weight = hyper['latency_weight']
        
        print(f" Model yüklendi: {filepath}")
        print(f"   Q-table boyutu: {len(self.q_table)} state")
        print(f"   Epsilon: {self.epsilon:.3f}")


# Kullanım örneği (standalone test)
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pim_simulator import PIMArray
    from baseline_models import GPUBaseline

    print(" Q-Learning Scheduler Demo")
    print("="*60)
    
    # Scheduler oluştur
    scheduler = AdaptiveQLearningScheduler(
        learning_rate=0.1,
        epsilon=0.3,
        energy_weight=0.7,
        latency_weight=0.3
    )
    
    # Örnek training data
    training_workloads = [
        (50000, 'Conv', 50.0),
        (5000000, 'Conv', 100.0),
        (100000, 'ReLU', 10.0),
        (10000000, 'FC', 50.0),
        (1000000, 'Conv', 20.0),
    ]
    
    # 10 episode eğit
    print("\n Q-Learning Eğitimi Başlıyor...")
    pim, gpu = PIMArray(), GPUBaseline()
    for episode in range(10):
        reward = scheduler.train_episode(pim, gpu, training_workloads * 5)
        print(f"Episode {episode+1}: Reward = {reward:.2f}, Epsilon = {scheduler.epsilon:.3f}")
    
    # İstatistikler
    print("\n Öğrenme İstatistikleri:")
    stats = scheduler.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Inference
    print("\n🔍 Test Predictions:")
    test_cases = [
        (50000, 'Conv', None),
        (10000000, 'FC', 10.0),
        (1000000, 'ReLU', 50.0)
    ]
    
    for workload, layer, deadline in test_cases:
        action, confidence = scheduler.predict(workload, layer, deadline)
        print(f"  Workload={workload}, Layer={layer}, Deadline={deadline}")
        print(f"    → {action} (confidence: {confidence:.2f})")
    
    # Modeli kaydet
    scheduler.save_model('q_learning_scheduler.json')