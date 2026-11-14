"""
Geliştirilmiş Q-Learning Eğitimi

Bu script Q-Learning'i daha çeşitli senaryolarla eğitir.
"""

import os
import random

from Q_Learning.adaptive_scheduler import AdaptiveQLearningScheduler
from pim_simulator import PIMArray
from baseline_models import GPUBaseline

_HERE = os.path.dirname(os.path.abspath(__file__))

def train_qlearning_extensive(episodes=50, seed=42):
    """
    Kapsamlı Q-Learning eğitimi
    
    - 50 episode (10 yerine)
    - Çeşitli workload kombinasyonları
    - Farklı deadline'lar
    - Tüm layer tipleri
    """
    print("🎓 GELİŞMİŞ Q-LEARNING EĞİTİMİ")
    print("="*70)
    
    scheduler = AdaptiveQLearningScheduler(
        learning_rate=0.15,      # Biraz daha hızlı öğrenme
        discount_factor=0.9,
        epsilon=0.4,             # Daha fazla exploration
        energy_weight=0.7,
        latency_weight=0.3
    )
    
    # ÇEŞİTLİ TRAINING DATA
    base_workloads = [
        # Small workloads (PIM favors)
        (10000, 'Conv', None),
        (20000, 'Conv', 100.0),
        (50000, 'Conv', 50.0),
        (30000, 'ReLU', None),
        (40000, 'Pool', None),
        
        # Medium workloads (HYBRID territory)
        (500000, 'Conv', None),
        (1000000, 'Conv', 100.0),
        (2000000, 'Conv', 50.0),
        (1500000, 'Conv', 20.0),    # Sıkı deadline
        (800000, 'FC', None),
        (1200000, 'ReLU', 30.0),
        
        # Large workloads (GPU favors)
        (10000000, 'Conv', None),
        (20000000, 'Conv', 100.0),
        (50000000, 'Conv', 50.0),
        (30000000, 'FC', None),
        (15000000, 'FC', 10.0),     # Çok sıkı deadline
        
        # Edge cases
        (100000, 'Conv', 5.0),      # Small ama sıkı deadline → GPU?
        (50000000, 'Conv', 200.0),  # Large ama rahat deadline → Hybrid?
        (1000000, 'Pool', None),    # Pool genelde PIM
        (5000000, 'ReLU', None),    # ReLU genelde PIM
        
        # Different layer types
        (100000, 'BatchNorm', None),
        (500000, 'Dropout', None),
        (2000000, 'Linear', 100.0),
        (10000000, 'Linear', 10.0),
    ]
    
    # 50 EPISODE EĞİTİM
    print(f"\n{'Episode':<10} {'Reward':<15} {'Epsilon':<15} {'Q-States':<12} {'Avg Reward':<15}")
    print("-" * 75)
    
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    pim, gpu = PIMArray(), GPUBaseline()

    for episode in range(episodes):
        # Her episode'da workload'ları karıştır ve çoğalt
        episode_workloads = base_workloads * 2  # 48 karar
        random.shuffle(episode_workloads)
        
        # Episode çalıştır
        reward = scheduler.train_episode(pim, gpu, episode_workloads)
        stats = scheduler.get_statistics()
        
        # Her 5 episode'da bir yazdır
        if (episode + 1) % 5 == 0:
            print(f"{episode+1:<10} {reward:<15.2f} {scheduler.epsilon:<15.3f} "
                  f"{stats['q_table_size']:<12} {stats['avg_reward']:<15.2f}")
    
    # Final istatistikler
    print("\n" + "="*70)
    print("📊 EĞİTİM SONUÇLARI")
    print("="*70)
    
    stats = scheduler.get_statistics()
    print(f"\nToplam Episode: {stats['episodes']}")
    print(f"Toplam Reward: {stats['total_reward']:.2f}")
    print(f"Ortalama Reward: {stats['avg_reward']:.2f}")
    print(f"Q-Table Boyutu: {stats['q_table_size']} states")
    print(f"Final Epsilon: {stats['epsilon']:.3f}")
    
    # Q-Table içeriğini göster
    print("\n📋 Öğrenilen Q-Table:")
    print(f"{'State':<50} {'PIM':<10} {'GPU':<10} {'HYBRID':<10} {'Best':<10}")
    print("-" * 90)
    
    for state, q_values in list(scheduler.q_table.items())[:15]:  # İlk 15 state
        best_action = scheduler.actions[q_values.argmax()]
        print(f"{str(state):<50} {q_values[0]:<10.2f} {q_values[1]:<10.2f} "
              f"{q_values[2]:<10.2f} {best_action:<10}")
    
    if len(scheduler.q_table) > 15:
        print(f"... ve {len(scheduler.q_table) - 15} state daha")
    
    # Modeli kaydet
    scheduler.save_model(os.path.join(_HERE, 'adaptive_scheduler_trained_v2.json'))
    print(f"\n✅ Geliştirilmiş model kaydedildi: adaptive_scheduler_trained_v2.json")
    
    # Test predictions
    print("\n🔍 Örnek Tahminler:")
    print(f"{'Scenario':<40} {'Decision':<10} {'Confidence':<12}")
    print("-" * 65)
    
    test_cases = [
        (50000, 'Conv', None, "Small Conv, no deadline"),
        (50000, 'Conv', 5.0, "Small Conv, strict deadline"),
        (1000000, 'Conv', None, "Medium Conv, no deadline"),
        (1000000, 'Conv', 10.0, "Medium Conv, tight deadline"),
        (10000000, 'Conv', None, "Large Conv, no deadline"),
        (10000000, 'FC', None, "Large FC, no deadline"),
        (100000, 'Pool', None, "Small Pool"),
        (500000, 'ReLU', None, "Medium ReLU"),
    ]
    
    for workload, layer, deadline, desc in test_cases:
        action, confidence = scheduler.predict(workload, layer, deadline)
        print(f"{desc:<40} {action:<10} {confidence:<12.2f}")
    
    return scheduler


if __name__ == "__main__":
    scheduler = train_qlearning_extensive()
    
    print("\n" + "="*70)
    print("🎉 EĞİTİM TAMAMLANDI!")
    print("="*70)
    print("\n💡 Şimdi demo_simulator.py dosyasını çalıştırabilirsiniz.")
    print("   AdvancedHybridScheduler yeni modeli (v2) otomatik yükler.")
