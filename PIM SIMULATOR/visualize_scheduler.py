import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from Q_Learning.adaptive_scheduler import AdaptiveQLearningScheduler   
def run_simulation_and_visualize():
    print("🚀 Simülasyon ve Görselleştirme Başlıyor...")
    
    # 1. Scheduler'ı Başlat
    scheduler = AdaptiveQLearningScheduler(
        learning_rate=0.1,
        epsilon=0.9,       # Yüksek başlatıp düşüreceğiz
        discount_factor=0.9,
        energy_weight=0.7,
        latency_weight=0.3
    )
    
    # 2. Eğitim Verisi Oluştur (Çeşitli senaryolar)
    # (Workload Size, Layer Type, Deadline ms)
    scenarios = [
        (50000, 'Conv', 5.0),       # Küçük, Strict -> PIM seçmeli
        (15000000, 'FC', 100.0),    # Büyük, Relaxed -> GPU seçmeli
        (500000, 'ReLU', 20.0),     # Orta -> Hybrid olabilir
        (80000, 'Conv', None),      # Küçük, No Deadline
        (20000000, 'Conv', 10.0),   # Çok büyük, Strict -> GPU (Hız lazım)
    ]
    
    # 3. Eğitim Döngüsü
    n_episodes = 300
    rewards_per_episode = []
    epsilon_history = []
    
    for episode in range(n_episodes):
        # Her episode'da senaryoları karıştırıp eğit
        episode_data = scenarios * 2  # Veriyi çoğalt
        np.random.shuffle(episode_data)
        
        # Senin sınıfındaki train_episode metodunu çağır
        total_reward = scheduler.train_episode(None, None, episode_data)
        
        rewards_per_episode.append(total_reward)
        epsilon_history.append(scheduler.epsilon)
    
    # 4. Verileri Görselleştirme için Hazırla
    
    # Grafik Çerçevesi Ayarları
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Q-Learning Scheduler Performans Analizi', fontsize=20, weight='bold')
    
    # --- GRAFİK 1: Öğrenme Eğrisi (Total Reward) ---
    # Moving Average ile gürültüyü azaltarak çizelim
    series_reward = pd.Series(rewards_per_episode)
    window_size = 20
    rolling_mean = series_reward.rolling(window=window_size).mean()
    
    axes[0, 0].plot(rewards_per_episode, alpha=0.3, color='gray', label='Raw Reward')
    axes[0, 0].plot(rolling_mean, color='blue', linewidth=2, label=f'{window_size}-Ep Mov. Avg')
    axes[0, 0].set_title('Öğrenme Eğrisi (Convergence)', fontsize=14)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    
    # --- GRAFİK 2: Epsilon Decay (Exploration vs Exploitation) ---
    axes[0, 1].plot(epsilon_history, color='orange', linewidth=2)
    axes[0, 1].set_title('Epsilon Decay (Keşif Oranı)', fontsize=14)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon Değeri')
    axes[0, 1].text(n_episodes*0.7, 0.5, 'Exploration', fontsize=12, color='orange')
    axes[0, 1].text(n_episodes*0.7, 0.1, 'Exploitation', fontsize=12, color='green')
    
    # --- GRAFİK 3: Q-Table Isı Haritası (Policy Heatmap) ---
    # Q-table'ı DataFrame'e çevirelim
    q_data = []
    for state, values in scheduler.q_table.items():
        state_str = f"{state[0]}\n{state[1]}\n{state[2]}" # (Size, Layer, Deadline)
        row = {
            'State': state_str,
            'PIM': values[0],
            'GPU': values[1],
            'HYBRID': values[2]
        }
        q_data.append(row)
    
    df_q = pd.DataFrame(q_data).set_index('State')
    
    # Sadece en çok karşılaşılan 10 durumu göster (tablo çok büyükse)
    if len(df_q) > 10:
        df_q = df_q.head(10)
        
    sns.heatmap(df_q, annot=True, cmap='RdYlGn', fmt='.1f', linewidths=.5, ax=axes[1, 0])
    axes[1, 0].set_title('Q-Table Isı Haritası (Tercih Edilen Aksiyonlar)', fontsize=14)
    axes[1, 0].set_ylabel('State (Size, Layer, Deadline)')
    
    # --- GRAFİK 4: Workload Bazlı Karar Dağılımı ---
    # Modeli test edip hangi boyutta ne seçtiğine bakalım
    test_results = {'Size': [], 'Action': []}
    test_sizes = [50000, 500000, 5000000, 15000000] # Küçükten büyüğe
    test_layers = ['Conv', 'FC']
    
    for size in test_sizes:
        for layer in test_layers:
            # Deadline'ı relaxed tutalım ki salt size etkisine bakalım
            action, _ = scheduler.predict(size, layer, deadline_ms=100)
            
            # Kategorik isimlendirme (grafik için)
            if size < 100000: cat = 'Small'
            elif size < 10000000: cat = 'Medium'
            else: cat = 'Large'
            
            test_results['Size'].append(cat)
            test_results['Action'].append(action)

    df_test = pd.DataFrame(test_results)
    
    sns.countplot(x='Size', hue='Action', data=df_test, ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Workload Boyutuna Göre Karar Dağılımı', fontsize=14)
    axes[1, 1].set_ylabel('Seçim Sayısı')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation_and_visualize()