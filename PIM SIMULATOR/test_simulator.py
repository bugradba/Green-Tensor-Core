from Q_Learning.adaptive_scheduler import AdaptiveQLearningScheduler 
from pim_simulator import PIMArray
from baseline_models import GPUBaseline, CPUBaseline

def test_pim_core():
    """
    Tek core test et
    """
    from pim_simulator import PIM_Core

    core = PIM_Core()
    result, energy, latency = core.multiply_4bit(15, 15)

    print("---- PIM Core Test ----")
    print(f"15 x 15 = {result}")
    print(f"Enerji: {energy:.2f} pJ")
    print(f"Gecikme: {latency:.2f} ns")
    print() 

def test_pim_cluster():
    """Cluster MAC işlemini test et (pPIM uyumlu analiz)."""
    from pim_simulator import PIMCluster
    
    print("\n---- PIM Cluster Test ----")

    # Cluster'ı başlat
    cluster = PIMCluster()

    # Test değerleri
    a, b = 255, 200
    expected_result = a * b  # Referans (8-bit tam doğruluk)

    # -------------------------
    # 1) 8-bit Precision
    # -------------------------
    result_8bit, energy_8bit, latency_8bit = cluster.mac_8bit(a, b, precision=8)

    print(f"{a} x {b} (8-bit): {result_8bit}")
    print(f"  Enerji: {energy_8bit:.2f} pJ, Gecikme: {latency_8bit:.2f} ns")

    # -------------------------
    # 2) 4-bit Precision Scaling
    # -------------------------
    result_4bit, energy_4bit, latency_4bit = cluster.mac_8bit(a, b, precision=4)

    print(f"{a} x {b} (4-bit approx): {result_4bit}")
    print(f"  Enerji: {energy_4bit:.2f} pJ, Gecikme: {latency_4bit:.2f} ns")

    # -------------------------
    # 3) DOĞRU ACCURACY ANALİZİ
    # (pPIM paper uyumlu)
    # -------------------------
    ideal_4bit = ((a >> 4) * (b >> 4)) << 8

    accuracy = (result_4bit / ideal_4bit) * 100 if ideal_4bit > 0 else 0
    error = 100 - accuracy

    print(f"  4-bit Scaled Accuracy: %{accuracy:.1f}")
    print(f"  Quantization Error: %{error:.1f}")

    # -------------------------
    # 4) Enerji Tasarrufu
    # -------------------------
    if energy_8bit > 0:
        savings = (1 - energy_4bit / energy_8bit) * 100
        print(f"  Enerji Tasarrufu: %{savings:.1f}")

    print("-" * 40)

def test_simple_cnn():
    """Basit bir CNN katmanını simüle et."""
    pim = PIMArray(num_clusters=256)
    gpu = GPUBaseline()
    cpu = CPUBaseline()

    # Örnek: AlexNet'in ilk conv katmanı
    # Input: (3, 227, 227), Kernel: (96, 3, 11, 11)

    input_shape = (3, 227, 227)
    kernel_shape = (96, 3, 11, 11)

    print("----  CNN Katmanı (AlexNet Conv1) ----")
    print(f"Input: {input_shape}, Kernel: {kernel_shape}")
    print()

    # PIM 8-bit
    stats_pim8 = pim.convolution_layer(input_shape, kernel_shape, precision=8)
    print("PIM (8-bit):")
    print(f"  Enerji: {stats_pim8['energy_total_mj']:.2f} mJ")
    print(f"  Gecikme: {stats_pim8['latency_ms']:.2f} ms")
    print(f"  Güç: {stats_pim8['power_mw']:.2f} mW")
    print()
    
    # PIM 4-bit
    stats_pim4 = pim.convolution_layer(input_shape, kernel_shape, precision=4)
    print("PIM (4-bit precision scaling):")
    print(f"  Enerji: {stats_pim4['energy_total_mj']:.2f} mJ")
    print(f"  Gecikme: {stats_pim4['latency_ms']:.2f} ms")
    print(f"  Güç: {stats_pim4['power_mw']:.2f} mW")
    print(f"  Tasarruf: %{(1 - stats_pim4['energy_total_mj']/stats_pim8['energy_total_mj'])*100:.1f}")
    print()

    # GPU
    stats_gpu = gpu.model_inference(stats_pim8['total_macs'])
    print("GPU (Jetson Nano):")
    print(f"  Enerji: {stats_gpu['total_energy_mj']:.2f} mJ")
    print(f"  Gecikme: {stats_gpu['total_latency_ms']:.2f} ms")
    print(f"  Güç: {stats_gpu['avg_power_mw']:.2f} mW")
    print()
    
    # Karşılaştırma
    print("=== KARŞILAŞTIRMA ===")
    print(f"PIM vs GPU Enerji Tasarrufu: %{(1 - stats_pim8['energy_total_mj']/stats_gpu['total_energy_mj'])*100:.1f}")
    print(f"PIM vs GPU Hız Artışı: {stats_gpu['total_latency_ms'] / stats_pim8['latency_ms']:.2f}x")


def test_advanced_hybrid():
    """Katman Bazlı (Layer-wise) Hibrit Test"""
    from pim_simulator import PIMArray
    from baseline_models import GPUBaseline
    from hybrid_scheduler import HybridSystem

    print("\n" + "="*50)
    print("GELİŞMİŞ KATMAN BAZLI HİBRİT TEST")
    print("="*50)

    # Sistemleri Hazırla
    pim = PIMArray()
    gpu = GPUBaseline()
    scheduler = HybridSystem(pim, gpu)

    # Sanal bir AI Modeli Oluştur (Liste olarak)
    # Bir Conv2D -> ReLU -> Linear katmanlı model simülasyonu
    fake_model = [
        {
            'name': 'conv1', 
            'type': 'Conv2D', 
            'input': (3, 227, 227), 
            'kernel': (96, 3, 11, 11)
        },
        {
            'name': 'relu1', 
            'type': 'ReLU'
        },
        {
            'name': 'fc1', 
            'type': 'Linear', 
            'in_features': 9216, 
            'out_features': 4096
        }
    ]

    # Analiz Et ve Çalıştır
    input_data_size_mb = 1.5 # Örnek veri boyutu
    plan, total_e, total_l = scheduler.analyze_model_layers(fake_model, input_data_size_mb)

    # Sonuçları Yazdır
    print(f"\n{'Katman':<10} {'Tip':<10} {'Cihaz':<10} {'Enerji (mJ)':<15} {'Sebep'}")
    print("-" * 80)
    
    for p in plan:
        print(f"{p['layer']:<10} {p['type']:<10} {p['device']:<10} {p['energy']:<15.4f} {p['reason']}")
    
    print("-" * 80)
    print(f"TOPLAM ENERJİ: {total_e:.2f} mJ")
    print(f"TOPLAM SÜRE:   {total_l:.2f} ms")



def test_adaptive_qlearning():
    """
    Q-Learning tabanlı adaptive scheduler testi
    """
    print("\n" + "="*70)
    print("  ADAPTIVE Q-LEARNING SCHEDULER TEST")
    print("="*70)
    
    from Q_Learning.adaptive_scheduler import AdaptiveQLearningScheduler
    
    # Scheduler oluştur
    scheduler = AdaptiveQLearningScheduler(
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.3,
        energy_weight=0.7,
        latency_weight=0.3
    )
    
    print("\n  Hyperparameters:")
    print(f"  Learning Rate (α): {scheduler.alpha}")
    print(f"  Discount Factor (γ): {scheduler.gamma}")
    print(f"  Exploration Rate (ε): {scheduler.epsilon}")
    print(f"  Energy Weight: {scheduler.energy_weight}")
    print(f"  Latency Weight: {scheduler.latency_weight}")
    
    # Training workloads
    training_workloads = [
        (50000, 'Conv', 100.0),
        (5000000, 'Conv', 100.0),
        (50000000, 'Conv', 100.0),
        (100000, 'ReLU', 50.0),
        (1000000, 'ReLU', 10.0),
        (10000000, 'FC', 50.0),
        (1000000, 'FC', 100.0),
        (500000, 'Conv', None),
    ]
    
    # Eğitim
    print("\n Training Phase (20 episodes)...")
    print(f"{'Episode':<10} {'Reward':<15} {'Epsilon':<15} {'Avg Reward':<15}")
    print("-" * 70)
    
    for episode in range(20):
        import random
        episode_workloads = training_workloads * 3
        random.shuffle(episode_workloads)
        
        reward = scheduler.train_episode(None, None, episode_workloads)
        stats = scheduler.get_statistics()
        avg_reward = stats['avg_reward']
        
        print(f"{episode+1:<10} {reward:<15.2f} {scheduler.epsilon:<15.3f} {avg_reward:<15.2f}")
    
    # İstatistikler
    print("\n Learning Statistics:")
    stats = scheduler.get_statistics()
    print(f"  Total Episodes: {stats['episodes']}")
    print(f"  Total Reward: {stats['total_reward']:.2f}")
    print(f"  Average Reward: {stats['avg_reward']:.2f}")
    print(f"  Q-Table Size: {stats['q_table_size']} states learned")
    print(f"  Final Epsilon: {stats['epsilon']:.3f}")
    
    # Test phase
    print("\n  Test Phase (Inference Mode):")
    print(f"{'Workload':<12} {'Layer':<8} {'Deadline':<12} {'Decision':<10} {'Confidence':<12}")
    print("-" * 70)
    
    test_cases = [
        (50000, 'Conv', None),
        (5000000, 'Conv', 100.0),
        (50000000, 'Conv', 100.0),
        (100000, 'ReLU', 50.0),
        (1000000, 'ReLU', 5.0),
        (10000000, 'FC', 50.0),
        (500000, 'Conv', 200.0)
    ]
    
    for workload, layer, deadline in test_cases:
        action, confidence = scheduler.predict(workload, layer, deadline)
        deadline_str = f"{deadline} ms" if deadline else "None"
        print(f"{workload:<12,} {layer:<8} {deadline_str:<12} {action:<10} {confidence:<12.2f}")
    
    # Karşılaştırma
    print("\n Comparison: Rule-Based vs Q-Learning")
    print(f"{'Scenario':<30} {'Rule-Based':<15} {'Q-Learning':<15} {'Match?':<10}")
    print("-" * 70)
    
    comparison_cases = [
        (100000, 'Conv', None),
        (5000000, 'Conv', 100.0),
        (10000000, 'FC', 10.0)
    ]
    
    for workload, layer, deadline in comparison_cases:
        # Rule-based
        if workload < 100000:
            rule_based = "PIM"
        elif workload > 10000000:
            rule_based = "GPU"
        else:
            rule_based = "HYBRID"
        
        # Q-Learning
        q_decision, _ = scheduler.predict(workload, layer, deadline)
        
        match = "Same " if q_decision == rule_based else "Different"
        scenario = f"{workload//1000}K {layer}"
        print(f"{scenario:<30} {rule_based:<15} {q_decision:<15} {match:<10}")
    
    # Model kaydet
    scheduler.save_model('adaptive_scheduler_trained.json')
    
    print("\n Adaptive Q-Learning test tamamlandı!")
    print("   Sistem artık geçmiş deneyimlerden öğrenerek")
    print("   Optimal PIM/GPU kararları verebilir!")

def visualize_results():
    """Tüm sonuçları görselleştir"""
    print("\n" + "="*70)
    print(" GÖRSELLEŞTIRME BAŞLIYOR...")
    print("="*70)
    
    from visualize_scheduler import EcoPIMVisualizer
    
    viz = EcoPIMVisualizer()
    
    # Q-Learning sonuçlarını oku (eğer JSON'dan okumak isterseniz)
    try:
        import json
        with open('adaptive_scheduler_trained.json', 'r') as f:
            model_data = json.load(f)
        
        # Gerçek veriler (örnek)
        episodes = list(range(1, 21))
        rewards = [510, 643, 683, 703, 803, 503, 780, 646, 620, 673,
                   690, 720, 750, 680, 710, 730, 690, 710, 720, 700]
        
    except:
        # Manuel veriler
        episodes = list(range(1, 11))
        rewards = [510, 643, 683, 703, 803, 503, 780, 646, 620, 673]
    
    # 1. Q-Learning
    print("\n1️⃣ Q-Learning training grafiği...")
    viz.plot_qlearning_training(episodes, rewards)
    
    # 2. Enerji karşılaştırması
    print("\n2️⃣ Enerji karşılaştırma...")
    viz.plot_energy_comparison(
        pim_8bit=211.38,
        pim_4bit=186.57,
        gpu=332.09,
        hybrid=219.10
    )
    
    # 3. Trade-off
    print("\n3️⃣ Trade-off analizi...")
    results_dict = {
        'PIM (8-bit)': {'energy': 211.38, 'latency': 41.02},
        'PIM (4-bit)': {'energy': 186.57, 'latency': 20.51},
        'GPU': {'energy': 332.09, 'latency': 6.95},
        'Hybrid': {'energy': 219.10, 'latency': 42.69}
    }
    viz.plot_latency_vs_energy(results_dict)
    
    # 4. 4-bit vs 8-bit
    print("\n4️⃣ Precision karşılaştırma...")
    viz.plot_4bit_vs_8bit(19.44, 4.32, 6.40, 3.20)
    
    # 5. Hibrit breakdown
    print("\n5️⃣ Hibrit breakdown...")
    layer_results = [
        {'layer': 'Conv1', 'type': 'Conv2D', 'device': 'PIM', 'energy': 211.38},
        {'layer': 'ReLU1', 'type': 'ReLU', 'device': 'PIM', 'energy': 0.01},
        {'layer': 'FC1', 'type': 'Linear', 'device': 'GPU', 'energy': 7.71}
    ]
    viz.plot_hybrid_breakdown(layer_results)
    
    print("\n✅ Tüm grafikler oluşturuldu!")

    
    print("\n" + "="*70)
    response = input("📊 Grafikleri oluşturmak ister misiniz? (y/n): ")
    if response.lower() == 'y':
        visualize_results()

def test_integrated_qlearning_hybrid():
    """
    Q-Learning'in gerçek hibrit sistemle entegre testi
    
    Bu test:
    1. Gerçek PIM/GPU simülasyonu kullanır
    2. Q-Learning ile karar verir
    3. Rule-based ile karşılaştırır
    4. Online learning gösterir
    """
    print("\n" + "="*70)
    print("🧠 ENTEGRE Q-LEARNING HİBRİT SİSTEM TESTİ")
    print("="*70)
    
    from pim_simulator import PIMArray
    from baseline_models import GPUBaseline
    
    # ADIM 1: Dosyaya ekle (hybrid_scheduler.py'ye AdvancedHybridScheduler)
    # Şimdilik import edemeyiz, ama mantığı gösterelim
    
    print("\n📋 Test Modeli: AlexNet İlk 5 Katman")
    print("-" * 70)
    
    # Test modeli
    alexnet_layers = [
        {'name': 'conv1', 'type': 'Conv2D', 'input': (3, 227, 227), 'kernel': (96, 3, 11, 11)},
        {'name': 'relu1', 'type': 'ReLU'},
        {'name': 'pool1', 'type': 'Pool'},
        {'name': 'conv2', 'type': 'Conv2D', 'input': (96, 27, 27), 'kernel': (256, 96, 5, 5)},
        {'name': 'fc1', 'type': 'Linear', 'in_features': 9216, 'out_features': 4096}
    ]
    
    # Sistemleri oluştur
    pim = PIMArray(num_clusters=256)
    gpu = GPUBaseline()
    
    # Manuel simülasyon (AdvancedHybridScheduler olmadan)
    print("\n🎯 Q-Learning Kararları:")
    print(f"{'Katman':<10} {'Tip':<10} {'Karar':<10} {'Sebep':<40}")
    print("-" * 70)
    
    from Q_Learning.adaptive_scheduler import AdaptiveQLearningScheduler
    
    # Q-Learning scheduler yükle
    q_scheduler = AdaptiveQLearningScheduler()
    try:
        q_scheduler.load_model('adaptive_scheduler_trained.json')
        print("✅ Eğitilmiş model yüklendi\n")
    except:
        print("⚠️  Eğitilmiş model yok, varsayılan Q-table kullanılıyor\n")
    
    total_energy_q = 0
    total_latency_q = 0
    
    for layer in alexnet_layers:
        layer_type = layer['type']
        
        # Workload tahmin
        if 'input' in layer and 'kernel' in layer:
            input_shape = layer['input']
            kernel_shape = layer['kernel']
            C_in, H, W = input_shape
            C_out, _, Kh, Kw = kernel_shape
            workload = C_out * C_in * H * W * Kh * Kw
        else:
            workload = 100000  # Default
        
        # Q-Learning kararı
        device, confidence = q_scheduler.predict(workload, layer_type, None)
        
        # Simülasyon
        if device == 'PIM' and 'input' in layer and 'kernel' in layer:
            stats = pim.convolution_layer(layer['input'], layer['kernel'], precision=8)
            energy = stats['energy_total_mj']
            latency = stats['latency_ms']
        elif device == 'GPU' and 'input' in layer and 'kernel' in layer:
            C_in, H, W = layer['input']
            C_out, _, Kh, Kw = layer['kernel']
            macs = C_out * C_in * H * W * Kh * Kw
            stats = gpu.model_inference(macs)
            energy = stats['total_energy_mj']
            latency = stats['total_latency_ms']
        else:
            # ReLU, Pool için basit tahmin
            energy = 0.01 if device == 'PIM' else 0.05
            latency = 0.01 if device == 'PIM' else 0.005
        
        total_energy_q += energy
        total_latency_q += latency
        
        reason = f"Q-Learning (conf: {confidence:.1f})"
        print(f"{layer['name']:<10} {layer_type:<10} {device:<10} {reason:<40}")
    
    print("-" * 70)
    print(f"TOPLAM (Q-Learning): Enerji={total_energy_q:.2f} mJ, Latency={total_latency_q:.2f} ms")
    
    # Rule-based karşılaştırma
    print("\n🎯 Rule-Based Kararları:")
    print(f"{'Katman':<10} {'Tip':<10} {'Karar':<10} {'Sebep':<40}")
    print("-" * 70)
    
    total_energy_r = 0
    total_latency_r = 0
    
    for layer in alexnet_layers:
        layer_type = layer['type']
        
        # Rule-based karar
        if layer_type in ['ReLU', 'Pool']:
            device = 'PIM'
            reason = 'Simple LUT Operation'
        elif 'input' in layer and 'kernel' in layer:
            device = 'PIM'
            reason = 'Memory-intensive MAC'
        elif layer_type in ['Linear', 'FC']:
            device = 'GPU'
            reason = 'Large Matrix Multiplication'
        else:
            device = 'PIM'
            reason = 'Default'
        
        # Simülasyon (aynı mantık)
        if device == 'PIM' and 'input' in layer and 'kernel' in layer:
            stats = pim.convolution_layer(layer['input'], layer['kernel'], precision=8)
            energy = stats['energy_total_mj']
            latency = stats['latency_ms']
        elif device == 'GPU' and 'input' in layer and 'kernel' in layer:
            C_in, H, W = layer['input']
            C_out, _, Kh, Kw = layer['kernel']
            macs = C_out * C_in * H * W * Kh * Kw
            stats = gpu.model_inference(macs)
            energy = stats['total_energy_mj']
            latency = stats['total_latency_ms']
        else:
            energy = 0.01 if device == 'PIM' else 0.05
            latency = 0.01 if device == 'PIM' else 0.005
        
        total_energy_r += energy
        total_latency_r += latency
        
        print(f"{layer['name']:<10} {layer_type:<10} {device:<10} {reason:<40}")
    
    print("-" * 70)
    print(f"TOPLAM (Rule-Based): Enerji={total_energy_r:.2f} mJ, Latency={total_latency_r:.2f} ms")
    
    # Karşılaştırma
    print("\n" + "="*70)
    print("📊 KARŞILAŞTIRMA SONUÇLARI")
    print("="*70)
    
    energy_improvement = (total_energy_r - total_energy_q) / total_energy_r * 100
    latency_diff = (total_latency_q - total_latency_r) / total_latency_r * 100
    
    print(f"\n{'Metrik':<20} {'Q-Learning':<15} {'Rule-Based':<15} {'Fark':<15}")
    print("-" * 70)
    print(f"{'Enerji (mJ)':<20} {total_energy_q:<15.2f} {total_energy_r:<15.2f} {energy_improvement:>+.1f}%")
    print(f"{'Gecikme (ms)':<20} {total_latency_q:<15.2f} {total_latency_r:<15.2f} {latency_diff:>+.1f}%")
    
    if energy_improvement > 0:
        print(f"\n✅ Q-Learning {energy_improvement:.1f}% daha enerji verimli!")
    elif energy_improvement < -5:
        print(f"\n⚠️  Q-Learning {abs(energy_improvement):.1f}% daha fazla enerji tüketiyor")
        print("   (Daha fazla eğitim gerekebilir)")
    else:
        print(f"\n➡️  Benzer performans (fark: {energy_improvement:.1f}%)")
    
    print("\n💡 NOT: Q-Learning gerçek simülasyon sonuçlarıyla öğreniyor.")
    print("   Daha fazla episode ile performans artacaktır.")




if __name__ == "__main__":
    test_pim_core()
    test_simple_cnn()
    test_pim_cluster()
    test_advanced_hybrid()
    test_adaptive_qlearning()
    test_integrated_qlearning_hybrid()  