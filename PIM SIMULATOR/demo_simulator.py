"""
Demo / benchmark script'i. Çalıştırma: python demo_simulator.py

Not: Bu dosya eskiden test_simulator.py idi. Fonksiyonlar sonuç yazdırır,
doğrulama yapmaz; gerçek (assert'li) testler tests/ klasöründedir.
"""
from Q_Learning.adaptive_scheduler import AdaptiveQLearningScheduler
from pim_simulator import PIMArray, conv_stats
from baseline_models import GPUBaseline, CPUBaseline
from hybrid_scheduler import simulate_layer

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
    # 4-bit sonucu, gerçek 8-bit çarpımla (a*b) karşılaştırılır.
    # (Eskiden 4-bit sonuç kendisiyle karşılaştırıldığı için hep %100 çıkıyordu.)
    error = abs(expected_result - result_4bit) / expected_result * 100

    print(f"  4-bit Göreli Hata (a*b'ye göre): %{error:.1f}")

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
    stride, padding = 4, 0

    print("----  CNN Katmanı (AlexNet Conv1) ----")
    print(f"Input: {input_shape}, Kernel: {kernel_shape}, Stride: {stride}")
    print()

    # PIM 8-bit
    stats_pim8 = pim.convolution_layer(input_shape, kernel_shape, precision=8, stride=stride, padding=padding)
    print("PIM (8-bit):")
    print(f"  Enerji: {stats_pim8['energy_total_mj']:.2f} mJ")
    print(f"  Gecikme: {stats_pim8['latency_ms']:.2f} ms")
    print(f"  Güç: {stats_pim8['power_mw']:.2f} mW")
    print()
    
    # PIM 4-bit
    stats_pim4 = pim.convolution_layer(input_shape, kernel_shape, precision=4, stride=stride, padding=padding)
    print("PIM (4-bit precision scaling):")
    print(f"  Enerji: {stats_pim4['energy_total_mj']:.2f} mJ")
    print(f"  Gecikme: {stats_pim4['latency_ms']:.2f} ms")
    print(f"  Güç: {stats_pim4['power_mw']:.2f} mW")
    print(f"  Tasarruf: %{(1 - stats_pim4['energy_total_mj']/stats_pim8['energy_total_mj'])*100:.1f}")
    print()

    # GPU
    cs = conv_stats(input_shape, kernel_shape, stride, padding)
    data_bits = (cs['input_elems'] + cs['weight_elems'] + cs['output_elems']) * 8
    stats_gpu = gpu.model_inference(stats_pim8['total_macs'], data_bits)
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
            'kernel': (96, 3, 11, 11),
            'stride': 4
        },
        {
            'name': 'relu1', 
            'type': 'ReLU',
            'input': (96, 55, 55)
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
    
    import random
    pim, gpu = PIMArray(), GPUBaseline()
    episode_rewards = []
    for episode in range(20):
        episode_workloads = training_workloads * 3
        random.shuffle(episode_workloads)
        
        reward = scheduler.train_episode(pim, gpu, episode_workloads)
        episode_rewards.append(reward)
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
    return episode_rewards

def visualize_results():
    """Tüm grafikleri simülatör sonuçlarından üret."""
    from visualize_scheduler import EcoPIMVisualizer

    print("\n" + "="*70)
    print(" GÖRSELLEŞTIRME")
    print("="*70)
    viz = EcoPIMVisualizer()
    viz.create_summary_dashboard()
    viz.create_qlearning_analysis()
    viz.create_alexnet_breakdown()


def test_integrated_qlearning_hybrid():
    """
    Q-Learning'in gerçek hibrit sistemle entegre testi:
    AlexNet katmanlarıyla eğitilir, kural tabanlı yöntem ve oracle ile karşılaştırılır.
    """
    from results import compute_results

    print("\n" + "="*70)
    print("🧠 ENTEGRE Q-LEARNING HİBRİT SİSTEM TESTİ (AlexNet)")
    print("="*70)

    r = compute_results()
    alex = r['alexnet']

    print(f"\n{'Katman':<10} {'Tip':<10} {'Kural':<8} {'Q-Learn':<8}")
    print("-" * 40)
    for rule, q in zip(alex['rule_based']['layers'], alex['qlearning']['layers']):
        print(f"{rule['layer']:<10} {rule['type']:<10} {rule['device']:<8} {q['device']:<8}")

    print(f"\n{'Strateji':<20} {'Enerji (mJ)':>12} {'Gecikme (ms)':>14}")
    print("-" * 48)
    for k, v in alex.items():
        print(f"{k:<20} {v['energy']:>12.2f} {v['latency']:>14.2f}")

    q, rb = alex['qlearning'], alex['rule_based']
    print(f"\nQ-Learning, kural tabanlıya göre: enerji tasarrufu %{(rb['energy'] - q['energy']) / rb['energy'] * 100:.1f}, "
          f"gecikme değişimi {(q['latency'] - rb['latency']) / rb['latency'] * 100:+.1f}%")


def test_full_alexnet():
    """
    Tam AlexNet modeli: kural tabanlı hibrit plan, gerçekten simüle edilmiş
    GPU-only plan ile karşılaştırılır.
    """
    from results import alexnet_strategies

    print("\n" + "="*70)
    print("🏗️  TAM ALEXNET MODELİ TESTİ")
    print("="*70)

    pim, gpu = PIMArray(num_clusters=256), GPUBaseline()
    res = alexnet_strategies(pim, gpu)
    hybrid, gpu_only = res['rule_based'], res['gpu_only']

    print(f"{'#':<4} {'Katman':<12} {'Tip':<10} {'Cihaz':<8} {'Enerji (mJ)':<12} {'Latency (ms)':<12}")
    print("-" * 70)
    for i, row in enumerate(hybrid['layers'], 1):
        print(f"{i:<4} {row['layer']:<12} {row['type']:<10} {row['device']:<8} "
              f"{row['energy']:<12.4f} {row['latency']:<12.4f}")

    print("-" * 70)
    print(f"  Hibrit (kural): {hybrid['energy']:.2f} mJ, {hybrid['latency']:.2f} ms")
    print(f"  GPU-only      : {gpu_only['energy']:.2f} mJ, {gpu_only['latency']:.2f} ms")
    saving = (gpu_only['energy'] - hybrid['energy']) / gpu_only['energy'] * 100
    print(f"\n  Enerji farkı (GPU-only'ye göre): %{saving:.1f}")


if __name__ == "__main__":
    test_pim_core()
    test_simple_cnn()
    test_pim_cluster()
    test_advanced_hybrid()
    test_adaptive_qlearning()
    test_integrated_qlearning_hybrid()
    test_full_alexnet()
