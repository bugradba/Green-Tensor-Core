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
    """Cluster MAC işlemini test et."""
    from pim_simulator import PIMCluster
    
    # Cluster'ı başlat
    cluster = PIMCluster()

    # Test değerleri
    a, b = 255, 200

    # 1. Adım: 8-bit Hassasiyet (Precision = 8)
    result_8bit, energy_8bit, latency_8bit = cluster.mac_8bit(a, b, precision=8)

    # 2. Adım: 4-bit Hassasiyet (Precision = 4 - Precision Scaling)
    result_4bit, energy_4bit, latency_4bit = cluster.mac_8bit(a, b, precision=4)

    # Sonuçları Ekrana Yazdır
    print("---- PIM Cluster Test ----")
    print(f"{a} x {b} (8-bit): {result_8bit}")
    print(f"  Enerji: {energy_8bit:.2f} pJ, Gecikme: {latency_8bit:.2f} ns")
    
    print(f"{a} x {b} (4-bit approx): {result_4bit}")
    print(f"  Enerji: {energy_4bit:.2f} pJ, Gecikme: {latency_4bit:.2f} ns")
    
    # Tasarruf Hesabı
    if energy_8bit > 0:
        savings = (1 - energy_4bit / energy_8bit) * 100
        print(f"  Tasarruf: %{savings:.1f}")
    
    print()

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

if __name__ == "__main__":
    # Diğer testler...
    test_advanced_hybrid() # <-- Bunu eklemeyi unutma

# Ana test fonksiyonuna ekleyin
if __name__ == "__main__":
    test_pim_core()
    test_simple_cnn() 
    test_pim_cluster()
    test_advanced_hybrid() 