"""
Eco-PIM Görselleştirme Modülü

Bu modül projenin tüm sonuçlarını profesyonel grafiklerle görselleştirir:
1. Q-Learning eğitim süreci
2. Enerji karşılaştırması
3. Gecikme vs Enerji trade-off
4. 4-bit vs 8-bit karşılaştırma
5. Hibrit sistem breakdown
6. Dashboard (tüm grafikler tek sayfada)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

# Seaborn stil ayarları
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.titlesize'] = 14

class EcoPIMVisualizer:
    """Eco-PIM projesinin tüm görselleştirmelerini oluşturur"""
    
    def __init__(self):
        self.colors = {
            'pim': '#2ecc71',      # Yeşil
            'gpu': '#e74c3c',      # Kırmızı
            'hybrid': '#3498db',   # Mavi
            'pim_4bit': '#27ae60', # Koyu yeşil
            'accent': '#f39c12'    # Turuncu
        }
    
    def plot_qlearning_training(self, episodes, rewards, save_path='qlearning_training.png'):
        """
        Q-Learning eğitim sürecini görselleştir
        
        Args:
            episodes: [1, 2, 3, ..., 20]
            rewards: [510, 643, 683, ...]
            save_path: Kayıt yolu
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Sol grafik: Episode rewards
        ax1.plot(episodes, rewards, 'o-', color=self.colors['pim'], 
                linewidth=2.5, markersize=8, label='Episode Reward')
        
        # Trend line (moving average)
        if len(rewards) >= 3:
            window = min(3, len(rewards))
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(episodes[window-1:], moving_avg, '--', 
                    color=self.colors['accent'], linewidth=2, label='Trend (MA-3)')
        
        ax1.axhline(y=np.mean(rewards), color='gray', linestyle=':', 
                   linewidth=1.5, label=f'Average: {np.mean(rewards):.1f}')
        
        ax1.set_xlabel('Episode', fontweight='bold')
        ax1.set_ylabel('Total Reward', fontweight='bold')
        ax1.set_title('Q-Learning Training Progress', fontweight='bold', pad=15)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Sağ grafik: Cumulative reward
        cumulative = np.cumsum(rewards)
        ax2.fill_between(episodes, 0, cumulative, alpha=0.3, color=self.colors['pim'])
        ax2.plot(episodes, cumulative, '-', color=self.colors['pim'], linewidth=2.5)
        
        ax2.set_xlabel('Episode', fontweight='bold')
        ax2.set_ylabel('Cumulative Reward', fontweight='bold')
        ax2.set_title('Learning Accumulation', fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3)
        
        # Final değerleri ekle
        final_reward = rewards[-1]
        final_cumulative = cumulative[-1]
        ax1.text(episodes[-1], final_reward, f'{final_reward:.0f}', 
                ha='left', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(episodes[-1], final_cumulative, f'{final_cumulative:.0f}', 
                ha='left', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Grafik kaydedildi: {save_path}")
        plt.show()
    
    def plot_energy_comparison(self, pim_8bit, pim_4bit, gpu, hybrid, 
                               save_path='energy_comparison.png'):
        """
        Enerji tüketimi karşılaştırması (bar chart)
        
        Args:
            pim_8bit: PIM 8-bit enerji (mJ)
            pim_4bit: PIM 4-bit enerji (mJ)
            gpu: GPU enerji (mJ)
            hybrid: Hibrit enerji (mJ)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        systems = ['PIM\n(8-bit)', 'PIM\n(4-bit)', 'GPU\n(Baseline)', 'Hybrid\n(Q-Learning)']
        energies = [pim_8bit, pim_4bit, gpu, hybrid]
        colors = [self.colors['pim'], self.colors['pim_4bit'], 
                 self.colors['gpu'], self.colors['hybrid']]
        
        bars = ax.bar(systems, energies, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        
        # Değerleri bar üzerine yaz
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{energy:.1f} mJ',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Tasarruf yüzdelerini ekle
        savings_vs_gpu = [(gpu - e) / gpu * 100 for e in energies]
        for i, (bar, saving) in enumerate(zip(bars, savings_vs_gpu)):
            if saving > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 0.5,
                       f'↓{saving:.1f}%',
                       ha='center', va='center', fontsize=10, 
                       color='white', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
        
        ax.set_ylabel('Enerji Tüketimi (mJ)', fontweight='bold', fontsize=12)
        ax.set_title('Sistem Enerji Karşılaştırması', fontweight='bold', fontsize=14, pad=20)
        ax.set_ylim(0, max(energies) * 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Grafik kaydedildi: {save_path}")
        plt.show()
    
    def plot_latency_vs_energy(self, results_dict, save_path='latency_vs_energy.png'):
        """
        Gecikme vs Enerji scatter plot (trade-off analizi)
        
        Args:
            results_dict: {
                'PIM': {'energy': 211, 'latency': 41},
                'GPU': {'energy': 332, 'latency': 6.95},
                'Hybrid': {'energy': 219, 'latency': 42.69}
            }
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for system, data in results_dict.items():
            energy = data['energy']
            latency = data['latency']
            
            if system == 'PIM':
                color = self.colors['pim']
                marker = 'o'
                size = 300
            elif system == 'GPU':
                color = self.colors['gpu']
                marker = 's'
                size = 300
            elif system == 'Hybrid':
                color = self.colors['hybrid']
                marker = '^'
                size = 350
            else:
                color = 'gray'
                marker = 'D'
                size = 250
            
            ax.scatter(latency, energy, s=size, c=color, marker=marker, 
                      alpha=0.7, edgecolor='black', linewidth=2, label=system)
            
            # Etiket ekle
            ax.annotate(f'{system}\n({energy:.0f} mJ, {latency:.1f} ms)',
                       xy=(latency, energy), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Pareto frontier çiz (ideal bölge)
        ax.axhline(y=min([d['energy'] for d in results_dict.values()]), 
                  color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Min Energy')
        ax.axvline(x=min([d['latency'] for d in results_dict.values()]), 
                  color='blue', linestyle=':', linewidth=1.5, alpha=0.5, label='Min Latency')
        
        ax.set_xlabel('Gecikme (ms)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Enerji (mJ)', fontweight='bold', fontsize=12)
        ax.set_title('Enerji vs Gecikme Trade-off Analizi', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # İdeal bölgeyi vurgula
        min_energy = min([d['energy'] for d in results_dict.values()])
        min_latency = min([d['latency'] for d in results_dict.values()])
        ax.fill_between([0, min_latency], 0, min_energy, alpha=0.1, color='green', label='Ideal Zone')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Grafik kaydedildi: {save_path}")
        plt.show()
    
    def plot_4bit_vs_8bit(self, bit_8_energy, bit_4_energy, 
                          bit_8_latency, bit_4_latency,
                          save_path='4bit_vs_8bit.png'):
        """
        4-bit vs 8-bit karşılaştırması (dual axis)
        
        Args:
            bit_8_energy, bit_4_energy: Enerji değerleri (pJ)
            bit_8_latency, bit_4_latency: Gecikme değerleri (ns)
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        x = ['8-bit\n(Tam Hassasiyet)', '4-bit\n(Quantized)']
        
        # Enerji barları (sol axis)
        energies = [bit_8_energy, bit_4_energy]
        bars1 = ax1.bar([0, 1], energies, width=0.4, align='edge',
                       color=self.colors['pim'], alpha=0.7, 
                       label='Enerji (pJ)', edgecolor='black', linewidth=1.5)
        
        ax1.set_ylabel('Enerji (pJ)', fontweight='bold', fontsize=12, color=self.colors['pim'])
        ax1.tick_params(axis='y', labelcolor=self.colors['pim'])
        ax1.set_xticks([0.2, 1.2])
        ax1.set_xticklabels(x, fontsize=11)
        
        # Enerji tasarrufu etiketi
        saving = (bit_8_energy - bit_4_energy) / bit_8_energy * 100
        ax1.text(1.2, bit_4_energy, f'↓{saving:.1f}%\nTasarruf',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=self.colors['pim'],
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=self.colors['pim'], linewidth=2))
        
        # Gecikme çizgisi (sağ axis)
        ax2 = ax1.twinx()
        latencies = [bit_8_latency, bit_4_latency]
        line = ax2.plot([0.2, 1.2], latencies, 'o-', 
                       color=self.colors['gpu'], linewidth=3, markersize=12,
                       label='Gecikme (ns)')
        
        ax2.set_ylabel('Gecikme (ns)', fontweight='bold', fontsize=12, color=self.colors['gpu'])
        ax2.tick_params(axis='y', labelcolor=self.colors['gpu'])
        
        # Gecikme iyileşmesi etiketi
        latency_improve = (bit_8_latency - bit_4_latency) / bit_8_latency * 100
        ax2.text(1.2, bit_4_latency, f'↓{latency_improve:.1f}%\nDaha Hızlı',
                ha='center', va='top', fontsize=10, fontweight='bold',
                color=self.colors['gpu'],
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=self.colors['gpu'], linewidth=2))
        
        # Değerleri ekle
        for bar, energy in zip(bars1, energies):
            ax1.text(bar.get_x() + bar.get_width()/2., energy,
                    f'{energy:.2f} pJ',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        for i, (x_pos, latency) in enumerate(zip([0.2, 1.2], latencies)):
            ax2.text(x_pos, latency, f'{latency:.2f} ns',
                    ha='center', va='top', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax1.set_title('4-bit Quantization Etkisi', fontweight='bold', fontsize=14, pad=20)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Grafik kaydedildi: {save_path}")
        plt.show()
    
    def plot_hybrid_breakdown(self, layer_results, save_path='hybrid_breakdown.png'):
        """
        Hibrit sistemin katman bazlı enerji dağılımı (pie chart + bar)
        
        Args:
            layer_results: [
                {'layer': 'conv1', 'type': 'Conv2D', 'device': 'PIM', 'energy': 211.38},
                {'layer': 'relu1', 'type': 'ReLU', 'device': 'PIM', 'energy': 0.01},
                {'layer': 'fc1', 'type': 'Linear', 'device': 'GPU', 'energy': 7.71}
            ]
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sol: Pie chart (enerji dağılımı)
        layers = [r['layer'] for r in layer_results]
        energies = [r['energy'] for r in layer_results]
        devices = [r['device'] for r in layer_results]
        
        colors_pie = [self.colors['pim'] if d == 'PIM' else self.colors['gpu'] 
                     for d in devices]
        
        wedges, texts, autotexts = ax1.pie(energies, labels=layers, autopct='%1.1f%%',
                                           colors=colors_pie, startangle=90,
                                           textprops={'fontsize': 10, 'fontweight': 'bold'},
                                           wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax1.set_title('Katman Bazlı Enerji Dağılımı', fontweight='bold', fontsize=12, pad=15)
        
        # Sağ: Bar chart (device breakdown)
        pim_total = sum([r['energy'] for r in layer_results if r['device'] == 'PIM'])
        gpu_total = sum([r['energy'] for r in layer_results if r['device'] == 'GPU'])
        
        devices_unique = ['PIM', 'GPU']
        device_energies = [pim_total, gpu_total]
        device_colors = [self.colors['pim'], self.colors['gpu']]
        
        bars = ax2.barh(devices_unique, device_energies, color=device_colors, 
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, energy in zip(bars, device_energies):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{energy:.2f} mJ\n({energy/(pim_total+gpu_total)*100:.1f}%)',
                    ha='left', va='center', fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Toplam Enerji (mJ)', fontweight='bold', fontsize=11)
        ax2.set_title('Cihaz Bazlı Toplam Enerji', fontweight='bold', fontsize=12, pad=15)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Grafik kaydedildi: {save_path}")
        plt.show()
    
    def create_full_dashboard(self, all_data, save_path='dashboard.png'):
        """
        Tüm metrikleri tek sayfada gösteren dashboard
        
        Args:
            all_data: Dict içinde tüm veriler
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Q-Learning Training (üst sol)
        ax1 = fig.add_subplot(gs[0, :2])
        episodes = all_data['qlearning']['episodes']
        rewards = all_data['qlearning']['rewards']
        ax1.plot(episodes, rewards, 'o-', color=self.colors['pim'], linewidth=2, markersize=6)
        ax1.set_title('Q-Learning Training', fontweight='bold', fontsize=11)
        ax1.set_xlabel('Episode', fontsize=9)
        ax1.set_ylabel('Reward', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Enerji Karşılaştırması (üst sağ)
        ax2 = fig.add_subplot(gs[0, 2])
        systems = ['PIM\n8-bit', 'GPU']
        energies = [all_data['energy']['pim_8bit'], all_data['energy']['gpu']]
        bars = ax2.bar(systems, energies, color=[self.colors['pim'], self.colors['gpu']], alpha=0.7)
        ax2.set_title('Enerji', fontweight='bold', fontsize=11)
        ax2.set_ylabel('mJ', fontsize=9)
        for bar, e in zip(bars, energies):
            ax2.text(bar.get_x() + bar.get_width()/2., e, f'{e:.0f}', 
                    ha='center', va='bottom', fontsize=8)
        
        # 3. Gecikme Karşılaştırması (orta sol)
        ax3 = fig.add_subplot(gs[1, 0])
        latencies = [all_data['latency']['pim'], all_data['latency']['gpu']]
        ax3.barh(systems, latencies, color=[self.colors['pim'], self.colors['gpu']], alpha=0.7)
        ax3.set_title('Gecikme', fontweight='bold', fontsize=11)
        ax3.set_xlabel('ms', fontsize=9)
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. 4-bit vs 8-bit (orta orta)
        ax4 = fig.add_subplot(gs[1, 1])
        bits = ['8-bit', '4-bit']
        bit_energies = [all_data['precision']['8bit'], all_data['precision']['4bit']]
        ax4.bar(bits, bit_energies, color=[self.colors['pim'], self.colors['pim_4bit']], alpha=0.7)
        ax4.set_title('Precision Impact', fontweight='bold', fontsize=11)
        ax4.set_ylabel('pJ', fontsize=9)
        saving = (bit_energies[0] - bit_energies[1]) / bit_energies[0] * 100
        ax4.text(1, bit_energies[1], f'↓{saving:.0f}%', ha='center', va='bottom', fontsize=9)
        
        # 5. Hibrit Breakdown (orta sağ)
        ax5 = fig.add_subplot(gs[1, 2])
        layer_data = all_data['hybrid']['layers']
        layers = [l['layer'] for l in layer_data]
        layer_energies = [l['energy'] for l in layer_data]
        ax5.pie(layer_energies, labels=layers, autopct='%1.0f%%', textprops={'fontsize': 8})
        ax5.set_title('Layer Breakdown', fontweight='bold', fontsize=11)
        
        # 6. Trade-off Scatter (alt, geniş)
        ax6 = fig.add_subplot(gs[2, :])
        for system in ['PIM', 'GPU', 'Hybrid']:
            e = all_data['tradeoff'][system]['energy']
            l = all_data['tradeoff'][system]['latency']
            color = self.colors[system.lower()]
            ax6.scatter(l, e, s=200, c=color, alpha=0.7, edgecolor='black', linewidth=2, label=system)
            ax6.text(l, e, f'  {system}', fontsize=9, fontweight='bold')
        
        ax6.set_xlabel('Latency (ms)', fontsize=10, fontweight='bold')
        ax6.set_ylabel('Energy (mJ)', fontsize=10, fontweight='bold')
        ax6.set_title('Energy vs Latency Trade-off', fontweight='bold', fontsize=12)
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
        
        # Ana başlık
        fig.suptitle('Eco-PIM Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard kaydedildi: {save_path}")
        plt.show()


# Kullanım örneği
if __name__ == "__main__":
    print("📊 Eco-PIM Görselleştirme Başlatılıyor...\n")
    
    viz = EcoPIMVisualizer()
    
    # 1. Q-Learning Training
    print("1️⃣ Q-Learning eğitim grafiği...")
    episodes = list(range(1, 21))
    rewards = [510, 643, 683, 703, 803, 503, 780, 646, 620, 673,
               690, 720, 750, 680, 710, 730, 690, 710, 720, 700]
    viz.plot_qlearning_training(episodes, rewards)
    
    # 2. Enerji Karşılaştırması
    print("\n2️⃣ Enerji karşılaştırma grafiği...")
    viz.plot_energy_comparison(
        pim_8bit=211.38,
        pim_4bit=186.57,
        gpu=332.09,
        hybrid=219.10
    )
    
    # 3. Latency vs Energy
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
    viz.plot_4bit_vs_8bit(
        bit_8_energy=19.44,
        bit_4_energy=4.32,
        bit_8_latency=6.40,
        bit_4_latency=3.20
    )
    
    # 5. Hibrit Breakdown
    print("\n5️⃣ Hibrit sistem breakdown...")
    layer_results = [
        {'layer': 'Conv1', 'type': 'Conv2D', 'device': 'PIM', 'energy': 211.38},
        {'layer': 'ReLU1', 'type': 'ReLU', 'device': 'PIM', 'energy': 0.01},
        {'layer': 'FC1', 'type': 'Linear', 'device': 'GPU', 'energy': 7.71}
    ]
    viz.plot_hybrid_breakdown(layer_results)
    
    # 6. Full Dashboard
    print("\n6️⃣ Tam dashboard oluşturuluyor...")
    all_data = {
        'qlearning': {
            'episodes': episodes,
            'rewards': rewards
        },
        'energy': {
            'pim_8bit': 211.38,
            'gpu': 332.09
        },
        'latency': {
            'pim': 41.02,
            'gpu': 6.95
        },
        'precision': {
            '8bit': 19.44,
            '4bit': 4.32
        },
        'hybrid': {
            'layers': layer_results
        },
        'tradeoff': {
            'PIM': {'energy': 211.38, 'latency': 41.02},
            'GPU': {'energy': 332.09, 'latency': 6.95},
            'Hybrid': {'energy': 219.10, 'latency': 42.69}
        }
    }
    viz.create_full_dashboard(all_data)
    
    print("\n✅ Tüm grafikler oluşturuldu!")
    print("\n📁 Oluşturulan dosyalar:")
    print("  - qlearning_training.png")
    print("  - energy_comparison.png")
    print("  - latency_vs_energy.png")
    print("  - 4bit_vs_8bit.png")
    print("  - hybrid_breakdown.png")
    print("  - dashboard.png")