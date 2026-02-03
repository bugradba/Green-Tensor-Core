"""
Eco-PIM Görselleştirme - Gerçek Proje Verileri İle Güncellenmiş

Bu script projenizin GERÇEK sonuçlarını görselleştirir.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Stil ayarları
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 14

class EcoPIMVisualizerUpdated:
    """Gerçek proje verilerini görselleştir"""
    
    def __init__(self):
        self.colors = {
            'pim': '#2ecc71',
            'gpu': '#e74c3c',
            'hybrid': '#3498db',
            'pim_4bit': '#27ae60',
            'accent': '#f39c12'
        }
    
    def create_summary_dashboard(self, save_path='eco_pim_summary.png'):
        """
        Tüm önemli metrikleri tek sayfada göster
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. Enerji Karşılaştırması (üst sol, geniş)
        ax1 = fig.add_subplot(gs[0, :2])
        systems = ['PIM\n8-bit', 'PIM\n4-bit', 'GPU', 'Hybrid\nAlexNet']
        energies = [211.38, 186.57, 332.09, 313.50]
        colors = [self.colors['pim'], self.colors['pim_4bit'], 
                 self.colors['gpu'], self.colors['hybrid']]
        
        bars = ax1.bar(systems, energies, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        # Tasarruf yüzdelerini ekle
        savings = [36.3, 43.8, 0, 33.3]  # GPU'ya göre
        for bar, energy, saving in zip(bars, energies, savings):
            ax1.text(bar.get_x() + bar.get_width()/2., energy,
                    f'{energy:.0f} mJ',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
            if saving > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., energy * 0.5,
                        f'↓{saving:.1f}%',
                        ha='center', va='center', fontsize=9,
                        color='white', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
        
        ax1.set_ylabel('Enerji (mJ)', fontweight='bold', fontsize=11)
        ax1.set_title('Enerji Tüketimi Karşılaştırması', fontweight='bold', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 4-bit vs 8-bit (üst sağ)
        ax2 = fig.add_subplot(gs[0, 2])
        categories = ['8-bit', '4-bit']
        cluster_energies = [19.44, 4.32]  # pJ
        
        bars2 = ax2.bar(categories, cluster_energies, 
                       color=[self.colors['pim'], self.colors['pim_4bit']],
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, energy in zip(bars2, cluster_energies):
            ax2.text(bar.get_x() + bar.get_width()/2., energy,
                    f'{energy:.2f} pJ',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax2.text(1, cluster_energies[1], '↓77.8%',
                ha='center', va='top', fontsize=9, fontweight='bold',
                color=self.colors['pim_4bit'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_ylabel('Enerji (pJ)', fontweight='bold', fontsize=10)
        ax2.set_title('4-bit Quantization Etkisi', fontweight='bold', fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. AlexNet Layer Distribution (orta sol)
        ax3 = fig.add_subplot(gs[1, 0])
        layer_types = ['PIM\n(85%)', 'GPU\n(15%)']
        layer_counts = [17, 3]
        layer_colors = [self.colors['pim'], self.colors['gpu']]
        
        wedges, texts, autotexts = ax3.pie(layer_counts, labels=layer_types,
                                           autopct='%d', colors=layer_colors,
                                           startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'},
                                           wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
        
        ax3.set_title('AlexNet Katman Dağılımı\n(20 Katman)', fontweight='bold', fontsize=11)
        
        # 4. Energy Breakdown (orta orta)
        ax4 = fig.add_subplot(gs[1, 1])
        energy_breakdown = ['PIM\n96.2%', 'GPU\n3.8%']
        energy_values = [301.63, 11.86]
        
        bars4 = ax4.barh(energy_breakdown, energy_values,
                        color=[self.colors['pim'], self.colors['gpu']],
                        alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, energy in zip(bars4, energy_values):
            ax4.text(energy, bar.get_y() + bar.get_height()/2.,
                    f'{energy:.1f} mJ',
                    ha='left', va='center', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax4.set_xlabel('Enerji (mJ)', fontweight='bold', fontsize=10)
        ax4.set_title('AlexNet Enerji Breakdown', fontweight='bold', fontsize=11)
        ax4.grid(axis='x', alpha=0.3)
        
        # 5. Gecikme Karşılaştırması (orta sağ)
        ax5 = fig.add_subplot(gs[1, 2])
        systems_lat = ['PIM', 'GPU', 'Hybrid']
        latencies = [41.02, 6.95, 58.88]
        colors_lat = [self.colors['pim'], self.colors['gpu'], self.colors['hybrid']]
        
        bars5 = ax5.bar(systems_lat, latencies, color=colors_lat, alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        
        for bar, latency in zip(bars5, latencies):
            ax5.text(bar.get_x() + bar.get_width()/2., latency,
                    f'{latency:.1f} ms',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax5.set_ylabel('Gecikme (ms)', fontweight='bold', fontsize=10)
        ax5.set_title('Gecikme Karşılaştırması', fontweight='bold', fontsize=11)
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Key Metrics (alt, geniş)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Metrik kutuları
        metrics = [
            ('Enerji Tasarrufu\n(PIM vs GPU)', '36.3%', self.colors['pim']),
            ('4-bit Tasarrufu', '77.8%', self.colors['pim_4bit']),
            ('AlexNet Tasarrufu', '33.3%', self.colors['hybrid']),
            ('4-bit Doğruluk', '100%', self.colors['accent']),
            ('PIM Katman Oranı', '85%', self.colors['pim']),
        ]
        
        x_positions = np.linspace(0.1, 0.9, len(metrics))
        
        for (x, (title, value, color)) in zip(x_positions, metrics):
            # Kutu
            rect = plt.Rectangle((x - 0.08, 0.3), 0.16, 0.4,
                                facecolor=color, alpha=0.2,
                                edgecolor=color, linewidth=2,
                                transform=ax6.transAxes)
            ax6.add_patch(rect)
            
            # Değer
            ax6.text(x, 0.55, value,
                    ha='center', va='center', fontsize=20, fontweight='bold',
                    color=color, transform=ax6.transAxes)
            
            # Başlık
            ax6.text(x, 0.35, title,
                    ha='center', va='center', fontsize=9,
                    color='black', transform=ax6.transAxes)
        
        # Ana başlık
        fig.suptitle('Eco-PIM: AI-Powered Energy Optimization Results',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard kaydedildi: {save_path}")
        plt.show()
    
    def create_qlearning_analysis(self, save_path='qlearning_analysis.png'):
        """
        Q-Learning eğitim sürecini görselleştir
        (Gerçek verilerle - negatif reward'lar dahil)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Training Progress (gerçek veriler)
        episodes = list(range(1, 21))
        rewards = [-4377.33, -2566.00, -671.00, -1657.33, -932.00,
                   -1819.67, -1312.33, -1484.00, -2420.33, -1384.33,
                   -901.00, -1002.33, -3292.33, -4297.67, -1112.67,
                   -2014.33, -374.67, -538.33, -1906.67, -2240.00]
        
        ax1.plot(episodes, rewards, 'o-', color=self.colors['gpu'],
                linewidth=2, markersize=6, label='Episode Reward')
        
        # Trend line
        z = np.polyfit(episodes, rewards, 2)
        p = np.poly1d(z)
        ax1.plot(episodes, p(episodes), '--', color=self.colors['accent'],
                linewidth=2, label='Trend')
        
        ax1.axhline(y=np.mean(rewards), color='gray', linestyle=':',
                   linewidth=1.5, label=f'Avg: {np.mean(rewards):.0f}')
        
        ax1.set_xlabel('Episode', fontweight='bold')
        ax1.set_ylabel('Reward', fontweight='bold')
        ax1.set_title('Q-Learning Training Progress\n(Negatif = Reward fonksiyonu dengesiz)',
                     fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Table Growth
        q_states = [2, 3, 5, 5, 6, 6, 7, 7, 8, 8,
                   8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        
        ax2.plot(episodes, q_states, 's-', color=self.colors['pim'],
                linewidth=2, markersize=6)
        ax2.fill_between(episodes, 0, q_states, alpha=0.3, color=self.colors['pim'])
        
        ax2.set_xlabel('Episode', fontweight='bold')
        ax2.set_ylabel('Q-Table Size (states)', fontweight='bold')
        ax2.set_title('State Space Exploration', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Decision Distribution
        decisions = {'PIM': 5, 'GPU': 2, 'HYBRID': 1}
        colors_dec = [self.colors['pim'], self.colors['gpu'], self.colors['hybrid']]
        
        wedges, texts, autotexts = ax3.pie(decisions.values(), labels=decisions.keys(),
                                           autopct='%d', colors=colors_dec,
                                           startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'},
                                           wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
        
        ax3.set_title('Q-Learning Decision Distribution\n(Test Phase)', fontweight='bold')
        
        # 4. Comparison: Q-Learning vs Rule-Based
        scenarios = ['100K\nConv', '5M\nConv', '10M\nFC']
        q_decisions = ['GPU', 'PIM', 'PIM']
        r_decisions = ['HYBRID', 'HYBRID', 'HYBRID']
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        # Q-Learning bars
        q_colors = []
        for d in q_decisions:
            if d == 'PIM':
                q_colors.append(self.colors['pim'])
            elif d == 'GPU':
                q_colors.append(self.colors['gpu'])
            else:
                q_colors.append(self.colors['hybrid'])
        
        bars1 = ax4.bar(x - width/2, [1]*len(scenarios), width,
                       color=q_colors, alpha=0.8, label='Q-Learning',
                       edgecolor='black', linewidth=1.5)
        
        # Rule-based bars
        bars2 = ax4.bar(x + width/2, [1]*len(scenarios), width,
                       color=self.colors['hybrid'], alpha=0.8, label='Rule-Based',
                       edgecolor='black', linewidth=1.5)
        
        # Labels
        for i, (bar1, bar2, q_d, r_d) in enumerate(zip(bars1, bars2, q_decisions, r_decisions)):
            ax4.text(bar1.get_x() + bar1.get_width()/2., 0.5,
                    q_d, ha='center', va='center', fontsize=9,
                    fontweight='bold', rotation=90, color='white')
            ax4.text(bar2.get_x() + bar2.get_width()/2., 0.5,
                    r_d, ha='center', va='center', fontsize=9,
                    fontweight='bold', rotation=90, color='white')
        
        ax4.set_ylabel('Decision', fontweight='bold')
        ax4.set_title('Q-Learning vs Rule-Based\n(3/8 farklı)', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(scenarios)
        ax4.set_yticks([])
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Q-Learning Adaptive Scheduler Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Q-Learning analizi kaydedildi: {save_path}")
        plt.show()
    
    def create_alexnet_breakdown(self, save_path='alexnet_breakdown.png'):
        """
        AlexNet katman bazlı detaylı analiz
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Layer-wise energy (sadece önemli katmanlar)
        layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']
        energies = [211.38, 41.87, 13.79, 20.69, 13.79, 7.64, 3.40, 0.83]
        devices = ['PIM', 'PIM', 'PIM', 'PIM', 'PIM', 'GPU', 'GPU', 'GPU']
        
        colors_layer = [self.colors['pim'] if d == 'PIM' else self.colors['gpu'] 
                       for d in devices]
        
        bars = ax1.barh(layers, energies, color=colors_layer, alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        
        for bar, energy, device in zip(bars, energies, devices):
            ax1.text(energy, bar.get_y() + bar.get_height()/2.,
                    f' {energy:.1f} mJ ({device})',
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('Enerji (mJ)', fontweight='bold', fontsize=11)
        ax1.set_title('Katman Bazlı Enerji Tüketimi', fontweight='bold', fontsize=12)
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Cumulative energy
        cumulative = np.cumsum(energies)
        
        ax2.plot(range(1, len(layers)+1), cumulative, 'o-',
                color=self.colors['hybrid'], linewidth=2.5, markersize=8)
        ax2.fill_between(range(1, len(layers)+1), 0, cumulative,
                        alpha=0.3, color=self.colors['hybrid'])
        
        # PIM/GPU bölgelerini vurgula
        ax2.axvline(x=5.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax2.text(2.5, cumulative[-1] * 0.9, 'PIM Zone\n(Conv Layers)',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=self.colors['pim'], alpha=0.3))
        ax2.text(7, cumulative[-1] * 0.5, 'GPU Zone\n(FC Layers)',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=self.colors['gpu'], alpha=0.3))
        
        ax2.set_xlabel('Katman Numarası', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Kümülatif Enerji (mJ)', fontweight='bold', fontsize=11)
        ax2.set_title('Kümülatif Enerji Profili', fontweight='bold', fontsize=12)
        ax2.set_xticks(range(1, len(layers)+1))
        ax2.set_xticklabels(layers, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('AlexNet-20 Detaylı Enerji Analizi',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ AlexNet breakdown kaydedildi: {save_path}")
        plt.show()


# ANA ÇALIŞTIRMA
if __name__ == "__main__":
    print("📊 Eco-PIM Görselleştirme (Gerçek Verilerle)")
    print("="*70)
    
    viz = EcoPIMVisualizerUpdated()
    
    print("\n1️⃣ Özet dashboard oluşturuluyor...")
    viz.create_summary_dashboard()
    
    print("\n2️⃣ Q-Learning analizi oluşturuluyor...")
    viz.create_qlearning_analysis()
    
    print("\n3️⃣ AlexNet breakdown oluşturuluyor...")
    viz.create_alexnet_breakdown()
    
    print("\n✅ Tüm grafikler oluşturuldu!")
    print("\n📁 Oluşturulan dosyalar:")
    print("  - eco_pim_summary.png (ana dashboard)")
    print("  - qlearning_analysis.png (Q-Learning analizi)")
    print("  - alexnet_breakdown.png (AlexNet detayları)")