"""
Eco-PIM Görselleştirme

Tüm değerler results.compute_results() ile simülatörden üretilir;
grafiklerde elle girilmiş sayı yoktur.

Çalıştırma:  python visualize_scheduler.py
"""
import os

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from results import compute_results

plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 14

_HERE = os.path.dirname(os.path.abspath(__file__))

STRATEGY_LABELS = {
    'gpu_only': 'GPU-only',
    'pim_only': 'PIM-only',
    'rule_based': 'Kural tabanlı',
    'qlearning': 'Q-Learning',
    'oracle_min_energy': 'Oracle\n(min enerji)',
}


class EcoPIMVisualizerUpdated:
    """Simülatör sonuçlarını görselleştir"""

    def __init__(self, results=None, show=False):
        self.r = results or compute_results()
        self.show = show
        self.colors = {
            'pim': '#2ecc71',
            'gpu': '#e74c3c',
            'hybrid': '#3498db',
            'pim_4bit': '#27ae60',
            'accent': '#f39c12',
        }

    def _device_color(self, d):
        return {'PIM': self.colors['pim'], 'GPU': self.colors['gpu']}.get(d, self.colors['hybrid'])

    def _finish(self, fig, save_path):
        path = save_path if os.path.isabs(save_path) else os.path.join(_HERE, save_path)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"✅ Kaydedildi: {path}")
        if self.show:
            plt.show()
        plt.close(fig)
        return path

    def create_summary_dashboard(self, save_path='eco_pim_summary.png'):
        r = self.r
        c1, mac, alex = r['conv1'], r['mac'], r['alexnet']
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

        # 1. AlexNet strateji enerjileri
        ax1 = fig.add_subplot(gs[0, :2])
        keys = [k for k in STRATEGY_LABELS if k in alex]
        energies = [alex[k]['energy'] for k in keys]
        base = alex['gpu_only']['energy']
        bars = ax1.bar([STRATEGY_LABELS[k] for k in keys], energies,
                       color=[self.colors['gpu'], self.colors['pim'], self.colors['accent'],
                              self.colors['hybrid'], 'gray'][:len(keys)],
                       alpha=0.85, edgecolor='black')
        for bar, e in zip(bars, energies):
            saving = (1 - e / base) * 100
            label = f'{e:.1f} mJ' + (f'\n(%{saving:.0f} tasarruf)' if abs(saving) > 0.5 else '\n(referans)')
            ax1.text(bar.get_x() + bar.get_width() / 2, e, label,
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.set_ylabel('Enerji (mJ)')
        ax1.set_title('AlexNet toplam enerji (GPU-only\'ye göre fark)', fontweight='bold')
        ax1.set_ylim(0, max(energies) * 1.25)

        # 2. Tek MAC: 8-bit vs 4-bit
        ax2 = fig.add_subplot(gs[0, 2])
        vals = [mac['energy_8bit_pj'], mac['energy_4bit_pj']]
        ax2.bar(['8-bit', '4-bit'], vals, color=[self.colors['pim'], self.colors['pim_4bit']],
                edgecolor='black')
        for i, v in enumerate(vals):
            ax2.text(i, v, f'{v:.2f} pJ', ha='center', va='bottom', fontweight='bold')
        ax2.set_title(f"MAC enerjisi\n4-bit ort. hata: %{mac['mean_rel_error_4bit']*100:.1f}",
                      fontweight='bold')
        ax2.set_ylabel('pJ / MAC')
        ax2.set_ylim(0, max(vals) * 1.2)

        # 3. conv1 enerji
        ax3 = fig.add_subplot(gs[1, 0])
        names = ['PIM 8-bit', 'PIM 4-bit', 'GPU']
        ce = [c1['pim8']['energy_total_mj'], c1['pim4']['energy_total_mj'],
              c1['gpu']['total_energy_mj']]
        ax3.bar(names, ce, color=[self.colors['pim'], self.colors['pim_4bit'], self.colors['gpu']],
                edgecolor='black')
        for i, v in enumerate(ce):
            ax3.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        ax3.set_title('conv1 enerji (mJ)', fontweight='bold')

        # 4. conv1 gecikme
        ax4 = fig.add_subplot(gs[1, 1])
        cl = [c1['pim8']['latency_ms'], c1['pim4']['latency_ms'], c1['gpu']['total_latency_ms']]
        ax4.bar(names, cl, color=[self.colors['pim'], self.colors['pim_4bit'], self.colors['gpu']],
                edgecolor='black')
        for i, v in enumerate(cl):
            ax4.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        ax4.set_title('conv1 gecikme (ms)', fontweight='bold')

        # 5. Strateji gecikmeleri
        ax5 = fig.add_subplot(gs[1, 2])
        lat = [alex[k]['latency'] for k in keys]
        ax5.barh([STRATEGY_LABELS[k].replace('\n', ' ') for k in keys], lat,
                 color=self.colors['hybrid'], edgecolor='black')
        ax5.set_xlabel('ms')
        ax5.set_title('AlexNet toplam gecikme', fontweight='bold')

        # 6. Metrik kutuları
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        q = alex.get('qlearning', alex['rule_based'])
        pim_layers = sum(1 for l in q['layers'] if l['device'] == 'PIM')
        metrics = [
            ('conv1 PIM vs GPU\nenerji', f"{(1 - ce[0]/ce[2])*100:+.0f}%", self.colors['pim']),
            ('4-bit MAC\nenerji', f"{(1 - vals[1]/vals[0])*100:+.0f}%", self.colors['pim_4bit']),
            ('Q-Learning vs GPU-only\nenerji', f"{(1 - q['energy']/base)*100:+.0f}%", self.colors['hybrid']),
            ('Q-Learning vs GPU-only\ngecikme', f"{alex['gpu_only']['latency']/q['latency']:.2f}×", self.colors['accent']),
            ('Q-Learning PIM\nkatmanları', f"{pim_layers}/{len(q['layers'])}", self.colors['pim']),
        ]
        for x, (title, value, color) in zip(np.linspace(0.1, 0.9, len(metrics)), metrics):
            ax6.add_patch(plt.Rectangle((x - 0.08, 0.2), 0.16, 0.6, facecolor=color, alpha=0.15,
                                        edgecolor=color, linewidth=2, transform=ax6.transAxes))
            ax6.text(x, 0.6, value, ha='center', va='center', fontsize=18, fontweight='bold',
                     color=color, transform=ax6.transAxes)
            ax6.text(x, 0.32, title, ha='center', va='center', fontsize=9, transform=ax6.transAxes)
        ax6.text(0.5, 0.02, "Yüzdeler tasarruftur (pozitif = daha az enerji). Değerler simülasyon çıktısıdır (results.py).",
                 ha='center', fontsize=8, color='gray', transform=ax6.transAxes)

        fig.suptitle('Eco-PIM: Simülasyon Sonuçları', fontsize=16, fontweight='bold')
        return self._finish(fig, save_path)

    def create_qlearning_analysis(self, save_path='qlearning_analysis.png'):
        ql = self.r['qlearning']
        rewards, q_states = ql['episode_rewards'], ql['q_states']
        episodes = np.arange(1, len(rewards) + 1)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        ax1.plot(episodes, rewards, 'o-', color=self.colors['hybrid'], label='Episode reward')
        if len(rewards) >= 5:
            k = 5
            ma = np.convolve(rewards, np.ones(k) / k, mode='valid')
            ax1.plot(episodes[k - 1:], ma, '--', color=self.colors['accent'], label=f'{k}-episode ort.')
        ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward')
        ax1.set_title('Q-Learning eğitimi (AlexNet katmanları)', fontweight='bold')
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(episodes, q_states, 's-', color=self.colors['pim'])
        ax2.set_xlabel('Episode'); ax2.set_ylabel('Q-table state sayısı')
        ax2.set_title('State uzayı keşfi', fontweight='bold'); ax2.grid(alpha=0.3)

        q_layers = self.r['alexnet']['qlearning']['layers']
        counts = {}
        for l in q_layers:
            counts[l['device']] = counts.get(l['device'], 0) + 1
        ax3.pie(list(counts.values()), labels=list(counts.keys()), autopct='%d',
                colors=[self._device_color(d) for d in counts],
                wedgeprops={'edgecolor': 'black'})
        ax3.set_title('Q-Learning kararları (AlexNet)', fontweight='bold')

        alex = self.r['alexnet']
        keys = [k for k in ('rule_based', 'qlearning', 'oracle_min_energy') if k in alex]
        x = np.arange(len(keys))
        e = [alex[k]['energy'] for k in keys]
        l = [alex[k]['latency'] for k in keys]
        ax4.bar(x - 0.2, e, 0.4, label='Enerji (mJ)', color=self.colors['pim'])
        ax4b = ax4.twinx()
        ax4b.bar(x + 0.2, l, 0.4, label='Gecikme (ms)', color=self.colors['gpu'])
        ax4.set_xticks(x); ax4.set_xticklabels([STRATEGY_LABELS[k].replace('\n', ' ') for k in keys])
        ax4.set_ylabel('mJ'); ax4b.set_ylabel('ms')
        ax4.set_title('Strateji karşılaştırması', fontweight='bold')
        ax4.legend(loc='upper left'); ax4b.legend(loc='upper right')

        fig.suptitle('Q-Learning Adaptive Scheduler Analizi', fontsize=14, fontweight='bold')
        fig.tight_layout()
        return self._finish(fig, save_path)

    def create_alexnet_breakdown(self, save_path='alexnet_breakdown.png', strategy='qlearning'):
        plan = self.r['alexnet'][strategy]['layers']
        plan = [p for p in plan if p['type'] in ('Conv2D', 'Linear')]
        names = [p['layer'] for p in plan]
        energies = [p['energy'] for p in plan]
        devices = [p['device'] for p in plan]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        bars = ax1.barh(names, energies, color=[self._device_color(d) for d in devices],
                        edgecolor='black')
        for bar, e, d in zip(bars, energies, devices):
            ax1.text(e, bar.get_y() + bar.get_height() / 2, f' {e:.2f} mJ ({d})',
                     va='center', fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('Enerji (mJ)')
        ax1.set_title('Katman bazlı enerji (Conv/FC)', fontweight='bold')

        cum = np.cumsum(energies)
        ax2.plot(range(1, len(names) + 1), cum, 'o-', color=self.colors['hybrid'])
        ax2.fill_between(range(1, len(names) + 1), 0, cum, alpha=0.3, color=self.colors['hybrid'])
        ax2.set_xticks(range(1, len(names) + 1)); ax2.set_xticklabels(names, rotation=45)
        ax2.set_ylabel('Kümülatif enerji (mJ)')
        ax2.set_title('Kümülatif enerji profili', fontweight='bold'); ax2.grid(alpha=0.3)

        fig.suptitle(f'AlexNet enerji analizi — {STRATEGY_LABELS[strategy]}',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        return self._finish(fig, save_path)


EcoPIMVisualizer = EcoPIMVisualizerUpdated


if __name__ == "__main__":
    viz = EcoPIMVisualizerUpdated()
    viz.create_summary_dashboard()
    viz.create_qlearning_analysis()
    viz.create_alexnet_breakdown()
