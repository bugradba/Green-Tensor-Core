# 🌿 Green-Tensor-Core

**Sustainable Hybrid (CPU + PIM) Computing Architecture for Next-Gen AI Workloads**

> *"Moving the processor to the data, instead of moving data to the processor."*

---

## 🚀 Executive Summary

**Green-Tensor-Core** is a simulation framework for a **Hybrid Computing Architecture** designed to tackle one of the biggest challenges in modern computing: **Data Movement Energy Costs**.

In traditional Von Neumann architectures, up to **62.7% of total system energy** is wasted on moving data between **Memory (DRAM)** and the **Processor (CPU/GPU)**.  
This project proposes a sustainable solution by integrating **Processing-in-Memory (PIM)** accelerators with a host CPU.

By offloading heavy **vector-matrix operations** to the memory module and keeping control logic on the CPU, **Green-Tensor-Core** aims to drastically reduce the **carbon footprint** of:

- 5G/6G Networks  
- Green Data Centers  
- Autonomous Systems  
- Edge AI & IoT Devices  

---

## 🧠 System Architecture

The system follows a **heterogeneous computing model** with three main layers:

### 1. Host CPU (Control Plane)

- **Role:** Manages OS, I/O requests, and light serial processing  
- **Logic:** Acts as the *"brain"* and decides which tasks should be offloaded  

### 2. PIM Accelerator (Data Plane)

- **Role:** Performs high-intensity parallel computations (e.g., Deep Learning Inference, FFT) directly inside memory  
- **Benefit:** Near-zero data movement cost for large datasets  

### 3. Intelligent Hybrid Scheduler ⚡

**Core Innovation:**  
A runtime scheduler that analyzes incoming tasks:

- If task is **Data-Intensive** (e.g., Matrix Multiplication) → Offload to **PIM**
- If task is **Control-Intensive** (e.g., Branching Logic) → Execute on **CPU**

---

## 🌍 Potential Use Cases

| Domain | Problem | Green-Tensor-Core Solution |
|--------|---------|----------------------------|
| 📡 5G & 6G Networks | Base stations consume massive power for signal processing | Reduces energy per bit via in-memory processing |
| ☁️ Green Data Centers | AI training (LLMs) causes high heat & carbon emissions | Lowers cooling cost and TDP |
| 🛸 Autonomous Systems | Limited battery for AI workloads | Extends flight time / driving range |
| 🔒 Edge AI & IoT | Cloud offloading is slow and risky | Enables secure, low-latency on-device AI |

---

## 📊 Simulation Results

All numbers below are produced by the simulator (`python results.py`); nothing is hand-entered.
Baseline: NVIDIA Jetson Nano (roofline model, optimistic data reuse — i.e. a *best-case* GPU).

**AlexNet conv1 (stride 4, 105.4 M MACs)**

| Platform | Energy | Latency |
|---|---|---|
| PIM 8-bit | 2.05 mJ | 2.64 ms |
| PIM 4-bit | 0.46 mJ | 1.32 ms |
| GPU (Jetson Nano) | 2.71 mJ | 0.45 ms |

**Full AlexNet (batch = 1)**

| Strategy | Energy | Latency | Energy vs GPU-only |
|---|---|---|---|
| GPU-only | 43.41 mJ | 6.96 ms | — |
| PIM-only | 22.36 mJ | 28.45 ms | −48.5 % |
| Rule-based hybrid | 36.03 mJ | 29.27 ms | −17.0 % |
| Q-Learning hybrid | 29.47 mJ | 6.07 ms | −32.1 % |

Key observations:

- PIM uses less energy per layer, but is slower than the GPU on large convolutions.
- With batch = 1, fully-connected layers are **memory-bound** on the GPU, so PIM wins there on both energy and latency. The learned policy discovers this (FC → PIM, Conv → GPU), while the static rule (FC → GPU) does not.
- 4-bit precision scaling cuts MAC energy by 78 %, at a mean relative error of ~16 % per multiplication.

<p align="center"><img src="PIM SIMULATOR/eco_pim_summary.png" width="800"></p>

---

## 🎚️ Layer-wise Precision Selection (with measured accuracy)

4-bit PIM saves energy, but it is only useful if accuracy survives. Instead of assuming an accuracy loss,
the planner **measures** it: for every candidate plan it runs integer inference with exactly the arithmetic
the PIM model charges for (`PIMCluster.mac_8bit`), on a real dataset.

- **Model / data:** small CNN (conv–conv–pool–fc–fc) trained in pure NumPy on scikit-learn's handwritten
  digits (offline, no PyTorch needed).
- **Choices per layer:** `GPU` (INT8) · `PIM8` · `PIM4`
- **Objective:** minimum energy, subject to accuracy ≥ INT8 accuracy − budget, and optionally latency ≤ deadline.
- **Two 4-bit schemes, same hardware cost (one 4×4 LUT multiply):**
  - `truncate` — the pPIM/simulator semantics (keep the high nibble of the 8-bit code)
  - `round` — round to a 4-bit code with its own scale

**Accuracy** (5 random train/test splits, mean ± std):

| Setting | Accuracy |
|---|---|
| Float32 | 98.52 ± 0.20 % |
| INT8, all layers | 98.56 ± 0.22 % |
| 4-bit `truncate`, all layers | 97.56 ± 0.74 % |
| 4-bit `round`, all layers | 98.26 ± 0.28 % |

**Energy saving vs. all-PIM8 under an accuracy budget** (5 splits):

| Budget | `truncate` | `round` |
|---|---|---|
| ≤ 0.5 pt | 70.5 % (loss 0.33 pt) | 73.7 % (loss 0.15 pt) |
| ≤ 1.0 pt | 72.1 % (loss 0.52 pt) | 73.7 % (loss 0.30 pt) |

Findings:

- **Sensitivity is very uneven.** Under `truncate`, putting only `fc1` in 4-bit costs 1.1 pt, while `conv1` costs 0.2 pt
  and `conv2` (82 % of the 8-bit PIM energy) costs 0.4 pt. Choosing layer by layer keeps most of the saving at a fraction of the loss.
- **Accuracy losses are not additive.** `conv1+conv2+fc1` together lose 0.93 pt, less than `fc1` alone, so every plan's accuracy is measured rather than summed.
- **Rounding beats truncation at zero extra hardware cost.** The high-nibble scheme in the pPIM paper discards precision that a rounded 4-bit code keeps.
- **Latency adds a real trade-off.** PIM is about 3× slower than the GPU here. Under a tight deadline the optimum moves the heaviest layer (`conv2`) to the GPU and keeps the cheap layers on 4-bit PIM.
- **Exhaustive vs. greedy.** The exhaustive search gives the exact optimum (3^L plans; 81 here). The greedy planner, meant for large networks, found the optimum in 170 of 174 scenarios. It missed only where several layers had to switch to 4-bit together, and when it misses the gap can be large, so use exhaustive search when L is small.

<p align="center"><img src="PIM SIMULATOR/precision_pareto.png" width="850"></p>

```bash
cd "PIM SIMULATOR"
python precision_experiment.py          # full experiment, ~1.5 min → precision_pareto.png, precision_results.json
python precision_experiment.py --quick  # seed 0 only
python quantization.py                  # retrain the CNN (digits_cnn.npz)
```

---

## 🛠️ Installation & Quick Start

### Prerequisites

- Python **3.8+**
- `pip`

```bash
git clone https://github.com/bugradba/Green-Tensor-Core.git
cd Green-Tensor-Core
pip install -r requirements.txt

pytest                                  # correctness tests

cd "PIM SIMULATOR"
python results.py                       # print all results
python visualize_scheduler.py           # regenerate the charts
python advanced_training.py             # retrain the Q-Learning scheduler
python demo_simulator.py                # step-by-step demo
python precision_experiment.py          # layer-wise precision experiment
```

```
Green-Tensor-Core/
├── PIM SIMULATOR/
│   ├── pim_simulator.py            # PIM core / cluster / array model (pPIM-based)
│   ├── baseline_models.py          # GPU & CPU roofline baselines
│   ├── cnn_models.py               # AlexNet, VGG16, ResNet50, MobileNetV2 layer definitions
│   ├── hybrid_scheduler.py         # simulate_layer(), rule-based & Q-Learning hybrid schedulers
│   ├── Q_Learning/
│   │   └── adaptive_scheduler.py   # Q-Learning agent
│   ├── advanced_training.py        # Q-Learning training script
│   ├── results.py                  # single source of truth for all reported numbers
│   ├── quantization.py             # NumPy CNN + integer 8/4-bit inference (PIM semantics)
│   ├── precision_planner.py        # layer-wise device + precision planner (exhaustive / greedy)
│   ├── precision_experiment.py     # accuracy/energy/latency experiment, Pareto chart
│   ├── digits_cnn.npz              # trained CNN weights
│   ├── visualize_scheduler.py      # charts generated from results.py
│   ├── demo_simulator.py           # printed demo walkthrough
│   └── adaptive_precision.py       # precision/energy trade-off experiments
├── tests/                          # pytest correctness tests
├── README.md
└── requirements.txt
```

### Modelling assumptions

- PIM: per-MAC energy/latency from the pPIM paper; MACs split evenly over 256 clusters; data movement charged per 8 KB DRAM row (RowClone), not per MAC.
- GPU/CPU: latency = max(compute time, DRAM transfer time); average power never exceeds TDP; each tensor is read from DRAM once.
- Q-Learning: state = (workload size, layer type, deadline); reward = energy saving vs GPU − latency penalty, both computed by the simulators.

##  System Architecture

The final hardware and software architecture of the project is built upon a heterogeneous computing framework. The system is controlled by a **Q-Learning-based Smart Hybrid Scheduler** that manages workload distribution between the **CPU** and **PIM (Processing-in-Memory)** units.

The architectural diagram below illustrates the details of this integration:

<p align="center">
  <img src="System Architecture.png" alt="Proposed Smart Hybrid Scheduler and PIM Architecture" width="800">
</p>

###  Operational Logic
To overcome the **Memory Wall** bottleneck, the scheduler dynamically analyzes tasks (profiling) and routes them based on the following logic:

```
IF data_intensive  → PIM (Processing-in-Memory)
ELSE               → CPU / GPU (Host Processor)
```



### Contact & Acknowledgements

- **Developer:** Muhammed Buğra Demirbaş
- **Context:** Developed for Tomorrow's Technology Leaders (Sustainability Track).

- **LinkedIn:** https://www.linkedin.com/in/m-bugra-demirbas/

- **Email:** mbugrademirbas@gmail.com



