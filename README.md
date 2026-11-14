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



