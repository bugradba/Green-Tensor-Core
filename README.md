# 🌿 Green-Tensor-Core

**Sustainable Hybrid (CPU + PIM) Computing Architecture for Next-Gen AI Workloads**

> *"Moving the processor to the data, instead of moving data to the processor."*

[View Demo](#) • [Documentation](#) • [Report Bug](#)

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

> 📌 *(Place your architecture diagram in `docs/architecture_diagram.png` and reference it here)*

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

Benchmarked against **CPU-only architectures** using synthetic workloads (ResNet-50-like matrix operations):

- ⚡ **Energy Savings:** ~**42%** reduction in total energy consumption  
- ⏱️ **Latency:** **1.8× speedup** for large batch workloads  
- 📉 **Bus Utilization:** **60% reduction** in memory bus traffic  

---

## 🛠️ Installation & Quick Start

### Prerequisites

- Python **3.8+**
- `pip`

### Steps

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/bugradba/Green-Tensor-Core.git
cd Green-Tensor-Core
pip install -r requirements.txt
python src/main.py --mode hybrid --workload large_matrix


Green-Tensor-Core/
├── src/
│   ├── components/       # CPU, PIM, Memory models
│   ├── scheduler/        # Task offloading logic
│   ├── analysis/         # Energy profiling tools
│   └── main.py           # Entry point
├── notebooks/            # Visualization & analysis notebooks
├── docs/                 # Diagrams and references
├── tests/                # Unit tests
├── requirements.txt
└── README.md


🤝 Contact & Acknowledgements

Developer: Muhammed Buğra Demirbaş
Context: Developed for Tomorrow's Technology Leaders (Sustainability Track).

LinkedIn: https://www.linkedin.com/in/m-bugra-demirbas/

Email: mbugrademirbas@gmail.com



