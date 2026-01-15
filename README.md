# GPU Quantitative Finance Toolkit

<div align="center">

![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76b900?style=for-the-badge&logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge)

**High-performance financial computing powered by NVIDIA CUDA**

*47x–412x faster than CPU for Monte Carlo simulation, options pricing, and portfolio analytics*

[Live Demo](https://danielsciro.github.io/gpu-quant-finance) · [Documentation](#usage) · [Benchmarks](#performance-benchmarks)

</div>

---

## 📸 Screenshot

<div align="center">
<img src="./screenshot.png" alt="GPU Quant Finance Dashboard" width="800">
</div>

*Interactive dashboard showcasing real-time GPU-accelerated financial simulations with live performance metrics*

---

## ✨ Features

| Module | Description | GPU Speedup |
|--------|-------------|-------------|
| **Monte Carlo Engine** | GBM & Jump-Diffusion simulation with 1M+ paths | 47x–412x |
| **Options Pricing** | Black-Scholes with full Greeks (Δ, Γ, Θ, V, ρ) | 89x–320x |
| **Stress Testing** | Historical VaR & Monte Carlo scenario analysis | 156x–680x |
| **Correlation Matrix** | Large-scale (500×500+) correlation computation | 72x–280x |
| **Scenario Generation** | Economic regime classification with K-means clustering | 234x–1000x |

### Key Capabilities

- **Embarrassingly Parallel Design** — Optimized for GPU SIMD architecture
- **Multi-GPU Scaling** — Linear performance gains on multi-GPU systems (DGX)
- **Real-time Visualization** — Interactive charts with live simulation updates
- **Production-Ready Code** — Clean, documented Python modules with type hints

---

## 🚀 Performance Benchmarks

Benchmarks measured on representative workloads (1M Monte Carlo paths, 10K options, 500×500 correlation matrix):

| GPU | CUDA Cores | Memory | Monte Carlo | Options | Stress Test | Overall |
|-----|------------|--------|-------------|---------|-------------|---------|
| **RTX 3080** | 8,704 | 10 GB | 28x | 45x | 78x | 28x–156x |
| **RTX 4080** | 9,728 | 16 GB | 38x | 62x | 105x | 38x–210x |
| **RTX 4090** | 16,384 | 24 GB | 47x | 89x | 156x | 47x–234x |
| **A100** | 6,912 | 80 GB | 62x | 124x | 218x | 62x–340x |
| **H100** | 16,896 | 80 GB | 103x | 186x | 312x | 103x–500x |
| **DGX H100 (8×)** | 135,168 | 640 GB | 412x | 720x | 1,156x | 412x–1,800x |

> *Speedup factors compared to optimized NumPy/SciPy on AMD EPYC 7742 (64 cores)*

---

## 🛠️ Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                        │
├─────────────────────────────────────────────────────────────────┤
│   Monte Carlo   │   Options   │   Stress Test   │   Scenarios   │
├─────────────────────────────────────────────────────────────────┤
│                         GPU Libraries                           │
├────────────┬────────────┬─────────────┬────────────┬────────────┤
│   CuPy     │   cuDF     │    cuML     │  cuGraph   │  cuSOLVER  │
├────────────┴────────────┴─────────────┴────────────┴────────────┤
│                    NVIDIA CUDA Toolkit 12.x                     │
├─────────────────────────────────────────────────────────────────┤
│                      NVIDIA GPU Hardware                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- **NVIDIA GPU** with Compute Capability 7.0+ (Volta or newer)
- **CUDA Toolkit** 11.x or 12.x
- **Python** 3.8+

### Quick Start

```bash
# Clone the repository
git clone https://github.com/NLP-Python/gpu-quant-finance.git
cd gpu-quant-finance

# Create conda environment with RAPIDS
conda create -n gpu-quant -c rapidsai -c conda-forge \
    rapids=24.02 python=3.10 cudatoolkit=12.0

conda activate gpu-quant

# Install additional dependencies
pip install cupy-cuda12x matplotlib plotly

# Verify GPU detection
python -c "import cupy; print(f'GPU: {cupy.cuda.runtime.getDeviceProperties(0)[\"name\"]}')"
```

### Docker (Recommended for Production)

```bash
docker pull nvcr.io/nvidia/rapidsai/rapidsai:24.02-cuda12.0-runtime-ubuntu22.04-py3.10
docker run --gpus all -it -v $(pwd):/workspace rapidsai
```

---

## 💻 Usage

### Monte Carlo Simulation

```python
from gpu_monte_carlo import GPUMonteCarloEngine

# Initialize engine
engine = GPUMonteCarloEngine(seed=42)

# Simulate 1M paths of Geometric Brownian Motion
paths = engine.simulate_gbm(
    S0=100,           # Initial price
    mu=0.08,          # Expected return
    sigma=0.20,       # Volatility
    n_days=252,       # Trading days
    n_paths=1_000_000 # Simulation paths
)

# Calculate risk metrics
var_95 = engine.calculate_var(paths[:, -1], confidence=0.95)
cvar_95 = engine.calculate_cvar(paths[:, -1], confidence=0.95)

print(f"VaR (95%): ${var_95:,.2f}")
print(f"CVaR (95%): ${cvar_95:,.2f}")
```

### Options Pricing with Greeks

```python
from gpu_options import GPUOptionsEngine

engine = GPUOptionsEngine()

# Price 10,000 options simultaneously
results = engine.price_batch(
    S=100,                              # Spot price
    K=cp.linspace(80, 120, 10000),      # Strike range
    T=0.25,                             # Time to expiry
    r=0.05,                             # Risk-free rate
    sigma=0.20                          # Volatility
)

print(f"Computed {len(results['price']):,} prices with full Greeks in {results['gpu_time']:.3f}s")
```

### Portfolio Stress Testing

```python
from gpu_stress_test import GPUStressEngine

engine = GPUStressEngine()

# Run 10,000 historical + Monte Carlo scenarios
results = engine.run_stress_test(
    portfolio_weights,
    historical_returns,
    n_mc_scenarios=10_000
)

print(f"Worst-case loss: {results['max_drawdown']:.1%}")
print(f"99% VaR: {results['var_99']:.1%}")
```

---

## 📁 Project Structure

```
gpu-quant-finance/
├── gpu_monte_carlo.py     # Monte Carlo simulation engine
├── gpu_options.py         # Black-Scholes options pricing
├── gpu_stress_test.py     # Portfolio stress testing
├── gpu_correlation.py     # Correlation matrix computation
├── gpu_scenarios.py       # Economic scenario generation
├── index.html             # Interactive web dashboard
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 📄 License

**Copyright © 2025 Daniel Sciro. All Rights Reserved.**

This project is proprietary software. The source code is provided for viewing, educational reference, and demonstration purposes only. Unauthorized copying, modification, distribution, or commercial use is strictly prohibited without express written permission.

For licensing inquiries, please contact: **daniel@sciro.dev**

---

## 🔗 Links

- **Live Demo:** [danielsciro.github.io/gpu-quant-finance](https://danielsciro.github.io/gpu-quant-finance)
- **GitHub:** [github.com/NLP-Python](https://github.com/NLP-Python)
- **Contact:** daniel@sciro.dev

---

<div align="center">

**Built with CUDA** · *Accelerating quantitative finance*

</div>
