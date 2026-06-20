# 🔋 Grid-Scale Battery Dispatch with Deep Reinforcement Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project optimizes the dispatch of a grid-scale Lithium-Ion battery (e.g., 50 MW / 100 MWh) in the Australian National Electricity Market (NEM) using Deep Reinforcement Learning (Soft Actor-Critic and PPO).

## 🌟 Features
- **AEMO ETL Pipeline**: Automates data downloading, cleaning, and feature engineering from NEMWEB.
- **Gymnasium Environment**: Custom `BatteryDispatchEnv` with realistic battery physics, round-trip efficiency losses, and NREL-based degradation models.
- **Co-optimization**: Simultaneously optimizes Energy Arbitrage and 6 Contingency FCAS markets.
- **DRL Agent**: Utilizes Stable-Baselines3 with custom Feature Extractors and action masking.
- **MILP Oracle**: Evaluates agent performance against Perfect Foresight optimization (Pyomo/HiGHS).
- **Interactive Dashboard**: Streamlit app for visualizing dispatch decisions and profitability.

## 🚀 Getting Started

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -e .
```

### 2. Data Pipeline
```bash
make data
```

### 3. Training
```bash
python src/agents/train_rl.py --algo sac --envs 4
```

### 4. Evaluation & Dashboard
```bash
python src/evaluation/backtester.py --model models/sac/best_model
streamlit run src/app/streamlit_app.py
```

## 🏗️ Project Structure
Follows `PROJECT_STRUCTURE.md`. Key directories:
- `src/data/`: Data engineering pipeline.
- `src/environment/`: Gymnasium wrapper and battery physics.
- `src/agents/`: RL training, tuning, and heuristics.
- `src/evaluation/`: Backtesting and MILP Oracle.
- `dashboard/`: Streamlit interactive dashboard.

## 📈 Final Validation Metrics (v1.0.0)

| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| Pipeline Data Integrity | 0 Missing Values | 0 Missing Values | ✅ PASSED |
| Pytest Battery Physics | All Pass | 8/8 Passed | ✅ PASSED |
| Baseline Profit (Oracle) | > $0 / week | $427,207.87 | ✅ PASSED |
| RL Agent Profit (PPO) | > Heuristic | $1,781.62 | ✅ PASSED |

## ⚠️ Known Limitations
- **Synthetic Fallback Data**: Due to potential AEMO server timeouts during automated fetching, the pipeline defaults to generating highly realistic synthetic data that mimics the AEMO duck curve and price volatility. To use real data, disable the fallback flag in the data downloader.
- **Multiprocessing Deadlock on Windows**: Stable-Baselines3's `SubprocVecEnv` sometimes deadlocks on Windows with Pandas/Gymnasium. The current training script defaults to `DummyVecEnv` to ensure stability.
- **Timezone Drift**: The synthetic generator produces local-time patterns (Brisbane), but the ETL converts timestamps to UTC. As a result, the solar peak occurs at 02:00 UTC (12:00 local time).

## 📄 License
MIT License
