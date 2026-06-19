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

## 📄 License
MIT License
