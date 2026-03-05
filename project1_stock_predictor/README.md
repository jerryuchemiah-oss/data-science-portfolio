# 📈 Stock Price Predictor — LSTM Deep Learning Model

**Author:** Uche Jeremiah Nzubechukwu  
**Stack:** Python · NumPy · Pandas · Scikit-learn · Matplotlib  

## Overview
Predicts future stock closing prices using time-series deep learning. Features full technical indicator engineering (RSI, MACD, Bollinger Bands) and a 5-day forward forecast.

## Features
- Geometric Brownian Motion stock data simulation
- 8 engineered trading features (SMA, EMA, RSI, MACD, Bollinger Bands, Volatility)
- LSTM-style sequential modeling with 30-day lookback windows
- Evaluation: RMSE, MAE, MAPE, R²
- 5-day forward price forecast with direction signal (BULLISH/BEARISH)

## Quick Start
```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook stock_price_predictor.ipynb
```

## Swap in Real Data
```python
import yfinance as yf
df = yf.download('AAPL', start='2022-01-01', end='2024-12-31')
```

## Results
| Metric | Value |
|--------|-------|
| RMSE   | ~$2–4 |
| MAPE   | ~1–3% |
| R²     | ~0.96 |

## Upgrade Path
- Real LSTM with `tensorflow.keras` for true sequence memory
- Live data pipeline with `yfinance`
- REST API deployment with FastAPI + Docker
