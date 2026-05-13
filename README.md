---
title: OMX30 Analyzer Pro
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 📈 OMX30 Analyzer Pro

Komplett aktieanalysverktyg för alla 30 aktier i OMX Stockholm 30-indexet.

## Funktioner

| Modul | Beskrivning |
|-------|-------------|
| 📊 Teknisk analys | Candlestick, RSI, MACD, Bollinger, SMA/EMA, Stochastic, ATR, Williams %R, CCI, OBV, Pivotpunkter |
| 📋 Fundamental analys | P/E, P/B, EV/EBITDA, ROE, ROA, marginaler, skuld, utdelning, analytiker-riktkurser |
| 💰 DCF Värdering | Fritt kassaflöde, WACC, terminal value, intrinsiskt värde, säkerhetsmarginal, känslighetsanalys 7×7 |
| 🎲 Monte Carlo | GBM-simulering (upp till 3 000 banor), VaR (90/95/99 %), percentilband P5–P95 |
| 🔍 OMX30 Screener | Alla 30 aktier med avkastning, heatmap-treemap, korrelationsmatris |
| 🔓 Admin-panel | Rådata, API-debug, bulk-export CSV, avancerad DCF, cache-kontroll |

## Kurs-uppdatering

All data cachas i **1 timme** via `@st.cache_data(ttl=3600)`.  
Knappen "Tvinga uppdatering" rensar cachen direkt.

## Installation

```bash
git clone https://github.com/DITT_ANVÄNDARNAMN/omx30-analyzer.git
cd omx30-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Eller kör startskriptet (skapar venv automatiskt):

```bash
bash run.sh
```

Appen öppnas på **http://localhost:8501**

## Backdoor / Admin-åtkomst

I sidopanelen under **⚙️ Avancerade inställningar**:

```
OMXPRO-BACKDOOR-2024
```

Ger åtkomst till:
- Rådata (OHLCV, finansiella rapporter)
- Fullständig API-debug och info-dict
- Avancerad DCF med alla parametrar
- Bulk-export för samtliga 30 aktier
- Cache-kontroll och systemstatus

## OMX30-komponenter

ABB, Alfa Laval, Autoliv SDB, ASSA ABLOY B, AstraZeneca, Atlas Copco A/B,
Boliden, Electrolux B, Ericsson B, Essity B, Evolution, Getinge B, Hexagon B,
H&M B, Industrivärden C, Investor B, Nordea, NIBE Industrier B, Saab B,
SCA B, SEB A, Handelsbanken A, Sinch, SKF B, SSAB A, Swedbank A,
Tele2 B, Telia, Volvo B

## Datakälla

Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance)

## Stack

- [Streamlit](https://streamlit.io/) — UI-ramverk
- [yfinance](https://github.com/ranaroussi/yfinance) — marknadsdata
- [Plotly](https://plotly.com/) — interaktiva diagram
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — databehandling
- [scipy](https://scipy.org/) — statistik
