#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  OMX30 Analyzer Pro — Startar"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Skapa virtualenv om det inte finns
if [ ! -d ".venv" ]; then
  echo "→ Skapar Python-miljö (.venv)..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "→ Installerar/uppdaterar beroenden..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "→ Startar OMX30 Analyzer på http://localhost:8501"
echo "   Backdoor-kod: OMXPRO-BACKDOOR-2024"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

streamlit run app.py \
  --server.port 8501 \
  --server.headless false \
  --theme.base dark \
  --theme.primaryColor "#40c4ff" \
  --theme.backgroundColor "#0a0a14" \
  --theme.secondaryBackgroundColor "#0d0d1e" \
  --theme.textColor "#c0c8e8"
