"""
OMX30 Analyzer — Teknisk, Fundamental, DCF & Monte Carlo
Kurs uppdateras i realtid var 1:a eller 5:e minut via auto-refresh.
"""
from __future__ import annotations

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import time
import warnings
import json
import io

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# RETRY — exponentiell backoff vid Yahoo Finance rate-limit
# ─────────────────────────────────────────────────────────────────────────────
def _ticker(sym: str) -> yf.Ticker:
    return yf.Ticker(sym)


def _retry(fn, retries: int = 4, base_delay: float = 5.0):
    """Kör fn(), retry med exponentiell backoff vid rate-limit."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            name = type(exc).__name__
            is_rate = "RateLimit" in name or "rate limit" in str(exc).lower()
            if is_rate and attempt < retries - 1:
                wait = base_delay * (2 ** attempt)
                time.sleep(wait)
                continue
            raise
    return None

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OMX30 Analyzer Pro",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, .stApp, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.stApp {
    background-color: #09090F;
    color: #D0D0E0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #0C0C14;
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] * { font-size: 13px; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #55556A !important;
    margin-top: 1.4rem !important;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #0F0F18;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 0;
    padding: 14px 16px;
    transition: border-color 0.15s;
}
div[data-testid="metric-container"]:hover {
    border-color: rgba(255,255,255,0.18);
}
div[data-testid="metric-container"] label {
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: #55556A !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #EEEEF6 !important;
    letter-spacing: -0.02em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 11px !important;
    font-weight: 500 !important;
}

/* ── Headings ── */
h1 { font-size: 20px !important; font-weight: 700 !important;
     color: #FFFFFF !important; letter-spacing: -0.02em !important; }
h2 { font-size: 14px !important; font-weight: 600 !important;
     color: #FFFFFF !important; letter-spacing: 0.01em !important;
     border-bottom: 1px solid rgba(255,255,255,0.07);
     padding-bottom: 8px; margin-top: 1.6rem !important; }
h3 { font-size: 11px !important; font-weight: 600 !important;
     color: #808099 !important; letter-spacing: 0.10em !important;
     text-transform: uppercase !important; }

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    background: #0C0C14 !important;
    border-radius: 0 !important;
    padding: 0;
    gap: 0;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}
button[data-baseweb="tab"] {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #55556A !important;
    border-radius: 0 !important;
    padding: 10px 20px !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.12s !important;
}
button[data-baseweb="tab"]:hover { color: #C0C0D8 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    background: transparent !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #FFFFFF !important;
}

/* ── Divider ── */
hr { border: none; border-top: 1px solid rgba(255,255,255,0.07) !important; margin: 1.6rem 0 !important; }

/* ── Signal badges ── */
.badge-buy  { background: rgba(34,197,94,0.10); color: #22C55E;
              border: 1px solid rgba(34,197,94,0.25);
              padding: 2px 10px; border-radius: 0;
              font-weight: 600; font-size: 11px; letter-spacing: 0.08em;
              text-transform: uppercase; }
.badge-sell { background: rgba(239,68,68,0.10); color: #EF4444;
              border: 1px solid rgba(239,68,68,0.25);
              padding: 2px 10px; border-radius: 0;
              font-weight: 600; font-size: 11px; letter-spacing: 0.08em;
              text-transform: uppercase; }
.badge-neu  { background: rgba(148,163,184,0.08); color: #94A3B8;
              border: 1px solid rgba(148,163,184,0.20);
              padding: 2px 10px; border-radius: 0;
              font-weight: 600; font-size: 11px; letter-spacing: 0.08em;
              text-transform: uppercase; }

/* ── Color helpers ── */
.pos { color: #22C55E; font-weight: 600; }
.neg { color: #EF4444; font-weight: 600; }
.neu { color: #94A3B8; }

/* ── Admin banner ── */
.admin-banner {
    background: rgba(239,68,68,0.05);
    border: 1px solid rgba(239,68,68,0.22);
    border-radius: 0;
    padding: 8px 14px;
    color: #EF4444;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Dataframes ── */
div[data-testid="stDataFrame"] { border-radius: 0; overflow: hidden; }
div[data-testid="stDataFrame"] th {
    background: #0F0F18 !important;
    color: #55556A !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] button {
    background: #111119;
    border: 1px solid rgba(255,255,255,0.10);
    color: #D0D0E0;
    font-size: 12px;
    font-weight: 500;
    border-radius: 0;
    letter-spacing: 0.04em;
    transition: all 0.12s;
}
div[data-testid="stButton"] button:hover {
    background: #1A1A26;
    border-color: rgba(255,255,255,0.22);
    color: #FFFFFF;
}

/* ── Inputs / Selects ── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stMultiSelect"] > div {
    background: #0F0F18;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 0;
}

/* ── Info / Warning ── */
div[data-testid="stAlert"] {
    border-radius: 0;
    border-left-width: 2px;
    font-size: 13px;
}

/* ── Caption / small text ── */
.stCaption, small, caption {
    color: #44445A !important;
    font-size: 11px !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 0 !important;
}

/* ── Slider / Radio ── */
div[data-testid="stSlider"] > div { border-radius: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# OMX30 KOMPONENTER
# ─────────────────────────────────────────────────────────────────────────────
OMX30: dict[str, str] = {
    "ABB":              "ABB.ST",
    "Alfa Laval":       "ALFA.ST",
    "Autoliv SDB":      "ALIV-SDB.ST",
    "ASSA ABLOY B":     "ASSA-B.ST",
    "AstraZeneca":      "AZN.ST",
    "Atlas Copco A":    "ATCO-A.ST",
    "Atlas Copco B":    "ATCO-B.ST",
    "Boliden":          "BOL.ST",
    "Electrolux B":     "ELUX-B.ST",
    "Ericsson B":       "ERIC-B.ST",
    "Essity B":         "ESSITY-B.ST",
    "Evolution":        "EVO.ST",
    "Getinge B":        "GETI-B.ST",
    "Hexagon B":        "HEXA-B.ST",
    "H&M B":            "HM-B.ST",
    "Industrivärden C": "INDU-C.ST",
    "Investor B":       "INVE-B.ST",
    "Nordea":           "NDA-SE.ST",
    "NIBE Industrier B":"NIBE-B.ST",
    "Saab B":           "SAAB-B.ST",
    "SCA B":            "SCA-B.ST",
    "SEB A":            "SEB-A.ST",
    "Handelsbanken A":  "SHB-A.ST",
    "Sinch":            "SINCH.ST",
    "SKF B":            "SKF-B.ST",
    "SSAB A":           "SSAB-A.ST",
    "Swedbank A":       "SWED-A.ST",
    "Tele2 B":          "TEL2-B.ST",
    "Telia":            "TELIA.ST",
    "Volvo B":          "VOLV-B.ST",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING  (cache TTL = 3600 s = 1 timme)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = _retry(lambda: _ticker(ticker).history(period=period))
    if df is None:
        return pd.DataFrame()
    if not df.empty:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def fetch_realtime_price(ticker: str) -> float | None:
    try:
        return float(_retry(lambda: _ticker(ticker).fast_info.last_price))
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker: str) -> dict:
    """
    Hämtar info-dict och berikar den med fast_info + beräknade fält
    så att kritiska nyckeltal alltid finns även vid rate-limiting.
    """
    t = _ticker(ticker)

    # 1. Primär info-dict (kan vara tom vid rate-limit)
    try:
        info = _retry(lambda: t.info) or {}
    except Exception:
        info = {}

    # 2. fast_info — mer tillförlitlig, fyll luckor
    try:
        fi = t.fast_info
        _fill = {
            "marketCap":          getattr(fi, "market_cap",               None),
            "sharesOutstanding":  getattr(fi, "shares",                   None),
            "currentPrice":       getattr(fi, "last_price",               None),
            "regularMarketPrice": getattr(fi, "last_price",               None),
            "fiftyTwoWeekHigh":   getattr(fi, "fifty_two_week_high",      None),
            "fiftyTwoWeekLow":    getattr(fi, "fifty_two_week_low",       None),
            "forwardPE":          getattr(fi, "forward_pe",               None),
            "currency":           getattr(fi, "currency",                 None),
            "exchange":           getattr(fi, "exchange",                 None),
        }
        for k, v in _fill.items():
            if v is not None and not info.get(k):
                info[k] = v
    except Exception:
        pass

    # 3. Beräkna börsvärde om det fortfarande saknas
    _cp     = float(info.get("currentPrice") or 0)
    _shares = float(info.get("sharesOutstanding") or 0)
    if not info.get("marketCap") and _cp > 0 and _shares > 0:
        info["marketCap"] = _cp * _shares

    # 4. Dividend: hämta faktisk utdelningshistorik om dividendRate saknas
    if not info.get("dividendRate"):
        try:
            divs = _retry(lambda: t.dividends)
            if divs is not None and not divs.empty:
                # Summera utdelningar senaste 12 månader
                cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=13)
                divs.index = pd.to_datetime(divs.index, utc=True)
                ttm_div = float(divs[divs.index >= cutoff].sum())
                if ttm_div > 0:
                    info["dividendRate"] = ttm_div
                    if _cp > 0:
                        info["dividendYield"] = ttm_div / _cp
        except Exception:
            pass

    # 5. 52-veckors förändring om den saknas
    if not info.get("52WeekChange"):
        try:
            _h52 = _retry(lambda: t.history(period="1y"))["Close"]
            if _h52 is not None and len(_h52) >= 2:
                info["52WeekChange"] = float(_h52.iloc[-1] / _h52.iloc[0] - 1)
        except Exception:
            pass

    return info


@st.cache_data(ttl=3600, show_spinner=False)
def calc_beta_omx(ticker: str) -> float:
    """Beräknar beta mot OMXS30 (^OMX) från 2 års daglig avkastning."""
    try:
        stock  = _retry(lambda: _ticker(ticker).history(period="2y"))["Close"].pct_change().dropna()
        omx    = _retry(lambda: _ticker("^OMX").history(period="2y"))["Close"].pct_change().dropna()
        common = stock.index.intersection(omx.index)
        if len(common) < 100:
            return float(fetch_info(ticker).get("beta") or 1.0)
        s, m = stock.loc[common].values, omx.loc[common].values
        return round(float(np.cov(s, m)[0][1] / np.var(m)), 3)
    except Exception:
        return float(fetch_info(ticker).get("beta") or 1.0)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fx_rate(pair: str) -> float:
    """Hämtar valutakurs, t.ex. 'EURSEK=X' eller 'USDSEK=X'."""
    try:
        return float(_retry(lambda: _ticker(pair).fast_info.last_price))
    except Exception:
        return {"EURSEK=X": 11.0, "USDSEK=X": 10.5}.get(pair, 1.0)


def get_corrected_pb(info: dict) -> float | None:
    """
    Korrigerar P/B när financial_currency ≠ trading_currency (SEK).
    Nordea (EUR), AstraZeneca/ABB/Autoliv (USD) rapporterar bookValue i
    sin redovisningsvaluta medan currentPrice är i SEK — kräver konvertering.
    """
    pb_raw  = info.get("priceToBook")
    fin_cur = info.get("financialCurrency", "SEK")
    cur     = info.get("currency", "SEK")

    if fin_cur == cur or fin_cur == "SEK" or not pb_raw:
        return float(pb_raw) if pb_raw else None

    price   = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    bvps_fc = info.get("bookValue")
    if not bvps_fc or not price:
        return float(pb_raw)

    pair   = f"{fin_cur}SEK=X"
    fx     = fetch_fx_rate(pair)
    bvps_sek = float(bvps_fc) * fx
    return round(price / bvps_sek, 3) if bvps_sek > 0 else None


def get_fx_multiplier(info: dict) -> float:
    """
    Returnerar valutaomräkningsfaktor för finansiella rapportvärden → SEK.
    Nordea=EUR, ABB/AZN/Autoliv=USD, övriga=SEK (faktor 1.0).
    """
    fin_cur = info.get("financialCurrency", "SEK")
    if fin_cur == "SEK":
        return 1.0
    return fetch_fx_rate(f"{fin_cur}SEK=X")


def get_dividend_yield(info: dict) -> float | None:
    """
    dividendRate (i lokal valuta SEK) / currentPrice är mest tillförlitlig.
    trailingAnnualDividendRate är ibland i USD/EUR för cross-listed bolag.
    """
    price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    if not price:
        return None

    dr = info.get("dividendRate")
    if dr and float(dr) > 0 and float(dr) < price * 0.40:
        return float(dr) / price

    # Fallback: trailingAnnualDividendYield om rimligt (0.1%–25%)
    tady = info.get("trailingAnnualDividendYield")
    if tady and 0.001 < float(tady) < 0.25:
        return float(tady)

    return None


def most_recent_quarter_label(info: dict) -> str:
    """Returnerar 'Q1 2026', 'Q4 2025' etc. från mostRecentQuarter timestamp."""
    ts = info.get("mostRecentQuarter")
    if not ts:
        return "N/A"
    try:
        from datetime import datetime
        dt = datetime.utcfromtimestamp(int(ts))
        q  = (dt.month - 1) // 3 + 1
        return f"Q{q} {dt.year}"
    except Exception:
        return "N/A"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    def safe(fn):
        try:
            r = _retry(fn)
            return r if r is not None and not r.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    t = _ticker(ticker)
    return {
        "income":     safe(lambda: t.income_stmt),
        "balance":    safe(lambda: t.balance_sheet),
        "cashflow":   safe(lambda: t.cashflow),
        "q_income":   safe(lambda: t.quarterly_income_stmt),
        "q_balance":  safe(lambda: t.quarterly_balance_sheet),
        "q_cashflow": safe(lambda: t.quarterly_cashflow),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_analyst_data(ticker: str) -> dict:
    """Hämtar analytikermål och rekommendationer via separata yfinance-endpoints."""
    t = _ticker(ticker)
    result = {"targets": {}, "rec_summary": None, "rec_key": ""}
    try:
        at = _retry(lambda: t.analyst_price_targets)
        if isinstance(at, dict) and at.get("mean"):
            result["targets"] = at
        elif hasattr(at, "to_dict"):
            result["targets"] = at.to_dict()
    except Exception:
        pass
    try:
        rs = _retry(lambda: t.recommendations_summary)
        if rs is not None and not rs.empty:
            result["rec_summary"] = rs.iloc[0].to_dict()
    except Exception:
        pass
    try:
        info_rec = _retry(lambda: t.info)
        result["rec_key"] = info_rec.get("recommendationKey", "")
    except Exception:
        pass
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dividends(ticker: str) -> float:
    """Returnerar TTM-utdelning i lokal valuta (senaste 13 mån)."""
    try:
        t = _ticker(ticker)
        divs = _retry(lambda: t.dividends)
        if divs is None or divs.empty:
            return 0.0
        divs.index = pd.to_datetime(divs.index, utc=True)
        cut = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=13)
        return float(divs[divs.index >= cut].sum())
    except Exception:
        return 0.0


def _row(df: pd.DataFrame, *keys) -> pd.Series | None:
    """Hämtar första matchande rad från DataFrame, returnerar None om saknas."""
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return None


def _val(df: pd.DataFrame, col_idx: int, *keys) -> float | None:
    """Hämtar ett enskilt värde från statement, None om saknas."""
    if df.empty or col_idx >= len(df.columns):
        return None
    for k in keys:
        if k in df.index:
            v = df.loc[k].iloc[col_idx]
            return float(v) if pd.notna(v) else None
    return None


def q_label(df: pd.DataFrame, idx: int) -> str:
    """Returnerar 'Q1 2026' för kolumn idx i quarterly statement."""
    if df.empty or idx >= len(df.columns):
        return "—"
    c = df.columns[idx]
    return f"Q{(c.month-1)//3+1} {c.year}"


def build_report_table(df: pd.DataFrame, rows: dict, scale: float = 1e9, unit: str = "Mdr SEK") -> pd.DataFrame | None:
    """Bygger en ren rapporttabell med svenska etiketter. rows = {etikett: [möjliga yfinance-nycklar]}"""
    if df.empty:
        return None
    cols = df.columns[:4]
    col_labels = [f"Q{(c.month-1)//3+1} {c.year}" for c in cols]
    data = {}
    for label, keys in rows.items():
        series = _row(df, *keys)
        if series is not None:
            data[label] = [
                f"{float(series.iloc[i])/scale:,.2f}" if i < len(series) and pd.notna(series.iloc[i]) else "—"
                for i in range(len(cols))
            ]
        else:
            data[label] = ["—"] * len(cols)
    return pd.DataFrame(data, index=col_labels).T


def yoy_change(df: pd.DataFrame, *keys) -> str:
    """Beräknar YoY förändring senaste kvartal vs samma kvartal föregående år."""
    if df.empty or len(df.columns) < 5:
        return ""
    for k in keys:
        if k in df.index:
            curr = df.loc[k].iloc[0]
            prev = df.loc[k].iloc[4] if len(df.columns) > 4 else None
            if prev and pd.notna(curr) and pd.notna(prev) and prev != 0:
                chg = (float(curr) - float(prev)) / abs(float(prev)) * 100
                arrow = "▲" if chg > 0 else "▼"
                color = "#00e676" if chg > 0 else "#ff1744"
                return f'<span style="color:{color}">{arrow} {chg:+.1f}% YoY</span>'
    return ""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_latest() -> pd.DataFrame:
    """Hämtar stängningskurser för alla OMX30-aktier (3 månader)."""
    frames = {}
    for name, ticker in OMX30.items():
        try:
            df = fetch_history(ticker, "3mo")
            if not df.empty:
                frames[name] = df["Close"]
        except Exception:
            pass
    return pd.DataFrame(frames) if frames else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# TEKNISKA INDIKATORER
# ─────────────────────────────────────────────────────────────────────────────
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(com=n - 1, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)


def macd(s: pd.Series):
    line   = ema(s, 12) - ema(s, 26)
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    m   = sma(s, n)
    std = s.rolling(n).std()
    return m + k * std, m, m - k * std


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    lo = df["Low"].rolling(k_period).min()
    hi = df["High"].rolling(k_period).max()
    k  = 100 * (df["Close"] - lo) / (hi - lo)
    return k, k.rolling(d_period).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hi_n = df["High"].rolling(n).max()
    lo_n = df["Low"].rolling(n).min()
    return -100 * (hi_n - df["Close"]) / (hi_n - lo_n)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - tp.rolling(n).mean()) / (0.015 * mad)


def compute_signals(df: pd.DataFrame) -> list[dict]:
    c     = df["Close"]
    curr  = c.iloc[-1]
    signals = []

    # RSI
    r = rsi(c).iloc[-1]
    if   r < 30: sig, desc = "KÖP",     f"RSI {r:.1f} — översåld"
    elif r > 70: sig, desc = "SÄLJ",    f"RSI {r:.1f} — överköpt"
    else:        sig, desc = "NEUTRAL", f"RSI {r:.1f}"
    signals.append({"ind": "RSI", "signal": sig, "desc": desc})

    # MACD
    ml, sl, _ = macd(c)
    if   ml.iloc[-1] > sl.iloc[-1] and ml.iloc[-2] <= sl.iloc[-2]:
        sig, desc = "KÖP",     "MACD korsade signal uppåt"
    elif ml.iloc[-1] < sl.iloc[-1] and ml.iloc[-2] >= sl.iloc[-2]:
        sig, desc = "SÄLJ",    "MACD korsade signal nedåt"
    else:
        sig, desc = "NEUTRAL", f"MACD {ml.iloc[-1]:.2f} / Signal {sl.iloc[-1]:.2f}"
    signals.append({"ind": "MACD", "signal": sig, "desc": desc})

    # Glidande medelvärde
    s50  = sma(c, 50).iloc[-1]
    s200 = sma(c, 200).iloc[-1]
    if   curr > s50 > s200: sig, desc = "KÖP",     "Kurs > SMA50 > SMA200 — Golden Cross"
    elif curr < s50 < s200: sig, desc = "SÄLJ",    "Kurs < SMA50 < SMA200 — Death Cross"
    else:                   sig, desc = "NEUTRAL", "Blandade MA-signaler"
    signals.append({"ind": "MA Trend", "signal": sig, "desc": desc})

    # Bollinger Bands
    ub, _, lb = bollinger(c)
    if   curr < lb.iloc[-1]: sig, desc = "KÖP",     "Kurs under nedre Bollinger-band"
    elif curr > ub.iloc[-1]: sig, desc = "SÄLJ",    "Kurs över övre Bollinger-band"
    else:                    sig, desc = "NEUTRAL", "Inom Bollinger-band"
    signals.append({"ind": "Bollinger", "signal": sig, "desc": desc})

    # Stochastic
    k_val, d_val = stochastic(df)
    kv, dv = k_val.iloc[-1], d_val.iloc[-1]
    if   kv < 20 and kv > dv: sig, desc = "KÖP",     f"Stoch %K={kv:.1f} — överköpt → upp"
    elif kv > 80 and kv < dv: sig, desc = "SÄLJ",    f"Stoch %K={kv:.1f} — översåld → ned"
    else:                     sig, desc = "NEUTRAL", f"Stoch %K={kv:.1f} %D={dv:.1f}"
    signals.append({"ind": "Stochastic", "signal": sig, "desc": desc})

    # Williams %R
    wr = williams_r(df).iloc[-1]
    if   wr < -80: sig, desc = "KÖP",     f"Williams %R={wr:.1f} — översåld"
    elif wr > -20: sig, desc = "SÄLJ",    f"Williams %R={wr:.1f} — överköpt"
    else:          sig, desc = "NEUTRAL", f"Williams %R={wr:.1f}"
    signals.append({"ind": "Williams %R", "signal": sig, "desc": desc})

    # CCI
    cc = cci(df).iloc[-1]
    if   cc < -100: sig, desc = "KÖP",     f"CCI={cc:.0f} — översåld"
    elif cc >  100: sig, desc = "SÄLJ",    f"CCI={cc:.0f} — överköpt"
    else:           sig, desc = "NEUTRAL", f"CCI={cc:.0f}"
    signals.append({"ind": "CCI", "signal": sig, "desc": desc})

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# DCF
# ─────────────────────────────────────────────────────────────────────────────
def dcf_valuation(
    info: dict,
    cashflow_df: pd.DataFrame,
    growth_rate: float | None = None,
    wacc: float | None = None,
    terminal_growth: float = 0.025,
    years: int = 5,
    current_price: float | None = None,
    q_cashflow_df: pd.DataFrame | None = None,
) -> dict:
    """
    DCF i SEK för alla OMX30-bolag inkl. EUR/USD-rapportörer.
    Alla kassaflödesvärden omräknas till SEK via live-valutakurs.
    """
    fin_cur = info.get("financialCurrency", "SEK")
    fx      = fetch_fx_rate(f"{fin_cur}SEK=X") if fin_cur != "SEK" else 1.0

    cp = float(current_price or info.get("currentPrice") or
               info.get("regularMarketPrice") or 0)
    if cp <= 0:
        return {"error": "Kan inte hämta aktuell kurs"}

    shares = float(info.get("sharesOutstanding") or 1)

    # ── FCF: TTM från kvartalssiffror (mest aktuellt) → årsrapport → info ───
    fcf = None
    if q_cashflow_df is not None and not q_cashflow_df.empty:
        try:
            ttm_ocf, ttm_cx = 0.0, 0.0
            for i in range(min(4, len(q_cashflow_df.columns))):
                if "Operating Cash Flow" in q_cashflow_df.index:
                    v = q_cashflow_df.loc["Operating Cash Flow"].iloc[i]
                    if pd.notna(v): ttm_ocf += float(v)
                if "Capital Expenditure" in q_cashflow_df.index:
                    v = q_cashflow_df.loc["Capital Expenditure"].iloc[i]
                    if pd.notna(v): ttm_cx += float(v)
            if ttm_ocf > 0:
                fcf_q = ttm_ocf + ttm_cx  # capex är negativt i yfinance
                if fcf_q > 0:
                    fcf = fcf_q * fx       # konvertera till SEK
        except Exception:
            pass

    if not fcf and not cashflow_df.empty:
        try:
            ocf  = cashflow_df.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cashflow_df.index else None
            capx = cashflow_df.loc["Capital Expenditure"].iloc[0]  if "Capital Expenditure"  in cashflow_df.index else None
            if ocf and capx:
                val = float(ocf) + float(capx)
                if val > 0:
                    fcf = val * fx
        except Exception:
            pass

    if not fcf:
        raw = info.get("freeCashflow")
        if raw and float(raw) > 0:
            fcf = float(raw) * fx

    if not fcf or fcf <= 0:
        return {"error": "Otillräcklig kassaflödesdata — DCF ej tillgänglig för detta bolag"}

    # ── Tillväxttakt ─────────────────────────────────────────────────────────
    if growth_rate is None:
        rg = float(info.get("revenueGrowth")  or 0.05)
        eg = float(info.get("earningsGrowth") or 0.05)
        growth_rate = float(np.clip((rg + eg) / 2, 0.01, 0.30))

    # ── WACC via CAPM + kapitalstruktur ──────────────────────────────────────
    if wacc is None:
        sym    = info.get("symbol", "")
        beta   = calc_beta_omx(sym) if sym else float(info.get("beta") or 1.0)
        rf     = 0.038           # Svensk 10-årig statsobligation
        erp    = 0.055           # Equity Risk Premium
        ke     = rf + beta * erp
        td_sek = float(info.get("totalDebt")  or 0) * fx
        mc_sek = float(info.get("marketCap")  or 0) or (cp * shares)
        tot    = td_sek + mc_sek
        wd     = td_sek / tot if tot > 0 else 0.20
        we     = 1 - wd
        kd     = 0.045
        tax    = 0.206
        wacc   = float(np.clip(we * ke + wd * kd * (1 - tax), 0.06, 0.25))

    # ── FCF-projektion: kumulativt korrekt med avtagande tillväxt ────────────
    decay    = 0.85
    g        = growth_rate
    proj_fcf = []
    fcf_t    = fcf
    for _ in range(years):
        fcf_t = fcf_t * (1 + g)
        proj_fcf.append(fcf_t)
        g = max(g * decay, terminal_growth)   # g avtar men ej under terminal

    # ── Present Value ─────────────────────────────────────────────────────────
    if wacc <= terminal_growth:
        terminal_growth = wacc - 0.01

    tv  = proj_fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv  = sum(cf / (1 + wacc) ** (i + 1) for i, cf in enumerate(proj_fcf))
    pvt = tv / (1 + wacc) ** years
    ev  = pv + pvt

    # ── Equity Value per aktie (SEK) ──────────────────────────────────────────
    td_sek   = float(info.get("totalDebt") or 0) * fx
    cash_sek = float(info.get("totalCash") or 0) * fx
    eq_val   = ev - td_sek + cash_sek
    iv       = eq_val / shares if shares > 1 else 0
    mos      = (iv - cp) / cp * 100 if cp > 0 else 0

    return {
        "fcf_base": fcf, "projected_fcf": proj_fcf,
        "pv_fcf": pv, "pv_terminal": pvt,
        "enterprise_value": ev, "intrinsic_value": iv,
        "current_price": cp, "margin_of_safety": mos,
        "wacc": wacc, "growth_rate": growth_rate,
        "terminal_growth": terminal_growth,
        "currency_note": f"({fin_cur}→SEK ×{fx:.2f})" if fin_cur != "SEK" else "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
def monte_carlo(
    df: pd.DataFrame,
    n_sim: int = 500,
    n_days: int = 252,
    ci: float = 0.95,
) -> dict:
    if df.empty or len(df) < 30:
        raise ValueError("Otillräcklig kurshistorik för Monte Carlo (minst 30 handelsdagar krävs)")

    close = df["Close"].dropna()
    ret   = np.log(close / close.shift(1)).dropna()

    if len(ret) < 20:
        raise ValueError("För få avkastningsdatapunkter")

    mu    = float(ret.mean())
    sigma = float(ret.std())
    last  = float(close.iloc[-1])

    # Geometric Brownian Motion med drift-korrigering
    dt   = 1.0
    drift = (mu - 0.5 * sigma ** 2) * dt

    rng = np.random.default_rng(seed=42)
    # Vektoriserad beräkning: (n_days, n_sim)
    shocks = rng.normal(0, sigma * np.sqrt(dt), size=(n_days, n_sim))
    log_returns = drift + shocks
    price_paths = last * np.exp(np.cumsum(log_returns, axis=0))

    finals = price_paths[-1]
    var_pct = (1 - ci) * 100

    return {
        "sim":         price_paths,
        "finals":      finals,
        "last":        last,
        "mu":          mu,
        "sigma":       sigma,
        "expected":    float(np.mean(finals)),
        "var":         float(np.percentile(finals, var_pct)),
        "prob_profit": float(np.mean(finals > last) * 100),
        "p5":          float(np.percentile(finals, 5)),
        "p25":         float(np.percentile(finals, 25)),
        "p50":         float(np.percentile(finals, 50)),
        "p75":         float(np.percentile(finals, 75)),
        "p95":         float(np.percentile(finals, 95)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
_DARK = dict(template="plotly_dark",
             paper_bgcolor="#09090F",
             plot_bgcolor="#0D0D15",
             font=dict(family="Inter, system-ui, sans-serif", color="#C8C8D8", size=12),
             title_font=dict(family="Inter, system-ui, sans-serif", color="#FFFFFF", size=14),
             xaxis=dict(gridcolor="#1C1C28", zerolinecolor="#2A2A38"),
             yaxis=dict(gridcolor="#1C1C28", zerolinecolor="#2A2A38"))

_LEGEND = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)")


def chart_candle(df: pd.DataFrame, ticker: str, indicators: list[str]) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.50, 0.14, 0.18, 0.18],
        subplot_titles=(f"{ticker} — OHLCV", "Volym", "RSI (14)", "MACD"),
    )
    c = df["Close"]

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=c, name="Kurs",
        increasing_line_color="#00e676", decreasing_line_color="#ff1744",
        increasing_fillcolor="rgba(0,230,118,0.27)", decreasing_fillcolor="rgba(255,23,68,0.27)",
    ), 1, 1)

    colors_map = {"SMA 20":"#ffd54f","SMA 50":"#40c4ff","SMA 200":"#ea80fc","EMA 20":"#69f0ae"}
    for ind in indicators:
        if ind == "SMA 20":
            fig.add_trace(go.Scatter(x=df.index, y=sma(c, 20), name="SMA 20",
                line=dict(color=colors_map[ind], width=1.2)), 1, 1)
        elif ind == "SMA 50":
            fig.add_trace(go.Scatter(x=df.index, y=sma(c, 50), name="SMA 50",
                line=dict(color=colors_map[ind], width=1.5)), 1, 1)
        elif ind == "SMA 200":
            fig.add_trace(go.Scatter(x=df.index, y=sma(c, 200), name="SMA 200",
                line=dict(color=colors_map[ind], width=2.0)), 1, 1)
        elif ind == "EMA 20":
            fig.add_trace(go.Scatter(x=df.index, y=ema(c, 20), name="EMA 20",
                line=dict(color=colors_map[ind], width=1.2, dash="dot")), 1, 1)

    if "Bollinger" in indicators:
        ub, mb, lb = bollinger(c)
        fig.add_trace(go.Scatter(x=df.index, y=ub, name="BB Övre",
            line=dict(color="rgba(120,120,255,0.6)", dash="dash")), 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=lb, name="BB Nedre",
            line=dict(color="rgba(120,120,255,0.6)", dash="dash"),
            fill="tonexty", fillcolor="rgba(100,100,255,0.06)"), 1, 1)

    # Volume
    bar_colors = ["#00e676" if cl >= op else "#ff1744"
                  for cl, op in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volym",
        marker_color=bar_colors, opacity=0.65), 2, 1)

    # RSI
    rsi_s = rsi(c)
    fig.add_trace(go.Scatter(x=df.index, y=rsi_s, name="RSI",
        line=dict(color="#ffd54f", width=1.5)), 3, 1)
    for lvl, col in [(70, "rgba(255,23,68,.4)"), (30, "rgba(0,230,118,.4)"), (50, "rgba(255,255,255,.15)")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=col, row=3, col=1)

    # MACD
    ml, sl, hist = macd(c)
    hc = ["#00e676" if h >= 0 else "#ff1744" for h in hist]
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Hist",
        marker_color=hc, opacity=0.6), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=ml, name="MACD",
        line=dict(color="#40c4ff", width=1.5)), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=sl, name="Signal",
        line=dict(color="#ff6d00", width=1.5)), 4, 1)

    fig.update_layout(height=820, xaxis_rangeslider_visible=False,
        legend=_LEGEND, **_DARK)
    return fig


def chart_mc(mc_res: dict, ticker: str) -> go.Figure:
    sim, days = mc_res["sim"], list(range(mc_res["sim"].shape[0]))
    fig = go.Figure()

    sample = np.random.choice(sim.shape[1], min(80, sim.shape[1]), replace=False)
    for i in sample:
        fig.add_trace(go.Scatter(x=days, y=sim[:, i], mode="lines",
            line=dict(width=0.4, color="rgba(64,196,255,.12)"),
            showlegend=False, hoverinfo="skip"))

    bands = {
        "P95": (np.percentile(sim, 95, axis=1), "#00e676", "dash"),
        "P75": (np.percentile(sim, 75, axis=1), "#69f0ae", "solid"),
        "P50": (np.percentile(sim, 50, axis=1), "#ffffff", "solid"),
        "P25": (np.percentile(sim, 25, axis=1), "#ff8a80", "solid"),
        "P5":  (np.percentile(sim,  5, axis=1), "#ff1744", "dash"),
    }
    widths = {"P50": 2.5}
    for name, (vals, col, dash) in bands.items():
        fig.add_trace(go.Scatter(x=days, y=vals, name=name,
            line=dict(color=col, width=widths.get(name, 1.5), dash=dash)))

    fig.add_hline(y=mc_res["last"], line_dash="dot", line_color="rgba(255,255,255,.6)",
                  annotation_text=f"Nuv. {mc_res['last']:.2f}")
    fig.update_layout(title=f"{ticker} — Monte Carlo ({mc_res['sim'].shape[1]} simuleringar, {len(days)} dagar)",
        xaxis_title="Handelsdagar", yaxis_title="Pris (SEK)",
        height=500, **_DARK)
    return fig


def chart_dcf_waterfall(dcf: dict) -> go.Figure:
    n    = len(dcf["projected_fcf"])
    lbls = [f"FCF År {i+1}" for i in range(n)] + ["Terminal PV", "Enterprise Value"]
    vals = [v / 1e9 for v in dcf["projected_fcf"]] + [dcf["pv_terminal"] / 1e9, 0]
    meas = ["relative"] * n + ["relative", "total"]

    fig = go.Figure(go.Waterfall(
        orientation="v", measure=meas, x=lbls, y=vals,
        connector=dict(line=dict(color="#444466")),
        increasing=dict(marker=dict(color="#00e676")),
        decreasing=dict(marker=dict(color="#ff1744")),
        totals=dict(marker=dict(color="#40c4ff")),
        texttemplate="%{y:.2f}Mdr",
    ))
    fig.update_layout(title="DCF — Komponentanalys (Mdr SEK)",
        yaxis_title="Mdr SEK", height=420, **_DARK)
    return fig


def chart_hist_financials(fin: pd.DataFrame, rows: list[str], title: str) -> go.Figure | None:
    available = [r for r in rows if r in fin.index]
    if not available:
        return None
    data = fin.loc[available].T / 1e9
    fig  = px.bar(data, barmode="group", title=title,
        color_discrete_sequence=["#40c4ff","#00e676","#ffd54f","#ff6d00","#ea80fc"])
    fig.update_layout(yaxis_title="Mdr SEK", **_DARK)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_big(v, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    av = abs(v)
    sfx = f" {suffix}" if suffix else ""
    if av >= 1e12: return f"{v/1e12:.2f}T{sfx}"
    if av >= 1e9:  return f"{v/1e9:.1f}Mdr{sfx}"
    if av >= 1e6:  return f"{v/1e6:.0f}M{sfx}"
    if av >= 1e3:  return f"{v/1e3:.1f}K{sfx}"
    return f"{v:.1f}{sfx}"


def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v*100:.2f}%"


def fmt_x(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{d}f}x"


def fmt_num(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{d}f}"


def badge(signal: str) -> str:
    cls = {"KÖP": "badge-buy", "SÄLJ": "badge-sell"}.get(signal, "badge-neu")
    return f'<span class="{cls}">{signal}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if "admin" not in st.session_state:
        st.session_state.admin = False
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 60
    if "dcf_interval_min" not in st.session_state:
        st.session_state.dcf_interval_min = 10
    if "mc_interval_min" not in st.session_state:
        st.session_state.mc_interval_min = 10
    if "dcf_cache" not in st.session_state:
        st.session_state.dcf_cache = {}   # {ticker: (result, params, timestamp)}
    if "mc_cache" not in st.session_state:
        st.session_state.mc_cache = {}    # {ticker: (mc_res, params, timestamp)}

    # ── AUTO-REFRESH ─────────────────────────────────────────────────────────
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 60

    refresh_count = st_autorefresh(
        interval=st.session_state.refresh_interval * 1000,
        key="price_autorefresh",
    )

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## OMX30 Analyzer Pro")
        st.caption(f"Senast uppdaterad: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        refresh_opt = st.radio(
            "Kursuppdatering",
            options=[60, 300],
            format_func=lambda x: "Var 1 minut" if x == 60 else "Var 5 minuter",
            horizontal=True,
            index=0 if st.session_state.refresh_interval == 60 else 1,
        )
        if refresh_opt != st.session_state.refresh_interval:
            st.session_state.refresh_interval = refresh_opt
            st.rerun()

        st.caption(f"Uppdatering #{refresh_count} &nbsp;|&nbsp; Intervall: {refresh_opt}s")
        st.divider()

        selected = st.selectbox("Välj aktie", list(OMX30.keys()),
                                index=list(OMX30.keys()).index("Volvo B"))
        ticker   = OMX30[selected]

        period = st.select_slider("Tidsperiod",
            options=["1mo","3mo","6mo","1y","2y","5y"], value="1y")

        inds = st.multiselect("Indikatorer på kursgraf",
            ["SMA 20","SMA 50","SMA 200","EMA 20","Bollinger"],
            default=["SMA 50","SMA 200","Bollinger"])

        st.divider()
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Rensa cache", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col_btn2:
            if st.button("Uppdatera nu", use_container_width=True):
                st.rerun()

        st.divider()

        # ── BACKDOOR ENTRY ────────────────────────────────────────────────
        with st.expander("Avancerade inställningar", expanded=False):
            st.caption("Systemkonfiguration")
            key = st.text_input("Åtkomstkod", type="password",
                                placeholder="●●●●●●●●●●●●●●●●")
            if key == "OMXPRO-BACKDOOR-2024":
                st.session_state.admin = True
                st.success("Fullständig åtkomst aktiverad")
            elif key and key != "OMXPRO-BACKDOOR-2024":
                st.error("Ogiltig kod")
                st.session_state.admin = False

        if st.session_state.admin:
            st.markdown(
                '<div class="admin-banner">ADMIN-LÄGE AKTIVT<br>'
                'Fullständig systemåtkomst</div>',
                unsafe_allow_html=True)

    # ── DATA ─────────────────────────────────────────────────────────────────
    try:
        with st.spinner(f"Hämtar data för {selected}…"):
            df       = fetch_history(ticker, period)
            info     = fetch_info(ticker)
            fin      = fetch_financials(ticker)
            beta_omx = calc_beta_omx(ticker)
    except Exception as exc:
        name = type(exc).__name__
        if "RateLimit" in name or "rate" in str(exc).lower():
            st.warning(
                "Yahoo Finance begränsar tillfälligt anropen (rate limit). "
                "Vänta 30 sekunder och tryck sedan på knappen nedan."
            )
        else:
            st.error(f"Fel vid datahämtning: {exc}")
        if st.button("Försök igen"):
            st.cache_data.clear()
            st.rerun()
        st.stop()

    if df is None or df.empty:
        st.warning(
            f"Ingen kursdata för **{ticker}**. "
            "Yahoo Finance kan vara tillfälligt otillgängligt — försök om en stund."
        )
        if st.button("Försök igen"):
            st.cache_data.clear()
            st.rerun()
        st.stop()

    # ── HEADER ───────────────────────────────────────────────────────────────
    # Realtidspris (5 min cache) — fallback till historik
    rt_price = fetch_realtime_price(ticker)
    cp   = rt_price if rt_price else float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-1]) if rt_price else (float(df["Close"].iloc[-2]) if len(df) > 1 else cp)
    if rt_price:
        prev = float(info.get("previousClose") or df["Close"].iloc[-1])
    chg  = cp - prev
    pchg = chg / prev * 100 if prev else 0

    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        color = "#00e676" if chg >= 0 else "#ff1744"
        arrow = "▲" if chg >= 0 else "▼"
        st.markdown(
            f"## {selected} &nbsp;"
            f'<span style="color:{color};font-size:1.1em">'
            f"{cp:.2f} SEK &nbsp; {arrow} {chg:+.2f} ({pchg:+.2f}%)</span>",
            unsafe_allow_html=True)
        interval_label = "1 min" if st.session_state.refresh_interval == 60 else "5 min"
        st.caption(f"Ticker: **{ticker}** &nbsp;|&nbsp; Börs: Nasdaq Stockholm &nbsp;|&nbsp; "
                   f"Valuta: SEK &nbsp;|&nbsp; Sektor: {info.get('sector','N/A')} &nbsp;|&nbsp; "
                   f"Live-kurs uppdateras var {interval_label}")
    with col_h2:
        if st.session_state.admin:
            st.markdown('<div class="admin-banner" style="text-align:center">ADMIN</div>',
                        unsafe_allow_html=True)

    # ── QUICK METRICS ─────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    # Snabba fallback-beräkningar för top-metrics
    _shares_top  = float(info.get("sharesOutstanding") or
                         _val(fin["q_balance"], 0, "Ordinary Shares Number","Share Issued") or 0)
    _mktcap_top  = info.get("marketCap") or (cp * _shares_top if _shares_top > 0 else None)
    _pe_top      = info.get("trailingPE")
    if not _pe_top and _mktcap_top:
        try:
            _qi = fin["q_income"]
            _ttm_ni = sum(v for i in range(4)
                          for k in ["Net Income","Net Income Common Stockholders"]
                          if not _qi.empty and k in _qi.index
                          for v in [float(_qi.loc[k].iloc[i])]
                          if pd.notna(v)) if not _qi.empty else 0
            if _ttm_ni > 0:
                _pe_top = _mktcap_top / _ttm_ni
        except Exception:
            pass

    m1.metric("Kurs (SEK)",     f"{cp:.2f}",         f"{chg:+.2f} ({pchg:+.2f}%)")
    m2.metric("52v Högst",      f"{df['High'].max():.2f}")
    m3.metric("52v Lägst",      f"{df['Low'].min():.2f}")
    m4.metric("Snittvolym",     fmt_big(df["Volume"].mean()))
    m5.metric("Börsvärde",      fmt_big(_mktcap_top))
    m6.metric("P/E (trailing)", fmt_num(_pe_top))
    st.divider()

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_labels = [
        "Teknisk analys",
        "Fundamental analys",
        "DCF Värdering",
        "Monte Carlo",
        "OMX30 Screener",
    ]
    if st.session_state.admin:
        tab_labels.append("Admin")

    tabs = st.tabs(tab_labels)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — TEKNISK ANALYS
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.plotly_chart(chart_candle(df, ticker, inds), use_container_width=True)

        # Signaltabell
        st.subheader("Handelssignaler")
        sigs = compute_signals(df)
        buy_count  = sum(1 for s in sigs if s["signal"] == "KÖP")
        sell_count = sum(1 for s in sigs if s["signal"] == "SÄLJ")
        neu_count  = len(sigs) - buy_count - sell_count

        cols = st.columns(3)
        cols[0].metric("KÖP-signaler",     buy_count,  help="Antal indikatorer med köpsignal")
        cols[1].metric("SÄLJ-signaler",    sell_count, help="Antal indikatorer med säljsignal")
        cols[2].metric("NEUTRAL-signaler", neu_count)

        sig_cols = st.columns(len(sigs))
        for i, s in enumerate(sigs):
            with sig_cols[i]:
                st.markdown(f"**{s['ind']}**")
                st.markdown(badge(s["signal"]), unsafe_allow_html=True)
                st.caption(s["desc"])

        st.divider()

        # Extra TA-nyckeltal
        st.subheader("Nyckeltal & Nivåer")
        c = df["Close"]
        rsi_v  = rsi(c).iloc[-1]
        atr_v  = atr(df).iloc[-1]
        wr_v   = williams_r(df).iloc[-1]
        cci_v  = cci(df).iloc[-1]
        obv_d  = obv(df)
        kv, dv = stochastic(df)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("RSI 14",          f"{rsi_v:.1f}")
        r2.metric("ATR 14",          f"{atr_v:.2f} SEK")
        r3.metric("Williams %R",     f"{wr_v:.1f}")
        r4.metric("CCI 20",          f"{cci_v:.0f}")

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Stoch %K",        f"{kv.iloc[-1]:.1f}")
        r6.metric("Stoch %D",        f"{dv.iloc[-1]:.1f}")
        r7.metric("OBV-trend",       "Stigande" if obv_d.iloc[-1] > obv_d.iloc[-20] else "Fallande")
        dist200 = (cp - sma(c, 200).iloc[-1]) / sma(c, 200).iloc[-1] * 100
        r8.metric("Avst. SMA200",    f"{dist200:+.1f}%")

        # Pivotpunkter
        st.subheader("Pivotpunkter (Daglig)")
        ph, pl, pc = float(df["High"].iloc[-1]), float(df["Low"].iloc[-1]), cp
        pivot = (ph + pl + pc) / 3
        r1p = 2*pivot - pl; s1p = 2*pivot - ph
        r2p = pivot + (ph - pl); s2p = pivot - (ph - pl)
        r3p = ph + 2*(pivot - pl); s3p = pl - 2*(ph - pivot)

        pp = st.columns(7)
        pp[0].metric("S3", f"{s3p:.2f}")
        pp[1].metric("S2", f"{s2p:.2f}")
        pp[2].metric("S1", f"{s1p:.2f}")
        pp[3].metric("Pivot", f"{pivot:.2f}")
        pp[4].metric("R1", f"{r1p:.2f}")
        pp[5].metric("R2", f"{r2p:.2f}")
        pp[6].metric("R3", f"{r3p:.2f}")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — FUNDAMENTAL ANALYS
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        qi  = fin["q_income"]
        qb  = fin["q_balance"]
        qcf = fin["q_cashflow"]
        ai  = fin["income"]
        acf = fin["cashflow"]

        # Hämta analytiker- och utdelningsdata via separata endpoints
        analyst  = fetch_analyst_data(ticker)
        ttm_div  = fetch_dividends(ticker)

        mrq     = q_label(qi, 0) if not qi.empty else most_recent_quarter_label(info)
        fin_cur = info.get("financialCurrency", "SEK")
        fx      = get_fx_multiplier(info)   # 1.0 för SEK-bolag, ~11 för EUR, ~10.5 för USD
        pb_corr = get_corrected_pb(info)
        rec_map = {"buy":"KÖP","strong_buy":"STARKT KÖP","hold":"HÅLL",
                   "sell":"SÄLJ","underperform":"UNDERPERFORM","neutral":"NEUTRAL"}

        # Utdelning från faktisk historik
        dy = ttm_div / cp if (ttm_div and cp) else get_dividend_yield(info)

        # Banner
        cur_note = f"  |  Redovisningsvaluta: **{fin_cur}** (omräknat ×{fx:.2f} → SEK)" if fin_cur != "SEK" else ""
        st.info(f"Senaste rapport: **{mrq}**  |  Källa: Yahoo Finance / Bolagets rapporter{cur_note}")

        # ── SEKTION 1: Senaste kvartal KPIer ─────────────────────────────────
        st.subheader(f"Nyckeltal från {mrq}")
        k1,k2,k3,k4,k5,k6 = st.columns(6)

        rev_q   = _val(qi, 0, "Total Revenue","Operating Revenue")
        oi_q    = _val(qi, 0, "Operating Income","EBIT")
        ni_q    = _val(qi, 0, "Net Income","Net Income Common Stockholders")
        ebitda_q= _val(qi, 0, "EBITDA","Normalized EBITDA")
        fcf_q   = _val(qcf, 0, "Free Cash Flow")
        eps_q   = _val(qi, 0, "Diluted EPS","Basic EPS")
        shares  = float(info.get("sharesOutstanding") or 1)
        eps_q_calc = ni_q / shares if ni_q and shares else None

        k1.metric(f"Intäkter ({mrq})",    fmt_big(rev_q),
                  f"{yoy_change(qi,'Total Revenue','Operating Revenue')}" if not qi.empty else "")
        k2.metric(f"Rörelseresultat",     fmt_big(oi_q))
        k3.metric(f"Nettoresultat",       fmt_big(ni_q),
                  f"{yoy_change(qi,'Net Income')}" if not qi.empty else "")
        k4.metric("EBITDA",               fmt_big(ebitda_q))
        k5.metric("Fritt kassaflöde",     fmt_big(fcf_q))
        k6.metric("EPS (kvartal)",        fmt_num(eps_q if eps_q else eps_q_calc))

        # Marginaler beräknade från faktiska siffror
        if rev_q and rev_q > 0:
            gp_q = _val(qi, 0, "Gross Profit")
            m1,m2,m3,m4 = st.columns(4)
            if gp_q:  m1.metric("Bruttomarginal",   f"{gp_q/rev_q*100:.1f}%")
            if oi_q:  m2.metric("Rörelsemarginal",   f"{oi_q/rev_q*100:.1f}%")
            if ni_q:  m3.metric("Nettomarginal",     f"{ni_q/rev_q*100:.1f}%")
            if ebitda_q: m4.metric("EBITDA-marginal",f"{ebitda_q/rev_q*100:.1f}%")

        st.divider()

        # ── TTM från 4 senaste kvartal, omräknat till SEK via fx ─────────────
        def _ttm(df, *keys):
            vals = [_val(df, i, *keys) for i in range(4)]
            total = sum(v for v in vals if v is not None)
            return (total * fx) if total != 0 else None

        def _bs(df, *keys):
            """Balansräkningsvärde omräknat till SEK."""
            v = _val(df, 0, *keys)
            return (v * fx) if v is not None else None

        ttm_ni     = _ttm(qi,  "Net Income","Net Income Common Stockholders")
        ttm_rev    = _ttm(qi,  "Total Revenue","Operating Revenue")
        ttm_oi     = _ttm(qi,  "Operating Income","Total Operating Income As Reported")
        ttm_ebit   = _ttm(qi,  "EBIT") or ttm_oi
        ttm_gp     = _ttm(qi,  "Gross Profit")
        ttm_ebitda = _ttm(qi,  "EBITDA","Normalized EBITDA")
        ttm_fcf    = _ttm(qcf, "Free Cash Flow")
        ttm_ocf    = _ttm(qcf, "Operating Cash Flow")
        ttm_int    = _ttm(qi,  "Interest Expense Non Operating","Interest Expense")

        net_debt  = _bs(qb, "Net Debt")
        tot_eq    = _bs(qb, "Common Stock Equity","Stockholders Equity","Total Equity Gross Minority Interest")
        tot_debt  = _bs(qb, "Total Debt")
        cash_q    = _bs(qb, "Cash And Cash Equivalents","Cash Cash Equivalents And Short Term Investments")
        tot_assets= _bs(qb, "Total Assets")
        cur_assets= _bs(qb, "Current Assets","Total Current Assets")
        cur_liab  = _bs(qb, "Current Liabilities","Total Current Liabilities")
        inv_q     = _bs(qb, "Inventory")
        capex_q   = _val(qcf, 0, "Capital Expenditure")
        if capex_q: capex_q *= fx

        # ── Aktier: info → fast_info → balansräkning ─────────────────────────
        shares_bs = _val(qb, 0, "Ordinary Shares Number","Share Issued")
        shares    = float(info.get("sharesOutstanding") or shares_bs or 1)

        # ── Börsvärde: info → cp × aktier ────────────────────────────────────
        mktcap = float(info.get("marketCap") or 0) or (cp * shares if shares > 1 else None)

        # ── EV: info → mktcap + nettoskuld ───────────────────────────────────
        ev = float(info.get("enterpriseValue") or 0) or None
        if not ev and mktcap:
            _nd = net_debt if net_debt is not None else (
                (tot_debt or 0) - (cash_q or 0) if tot_debt else 0)
            ev = mktcap + _nd

        # ── P/B beräknad: kurs / (eget kapital per aktie) ────────────────────
        bvps_calc = tot_eq / shares if tot_eq and shares > 1 else None
        pb_stmt   = cp / bvps_calc  if bvps_calc and bvps_calc > 0 else None
        pb_corr   = pb_corr or pb_stmt   # info-korrektionen tar prio, annars rapport

        # Beräknade nyckeltal
        roe_calc   = ttm_ni / tot_eq         if ttm_ni and tot_eq and tot_eq > 0 else None
        roa_calc   = ttm_ni / tot_assets     if ttm_ni and tot_assets else None
        invested_k = (tot_eq + tot_debt)     if tot_eq and tot_debt else None
        roic_calc  = ttm_oi / invested_k     if ttm_oi and invested_k and invested_k > 0 else None
        nd_ebitda  = net_debt / ttm_ebitda   if net_debt and ttm_ebitda and ttm_ebitda > 0 else None
        fcf_yield  = ttm_fcf / mktcap        if ttm_fcf and mktcap else None
        fcf_margin = ttm_fcf / ttm_rev       if ttm_fcf and ttm_rev else None
        ni_margin  = ttm_ni  / ttm_rev       if ttm_ni  and ttm_rev else None
        oi_margin  = ttm_oi  / ttm_rev       if ttm_oi  and ttm_rev else None
        gp_margin  = ttm_gp  / ttm_rev       if ttm_gp  and ttm_rev else None
        ebitda_mg  = ttm_ebitda / ttm_rev    if ttm_ebitda and ttm_rev else None
        quick_r    = (cur_assets - (inv_q or 0)) / cur_liab if cur_assets and cur_liab else None
        int_cov    = ttm_oi / abs(ttm_int)   if ttm_oi and ttm_int and ttm_int != 0 else None
        # ── Beräkna allt från kvartalsdata — info används bara om det är bättre ─
        eps_ttm    = ttm_ni / shares         if ttm_ni and shares > 1 else None
        pe_calc    = mktcap / ttm_ni         if mktcap and ttm_ni and ttm_ni > 0 else None
        ps_calc    = mktcap / ttm_rev        if mktcap and ttm_rev else None
        ev_ebitda_c= ev / ttm_ebitda         if ev and ttm_ebitda and ttm_ebitda > 0 else None
        ev_rev_c   = ev / ttm_rev            if ev and ttm_rev else None

        # P/E: rapport-beräknad är alltid korrekt, info kan vara försenad
        pe_v   = pe_calc or info.get("trailingPE")
        # Forward P/E: info eller fast_info
        fpe    = info.get("forwardPE")
        if fpe and (fpe <= 0 or fpe > 300): fpe = None
        # P/S: rapport-beräknad
        ps_v   = ps_calc or info.get("priceToSalesTrailingTwelveMonths")
        # EV-multiplar: rapport-beräknade
        evebit = ev_ebitda_c or info.get("enterpriseToEbitda")
        evrev  = ev_rev_c    or info.get("enterpriseToRevenue")
        ev_ebit_r = ev / ttm_ebit if ev and ttm_ebit and ttm_ebit > 0 else None
        # PEG: pe / tillväxt
        peg_v  = info.get("pegRatio")
        if not peg_v and pe_v and rev_yoy and rev_yoy > 0:
            peg_v = pe_v / (rev_yoy * 100)

        # YoY-tillväxt
        rev_yoy = None
        if not qi.empty and len(qi.columns) >= 5:
            r_now  = _val(qi, 0, "Total Revenue","Operating Revenue")
            r_year = _val(qi, 4, "Total Revenue","Operating Revenue")
            if r_now and r_year and r_year != 0:
                rev_yoy = (r_now - r_year) / abs(r_year)
        ni_yoy = None
        if not qi.empty and len(qi.columns) >= 5:
            n_now  = _val(qi, 0, "Net Income")
            n_year = _val(qi, 4, "Net Income")
            if n_now and n_year and n_year != 0:
                ni_yoy = (n_now - n_year) / abs(n_year)

        # Direktavkastning: dividendRate från info (berikat med historik i fetch_info)
        dy   = get_dividend_yield(info)
        # ROE/ROA: rapport-beräknad tar prio
        roe_v = f"{roe_calc*100:.1f}%" if roe_calc else fmt_pct(info.get("returnOnEquity"))
        roa_v = f"{roa_calc*100:.1f}%" if roa_calc else fmt_pct(info.get("returnOnAssets"))

        # ── SEKTION 2: Värdering, hälsa, lönsamhet, utdelning/analytiker ────
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.subheader("Värdering")
            items = {
                "P/E (TTM)":        fmt_num(pe_v),
                "P/E (forward)":    fmt_num(fpe) if fpe else "N/A",
                "PEG":              fmt_num(peg_v),
                "P/B":              fmt_num(pb_corr),
                "P/S (TTM)":        fmt_num(ps_v),
                "EV/EBITDA":        fmt_num(evebit),
                "EV/EBIT":          fmt_num(ev_ebit_r),
                "EV/Revenue":       fmt_num(evrev),
                "FCF-yield":        fmt_pct(fcf_yield),
                "Börsvärde":        fmt_big(mktcap),
                "Enterprise Value": fmt_big(ev),
            }
            for k, v in items.items(): st.metric(k, v)

        with c2:
            st.subheader("Finansiell hälsa")
            ocf_q = _val(qcf, 0, "Operating Cash Flow")
            items = {
                "Nettoskuld":          fmt_big(net_debt),
                "Nettoskuld/EBITDA":   fmt_num(nd_ebitda),
                "Total skuld":         fmt_big(tot_debt),
                "Skuld/EK":            fmt_num(info.get("debtToEquity") or (tot_debt/tot_eq if tot_debt and tot_eq and tot_eq > 0 else None)),
                "Eget kapital":        fmt_big(tot_eq),
                "Kassa & likvida":     fmt_big(cash_q),
                "Totala tillgångar":   fmt_big(tot_assets),
                "Likviditetsgrad":     fmt_num(info.get("currentRatio") or (cur_assets/cur_liab if cur_assets and cur_liab else None)),
                "Snabblikviditet":     fmt_num(quick_r),
                "Räntetäckning":       fmt_num(int_cov),
                "Operativt CF (TTM)":  fmt_big(ttm_ocf),
                "CapEx (senaste Q)":   fmt_big(capex_q),
            }
            for k, v in items.items(): st.metric(k, v)

        with c3:
            st.subheader("Lönsamhet & Resultat (TTM)")
            items = {
                "Intäkter (TTM)":       fmt_big(ttm_rev),
                "Bruttoresultat (TTM)": fmt_big(ttm_gp),
                "EBIT (TTM)":           fmt_big(ttm_ebit),
                "EBITDA (TTM)":         fmt_big(ttm_ebitda),
                "Nettoresultat (TTM)":  fmt_big(ttm_ni),
                "FCF (TTM)":            fmt_big(ttm_fcf),
                "Bruttomarginal":       f"{gp_margin*100:.1f}%"  if gp_margin  else "N/A",
                "EBIT-marginal":        f"{ttm_ebit/ttm_rev*100:.1f}%" if ttm_ebit and ttm_rev else "N/A",
                "EBITDA-marginal":      f"{ebitda_mg*100:.1f}%"  if ebitda_mg  else "N/A",
                "Nettomarginal":        f"{ni_margin*100:.1f}%"  if ni_margin  else "N/A",
                "FCF-marginal":         f"{fcf_margin*100:.1f}%" if fcf_margin else "N/A",
                "ROE":                  roe_v,
                "ROA":                  roa_v,
                "ROIC":                 f"{roic_calc*100:.1f}%"  if roic_calc  else "N/A",
                "Intäktstillväxt (YoY)":fmt_pct(rev_yoy or info.get("revenueGrowth")),
                "Vinsttillväxt (YoY)":  fmt_pct(ni_yoy  or info.get("earningsGrowth")),
                "EPS (TTM)":            fmt_num(eps_ttm or info.get("trailingEps")),
                "R&D":                  fmt_big(_val(qi, 0, "Research And Development")),
            }
            for k, v in items.items(): st.metric(k, v)

        with c4:
            st.subheader("Utdelning & Analytiker")
            # Analytikermål — från fetch_analyst_data (separat endpoint)
            at       = analyst.get("targets", {})
            tgt_mean = at.get("mean") or info.get("targetMeanPrice")
            tgt_med  = at.get("median") or info.get("targetMedianPrice")
            tgt_low  = at.get("low")  or info.get("targetLowPrice")
            tgt_high = at.get("high") or info.get("targetHighPrice")
            rec_key  = analyst.get("rec_key") or info.get("recommendationKey","")
            # Rekommendationsfördelning
            rs = analyst.get("rec_summary") or {}
            n_buy  = (rs.get("strongBuy",0) or 0) + (rs.get("buy",0) or 0)
            n_hold = rs.get("hold",0) or 0
            n_sell = (rs.get("sell",0) or 0) + (rs.get("strongSell",0) or 0)
            n_tot  = n_buy + n_hold + n_sell
            rec_dist = f"Köp {n_buy}  Håll {n_hold}  Sälj {n_sell}" if n_tot > 0 else "N/A"
            # Utdelning
            eps_val = eps_ttm or info.get("trailingEps")
            payout  = (ttm_div / (eps_val * shares) * 100) if (ttm_div and eps_val and shares > 1 and eps_val > 0) else None
            payout_str = f"{payout:.1f}%" if payout else fmt_pct(info.get("payoutRatio"))
            ex_div = info.get("exDividendDate")
            if ex_div:
                try:
                    ex_div_str = datetime.utcfromtimestamp(int(ex_div)).strftime("%Y-%m-%d")
                except Exception:
                    ex_div_str = str(ex_div)[:10]
            else:
                ex_div_str = "N/A"

            items = {
                "Direktavkastning":    fmt_pct(dy),
                "Utdelning/aktie":     fmt_num(ttm_div) if ttm_div else fmt_num(info.get("dividendRate")),
                "Utdelningsandel":     payout_str,
                "Ex-utdelningsdatum":  ex_div_str,
                "5å snitt dir.avk.":   fmt_pct(info.get("fiveYearAvgDividendYield",0)/100 if info.get("fiveYearAvgDividendYield") else None),
                "Beta (vs OMXS30)":    fmt_num(beta_omx),
                "Antal aktier":        fmt_big(shares),
                "Insiderägande":       fmt_pct(info.get("heldPercentInsiders")),
                "Institutionellt äg.": fmt_pct(info.get("heldPercentInstitutions")),
                "52v förändring":      fmt_pct(info.get("52WeekChange")),
                "Riktkurs (medel)":    fmt_num(tgt_mean),
                "Riktkurs (median)":   fmt_num(tgt_med),
                "Riktkurs (låg/hög)":  f"{fmt_num(tgt_low)} / {fmt_num(tgt_high)}" if tgt_low and tgt_high else "N/A",
                "Uppside mot riktkurs":f"{(tgt_mean/cp-1)*100:+.1f}%" if tgt_mean and cp else "N/A",
                "Rekommendation":      rec_map.get(rec_key, rec_key.upper() if rec_key else "N/A"),
                "Fördelning (K/H/S)":  rec_dist,
                "Antal analytiker":    str(n_tot) if n_tot else "N/A",
            }
            for k, v in items.items(): st.metric(k, v)

        # ── SEKTION 3: Kvartalsrapporter (4 senaste) ──────────────────────────
        st.divider()
        st.subheader("Kvartalsrapporter — Resultaträkning")

        inc_rows = {
            "Intäkter (Mdr)":         ["Total Revenue","Operating Revenue"],
            "Bruttoresultat (Mdr)":   ["Gross Profit"],
            "EBITDA (Mdr)":           ["EBITDA","Normalized EBITDA"],
            "Rörelseresultat (Mdr)":  ["Operating Income","EBIT"],
            "Nettoresultat (Mdr)":    ["Net Income","Net Income Common Stockholders"],
            "F&U (Mdr)":              ["Research And Development"],
        }
        tbl_inc = build_report_table(qi, inc_rows)
        if tbl_inc is not None:
            st.dataframe(tbl_inc, use_container_width=True)

            # Graf
            plot_rows = [k for k in ["Intäkter (Mdr)","Bruttoresultat (Mdr)","Rörelseresultat (Mdr)","Nettoresultat (Mdr)"] if k in tbl_inc.index]
            if plot_rows:
                plot_df = tbl_inc.loc[plot_rows].replace("—", None)
                for col in plot_df.columns:
                    plot_df[col] = pd.to_numeric(plot_df[col].astype(str).str.replace(",",""), errors="coerce")
                fig_qi = px.bar(plot_df.T, barmode="group", title="Kvartalsresultat (Mdr)",
                    color_discrete_sequence=["#40c4ff","#00e676","#ffd54f","#ff6d00"])
                fig_qi.update_layout(yaxis_title="Mdr", **_DARK)
                st.plotly_chart(fig_qi, use_container_width=True)

        st.subheader("Kvartalsrapporter — Balansräkning")
        bal_rows = {
            "Totala tillgångar (Mdr)": ["Total Assets"],
            "Eget kapital (Mdr)":      ["Common Stock Equity","Stockholders Equity"],
            "Total skuld (Mdr)":       ["Total Debt"],
            "Nettoskuld (Mdr)":        ["Net Debt"],
            "Kassa (Mdr)":             ["Cash And Cash Equivalents","Cash Cash Equivalents And Short Term Investments"],
            "Goodwill & immat. (Mdr)": ["Goodwill And Other Intangible Assets"],
            "Rörelsekapital (Mdr)":    ["Working Capital"],
        }
        tbl_bal = build_report_table(qb, bal_rows)
        if tbl_bal is not None:
            st.dataframe(tbl_bal, use_container_width=True)

        st.subheader("Kvartalsrapporter — Kassaflöde")
        cf_rows = {
            "Operativt kassaflöde (Mdr)": ["Operating Cash Flow"],
            "CapEx (Mdr)":                ["Capital Expenditure"],
            "Fritt kassaflöde (Mdr)":     ["Free Cash Flow"],
            "D&A (Mdr)":                  ["Depreciation And Amortization"],
            "Förändring rörelsekapital":  ["Change In Working Capital"],
        }
        tbl_cf = build_report_table(qcf, cf_rows)
        if tbl_cf is not None:
            st.dataframe(tbl_cf, use_container_width=True)

        # ── SEKTION 4: Årsdata ────────────────────────────────────────────────
        st.divider()
        st.subheader("Årsvis historik")

        if not ai.empty:
            ann_rows = {
                "Intäkter (Mdr)":        ["Total Revenue","Operating Revenue"],
                "Bruttoresultat (Mdr)":  ["Gross Profit"],
                "EBITDA (Mdr)":          ["EBITDA","Normalized EBITDA"],
                "Rörelseresultat (Mdr)": ["Operating Income"],
                "Nettoresultat (Mdr)":   ["Net Income"],
                "F&U (Mdr)":             ["Research And Development"],
            }
            ann_cols  = ai.columns[:5]
            ann_lbls  = [str(c.year) for c in ann_cols]
            ann_data  = {}
            for label, keys in ann_rows.items():
                s = _row(ai, *keys)
                if s is not None:
                    ann_data[label] = [f"{float(s.iloc[i])/1e9:,.2f}" if i < len(s) and pd.notna(s.iloc[i]) else "—" for i in range(len(ann_cols))]
            if ann_data:
                tbl_ann = pd.DataFrame(ann_data, index=ann_lbls).T
                st.dataframe(tbl_ann, use_container_width=True)

                plot_rows_ann = [k for k in ["Intäkter (Mdr)","Rörelseresultat (Mdr)","Nettoresultat (Mdr)"] if k in tbl_ann.index]
                if plot_rows_ann:
                    pann = tbl_ann.loc[plot_rows_ann].replace("—", None)
                    for col in pann.columns:
                        pann[col] = pd.to_numeric(pann[col].astype(str).str.replace(",",""), errors="coerce")
                    fig_ann = px.bar(pann.T, barmode="group", title="Årsresultat (Mdr)",
                        color_discrete_sequence=["#40c4ff","#ffd54f","#ff6d00"])
                    fig_ann.update_layout(yaxis_title="Mdr", **_DARK)
                    st.plotly_chart(fig_ann, use_container_width=True)

        if not acf.empty:
            cf_rows_ann = {
                "Operativt CF (Mdr)":  ["Operating Cash Flow"],
                "CapEx (Mdr)":         ["Capital Expenditure"],
                "Fritt CF (Mdr)":      ["Free Cash Flow"],
            }
            ann_cf_cols = acf.columns[:5]
            ann_cf_lbls = [str(c.year) for c in ann_cf_cols]
            ann_cf_data = {}
            for label, keys in cf_rows_ann.items():
                s = _row(acf, *keys)
                if s is not None:
                    ann_cf_data[label] = [f"{float(s.iloc[i])/1e9:,.2f}" if i < len(s) and pd.notna(s.iloc[i]) else "—" for i in range(len(ann_cf_cols))]
            if ann_cf_data:
                tbl_cf_ann = pd.DataFrame(ann_cf_data, index=ann_cf_lbls).T
                st.dataframe(tbl_cf_ann, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — DCF
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("DCF-värdering (Discounted Cash Flow)")

        col_dcf, col_params = st.columns([3, 1])
        with col_params:
            st.markdown("**Anpassa parametrar**")
            raw_growth   = info.get("revenueGrowth") or 0.07
            init_growth  = int(round(float(np.clip(raw_growth, 0.0, 0.40)) * 100))

            dcf_growth   = st.slider("Tillväxttakt år 1 (%)",  0,    40,  init_growth, 1,
                                     format="%d%%") / 100
            dcf_wacc     = st.slider("WACC (%)",                4,    20,  9,           1,
                                     format="%d%%") / 100
            dcf_terminal = st.slider("Terminal tillväxt (%)",   1,     5,  2,           1,
                                     format="%d%%") / 100
            dcf_years    = st.slider("Prognosår",               3,    15,  5)

            st.divider()
            st.markdown("**Auto-uppdatering**")
            dcf_interval_opt = st.radio(
                "Uppdateringsintervall",
                options=[10, 30],
                format_func=lambda x: f"{x} min",
                index=0 if st.session_state.dcf_interval_min == 10 else 1,
                key="dcf_interval_radio",
                horizontal=True,
            )
            if dcf_interval_opt != st.session_state.dcf_interval_min:
                st.session_state.dcf_interval_min = dcf_interval_opt

        with col_dcf:
            dcf_params_now = (ticker, dcf_growth, dcf_wacc, dcf_terminal, dcf_years, round(cp, 1))
            cached = st.session_state.dcf_cache.get(ticker)
            now_ts = time.time()
            need_recalc = (
                cached is None
                or cached[1] != dcf_params_now
                or (now_ts - cached[2]) >= st.session_state.dcf_interval_min * 60
            )
            if need_recalc:
                result = dcf_valuation(info, fin["cashflow"],
                                       dcf_growth, dcf_wacc, dcf_terminal, dcf_years,
                                       current_price=cp, q_cashflow_df=fin["q_cashflow"])
                st.session_state.dcf_cache[ticker] = (result, dcf_params_now, now_ts)
            else:
                result = cached[0]
                now_ts = cached[2]

            calc_time = time.strftime("%H:%M:%S", time.localtime(now_ts))
            st.caption(f"Senast beräknad: **{calc_time}** &nbsp;|&nbsp; Uppdateras var **{st.session_state.dcf_interval_min} min**")

            if "error" in result:
                st.warning(result["error"])
                st.info("Tips: Aktien saknar tillräcklig kassaflödesdata hos Yahoo Finance.")
            else:
                d1, d2, d3, d4 = st.columns(4)
                iv   = result["intrinsic_value"]
                curr = result["current_price"]
                mos  = result["margin_of_safety"]
                d1.metric("Intrinsiskt värde", f"{iv:.2f} SEK")
                d2.metric("Marknadspris",       f"{curr:.2f} SEK")
                d3.metric("Säkerhetsmarginal",  f"{mos:.1f}%",
                          "Undervärderad" if mos > 0 else "Övervärderad")
                d4.metric("WACC (använt)",       fmt_pct(result["wacc"]))

                st.divider()
                d5, d6, d7 = st.columns(3)
                d5.metric("Enterprise Value",   fmt_big(result["enterprise_value"]))
                d6.metric("PV Fria Kassaflöden",fmt_big(result["pv_fcf"]))
                d7.metric("PV Terminal Value",  fmt_big(result["pv_terminal"]))

                st.plotly_chart(chart_dcf_waterfall(result), use_container_width=True)

                st.subheader("Projicerade fria kassaflöden")
                fcf_rows = []
                g = dcf_growth
                for i, v in enumerate(result["projected_fcf"]):
                    fcf_rows.append({
                        "År": f"År {i+1}",
                        "Tillväxttakt": f"{g*100:.1f}%",
                        "FCF (MSEK)":   f"{v/1e6:,.0f}",
                        "FCF (Mdr)":    f"{v/1e9:.2f}",
                    })
                    g *= 0.85
                st.dataframe(pd.DataFrame(fcf_rows), hide_index=True, use_container_width=True)

                st.divider()
                st.subheader("Känslighetsanalys (Säkerhetsmarginal %)")
                growth_range = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
                wacc_range   = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
                sens_data    = {}
                for w in wacc_range:
                    row = {}
                    for g2 in growth_range:
                        r = dcf_valuation(info, fin["cashflow"], g2, w, dcf_terminal, dcf_years, current_price=cp, q_cashflow_df=fin["q_cashflow"])
                        row[f"g={g2*100:.0f}%"] = round(r.get("margin_of_safety", 0), 1) if "error" not in r else 0
                    sens_data[f"WACC={w*100:.0f}%"] = row
                sens_df = pd.DataFrame(sens_data).T

                def color_mos(v):
                    if isinstance(v, (int, float)):
                        return f"color: {'#00e676' if v > 20 else '#ff1744' if v < -10 else '#ffd54f'}"
                    return ""
                st.dataframe(sens_df.style.applymap(color_mos), use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4 — MONTE CARLO
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("Monte Carlo — Prissimulering")

        mc_col, mc_par = st.columns([3, 1])
        with mc_par:
            n_sim  = st.slider("Simuleringar",   100, 3000, 500, 100)
            n_days = st.slider("Handelsdagar",    60,  504, 252,  21)
            ci_val = st.slider("Konfidensgrad",  0.90, 0.99, 0.95, 0.01)

            st.divider()
            st.markdown("**Auto-uppdatering**")
            mc_interval_opt = st.radio(
                "Uppdateringsintervall",
                options=[10, 30],
                format_func=lambda x: f"{x} min",
                index=0 if st.session_state.mc_interval_min == 10 else 1,
                key="mc_interval_radio",
                horizontal=True,
            )
            if mc_interval_opt != st.session_state.mc_interval_min:
                st.session_state.mc_interval_min = mc_interval_opt

        with mc_col:
            mc_params_now = (ticker, n_sim, n_days, round(ci_val, 2), round(cp, 1))
            mc_cached = st.session_state.mc_cache.get(ticker)
            mc_now_ts = time.time()
            mc_need_recalc = (
                mc_cached is None
                or mc_cached[1] != mc_params_now
                or (mc_now_ts - mc_cached[2]) >= st.session_state.mc_interval_min * 60
            )
            if mc_need_recalc:
                with st.spinner("Kör Monte Carlo…"):
                    mc_res = monte_carlo(df, n_sim, n_days, ci_val)
                st.session_state.mc_cache[ticker] = (mc_res, mc_params_now, mc_now_ts)
            else:
                mc_res    = mc_cached[0]
                mc_now_ts = mc_cached[2]

            mc_calc_time = time.strftime("%H:%M:%S", time.localtime(mc_now_ts))
            st.caption(f"Senast beräknad: **{mc_calc_time}** &nbsp;|&nbsp; Uppdateras var **{st.session_state.mc_interval_min} min**")

            st.plotly_chart(chart_mc(mc_res, selected), use_container_width=True)

        st.divider()
        c1,c2,c3,c4,c5 = st.columns(5)
        var_chg = (mc_res["var"] - mc_res["last"]) / mc_res["last"] * 100
        c1.metric("Förväntat slutpris",     f"{mc_res['expected']:.2f} SEK")
        c2.metric(f"VaR ({ci_val:.0%})",    f"{mc_res['var']:.2f} SEK", f"{var_chg:.1f}%")
        c3.metric("Sannolikhet vinst",       f"{mc_res['prob_profit']:.1f}%")
        c4.metric("Daglig drift (μ)",         f"{mc_res['mu']*100:.3f}%")
        c5.metric("Årsvolatilitet (σ)",       f"{mc_res['sigma']*np.sqrt(252)*100:.1f}%")

        st.subheader("Prispercentilar på slutdagen")
        pc1,pc2,pc3,pc4,pc5 = st.columns(5)
        for col, lbl, key in [
            (pc1,"P5 (Pessimistisk)","p5"), (pc2,"P25","p25"),
            (pc3,"P50 (Median)","p50"),     (pc4,"P75","p75"),
            (pc5,"P95 (Optimistisk)","p95"),
        ]:
            v   = mc_res[key]
            chg = (v - mc_res["last"]) / mc_res["last"] * 100
            col.metric(lbl, f"{v:.2f}", f"{chg:+.1f}%")

        # Distribution
        st.subheader("Fördelning av simulerade slutpriser")
        fig_d = go.Figure()
        fig_d.add_trace(go.Histogram(x=mc_res["finals"], nbinsx=60,
            marker_color="#40c4ff", opacity=0.75, name="Slutpriser"))
        fig_d.add_vline(x=mc_res["last"],    line_dash="dot", line_color="#fff",
                        annotation_text="Nuv. pris")
        fig_d.add_vline(x=mc_res["expected"],line_dash="dot", line_color="#00e676",
                        annotation_text="Förväntat")
        fig_d.add_vline(x=mc_res["var"],     line_dash="dot", line_color="#ff1744",
                        annotation_text=f"VaR {ci_val:.0%}")
        fig_d.update_layout(xaxis_title="Pris (SEK)", yaxis_title="Frekvens",
                            height=350, **_DARK)
        st.plotly_chart(fig_d, use_container_width=True)

        # VaR-tabell
        st.subheader("Value at Risk — Sammanfattning")
        var_df = pd.DataFrame([
            {"Konfidensgrad": "90%", "VaR (SEK)": f"{np.percentile(mc_res['finals'],10):.2f}",
             "Förlust %": f"{(np.percentile(mc_res['finals'],10)-mc_res['last'])/mc_res['last']*100:.1f}%"},
            {"Konfidensgrad": "95%", "VaR (SEK)": f"{np.percentile(mc_res['finals'],5):.2f}",
             "Förlust %": f"{(np.percentile(mc_res['finals'],5)-mc_res['last'])/mc_res['last']*100:.1f}%"},
            {"Konfidensgrad": "99%", "VaR (SEK)": f"{np.percentile(mc_res['finals'],1):.2f}",
             "Förlust %": f"{(np.percentile(mc_res['finals'],1)-mc_res['last'])/mc_res['last']*100:.1f}%"},
        ])
        st.dataframe(var_df, hide_index=True, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 5 — OMX30 SCREENER
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader("OMX30 — Komplett Screener")
        st.caption("Visar samtliga 30 aktier. Uppdateras var 60:e minut.")

        with st.spinner("Hämtar kurser för alla OMX30-aktier…"):
            all_p = fetch_all_latest()

        if all_p.empty:
            st.error("Kunde inte hämta marknadsdata.")
        else:
            last  = all_p.iloc[-1]
            d1r   = all_p.pct_change().iloc[-1]  * 100
            d1m   = (all_p.iloc[-1] / all_p.iloc[max(0, len(all_p)-21)]   - 1) * 100
            d3m   = (all_p.iloc[-1] / all_p.iloc[0]                        - 1) * 100
            vols  = all_p.pct_change().std() * np.sqrt(252) * 100

            scr = pd.DataFrame({
                "Pris (SEK)": last.round(2),
                "1D %":       d1r.round(2),
                "1M %":       d1m.round(2),
                "3M %":       d3m.round(2),
                "Årsvolat.":  vols.round(1),
            })

            def color_col(v):
                if isinstance(v, float):
                    return f"color: {'#00e676' if v > 0 else '#ff1744'}"
                return ""

            st.dataframe(
                scr.style
                   .applymap(color_col, subset=["1D %","1M %","3M %"])
                   .format({"Pris (SEK)":"{:.2f}","1D %":"{:+.2f}%",
                             "1M %":"{:+.2f}%","3M %":"{:+.2f}%","Årsvolat.":"{:.1f}%"}),
                use_container_width=True, height=900,
            )

            st.divider()

            # Heatmap (treemap)
            st.subheader("OMX30 Avkastningskarta — 3 månader")
            fig_tm = go.Figure(go.Treemap(
                labels=list(d3m.index),
                parents=[""] * len(d3m),
                values=d3m.abs().clip(lower=0.5),
                customdata=d3m.round(2).values,
                texttemplate="%{label}<br><b>%{customdata:+.1f}%</b>",
                marker=dict(
                    colors=d3m.values,
                    colorscale=[[0,"#c62828"],[0.5,"#1a1a35"],[1,"#1b5e20"]],
                    cmid=0, showscale=True,
                    colorbar=dict(title="3M %"),
                ),
            ))
            fig_tm.update_layout(height=450, **_DARK)
            st.plotly_chart(fig_tm, use_container_width=True)

            # Korrelationsmatris
            st.subheader("Korrelationsmatris (daglig avkastning)")
            corr  = all_p.pct_change().dropna().corr()
            fig_c = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale="RdBu", zmid=0,
                text=corr.round(2).values,
                texttemplate="%{text}", textfont=dict(size=7),
                colorbar=dict(title="ρ"),
            ))
            fig_c.update_layout(height=650, **_DARK)
            st.plotly_chart(fig_c, use_container_width=True)

            # Top/Bottom performers
            st.divider()
            col_top, col_bot = st.columns(2)
            with col_top:
                st.subheader("Bästa 5 (3M)")
                st.dataframe(
                    d3m.nlargest(5).rename("3M %").reset_index()
                       .rename(columns={"index":"Aktie"})
                       .style.format({"3M %":"{:+.2f}%"}),
                    hide_index=True, use_container_width=True
                )
            with col_bot:
                st.subheader("Sämsta 5 (3M)")
                st.dataframe(
                    d3m.nsmallest(5).rename("3M %").reset_index()
                       .rename(columns={"index":"Aktie"})
                       .style.format({"3M %":"{:+.2f}%"}),
                    hide_index=True, use_container_width=True
                )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 6 — ADMIN / BACKDOOR  (dold utan korrekt kod)
    # ═════════════════════════════════════════════════════════════════════════
    if st.session_state.admin:
        with tabs[5]:
            st.markdown("""
            <div class="admin-banner">
            <strong>ADMIN-LÄGE AKTIVT</strong> — Du har full åtkomst till rådata,
            API-debug, anpassade modeller och bulk-export.
            Åtkomstkod: <code>OMXPRO-BACKDOOR-2024</code>
            </div>
            """, unsafe_allow_html=True)

            a_tabs = st.tabs([
                "Rådata", "API Debug", "Avancerad DCF",
                "Bulk Export", "Systemkontroll",
            ])

            # ── Rådata ───────────────────────────────────────────────────────
            with a_tabs[0]:
                st.subheader("Rådata — direkt från Yahoo Finance")
                choice = st.selectbox("Dataset", [
                    "Prishistorik (OHLCV)",
                    "Bolagsinformation (info-dict)",
                    "Resultaträkning (årsvis)",
                    "Resultaträkning (kvartalsvis)",
                    "Balansräkning",
                    "Kassaflödesanalys",
                ])
                dataset_map = {
                    "Prishistorik (OHLCV)":            df,
                    "Bolagsinformation (info-dict)":    pd.DataFrame(list(info.items()), columns=["Nyckel","Värde"]),
                    "Resultaträkning (årsvis)":         fin["income"],
                    "Resultaträkning (kvartalsvis)":    fin["q_income"],
                    "Balansräkning":                    fin["balance"],
                    "Kassaflödesanalys":                fin["cashflow"],
                }
                raw = dataset_map[choice]
                if isinstance(raw, pd.DataFrame) and not raw.empty:
                    st.dataframe(raw, use_container_width=True, height=500)
                    csv_bytes = raw.to_csv().encode()
                    st.download_button(f"Ladda ner {choice}.csv", csv_bytes,
                                       f"{ticker}_{choice.split()[0]}.csv", "text/csv")
                else:
                    st.info("Ingen data tillgänglig för detta dataset.")

            # ── API Debug ────────────────────────────────────────────────────
            with a_tabs[1]:
                st.subheader("API Debug — Yahoo Finance metadata")
                st.json({
                    "ticker":          ticker,
                    "period":          period,
                    "data_points":     len(df),
                    "first_date":      str(df.index[0].date()) if not df.empty else "N/A",
                    "last_date":       str(df.index[-1].date()) if not df.empty else "N/A",
                    "cache_ttl_s":     3600,
                    "source":          "Yahoo Finance via yfinance",
                    "info_keys_count": len(info),
                    "info_keys_sample": list(info.keys())[:40] if info else [],
                    "fin_shapes": {
                        "income":   str(fin["income"].shape)   if not fin["income"].empty   else "empty",
                        "balance":  str(fin["balance"].shape)  if not fin["balance"].empty  else "empty",
                        "cashflow": str(fin["cashflow"].shape) if not fin["cashflow"].empty else "empty",
                    },
                    "info_snapshot": {k: v for k, v in list(info.items())[:20]},
                })
                st.divider()
                st.subheader("Fullständig info-dict (filtrerad)")
                search = st.text_input("Sök nyckel", placeholder="t.ex. revenue, debt, cash…")
                filtered = {k: v for k, v in info.items()
                            if not search or search.lower() in k.lower()}
                st.json(filtered)

            # ── Avancerad DCF ────────────────────────────────────────────────
            with a_tabs[2]:
                st.subheader("Avancerad DCF — Fullständiga parametrar")
                col1, col2, col3 = st.columns(3)
                with col1:
                    ag = st.number_input("Tillväxttakt år 1 (%)", 0.0, 100.0, 8.0, 0.5) / 100
                    aw = st.number_input("WACC (%)", 1.0, 30.0, 9.0, 0.5) / 100
                    at = st.number_input("Terminal tillväxt (%)", 0.5, 5.0, 2.5, 0.25) / 100
                with col2:
                    ay = st.number_input("Prognosår", 3, 20, 10)
                    adecay = st.number_input("Tillväxtavtagning/år (%)", 50.0, 100.0, 85.0, 5.0) / 100
                with col3:
                    a_rf  = st.number_input("Riskfri ränta (%)", 0.0, 10.0, 3.8, 0.1) / 100
                    a_erp = st.number_input("Aktieriskpremie (%)", 0.0, 15.0, 5.5, 0.5) / 100
                    a_tax = st.number_input("Skattesats (%)", 0.0, 50.0, 20.6, 0.5) / 100

                if st.button("▶ Kör avancerad DCF", type="primary"):
                    adv = dcf_valuation(info, fin["cashflow"], ag, aw, at, ay, current_price=cp, q_cashflow_df=fin["q_cashflow"])
                    if "error" in adv:
                        st.error(adv["error"])
                    else:
                        st.success(f"Intrinsiskt värde: **{adv['intrinsic_value']:.2f} SEK** "
                                   f"| Säkerhetsmarginal: **{adv['margin_of_safety']:.1f}%**")
                        st.json({k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in adv.items() if k != "projected_fcf"})

            # ── Bulk Export ──────────────────────────────────────────────────
            with a_tabs[3]:
                st.subheader("Bulk-export — Samtliga OMX30-aktier")
                st.info("Hämtar nyckeltal för alla 30 aktier. Kan ta 30–60 sekunder.")

                if st.button("▶ Starta bulk-analys", type="primary"):
                    rows    = []
                    prog    = st.progress(0)
                    status  = st.empty()
                    for i, (name, t) in enumerate(OMX30.items()):
                        status.caption(f"Hämtar {name}…")
                        try:
                            inf = fetch_info(t)
                            rows.append({
                                "Aktie":          name,
                                "Ticker":         t,
                                "Pris (SEK)":     inf.get("currentPrice"),
                                "P/E":            inf.get("trailingPE"),
                                "P/B":            inf.get("priceToBook"),
                                "EV/EBITDA":      inf.get("enterpriseToEbitda"),
                                "ROE (%)":        round((inf.get("returnOnEquity") or 0)*100, 2),
                                "Nettomarginal (%)": round((inf.get("profitMargins") or 0)*100, 2),
                                "Tillväxt (%)":   round((inf.get("revenueGrowth") or 0)*100, 2),
                                "Beta (OMXS30)":  calc_beta_omx(t),
                                "Utdelning (%)":  round((get_dividend_yield(inf) or 0)*100, 2),
                                "Börsvärde":      inf.get("marketCap"),
                            })
                        except Exception as e:
                            rows.append({"Aktie": name, "Ticker": t, "Pris (SEK)": f"FEL: {e}"})
                        prog.progress((i + 1) / len(OMX30))

                    status.empty()
                    bulk = pd.DataFrame(rows)
                    st.dataframe(bulk, use_container_width=True)
                    st.download_button(
                        "Ladda ner OMX30_bulk.csv",
                        bulk.to_csv(index=False).encode(),
                        "OMX30_bulk_export.csv", "text/csv"
                    )

            # ── Systemkontroll ───────────────────────────────────────────────
            with a_tabs[4]:
                st.subheader("Systemkontroll & Cache")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("Cache TTL",            "3 600 sekunder (1 h)")
                    st.metric("Sessionsadmin",        "Aktiv")
                    st.metric("Datakälla",            "Yahoo Finance")
                    st.metric("Uppdateringsfrekvens", "Automatisk (1 h)")

                with col_s2:
                    if st.button("Rensa all cache", type="primary"):
                        st.cache_data.clear()
                        st.success("Cache rensad — all data hämtas på nytt vid nästa anrop.")
                        st.rerun()

                    if st.button("Ladda om sidan"):
                        st.rerun()

                    if st.button("Logga ut från admin"):
                        st.session_state.admin = False
                        st.rerun()

                st.divider()
                st.subheader("OMX30-konfiguration (live)")
                st.json(OMX30)


if __name__ == "__main__":
    main()
