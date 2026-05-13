"""
OMX30 Analyzer — Teknisk, Fundamental, DCF & Monte Carlo
Kurs uppdateras automatiskt varje timme via Streamlit cache.
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
from datetime import datetime, timedelta
import warnings
import json
import io

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OMX30 Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global dark theme */
    .stApp { background-color: #0a0a14; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1e 0%, #111128 100%);
        border-right: 1px solid #1e1e3a;
    }
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #12122a 0%, #1a1a35 100%);
        border: 1px solid #252550;
        border-radius: 10px;
        padding: 12px 16px;
    }
    /* Positive/Negative colors */
    .pos  { color: #00e676; font-weight: 700; }
    .neg  { color: #ff1744; font-weight: 700; }
    .neu  { color: #90a4ae; }
    /* Signal badges */
    .badge-buy  { background:#00e676; color:#000; padding:3px 10px;
                  border-radius:5px; font-weight:700; font-size:13px; }
    .badge-sell { background:#ff1744; color:#fff; padding:3px 10px;
                  border-radius:5px; font-weight:700; font-size:13px; }
    .badge-neu  { background:#f59e0b; color:#000; padding:3px 10px;
                  border-radius:5px; font-weight:700; font-size:13px; }
    /* Admin banner */
    .admin-banner {
        background: linear-gradient(90deg, #1a0000, #330000);
        border: 1px solid #ff1744;
        border-radius: 8px;
        padding: 10px 16px;
        color: #ff6666;
        font-size: 13px;
        margin-bottom: 10px;
    }
    /* Section headers */
    h2, h3 { color: #7c9eff !important; }
    /* Tabs */
    button[data-baseweb="tab"] { font-size: 14px; }
    /* Divider */
    hr { border-color: #1e1e3a; }
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
    df = yf.Ticker(ticker).history(period=period)
    if not df.empty:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@st.cache_data(ttl=300, show_spinner=False)   # 5 min för realtidspris
def fetch_realtime_price(ticker: str) -> float | None:
    try:
        fi = yf.Ticker(ticker).fast_info
        return float(fi.last_price)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def calc_beta_omx(ticker: str) -> float:
    """Beräknar beta mot OMXS30 (^OMX) från 2 års daglig avkastning."""
    try:
        stock = yf.Ticker(ticker).history(period="2y")["Close"].pct_change().dropna()
        omx   = yf.Ticker("^OMX").history(period="2y")["Close"].pct_change().dropna()
        common = stock.index.intersection(omx.index)
        if len(common) < 100:
            return float(fetch_info(ticker).get("beta") or 1.0)
        s, m = stock.loc[common].values, omx.loc[common].values
        beta = float(np.cov(s, m)[0][1] / np.var(m))
        return round(beta, 3)
    except Exception:
        return float(fetch_info(ticker).get("beta") or 1.0)


def get_dividend_yield(info: dict) -> float | None:
    """yfinance returnerar ibland dividendYield som % istället för decimal — använd trailing."""
    v = info.get("trailingAnnualDividendYield")
    if v and v > 0:
        return float(v)
    raw = info.get("dividendYield")
    if raw and raw > 0:
        # Om värdet är > 0.20 är det troligen fel (t.ex. 0.47 = 47%) — ignorera
        return float(raw) if raw < 0.20 else None
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    def safe(fn):
        try:
            r = fn()
            return r if r is not None and not r.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    return {
        "income":    safe(lambda: t.income_stmt),
        "balance":   safe(lambda: t.balance_sheet),
        "cashflow":  safe(lambda: t.cashflow),
        "q_income":  safe(lambda: t.quarterly_income_stmt),
    }


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
) -> dict:
    # Hämta fritt kassaflöde
    fcf = info.get("freeCashflow")
    if not fcf or fcf <= 0:
        try:
            if not cashflow_df.empty:
                ocf  = cashflow_df.loc["Operating Cash Flow"].iloc[0]  if "Operating Cash Flow"  in cashflow_df.index else None
                capx = cashflow_df.loc["Capital Expenditure"].iloc[0]  if "Capital Expenditure"  in cashflow_df.index else None
                if ocf and capx:
                    fcf = float(ocf) + float(capx)
        except Exception:
            pass

    if not fcf or fcf <= 0:
        return {"error": "Otillräcklig kassaflödesdata — DCF ej tillgänglig"}

    # Tillväxttakt
    if growth_rate is None:
        rg = info.get("revenueGrowth") or 0.05
        eg = info.get("earningsGrowth") or 0.05
        growth_rate = float(np.clip((rg + eg) / 2, 0.01, 0.30))

    # WACC — beta beräknas mot OMXS30 (inte S&P 500)
    if wacc is None:
        ticker_sym = info.get("symbol", "")
        beta      = calc_beta_omx(ticker_sym) if ticker_sym else float(info.get("beta") or 1.0)
        rf        = 0.038
        erp       = 0.055
        ke        = rf + beta * erp
        td        = float(info.get("totalDebt")  or 0)
        mc        = float(info.get("marketCap")  or 1)
        total_cap = td + mc
        wd        = td / total_cap if total_cap > 0 else 0.2
        we        = 1 - wd
        kd        = 0.04
        tax       = 0.206
        wacc      = float(np.clip(we * ke + wd * kd * (1 - tax), 0.06, 0.25))

    # Projicera FCF med avtagande tillväxt
    g, decay = growth_rate, 0.85
    proj_fcf = []
    for i in range(1, years + 1):
        proj_fcf.append(fcf * (1 + g) ** i)
        g *= decay

    # Terminal Value (Gordon Growth)
    tv  = proj_fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv  = sum(cf / (1 + wacc) ** (i + 1) for i, cf in enumerate(proj_fcf))
    pvt = tv / (1 + wacc) ** years
    ev  = pv + pvt

    td      = float(info.get("totalDebt")         or 0)
    cash    = float(info.get("totalCash")          or 0)
    shares  = float(info.get("sharesOutstanding")  or 1)
    eq_val  = ev - td + cash
    iv      = eq_val / shares if shares > 0 else 0
    cp      = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    mos     = (iv - cp) / iv * 100 if iv > 0 else 0

    return {
        "fcf_base": fcf, "projected_fcf": proj_fcf,
        "pv_fcf": pv, "pv_terminal": pvt,
        "enterprise_value": ev, "intrinsic_value": iv,
        "current_price": cp, "margin_of_safety": mos,
        "wacc": wacc, "growth_rate": growth_rate,
        "terminal_growth": terminal_growth,
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
    ret         = df["Close"].pct_change().dropna()
    mu, sigma   = float(ret.mean()), float(ret.std())
    last        = float(df["Close"].iloc[-1])

    rng  = np.random.default_rng(seed=42)
    sim  = np.zeros((n_days, n_sim))
    for i in range(n_sim):
        dr         = rng.normal(mu, sigma, n_days)
        sim[:, i]  = last * np.exp(np.cumsum(np.log1p(dr)))

    finals = sim[-1]
    return {
        "sim":          sim,
        "finals":       finals,
        "last":         last,
        "mu":           mu,
        "sigma":        sigma,
        "expected":     float(np.mean(finals)),
        "var":          float(np.percentile(finals, (1 - ci) * 100)),
        "prob_profit":  float(np.mean(finals > last) * 100),
        "p5":           float(np.percentile(finals, 5)),
        "p25":          float(np.percentile(finals, 25)),
        "p50":          float(np.percentile(finals, 50)),
        "p75":          float(np.percentile(finals, 75)),
        "p95":          float(np.percentile(finals, 95)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
_DARK = dict(template="plotly_dark",
             paper_bgcolor="#0a0a14",
             plot_bgcolor="#0d0d1e",
             font=dict(color="#c0c8e8"))


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
        legend=dict(bgcolor="rgba(0,0,0,.5)", bordercolor="#333"),
        **_DARK)
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
def fmt_big(v, suffix="SEK"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    av = abs(v)
    if av >= 1e12: return f"{v/1e12:.2f}T {suffix}"
    if av >= 1e9:  return f"{v/1e9:.2f}Mdr {suffix}"
    if av >= 1e6:  return f"{v/1e6:.1f}M {suffix}"
    return f"{v:.2f} {suffix}"


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

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📈 OMX30 Analyzer Pro")
        st.caption(f"⏱ Senast uppdaterad: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption("Kurs uppdateras automatiskt var 60:e minut")
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
        if st.button("🔄 Tvinga uppdatering", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # ── BACKDOOR ENTRY ────────────────────────────────────────────────
        with st.expander("⚙️ Avancerade inställningar", expanded=False):
            st.caption("Systemkonfiguration")
            key = st.text_input("Åtkomstkod", type="password",
                                placeholder="●●●●●●●●●●●●●●●●")
            if key == "OMXPRO-BACKDOOR-2024":
                st.session_state.admin = True
                st.success("✅ Fullständig åtkomst aktiverad")
            elif key and key != "OMXPRO-BACKDOOR-2024":
                st.error("Ogiltig kod")
                st.session_state.admin = False

        if st.session_state.admin:
            st.markdown(
                '<div class="admin-banner">🔓 ADMIN-LÄGE AKTIVT<br>'
                'Fullständig systemåtkomst</div>',
                unsafe_allow_html=True)

    # ── DATA ─────────────────────────────────────────────────────────────────
    with st.spinner(f"Hämtar data för {selected}…"):
        df   = fetch_history(ticker, period)
        info = fetch_info(ticker)
        fin  = fetch_financials(ticker)
        beta_omx = calc_beta_omx(ticker)

    if df is None or df.empty:
        st.error(f"Kunde inte hämta data för {ticker}.")
        return

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
        st.caption(f"Ticker: **{ticker}** &nbsp;|&nbsp; Börs: Nasdaq Stockholm &nbsp;|&nbsp; "
                   f"Valuta: SEK &nbsp;|&nbsp; Sektor: {info.get('sector','N/A')}")
    with col_h2:
        if st.session_state.admin:
            st.markdown('<div class="admin-banner" style="text-align:center">🔓 ADMIN</div>',
                        unsafe_allow_html=True)

    # ── QUICK METRICS ─────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Kurs (SEK)",     f"{cp:.2f}",         f"{chg:+.2f} ({pchg:+.2f}%)")
    m2.metric("52v Högst",      f"{df['High'].max():.2f}")
    m3.metric("52v Lägst",      f"{df['Low'].min():.2f}")
    m4.metric("Snittvolym",     fmt_big(df["Volume"].mean(), ""))
    m5.metric("Börsvärde",      fmt_big(info.get("marketCap")))
    m6.metric("P/E (trailing)", fmt_num(info.get("trailingPE")))
    st.divider()

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_labels = [
        "📊 Teknisk analys",
        "📋 Fundamental analys",
        "💰 DCF Värdering",
        "🎲 Monte Carlo",
        "🔍 OMX30 Screener",
    ]
    if st.session_state.admin:
        tab_labels.append("🔓 Admin / Backdoor")

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
        f1, f2, f3 = st.columns(3)

        with f1:
            st.subheader("Värderingsmultiplar")
            data = {
                "P/E (trailing)":   fmt_num(info.get("trailingPE")),
                "P/E (forward)":    fmt_num(info.get("forwardPE")),
                "PEG":              fmt_num(info.get("pegRatio")),
                "P/B":              fmt_num(info.get("priceToBook")),
                "P/S (TTM)":        fmt_num(info.get("priceToSalesTrailingTwelveMonths")),
                "EV/EBITDA":        fmt_num(info.get("enterpriseToEbitda")),
                "EV/Revenue":       fmt_num(info.get("enterpriseToRevenue")),
                "Enterprise Value": fmt_big(info.get("enterpriseValue")),
            }
            for k, v in data.items():
                st.metric(k, v)

        with f2:
            st.subheader("Lönsamhet")
            prof = {
                "Bruttomarginal":     fmt_pct(info.get("grossMargins")),
                "Rörelsemarginal":    fmt_pct(info.get("operatingMargins")),
                "Nettomarginal":      fmt_pct(info.get("profitMargins")),
                "ROE":                fmt_pct(info.get("returnOnEquity")),
                "ROA":                fmt_pct(info.get("returnOnAssets")),
                "EBITDA":             fmt_big(info.get("ebitda")),
                "Intäkter (TTM)":     fmt_big(info.get("totalRevenue")),
                "Nettoresultat":      fmt_big(info.get("netIncomeToCommon")),
            }
            for k, v in prof.items():
                st.metric(k, v)

        with f3:
            st.subheader("Finansiell hälsa & Aktie")
            health = {
                "Skuld/EK":           fmt_num(info.get("debtToEquity")),
                "Likviditetsgrad":    fmt_num(info.get("currentRatio")),
                "Snabblikviditet":    fmt_num(info.get("quickRatio")),
                "Total skuld":        fmt_big(info.get("totalDebt")),
                "Kassa":              fmt_big(info.get("totalCash")),
                "Fritt kassaflöde":   fmt_big(info.get("freeCashflow")),
                "Intäktstillväxt":    fmt_pct(info.get("revenueGrowth")),
                "Vinsttillväxt":      fmt_pct(info.get("earningsGrowth")),
            }
            for k, v in health.items():
                st.metric(k, v)

        st.divider()
        f4, f5 = st.columns(2)
        with f4:
            st.subheader("Utdelning & Aktie")
            div_data = {
                "Direktavkastning":   fmt_pct(get_dividend_yield(info)),
                "Utdelningsandel":    fmt_pct(info.get("payoutRatio")),
                "Utdelning/aktie":    fmt_num(info.get("trailingAnnualDividendRate") or info.get("dividendRate")),
                "Beta (vs OMXS30)":   fmt_num(beta_omx),
                "Antal aktier":       fmt_big(info.get("sharesOutstanding"), ""),
                "Float":              fmt_big(info.get("floatShares"), ""),
                "Blankning %":        fmt_pct(info.get("shortPercentOfFloat")),
                "52v förändring":     fmt_pct(info.get("52WeekChange")),
            }
            for k, v in div_data.items():
                st.metric(k, v)

        with f5:
            st.subheader("Insider & Analyst")
            analyst = {
                "Analytiker-riktkurs":  fmt_num(info.get("targetMeanPrice")),
                "Riktkurs Hög":         fmt_num(info.get("targetHighPrice")),
                "Riktkurs Låg":         fmt_num(info.get("targetLowPrice")),
                "Rekommendation":       info.get("recommendationKey", "N/A"),
                "Antal analytiker":     fmt_num(info.get("numberOfAnalystOpinions"), 0),
                "Insider-ägarandel":    fmt_pct(info.get("heldPercentInsiders")),
                "Institutionell äg.":  fmt_pct(info.get("heldPercentInstitutions")),
                "Kortränta":            fmt_pct(info.get("shortRatio")),
            }
            for k, v in analyst.items():
                st.metric(k, v)

        # Historiska finansiella grafer
        if not fin["income"].empty:
            st.divider()
            st.subheader("Historisk Resultaträkning")
            fig_inc = chart_hist_financials(
                fin["income"],
                ["Total Revenue","Gross Profit","Operating Income","Net Income"],
                "Resultaträkning (Mdr SEK)"
            )
            if fig_inc:
                st.plotly_chart(fig_inc, use_container_width=True)

        if not fin["cashflow"].empty:
            fig_cf = chart_hist_financials(
                fin["cashflow"],
                ["Operating Cash Flow","Free Cash Flow","Capital Expenditure"],
                "Kassaflöde (Mdr SEK)"
            )
            if fig_cf:
                st.plotly_chart(fig_cf, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — DCF
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("DCF-värdering (Discounted Cash Flow)")

        col_dcf, col_params = st.columns([3, 1])
        with col_params:
            st.markdown("**Anpassa parametrar**")
            raw_growth    = info.get("revenueGrowth") or 0.07
            init_growth   = float(np.clip(raw_growth, 0.0, 0.40))
            dcf_growth    = st.slider("Tillväxttakt år 1", 0.0, 0.40, init_growth, 0.01,
                                      format="%.0f%%")
            dcf_wacc      = st.slider("WACC",               0.04, 0.20, 0.09,        0.005,
                                      format="%.1f%%")
            dcf_terminal  = st.slider("Terminal tillväxt",  0.01, 0.05, 0.025,       0.005,
                                      format="%.1f%%")
            dcf_years     = st.slider("Prognosår",          3,    15,   5)

        with col_dcf:
            result = dcf_valuation(info, fin["cashflow"],
                                   dcf_growth, dcf_wacc, dcf_terminal, dcf_years)

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

                # Tabellöversikt
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

                # Scenarioanalys
                st.divider()
                st.subheader("Känslighetsanalys (Säkerhetsmarginal %)")
                growth_range = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
                wacc_range   = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
                sens_data    = {}
                for w in wacc_range:
                    row = {}
                    for g2 in growth_range:
                        r = dcf_valuation(info, fin["cashflow"], g2, w, dcf_terminal, dcf_years)
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
            n_sim   = st.slider("Simuleringar",   100, 3000, 500, 100)
            n_days  = st.slider("Handelsdagar",    60,  504, 252,  21)
            ci_val  = st.slider("Konfidensgrad",  0.90, 0.99, 0.95, 0.01)

        with mc_col:
            with st.spinner("Kör Monte Carlo…"):
                mc_res = monte_carlo(df, n_sim, n_days, ci_val)

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
                st.subheader("🏆 Bästa 5 (3M)")
                st.dataframe(
                    d3m.nlargest(5).rename("3M %").reset_index()
                       .rename(columns={"index":"Aktie"})
                       .style.format({"3M %":"{:+.2f}%"}),
                    hide_index=True, use_container_width=True
                )
            with col_bot:
                st.subheader("📉 Sämsta 5 (3M)")
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
            🔓 <strong>ADMIN-LÄGE AKTIVT</strong> — Du har full åtkomst till rådata,
            API-debug, anpassade modeller och bulk-export.
            Åtkomstkod: <code>OMXPRO-BACKDOOR-2024</code>
            </div>
            """, unsafe_allow_html=True)

            a_tabs = st.tabs([
                "📦 Rådata", "🛠 API Debug", "🔧 Avancerad DCF",
                "📤 Bulk Export", "⚡ Systemkontroll",
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
                    st.download_button(f"⬇ Ladda ner {choice}.csv", csv_bytes,
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
                    adv = dcf_valuation(info, fin["cashflow"], ag, aw, at, ay)
                    if "error" in adv:
                        st.error(adv["error"])
                    else:
                        st.success(f"✅ Intrinsiskt värde: **{adv['intrinsic_value']:.2f} SEK** "
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
                        "⬇ Ladda ner OMX30_bulk.csv",
                        bulk.to_csv(index=False).encode(),
                        "OMX30_bulk_export.csv", "text/csv"
                    )

            # ── Systemkontroll ───────────────────────────────────────────────
            with a_tabs[4]:
                st.subheader("Systemkontroll & Cache")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("Cache TTL",            "3 600 sekunder (1 h)")
                    st.metric("Sessionsadmin",        "Aktiv ✅")
                    st.metric("Datakälla",            "Yahoo Finance")
                    st.metric("Uppdateringsfrekvens", "Automatisk (1 h)")

                with col_s2:
                    if st.button("🗑 Rensa all cache", type="primary"):
                        st.cache_data.clear()
                        st.success("✅ Cache rensad — all data hämtas på nytt vid nästa anrop.")
                        st.rerun()

                    if st.button("🔄 Ladda om sidan"):
                        st.rerun()

                    if st.button("🔒 Logga ut från admin"):
                        st.session_state.admin = False
                        st.rerun()

                st.divider()
                st.subheader("OMX30-konfiguration (live)")
                st.json(OMX30)


if __name__ == "__main__":
    main()
