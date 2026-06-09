"""
etf_gen.py — Investment Performance & Portfolio Analytics Dashboard
====================================================================
Author : Shreyas Jagadish Kanagal
Stack  : Python · Streamlit · yfinance · Plotly · SciPy · pandas · NumPy

Overview
--------
A full-stack, single-file financial analytics web application built with Streamlit.
The platform ingests live and historical market data for 110+ institutional ETFs
across 14 distinct market sectors and exposes four interactive pages:

    1. Home            — Live benchmark ticker tape + key metric glossary
    2. ETF Screener    — Multi-filter screener with risk/return scatter visualization
    3. Explore ETFs    — Sector deep-dive: cumulative returns, rolling volatility,
                         correlation heatmap, and per-fund performance tables
    4. Portfolio Builder — Markowitz Mean-Variance Optimization (Maximum Sharpe &
                           Minimum Volatility) with efficient frontier simulation

Architecture
------------
- All application state is managed through Streamlit's session_state dictionary.
- Historical price data is fetched via yfinance and cached (TTL=300s) to minimize
  redundant API calls during user interactions.
- Quantitative optimization is performed with SciPy's SLSQP (Sequential Least
  Squares Programming) constrained minimizer.
- The entire design system is implemented as a single injected CSS block using
  CSS custom properties (variables) for full theme consistency.

Key Design Decisions
--------------------
- Single-file architecture: keeps the project portable and easy to deploy on
  Streamlit Cloud with zero build configuration.
- @st.cache_data decorators: historical data and live quotes are separately cached
  with different TTLs to balance freshness vs. performance.
- Graceful degradation: every data fetch is wrapped in try/except so the UI never
  crashes on a bad ticker or network timeout; it displays a contextual error instead.
"""

# ── Standard Library ──────────────────────────────────────────────────────────
from datetime import datetime, timedelta   # Date arithmetic for historical windows
from typing import Any, Optional           # Type hints for BASE_LAYOUT and yfinance returns
import warnings                            # Suppress noisy pandas/yfinance deprecation logs

# ── Third-Party: Data & Computation ───────────────────────────────────────────
import numpy as np                         # Vectorized math: portfolio variance, annualization
import pandas as pd                        # DataFrame operations: returns, rolling stats, YTD slicing
import yfinance as yf                      # Yahoo Finance API wrapper for price history & live quotes
from scipy.optimize import minimize        # SLSQP optimizer for Max-Sharpe / Min-Vol problems

# ── Third-Party: UI & Visualization ───────────────────────────────────────────
import streamlit as st                     # Web application framework — renders all UI components
import plotly.graph_objects as go          # Low-level Plotly: Scatter, Pie, Heatmap traces
from plotly.subplots import make_subplots  # Multi-panel chart layouts (e.g., 3-pie allocation chart)

warnings.filterwarnings("ignore")  # Suppress pandas FutureWarning and yfinance chatter

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIGURATION
# Must be the first Streamlit call in the script.
# Sets the browser tab title, favicon, overall layout width, and sidebar state.
# "wide" layout maximizes horizontal space for multi-column chart dashboards.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Investment Analytics Platform",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL DESIGN SYSTEM — CSS Custom Properties + Component Overrides
# ─────────────────────────────────────────────────────────────────────────────
# This single injected <style> block implements the full visual design system.
#
# Architecture:
#   - :root defines a complete CSS custom-property (variable) token set covering
#     color palette, typographic scale, and font stack — all component styles
#     reference these tokens so the entire theme can be changed in one place.
#   - Streamlit's internal component selectors (data-testid="...") are targeted
#     with !important overrides to replace its default light theme.
#   - Custom utility classes (.kpi-card, .badge, .section-title, .styled-table,
#     .pos/.neg/.neu etc.) are defined here and applied throughout all page
#     render functions via st.markdown(unsafe_allow_html=True).
#
# Font Choice:
#   EB Garamond (Google Fonts) was selected for its editorial, financial-
#   publication aesthetic — distinguishing the dashboard from typical
#   Bootstrap/Material app defaults.
#
# Color Palette (dark-mode financial terminal):
#   --bg        #0b0e14   Deep navy-black canvas
#   --surface   #111520   Card / panel background
#   --surface2  #161c2a   Elevated / hover state surface
#   --border    #1e2840   Subtle structural separator
#   --accent    #00d4aa   Primary teal — positive returns, CTAs, active states
#   --accent2   #4f8ef7   Secondary blue — secondary data series
#   --accent3   #f7c948   Amber — warning states, tertiary highlights
#   --red       #f74b5c   Negative returns, drawdowns, alerts
#   --text      #ffffff   Primary body copy
#   --text-sub  #e2e8f0   Secondary body copy
#   --muted     #a0aec0   Labels, metadata, disabled states
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400..800;1,400..800&display=swap');

:root {
    --bg:        #0b0e14;
    --surface:   #111520;
    --surface2:  #161c2a;
    --border:    #1e2840;
    --accent:    #00d4aa;
    --accent2:   #4f8ef7;
    --accent3:   #f7c948;
    --red:       #f74b5c;
    --text:      #ffffff;
    --text-sub:  #e2e8f0;
    --muted:     #a0aec0;
    --font:      'EB Garamond', Garamond, Georgia, serif;

    /* Type scale — one authoritative set used everywhere */
    --fs-xs:   0.85rem;   /* labels, badges, tags          */
    --fs-sm:   0.95rem;   /* table headers, secondary text  */
    --fs-body: 1.1rem;    /* body copy, descriptions        */
    --fs-ui:   1.05rem;   /* inputs, buttons, selects       */
    --fs-sub:  1.2rem;    /* sub-headers, card titles       */
    --fs-h3:   1.4rem;    /* section sub-titles             */
    --fs-h2:   1.8rem;    /* section titles                 */
    --fs-h1:   3.2rem;    /* page hero title                */
}

/* ── Reset & Global ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-size: var(--fs-body) !important;
}

#MainMenu, footer { visibility: hidden; }
.block-container { padding: 2.5rem 3.5rem 4rem 3.5rem !important; max-width: 1650px !important; overflow: visible !important; }
/* Prevent stAppViewBlockContainer from clipping badge */
[data-testid="stAppViewBlockContainer"] { overflow: visible !important; }
[data-testid="stVerticalBlock"] { overflow: visible !important; }

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 300px !important;
    max-width: 300px !important;
    width: 300px !important;
    overflow: visible !important;
}
[data-testid="stSidebar"] > div:first-child {
    overflow: visible !important;
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
    font-family: var(--font) !important;
}
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Sidebar collapse button — « tab that peeks out past the sidebar edge ── */
[data-testid="stSidebarCollapseButton"] {
    position: fixed !important;
    top: 1rem !important;
    left: 300px !important;
    z-index: 9999 !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-left: none !important;
    border-radius: 0 5px 5px 0 !important;
    width: 1.4rem !important;
    height: 2.2rem !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    overflow: visible !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebarCollapseButton"]:hover {
    background: rgba(0,212,170,0.1) !important;
}
[data-testid="stSidebarCollapseButton"]::after {
    content: "«" !important;
    color: var(--muted) !important;
    font-size: 0.85rem !important;
    font-family: serif !important;
    display: block !important;
    line-height: 1 !important;
}
[data-testid="stSidebarCollapseButton"] > *,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapseButton"] kbd { display: none !important; }

/* ── Keyboard shortcut tooltip — hide everywhere ── */
div[class*="keyboardShortcut"],
div[class*="keyboard-shortcut"],
[aria-label*="keyboard"],
[aria-label*="shortcut"] { display: none !important; visibility: hidden !important; }
div[data-testid="stSidebar"] kbd { display: none !important; }

/* ── Re-open button when sidebar is fully collapsed ── */
[data-testid="collapsedControl"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-left: none !important;
    border-radius: 0 5px 5px 0 !important;
    width: 1.4rem !important;
    height: 2.2rem !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
}
[data-testid="collapsedControl"]::after {
    content: "»" !important;
    color: var(--muted) !important;
    font-size: 0.85rem !important;
    font-family: serif !important;
    display: block !important;
    line-height: 1 !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] span { display: none !important; }

/* ── Sidebar logo / wordmark ── */
.sidebar-logo {
    padding: 1.8rem 1.4rem 1.4rem 1.4rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.4rem;
}
.sidebar-logo-mark {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.5rem;
}
.sidebar-logo-icon {
    width: 28px;
    height: 28px;
    background: var(--accent);
    border-radius: 5px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.sidebar-logo-wordmark {
    font-family: var(--font);
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #ffffff;
    line-height: 1;
}
.sidebar-logo-wordmark span { color: var(--accent); }
.sidebar-logo-sub {
    font-family: var(--font);
    font-size: 0.68rem;
    font-weight: 400;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #a0aec0;
    padding-left: 2px;
    margin-top: 0.1rem;
}

/* ── Navigation Buttons ── */
[data-testid="stSidebar"] button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-left: 4px solid transparent !important;
    color: var(--text-sub) !important;
    font-family: var(--font) !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    padding: 0.75rem 1.2rem !important;
    text-align: left !important;
    width: calc(100% - 28px) !important; 
    margin-left: 16px !important; 
    margin-bottom: 0.6rem !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
/* Hide native browser tooltip on sidebar nav */
[data-testid="stSidebar"] button[title],
[data-testid="stSidebar"] button[data-testid] {
    pointer-events: auto !important;
}
[data-testid="stSidebar"] button::after {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"],
[data-testid="stSidebar"] [data-testid="stTooltipContent"],
[data-testid="stSidebar"] .stTooltipIcon {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stSidebar"] button:hover {
    border-left-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(0,212,170,0.06) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
}

/* ── Main canvas action buttons ── */
section.main button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text-sub) !important;
    font-family: var(--font) !important;
    font-size: var(--fs-ui) !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.6rem !important;
    transition: border-color 0.15s, color 0.15s !important;
    box-shadow: none !important;
}
section.main button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--surface2) !important;
    box-shadow: none !important;
}
section.main button:focus {
    box-shadow: none !important;
    outline: none !important;
}

/* ── Metric containers ── */
div[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem 1.2rem !important;
}
div[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: var(--fs-xs) !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-family: var(--font) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.7rem !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: var(--font) !important;
    font-size: var(--fs-sm) !important;
}

/* ── Dropdowns / Selects / Multiselect ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-size: var(--fs-ui) !important;
    font-family: var(--font) !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label {
    font-family: var(--font) !important;
    font-size: var(--fs-sm) !important;
    color: var(--muted) !important;
    letter-spacing: 0.04em !important;
}

/* ── Sliders ── */
div[data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}
div[data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
    background: var(--accent) !important;
}

/* ── Spinner / status text ── */
[data-testid="stStatusWidget"] p,
div[data-testid="stSpinner"] p {
    font-family: var(--font) !important;
    font-size: var(--fs-sm) !important;
}

/* ── Warnings / Info boxes (native Streamlit) ── */
div[data-testid="stAlert"] p {
    font-family: var(--font) !important;
    font-size: var(--fs-body) !important;
}

/* ════════════════════════════════════════════
   TYPE SYSTEM
   ════════════════════════════════════════════ */

.page-title {
    font-family: var(--font);
    font-size: var(--fs-h1);
    font-weight: 700;
    line-height: 1.1;
    color: var(--text);
    margin: 0.5rem 0 1.2rem 0;
}
.page-title span { color: var(--accent); }

.badge {
    display: inline-block;
    font-family: var(--font);
    font-size: var(--fs-xs);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 3px;
    padding: 0.22rem 0.8rem;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
    font-weight: 600;
    position: relative;
    z-index: 1;
}

.section-title {
    font-family: var(--font);
    font-size: var(--fs-h2);
    font-weight: 600;
    color: var(--text);
    margin: 0 0 0.4rem 0;
}
.section-sub {
    font-family: var(--font);
    font-size: var(--fs-body);
    color: var(--text-sub);
    margin-bottom: 1.6rem;
    line-height: 1.7;
}

.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.5rem;
}
.kpi-label {
    font-family: var(--font);
    font-size: var(--fs-xs);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-family: var(--font);
    font-size: 1.65rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.2;
}
.kpi-delta {
    font-family: var(--font);
    font-size: var(--fs-sm);
    color: var(--muted);
    margin-top: 0.25rem;
}
.up   { color: #00d4aa !important; }
.down { color: #f74b5c !important; }

.info-box, .warn-box {
    background: var(--surface2);
    border-radius: 4px;
    padding: 1rem 1.4rem;
    font-family: var(--font);
    font-size: var(--fs-body);
    color: var(--text-sub);
    line-height: 1.7;
    margin: 1rem 0;
}
.info-box { border-left: 3px solid var(--accent2); }
.warn-box { border-left: 3px solid var(--accent3); }

.divider { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

/* ── Tabular Matrix Displays ── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font);
    font-size: var(--fs-body);
    margin-top: 0.5rem;
}
.styled-table th {
    font-size: var(--fs-sm);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding: 0.75rem 0.9rem;
    text-align: right;
    white-space: nowrap;
    font-family: var(--font);
    font-weight: 500;
}
.styled-table th:first-child,
.styled-table th:nth-child(2) { text-align: left; }
.styled-table td {
    padding: 0.75rem 0.9rem;
    border-bottom: 1px solid rgba(30,40,64,0.5);
    color: var(--text-sub);
    text-align: right;
    white-space: nowrap;
    font-family: var(--font);
    font-size: var(--fs-body);
}
.styled-table td:first-child { color: var(--accent2); font-weight: 600; text-align: left; }
.styled-table td:nth-child(2) { color: var(--text); text-align: left; font-weight: 500; }
.styled-table tr:hover { background: var(--surface2); }
.pos { color: #00d4aa !important; font-weight: 600; }
.neg { color: #f74b5c !important; font-weight: 600; }
.neu { color: #a0aec0 !important; }

.metric-def {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}
.metric-def-name {
    font-family: var(--font);
    font-size: var(--fs-sub);
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 0.4rem;
}
.metric-def-body {
    font-family: var(--font);
    font-size: var(--fs-body);
    color: var(--text-sub);
    line-height: 1.65;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY GLOBAL LAYOUT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
# BASE_LAYOUT is a dict of Plotly layout properties shared by every chart in the
# application.  Each chart calls fig.update_layout(**BASE_LAYOUT) first, then
# applies its own overrides (title, height, axis labels etc.) on top.
#
# Key decisions:
#   - paper_bgcolor is transparent so charts sit flush on the CSS --surface panel.
#   - plot_bgcolor matches --surface (#111520) to keep the axis area on-brand.
#   - gridcolor uses --border (#1e2840) for subtle reference lines.
#   - hovermode="x unified" shows a single vertical tooltip bar across all series,
#     which is cleaner than per-point tooltips for time-series comparisons.
#   - All font references use EB Garamond to match the design system.
# ─────────────────────────────────────────────────────────────────────────────
BASE_LAYOUT: dict[str, Any] = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "#111520",
    "font": dict(family="EB Garamond, Garamond, serif", color="#a0aec0", size=14),
    "title_font": dict(family="EB Garamond, Garamond, serif", color="#ffffff", size=18),
    "xaxis": dict(gridcolor="#1e2840", showgrid=True, zeroline=False, tickfont=dict(color="#e2e8f0", family="EB Garamond, Garamond, serif")),
    "yaxis": dict(gridcolor="#1e2840", showgrid=True, zeroline=False, tickfont=dict(color="#e2e8f0", family="EB Garamond, Garamond, serif")),
    "hovermode": "x unified",
}

# Ordered color palette for multi-series charts (up to 10 distinct data series).
# Colors are chosen for contrast against the dark background and to avoid
# red/green-only reliance (accessibility consideration).
COLORS = ["#00d4aa","#4f8ef7","#f7c948","#f74b5c","#a78bfa",
          "#fb923c","#34d399","#f472b6","#38bdf8","#facc15"]

# ─────────────────────────────────────────────────────────────────────────────
# DATA INGESTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
# Two cached functions handle all external data fetching:
#
#   download_data()   — Bulk historical OHLCV via yfinance batch download.
#                       Returns a DataFrame of adjusted Close prices indexed by date.
#                       TTL=300s (5 min) balances freshness vs. API rate limits.
#
#   get_live_quotes() — Per-ticker fast_info (last price + previous close) for
#                       the live ticker tape on the Home page.
#                       TTL=60s for near-real-time quote display.
#
# Both functions are decorated with @st.cache_data so Streamlit memoizes the
# result keyed by the function arguments.  Re-running the app (e.g., on a slider
# change) will return the cached result instead of re-fetching, dramatically
# improving perceived performance.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def download_data(tickers: list, start: str = "2020-01-01") -> pd.DataFrame:
    """
    Fetch adjusted daily closing prices for a list of ticker symbols.

    Uses yfinance's batch downloader which issues a single HTTP request for all
    tickers, then normalises the result into a clean (date × ticker) DataFrame.

    Handles three yfinance response shapes:
      1. MultiIndex columns (batch download of 2+ tickers) → slice ["Close"] level
      2. Flat columns with a "Close" column (single ticker batch)  → extract directly
      3. pd.Series result (edge case for single tickers) → promote to DataFrame

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols (e.g. ["SPY", "QQQ", "AGG"]).
    start : str
        ISO-format start date string, e.g. "2020-01-01".

    Returns
    -------
    pd.DataFrame
        Columns = ticker symbols, index = trading dates.
        Columns with all-NaN data are dropped (delisted / bad tickers).
        Returns an empty DataFrame on any fetch failure so callers can handle
        gracefully without crashing the UI.
    """
    if not tickers:
        return pd.DataFrame()
    try:
        # auto_adjust=True applies split/dividend adjustments so all prices
        # are on a comparable scale for return calculations.
        raw: Optional[pd.DataFrame] = yf.download(tickers, start=start, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return pd.DataFrame()
        # Normalise column structure regardless of single vs. multi-ticker download
        if isinstance(raw.columns, pd.MultiIndex):
            data = raw["Close"]               # Multi-ticker: (metric, ticker) → slice Close level
        else:
            data = raw[["Close"]] if "Close" in raw.columns else raw  # Single-ticker flat layout
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])   # Ensure consistent DataFrame output
        # Drop entirely empty columns (bad tickers) and rows (non-trading days at boundaries)
        data = data.dropna(axis=1, how="all").dropna(how="all")
        return data
    except Exception as e:
        st.error(f"Ingestion Exception: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def get_live_quotes(tickers: list) -> dict:
    """
    Retrieve current price and daily change data for a list of tickers.

    Uses yfinance's Ticker.fast_info — a lightweight metadata endpoint that
    avoids pulling full OHLCV history just for the current price.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols.

    Returns
    -------
    dict
        Keys are ticker symbols.  Each value is a dict with:
          "price"  : float | None — Last traded price (rounded to 2 dp)
          "change" : float | None — Absolute price change from previous close
          "pct"    : float | None — Percentage change from previous close
        Missing data fields are set to None so the UI can display "—" placeholders.
    """
    results = {}
    for t in tickers:
        try:
            fi    = yf.Ticker(t).fast_info
            price = fi.last_price
            prev  = fi.previous_close
            results[t] = {
                "price":  round(price, 2) if price else None,
                "change": round(price - prev, 2) if price and prev else None,
                "pct":    round((price - prev) / prev * 100, 2) if price and prev else None,
            }
        except Exception:
            # Silently absorb per-ticker failures; partial data is still useful
            results[t] = {"price": None, "change": None, "pct": None}
    return results


def safe_stats(returns: pd.DataFrame, rf: float = 0.045):
    """
    Compute annualised risk/return statistics for a DataFrame of daily returns.

    All metrics are annualized assuming 252 trading days per year.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily percentage returns, columns = tickers.
    rf : float
        Annualised risk-free rate (decimal).  Default 4.5% reflects typical
        short-duration Treasury yield at time of writing.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        ann_ret  — Annualised arithmetic mean return per ticker
        ann_vol  — Annualised standard deviation (volatility) per ticker
        sharpe   — Sharpe ratio: (ann_ret - rf) / ann_vol (NaN where vol = 0)
        max_dd   — Maximum drawdown: largest peak-to-trough decline in the period
    """
    ann_ret = returns.mean() * 252                          # Scale daily mean to annual
    ann_vol = returns.std() * np.sqrt(252)                  # Scale daily σ to annual via √T rule
    sharpe  = (ann_ret - rf) / ann_vol.replace(0, np.nan)  # Avoid division by zero on flat series
    cum     = (1 + returns).cumprod()                       # Cumulative wealth index
    max_dd  = (cum / cum.cummax() - 1).min()               # Drawdown series minimum = max drawdown
    return ann_ret, ann_vol, sharpe, max_dd

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO OPTIMIZATION ENGINE — Markowitz Mean-Variance Framework (SciPy)
# ─────────────────────────────────────────────────────────────────────────────
# Implements Modern Portfolio Theory (MPT) as introduced by Harry Markowitz (1952).
#
# The core idea: for any set of assets, there exists an "efficient" portfolio that
# maximises expected return for a given level of risk (or minimises risk for a given
# return).  The collection of all such portfolios traces the Efficient Frontier.
#
# Three portfolio strategies are computed:
#   1. Equal Weight  — Naive baseline: 1/N allocation across all assets.
#   2. Max Sharpe    — Tangency portfolio: maximises risk-adjusted return.
#                      Found by minimising the negative Sharpe ratio subject to
#                      weights summing to 1 (long-only, no leverage).
#   3. Min Volatility — Global Minimum Variance portfolio: minimises portfolio
#                       standard deviation regardless of expected return.
#
# Optimisation method: SLSQP (Sequential Least Squares Programming) from
# scipy.optimize.minimize.  Chosen because it natively handles:
#   - Equality constraints  (weights sum to 1.0)
#   - Box constraints       (0 ≤ w_i ≤ 1, i.e., long-only, no short selling)
# ─────────────────────────────────────────────────────────────────────────────

def port_perf(w, mu, cov, rf=0.045):
    """
    Calculate annualised return, volatility, and Sharpe ratio for a portfolio.

    This is the core performance function called both by the optimizer's objective
    functions and by the post-optimization result display.

    Parameters
    ----------
    w : array-like
        Portfolio weight vector (must sum to 1).
    mu : pd.Series
        Mean daily returns for each asset.
    cov : pd.DataFrame
        Covariance matrix of daily returns (n × n).
    rf : float
        Annualised risk-free rate (decimal).

    Returns
    -------
    tuple[float, float, float]
        ret    — Annualised portfolio expected return (w · μ × 252)
        vol    — Annualised portfolio volatility  (√(wᵀ Σ w × 252))
        sharpe — Sharpe ratio (ret - rf) / vol;  NaN if vol = 0
    """
    w      = np.asarray(w, float)
    ret    = float(np.dot(w, mu) * 252)                        # Weighted daily mean → annualised
    vol    = float(np.sqrt(np.dot(w, np.dot(cov * 252, w))))   # Portfolio variance formula: wᵀ Σ w
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def max_sharpe(mu, cov, rf=0.045):
    """
    Find the Maximum Sharpe Ratio (Tangency) portfolio weights via SLSQP.

    Minimises the negative Sharpe ratio subject to:
      - Sum of weights = 1  (fully invested)
      - 0 ≤ w_i ≤ 1         (long-only, no leverage or short positions)

    The equal-weight vector is used as the starting point (warm start) because
    it lies in the feasible region and produces a valid Sharpe estimate, giving
    the optimizer a reasonable gradient direction.

    Tight convergence tolerance (ftol=1e-10) and a high iteration cap (1000)
    ensure the optimizer reaches a true local optimum for well-conditioned inputs.

    Parameters
    ----------
    mu  : pd.Series    — Mean daily returns per asset
    cov : pd.DataFrame — Daily return covariance matrix
    rf  : float        — Annualised risk-free rate

    Returns
    -------
    np.ndarray
        Optimal weight vector.  Falls back to equal weights if optimization fails.
    """
    n    = len(mu)
    init = np.full(n, 1/n)   # Equal-weight warm start
    try:
        res = minimize(
            lambda w: -port_perf(w, mu, cov, rf)[2],   # Objective: maximise Sharpe (minimise negative)
            init,
            method="SLSQP",
            bounds=[(0, 1)] * n,                         # Long-only constraint per asset
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],  # Fully-invested constraint
            options={"ftol": 1e-10, "maxiter": 1000}
        )
        return res.x if res.success else init   # Return optimal weights or fall back to equal weight
    except Exception:
        return init


def min_vol(mu, cov):
    """
    Find the Minimum Variance portfolio weights via SLSQP.

    Minimises portfolio standard deviation (vol) subject to the same constraints
    as max_sharpe() above.  This portfolio sits at the leftmost tip of the
    Efficient Frontier — it has the lowest achievable volatility for any fully-
    invested, long-only combination of the given assets.

    Note: the expected return (mu) is still required for port_perf() internally,
    but the objective function only optimises on volatility — return is ignored.

    Parameters
    ----------
    mu  : pd.Series    — Mean daily returns (needed for port_perf signature)
    cov : pd.DataFrame — Daily return covariance matrix

    Returns
    -------
    np.ndarray
        Optimal weight vector.  Falls back to equal weights if optimization fails.
    """
    n    = len(mu)
    init = np.full(n, 1/n)   # Equal-weight warm start
    try:
        res = minimize(
            lambda w: port_perf(w, mu, cov)[1],          # Objective: minimise annualised volatility
            init,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
            options={"ftol": 1e-10, "maxiter": 1000}
        )
        return res.x if res.success else init
    except Exception:
        return init


def frontier(mu, cov, n=800):
    """
    Monte Carlo simulation of the Efficient Frontier scatter cloud.

    Generates n random portfolios by sampling Dirichlet-distributed weights
    (which guarantees the simplex constraint: all weights ≥ 0 and sum to 1).
    Each portfolio's annualised return, volatility, and Sharpe ratio are
    recorded, producing the characteristic "cloud of dots" used in the
    Efficient Frontier scatter plot.

    Dirichlet(ones) is equivalent to a uniform distribution over the simplex,
    so all weight combinations are equally likely — this gives an unbiased
    picture of the full opportunity set.

    Parameters
    ----------
    mu  : pd.Series    — Mean daily returns
    cov : pd.DataFrame — Daily return covariance matrix
    n   : int          — Number of random portfolios to simulate (default 800)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        rs — Annualised returns array  (length n)
        vs — Annualised volatility array (length n)
        ss — Sharpe ratio array (length n)  — used to color-code the scatter points
    """
    n_assets = len(mu)
    rs, vs, ss = [], [], []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_assets))   # Random point on the weight simplex
        r, v, s = port_perf(w, mu, cov)
        rs.append(r); vs.append(v); ss.append(s)
    return np.array(rs), np.array(vs), np.array(ss)

# ─────────────────────────────────────────────────────────────────────────────
# ETF UNIVERSE — Curated Institutional Fund Registry (110+ ETFs, 14 Sectors)
# ─────────────────────────────────────────────────────────────────────────────
# ETF_UNIVERSE is the central data registry that powers the screener, sector
# explorer, and portfolio builder.
#
# Structure:
#   ETF_UNIVERSE = {
#       "Sector Name": {
#           "TICKER": (
#               "Fund Full Name",         # str — displayed in tables and charts
#               "One-sentence description" # str — shown in the Explore ETFs cards
#           ),
#           ...
#       },
#       ...
#   }
#
# Sector coverage:
#   Broad Market · Technology · Financials · Energy · Healthcare · Consumer
#   Industrials · Defense & Aerospace · Bonds & Fixed Income · Small & Mid Cap
#   International · Real Estate · Precious Metals & Commodities
#   Dividends & Income · ESG & Thematic
#
# Design rationale:
#   - Storing descriptions inline (rather than in a separate CSV/DB) keeps the
#     project self-contained and deployable with zero additional file dependencies.
#   - The dict structure allows O(1) lookup by ticker symbol after the flat
#     ALL_TICKERS index is constructed below.
# ─────────────────────────────────────────────────────────────────────────────
ETF_UNIVERSE = {
    "Broad Market": {
        "SPY":  ("SPDR S&P 500 ETF",            "Tracks the S&P 500 — 500 of the largest US companies. The most widely traded ETF in the world."),
        "VOO":  ("Vanguard S&P 500 ETF",         "Vanguard's low-cost S&P 500 fund. Near-identical to SPY but with a lower expense ratio (0.03%)."),
        "SPLG": ("SPDR Portfolio S&P 500 ETF",   "Core institutional large-cap holding offering ultra-low cost baseline index tracking."),
        "VTI":  ("Vanguard Total Stock Market",  "Covers the entire US stock market — large, mid, small, and micro cap stocks (~4,000 companies)."),
        "IVV":  ("iShares Core S&P 500",         "iShares version of the S&P 500 tracker. One of the cheapest ways to own the US large-cap index."),
        "SCHB": ("Schwab US Broad Market",       "Tracks the Dow Jones US Broad Stock Market Index — similar to VTI with ultra-low fees."),
        "VOOG": ("Vanguard S&P 500 Growth",      "S&P 500 growth subset — companies with above-average revenue and earnings growth rates."),
        "VOOV": ("Vanguard S&P 500 Value",       "S&P 500 value subset — companies trading at lower valuations relative to fundamentals."),
        "IVW":  ("iShares S&P 500 Growth",       "Concentrated exposure to large-cap equities exhibiting classic cross-sectional growth signals."),
        "IVE":  ("iShares S&P 500 Value",        "Large-cap value allocation matching standard fundamentals metrics like low P/E ratios.")
    },
    "Technology": {
        "QQQ":  ("Invesco Nasdaq-100",           "Tracks the 100 largest non-financial Nasdaq companies. Heavy weight in Apple, Microsoft, Nvidia, Amazon."),
        "QQQM": ("Invesco Nasdaq 100 ETF",       "Identical underlying tracker structure as QQQ but scaled for retail holding with lower expense ratios."),
        "XLK":  ("Technology Select SPDR",       "S&P 500 tech sector only — semiconductors, software, IT services, and hardware companies."),
        "VGT":  ("Vanguard Information Technology","Broader than XLK; includes Apple, Microsoft, Nvidia, Visa. One of the most popular tech ETFs."),
        "IWF":  ("iShares Russell 1000 Growth",  "Broad large-cap growth fund matching secular momentum parameters in US technical leadership."),
        "SOXX": ("iShares Semiconductor",        "Focused on semiconductor companies — Nvidia, AMD, Intel, TSMC, Broadcom. High growth, high volatility."),
        "SMH":  ("VanEck Semiconductor",         "Tracks the 25 largest US-listed semiconductor stocks. Often used as a chip-sector benchmark."),
        "ARKK": ("ARK Innovation ETF",           "Actively managed; bets on disruptive tech — AI, genomics, fintech, robotics. High risk, high reward."),
        "CIBR": ("First Trust Cybersecurity",    "Tracks cybersecurity companies including Palo Alto, CrowdStrike, Fortinet, Okta."),
        "CLOU": ("Global X Cloud Computing",     "Targets cloud software companies with subscription revenue models — SaaS, PaaS, IaaS providers."),
        "BOTZ": ("Global X Robotics & AI",       "Focuses on robotics, automation, and AI companies globally — Intuitive Surgical, Fanuc, Keyence."),
    },
    "Financials": {
        "XLF":  ("Financial Select SPDR",        "S&P 500 financials — banks, insurance, asset managers. Moves with interest rates and credit cycles."),
        "VFH":  ("Vanguard Financials",          "Broader financial sector exposure than XLF; includes smaller regional banks and diversified financials."),
        "KBE":  ("SPDR S&P Bank ETF",            "Equal-weighted US bank ETF — more exposure to regional banks vs. mega-cap names like JPMorgan."),
        "KRE":  ("SPDR S&P Regional Banking",    "Focused entirely on regional and community banks. Sensitive to Fed rate decisions and credit conditions."),
        "IAI":  ("iShares US Broker-Dealers",    "Targets investment banks, brokerages, and asset managers — Goldman Sachs, Morgan Stanley, Schwab."),
        "KBWP": ("Invesco KBW Property & Casualty","Insurance-focused — property, casualty, and reinsurance companies like Travelers, Allstate, Markel."),
    },
    "Energy": {
        "XLE":  ("Energy Select SPDR",           "S&P 500 energy sector — oil, gas, and energy equipment companies. ExxonMobil and Chevron dominate."),
        "VDE":  ("Vanguard Energy",              "Broad energy exposure including exploration, production, refining, and distribution companies."),
        "XOP":  ("SPDR Oil & Gas E&P",           "Equal-weighted exploration & production companies — more volatile than XLE, no refining/services."),
        "OIH":  ("VanEck Oil Services",          "Oilfield services companies — Schlumberger, Halliburton, Baker Hughes. Amplifies oil price moves."),
        "AMLP": ("Alerian MLP",                  "Master Limited Partnerships — pipeline and midstream infrastructure. Known for high dividend yields."),
        "USO":  ("United States Oil Fund",       "Tracks crude oil futures prices. Used for short-term oil price exposure, not long-term holding."),
        "UNG":  ("United States Natural Gas Fund","Tracks natural gas futures. Highly volatile; often used for short-term directional bets on gas prices."),
    },
    "Healthcare": {
        "XLV":  ("Health Care Select SPDR",      "S&P 500 healthcare — pharma, biotech, medical devices, managed care. Defensive sector."),
        "VHT":  ("Vanguard Health Care",         "Broader healthcare exposure; includes companies like UnitedHealth, Johnson & Johnson, Pfizer, Merck."),
        "IBB":  ("iShares Biotechnology",        "Tracks the Nasdaq biotech index — clinical-stage and commercial biotech companies. High volatility."),
        "XBI":  ("SPDR S&P Biotech",             "Equal-weighted biotech ETF — gives smaller companies more weight than IBB. Very volatile."),
        "IHI":  ("iShares Medical Devices",      "Medical device manufacturers — Medtronic, Abbott, Boston Scientific, Intuitive Surgical."),
        "LABU": ("Direxion Daily Biotech Bull 3x","3x leveraged biotech ETF. Extreme risk — for traders only, not long-term investors."),
    },
    "Consumer": {
        "XLY":  ("Consumer Discretionary SPDR",  "Cyclical consumer spending — Amazon, Tesla, Home Depot, Nike. Performs well in economic expansions."),
        "XLP":  ("Consumer Staples SPDR",        "Defensive consumer goods — food, beverages, household products. Less volatile; holds up in downturns."),
        "VCR":  ("Vanguard Consumer Discretionary","Broad consumer discretionary exposure across retail, autos, hotels, restaurants, media."),
        "VDC":  ("Vanguard Consumer Staples",    "Staples companies with stable revenue — Procter & Gamble, Coca-Cola, Walmart, Costco."),
        "ONLN": ("ProShares Online Retail",      "Targets e-commerce companies globally — Amazon, Alibaba, JD.com, Shopify, Wayfair."),
    },
    "Industrials": {
        "XLI":  ("Industrials Select SPDR",      "S&P 500 industrials — aerospace, defense contractors, machinery, transportation, construction."),
        "VIS":  ("Vanguard Industrials",         "Broad industrials sector including Honeywell, UPS, Caterpillar, Deere, Lockheed Martin."),
        "ITA":  ("iShares US Aerospace & Defense","Pure-play defense and aerospace — Lockheed, Raytheon, Northrop, General Dynamics, Boeing."),
        "JETS": ("U.S. Global Jets ETF",         "Airlines and aircraft manufacturers globally — Delta, United, Southwest, Boeing, Airbus."),
    },
    "Defense & Aerospace": {
        "PPA":  ("Invesco Aerospace & Defense",  "Tracks defense and aerospace companies with government contract revenue globally."),
        "XAR":  ("SPDR Aerospace & Defense",     "Equal-weighted defense ETF — spreads weight across small and large defense companies."),
        "DFEN": ("Direxion Daily Aerospace 3x",  "3x leveraged defense ETF. Extreme amplification of daily moves — short-term trading only."),
        "SHLD": ("Global X Defense Tech ETF",    "Focuses on advanced defense technology including drones, cybersecurity, and space defense."),
    },
    "Bonds & Fixed Income": {
        "AGG":  ("iShares Core US Aggregate Bond","Tracks the entire US investment-grade bond market — Treasuries, corporate, MBS. Low risk, low return."),
        "BND":  ("Vanguard Total Bond Market",   "Vanguard's version of the broad US bond market. Near-identical to AGG with slightly lower fees."),
        "TLT":  ("iShares 20+ Year Treasury",    "Long-duration US Treasuries. Very sensitive to interest rate changes — moves opposite to rate hikes."),
        "SHY":  ("iShares 1-3 Year Treasury",    "Short-term Treasuries. Minimal interest rate risk; used as a cash-like safe haven."),
        "HYG":  ("iShares High Yield Corporate", "Junk bonds — higher yield but higher default risk. Moves more like stocks than investment-grade bonds."),
        "LQD":  ("iShares Investment Grade Corp","Investment-grade corporate bonds from companies like Apple, Microsoft, JPMorgan. More yield than Treasuries."),
        "TIPS": ("iShares TIPS Bond ETF",        "Treasury Inflation-Protected Securities — principal adjusts with CPI inflation. Hedge against inflation."),
        "MUB":  ("iShares National Muni Bond",   "Municipal bonds — tax-exempt interest income. Popular with high-income investors in upper tax brackets."),
        "BNDX": ("Vanguard Total International Bond", "Offers non-US international bond exposure with currency-hedged yield structural layers.")
    },
    "Small & Mid Cap": {
        "IWM":  ("iShares Russell 2000",         "The benchmark small-cap ETF — 2,000 smaller US companies. Higher growth potential, higher volatility."),
        "MDY":  ("SPDR S&P MidCap 400",          "Mid-cap US companies — often overlooked but historically strong long-term performers."),
        "VB":   ("Vanguard Small-Cap",           "Tracks the CRSP US Small Cap Index — broader than IWM, lower cost, slightly different composition."),
        "IJR":  ("iShares Core S&P Small-Cap",   "S&P 600 small-cap index — requires profitability to qualify, making it higher quality than IWM."),
        "VIOO": ("Vanguard S&P Small-Cap 600",   "Same S&P 600 index as IJR but from Vanguard — ultra-low cost, high quality small-cap exposure."),
        "IWD":  ("iShares Russell 1000 Value",   "Mid-to-large asset screening targeting value multipliers under institutional constraints.")
    },
    "International": {
        "EFA":  ("iShares MSCI EAFE",            "Developed markets outside the US — Europe, Australia, Japan, Asia. Currency risk included."),
        "EEM":  ("iShares MSCI Emerging Markets","Developing countries — China, India, Brazil, Taiwan, South Korea. High growth, higher risk."),
        "VEA":  ("Vanguard Developed Markets",   "Vanguard's low-cost developed market fund; similar to EFA but includes Canada."),
        "VWO":  ("Vanguard Emerging Markets",    "Vanguard's emerging market fund; slightly different country weights vs. EEM, lower fees."),
        "IEMG": ("iShares Core Emerging Markets","Broader emerging market coverage than EEM; includes small caps. iShares' core EM fund."),
        "FXI":  ("iShares China Large-Cap",      "Top 50 Chinese companies listed in Hong Kong. High concentration in tech and finance."),
        "EWJ":  ("iShares MSCI Japan",           "Japanese equities — Toyota, Sony, Mitsubishi, SoftBank. World's third-largest economy."),
        "INDA": ("iShares MSCI India",           "Indian equities — one of the fastest-growing major economies. Exposure to HDFC, Infosys, Reliance."),
        "VXUS": ("Vanguard Total International Stock", "Total ex-US equity coverage across developed and emerging market jurisdictions.")
    },
    "Real Estate": {
        "VNQ":  ("Vanguard Real Estate",         "US REITs — commercial real estate, apartments, offices, industrial, data centers. Pays high dividends."),
        "IYR":  ("iShares US Real Estate",       "Broad US REIT exposure; includes retail, residential, healthcare, and specialty real estate."),
        "SCHH": ("Schwab US REIT",               "Low-cost REIT ETF from Schwab. Similar holdings to VNQ but with slightly lower expense ratio."),
        "XLRE": ("Real Estate Select SPDR",      "S&P 500 real estate sector only — excludes mortgage REITs, focuses on equity REITs."),
        "REM":  ("iShares Mortgage Real Estate", "Mortgage REITs — companies that finance mortgages. Higher yield, sensitive to interest rates."),
    },
    "Precious Metals & Commodities": {
        "GLD":  ("SPDR Gold Shares",             "Backed by physical gold stored in vaults. The largest and most liquid gold ETF globally."),
        "IAU":  ("iShares Gold Trust",           "Physical gold ETF from iShares. Lower expense ratio than GLD (0.25% vs 0.40%)."),
        "SLV":  ("iShares Silver Trust",         "Physical silver. More volatile than gold — used in both investment and industrial applications."),
        "GDX":  ("VanEck Gold Miners",           "Stocks of gold mining companies — amplifies gold price moves (leveraged to gold prices)."),
        "GDXJ": ("VanEck Junior Gold Miners",    "Smaller gold mining companies — even higher leverage to gold prices than GDX. Very volatile."),
        "PDBC": ("Invesco Optimum Yield Commodities","Broad commodity basket — energy, metals, agriculture. Rolls futures to maximize yield."),
        "DJP":  ("iPath Bloomberg Commodity",    "Diversified commodity index — energy, metals, agriculture. Structured as an ETN, not ETF."),
        "CPER": ("United States Copper Index",   "Tracks copper futures prices. Copper is a leading economic indicator ('Dr. Copper')."),
    },
    "Dividends & Income": {
        "VYM":  ("Vanguard High Dividend Yield", "High dividend-paying US stocks — financials, healthcare, energy, consumer staples. Lower volatility."),
        "DVY":  ("iShares Select Dividend",      "100 high-dividend US stocks — utilities, financials, energy. Higher yield than VYM."),
        "SCHD": ("Schwab US Dividend Equity",    "Quality dividend growers — screened for financial health and dividend growth consistency. Very popular."),
        "DGRO": ("iShares Dividend Growth",      "Companies with 5+ years of consistent dividend growth — quality filter makes this more growth-oriented."),
        "JEPI": ("JPMorgan Equity Premium Income","Covered call strategy on S&P 500 stocks. Very high monthly income (~7-10% yield) with capped upside."),
        "JEPQ": ("JPMorgan Nasdaq Equity Premium","Covered call strategy on Nasdaq stocks. High income with tech exposure. Launched 2022."),
        "PFF":  ("iShares Preferred & Income",   "Preferred stock ETF — hybrid between bonds and stocks. High income, limited price appreciation."),
    },
    "ESG & Thematic": {
        "ESGU": ("iShares MSCI USA ESG",         "ESG-screened US large/mid caps — excludes tobacco, weapons, and companies with poor ESG scores."),
        "ICLN": ("iShares Global Clean Energy",  "Clean energy producers globally — wind, solar, hydrogen, and renewable utilities."),
        "TAN":  ("Invesco Solar Energy",         "Global solar energy companies — First Solar, SolarEdge, Enphase, Sunrun."),
        "FAN":  ("First Trust Global Wind Energy","Wind energy companies globally — Vestas, Orsted, Siemens Gamesa, Northland Power."),
        "LIT":  ("Global X Lithium & Battery Tech","Lithium mining and battery technology — Albemarle, SQM, Panasonic. Tied to EV growth."),
        "DRIV": ("Global X Autonomous & EV",     "Electric vehicles, autonomous driving, and EV infrastructure — Tesla, NXP, Aptiv."),
        "ROBO": ("ROBO Global Robotics & Automation","Robotics and automation companies globally. More diversified than BOTZ."),
    },
}

# ── Flat lookup indexes derived from the nested ETF_UNIVERSE dict ─────────────
# ALL_TICKERS : dict[str, tuple] — ticker → (name, description) for O(1) lookup
# ALL_SECTORS : list[str]        — ordered sector name list for UI dropdowns
# TICKER_LIST : list[str]        — flat list of all ticker symbols for multiselects
ALL_TICKERS  = {t: info for sector in ETF_UNIVERSE.values() for t, info in sector.items()}
ALL_SECTORS  = list(ETF_UNIVERSE.keys())
TICKER_LIST  = list(ALL_TICKERS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# CLIENT-SIDE ROUTING & SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit re-executes the entire script on every user interaction.
# st.session_state persists values across re-runs within a browser session.
#
# "page" stores the currently active page name (string).  It is initialised to
# "Home" on first load via setdefault(), which is a no-op on subsequent re-runs.
#
# PAGE_MAP (defined at the bottom of the file) maps page name strings to their
# render functions so the router can call the correct function with one lookup.
# ─────────────────────────────────────────────────────────────────────────────
PAGES = ["Home", "ETF Screener", "Explore ETFs", "Portfolio Builder"]
st.session_state.setdefault("page", "Home")   # Initialize routing state on first load


def navigate(p):
    """
    Streamlit button on_click callback — updates the active page in session state.

    Because Streamlit re-runs the script on every widget interaction, updating
    session_state["page"] here causes the router at the bottom of the file to
    call the newly selected page's render function on the next re-run.

    Parameters
    ----------
    p : str
        The page name to navigate to (must be a key in PAGE_MAP).
    """
    st.session_state["page"] = p

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR COMPONENT — Navigation Panel
# ─────────────────────────────────────────────────────────────────────────────
# The sidebar is rendered once at module level (outside any page function) so it
# persists and stays in sync regardless of which page is active.
#
# Structure:
#   1. Branded logo mark with SVG bar-chart icon and "ETFAnalytics" wordmark
#   2. Navigation buttons — one per page in PAGES.
#      The active page button is prefixed with "●" to provide a visual
#      active-state indicator (CSS :active / :focus states are unreliable
#      across Streamlit re-runs).
#   3. Footer metadata bar — data source credit, author attribution,
#      LinkedIn and GitHub profile links.
#
# Note: st.button's on_click=navigate + args=(p,) pattern is used instead of
# checking the return value because Streamlit's on_click fires before the
# script re-runs, ensuring session_state["page"] is updated before the router
# at the bottom of the file evaluates PAGE_MAP[st.session_state["page"]].
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '''<div class="sidebar-logo">
            <div class="sidebar-logo-mark">
                <div class="sidebar-logo-icon">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <rect x="1" y="9" width="2.5" height="6" rx="1" fill="#0b0e14"/>
                        <rect x="4.5" y="6" width="2.5" height="9" rx="1" fill="#0b0e14"/>
                        <rect x="8" y="3" width="2.5" height="12" rx="1" fill="#0b0e14"/>
                        <rect x="11.5" y="1" width="2.5" height="14" rx="1" fill="#0b0e14"/>
                    </svg>
                </div>
                <div class="sidebar-logo-wordmark">ETF<span>Analytics</span></div>
            </div>
            <div class="sidebar-logo-sub">Portfolio Intelligence</div>
        </div>''',
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    current_page = st.session_state["page"]
    for p in PAGES:
        is_active = current_page == p
        label = f"● {p}" if is_active else p
        st.button(
            label,
            key=f"nav_{p}",
            on_click=navigate,
            args=(p,),
            use_container_width=True,
        )

    st.markdown("<hr style='border:none;border-top:1px solid #1e2840;margin:1.4rem 0 1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        '''<div style="font-family:'EB Garamond',Georgia,serif;font-size:0.88rem;
        color:#a0aec0;text-align:center;line-height:1.9;padding-bottom:1rem;">
        Data: Yahoo Finance API<br>Universe: 110+ ETFs<br>Engine: SciPy Optimizer<br>
        Author: Shreyas Jagadish Kanagal
        <div style="margin-top:0.6rem;display:flex;justify-content:center;gap:1rem;">
            <a href="https://www.linkedin.com/in/shreyaskanagal" target="_blank"
               style="color:#00d4aa;text-decoration:none;border-bottom:1px solid rgba(0,212,170,0.4);padding-bottom:1px;letter-spacing:0.03em;">
               LinkedIn
            </a>
            <span style="color:#1e2840;">|</span>
            <a href="https://github.com/shreyaskanagal" target="_blank"
               style="color:#00d4aa;text-decoration:none;border-bottom:1px solid rgba(0,212,170,0.4);padding-bottom:1px;letter-spacing:0.03em;">
               GitHub
            </a>
        </div>
        </div>''',
        unsafe_allow_html=True
    )

def page_header(title: str, subtitle: str = ""):
    """
    Render a standardised page header with a title and optional subtitle.

    Injects an HTML block that uses the .section-title CSS class from the
    design system.  Called at the top of every page function (except page_home,
    which has a custom hero layout).

    Parameters
    ----------
    title    : str — The primary page heading (e.g. "ETF Screener")
    subtitle : str — Optional descriptive sub-line rendered below the title
    """
    st.markdown(f"""
    <div style="padding:0.2rem 0 1.4rem 0; border-bottom:1px solid #1e2840; margin-bottom:2rem;">
        <p class="section-title" style="font-size:2.2rem; margin:0 0 0.3rem 0;">{title}</p>
        {"<p style='font-family:var(--font,\"EB Garamond\",Georgia,serif);font-size:1.05rem;color:#a0aec0;margin:0;'>" + subtitle + "</p>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME — Live Benchmark Ticker Tape + Platform Overview
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    """
    Render the Home landing page.

    Layout (top to bottom):
      1. Live benchmark ticker tape — 6 KPI cards showing real-time price and
         daily % change for SPY, QQQ, IWM, AGG, GLD, TLT.
         Arrow glyphs (▲/▼) and CSS classes (.up / .down) apply the teal/red
         color scheme for positive/negative moves.
      2. Platform overview — two-column section describing the application's
         purpose, tech stack, and business value proposition.
      3. Metric glossary — 8 financial terms (Sharpe Ratio, Max Drawdown, etc.)
         rendered in a two-column grid using the .metric-def CSS card style.
    """
    # Six cross-asset benchmarks used as representative market health indicators
    benchmarks = ["SPY","QQQ","IWM","AGG","GLD","TLT"]
    with st.spinner("Synchronizing market terminal arrays..."):
        quotes = get_live_quotes(benchmarks)   # Cached 60s — near-realtime without hammering the API

    st.markdown('<div class="badge">Portfolio Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""<h1 class="page-title">Track, Analyze &amp;<br><span>Optimize</span> Investments</h1>""",
                unsafe_allow_html=True)
    st.markdown("""<p style="font-family:'EB Garamond',Georgia,serif;font-size:1.15rem;color:#e2e8f0;line-height:1.8;max-width:850px;margin-bottom:2rem;">
    A professional-grade analytics platform tracking over <b style="color:#00d4aa">110+ ETFs</b> across
    <b style="color:#00d4aa">14 market sectors</b>. Built to replace manual spreadsheet reporting with automated data pipelines, interactive historical analysis, and institutional-grade portfolio optimization modeling.
    </p>""", unsafe_allow_html=True)

    # Human-readable labels for the 6 benchmark tickers displayed in the ticker tape
    bench_labels = {"SPY":"S&P 500 Index","QQQ":"Nasdaq-100 Core","IWM":"Russell 2000 Small-Cap",
                    "AGG":"US Aggregate Bond","GLD":"Physical Gold Shares","TLT":"Long-Term Treasury"}
    cols = st.columns(6)
    for i, t in enumerate(benchmarks):
        q     = quotes.get(t, {})
        p     = q.get("price"); pct = q.get("pct")
        # Determine CSS class and directional arrow based on daily price movement
        cls   = "up" if pct and pct > 0 else "down" if pct and pct < 0 else ""
        arrow = "▲" if pct and pct > 0 else "▼" if pct and pct < 0 else "—"
        with cols[i]:
            # Render the KPI card; fall back to an "offline" placeholder if data is unavailable
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">{bench_labels[t]}</div>
                <div class="kpi-value {cls}">${p:,.2f}</div>
                <div class="kpi-delta {cls}">{arrow} {abs(pct):.2f}%</div>
            </div>""" if p and pct else f"""<div class="kpi-card">
                <div class="kpi-label">{bench_labels[t]} ({t})</div>
                <div class="kpi-value">—</div>
                <div class="kpi-delta">Data Offline</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown('<p class="section-title">Functional Framework</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 2], gap="large")
    with col_a:
        st.markdown("""<div style="font-family:'EB Garamond',Georgia,serif;font-size:1.1rem;color:#e2e8f0;line-height:1.85;">
        <p style="margin-bottom:1.1rem;">This dashboard is designed to help analysts and portfolio managers easily track, evaluate, and optimize ETF investments. It eliminates manual spreadsheet work by providing a fully automated, interactive platform that visualizes the market in real time.</p>
        <p style="margin-bottom:1.1rem;">Operating seamlessly in the background, the application pulls live market data directly from Yahoo Finance APIs, cleans the historical records, and instantly calculates critical performance metrics. This allows stakeholders to quickly unpack historical trends, identify correlations, and understand risk factors before executing trades.</p>
        <p>Finally, the built-in Portfolio Builder utilizes established quantitative models—specifically the Markowitz Mean-Variance framework—to construct the most efficient asset mixes. It demonstrates exactly how to balance risk and reward, empowering users to build stronger, data-backed portfolios.</p>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div style="font-family:'EB Garamond',Georgia,serif;font-size:1.1rem;color:#e2e8f0;line-height:1.9;">
        <b style="color:#00d4aa;font-size:1.0rem;letter-spacing:0.05em;text-transform:uppercase;">Core Technology</b><br>
        Framework: Python · pandas · NumPy · SciPy<br>
        Data Pipeline: yfinance API<br>
        Visualization: Streamlit · Plotly<br><br>
        <b style="color:#00d4aa;font-size:1.0rem;letter-spacing:0.05em;text-transform:uppercase;">Business Value</b><br>
        Automated Financial Data Processing<br>
        Risk-Adjusted Performance Analysis<br>
        Data-Driven Portfolio Optimization
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Key Performance Metrics Defined</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Standard parameters used to evaluate assets and structure portfolios throughout this dashboard.</p>', unsafe_allow_html=True)

    defs = [
        ("Annualized Return", "The average yearly growth rate of an investment over a specific period, factoring in the power of compound interest. It normalizes returns so different timeframes can be compared equally."),
        ("Annualized Volatility", "A statistical measure of how much an investment's price fluctuates over a year. Higher volatility indicates wider price swings and, generally, higher investment risk."),
        ("Sharpe Ratio", "A metric that measures an investment's return compared to its risk. A higher Sharpe Ratio means the asset provides better returns for every unit of risk taken."),
        ("Maximum Drawdown", "The largest single drop in an asset's price from its highest peak to its lowest point before recovering. It acts as an indicator of worst-case scenario risk."),
        ("Year-to-Date (YTD) Return", "The total gain or loss of an investment from the very first trading day of the current calendar year to today."),
        ("Expense Ratio", "The annual fee that ETF providers charge shareholders to manage the fund, expressed as a percentage of the total investment."),
        ("Correlation Matrix", "A scale from -1.0 to +1.0 showing how similarly two assets perform. Mixing assets with low correlation is the key to building a properly diversified portfolio."),
        ("Efficient Frontier", "A visual curve representing the best possible portfolio combinations. Any portfolio on the curve offers the highest expected return for a specific level of risk.")
    ]

    col1, col2 = st.columns(2, gap="large")
    for i, (name, desc) in enumerate(defs):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""<div class="metric-def">
                <div class="metric-def-name">{name}</div>
                <div class="metric-def-body">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ETF SCREENER — Multi-Criteria Filter Engine
# ─────────────────────────────────────────────────────────────────────────────
def page_screener():
    """
    Render the ETF Screener page.

    Provides a sidebar-style filter panel (left column) and a results panel
    (right column) displaying a sortable HTML table and a risk/return scatter
    plot for all ETFs that pass the user-defined constraints.

    Filter controls:
      - Sector multiselect  — limits the screened universe to chosen sectors
      - Historical window   — 1Y / 2Y / 3Y / 5Y lookback for statistics
      - Risk-free rate      — adjustable benchmark rate for Sharpe calculations
      - Sort column / order — sort the result table by any performance metric
      - Min/Max annual return thresholds  — hard filters on ann_ret
      - Minimum Sharpe ratio              — exclude low-efficiency funds

    Result table columns:
      Ticker | Asset Name | Sector | Ann. Return | Volatility | Sharpe | Max Drawdown | YTD Return

    Scatter chart:
      X = Annualised Volatility, Y = Annualised Return
      Each dot = one ETF, colored by sector — makes the risk/return tradeoff
      across sectors immediately visible.
    """
    page_header("ETF Screener", "Filter, sort, and compare funds across the entire market universe")

    col_filters, col_results = st.columns([1, 3], gap="large")

    with col_filters:
        st.markdown("<p style='font-family:\"EB Garamond\",Georgia,serif;font-size:1.05rem;font-weight:600;color:#ffffff;margin-bottom:0.3rem;letter-spacing:0.04em;text-transform:uppercase;'>Filter Settings</p>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

        selected_sectors = st.multiselect("Select Sectors", ALL_SECTORS, default=ALL_SECTORS[:5])
        period_map = {"1Y": 365, "2Y": 730, "3Y": 1095, "5Y": 1825}
        period = st.selectbox("Historical Window", list(period_map.keys()), index=1)
        rf = st.slider("Risk-Free Benchmark (%)", 0.0, 6.0, 4.5, 0.25, format="%.2f%%") / 100

        st.markdown("<p style='font-family:\"EB Garamond\",Georgia,serif;font-size:1.05rem;font-weight:600;color:#ffffff;margin-bottom:0.3rem;letter-spacing:0.04em;text-transform:uppercase;'>Sort Results</p>", unsafe_allow_html=True)
        sort_col = st.selectbox("", ["Ann. Return","Volatility","Sharpe","Max Drawdown","YTD Return"])
        sort_asc = st.radio("Order", ["Descending","Ascending"]) == "Ascending"

        st.markdown("<p style='font-family:\"EB Garamond\",Georgia,serif;font-size:1.05rem;font-weight:600;color:#ffffff;margin-bottom:0.3rem;letter-spacing:0.04em;text-transform:uppercase;'>Performance Constraints</p>", unsafe_allow_html=True)
        min_ret = st.slider("Minimum Annual Return (%)", -50, 100, -50, 5)
        max_ret = st.slider("Maximum Annual Return (%)", -50, 200, 200, 5)
        min_sharpe = st.slider("Minimum Sharpe Ratio", -3.0, 5.0, -3.0, 0.1)

        run = st.button("▶ Run Screener", use_container_width=True)

    with col_results:
        if not selected_sectors:
            st.markdown('<div class="warn-box">Select at least one sector filter on the left to view results.</div>', unsafe_allow_html=True)
            return

        # Build the universe for this screen by flattening selected sectors.
        # dict.fromkeys() preserves insertion order while deduplicating any
        # tickers that might appear in multiple sectors.
        screen_tickers = list(dict.fromkeys([t for s in selected_sectors for t in ETF_UNIVERSE.get(s, {})]))
        start = (datetime.today() - timedelta(days=period_map[period])).strftime("%Y-%m-%d")

        with st.spinner("Compiling market data..."):
            data = download_data(screen_tickers, start=start)

        if data.empty:
            st.error("No historical data could be retrieved.")
            return

        # Filter to tickers for which data actually came back (some may be delisted)
        valid   = [t for t in screen_tickers if t in data.columns]
        returns = data[valid].pct_change().dropna()   # Convert prices → daily % returns
        ann_ret, ann_vol, sharpe, max_dd = safe_stats(returns, rf)

        # YTD return: compound returns from the first trading day of the current year.
        # Falls back to NaN series if no trading data exists yet this year.
        try:
            idx       = pd.DatetimeIndex(returns.index)
            ytd_start = idx[idx.year == datetime.today().year][0]   # First trading day of this year
            ytd_ret   = (1 + returns.loc[ytd_start:]).cumprod().iloc[-1] - 1
        except Exception:
            ytd_ret = pd.Series({t: np.nan for t in valid})

        # Build the row list — apply user-defined performance filters in the loop
        rows = []
        for t in valid:
            sector = next((s for s, etfs in ETF_UNIVERSE.items() if t in etfs), "—")
            name   = ALL_TICKERS[t][0] if t in ALL_TICKERS else t
            r  = ann_ret.get(t, np.nan)
            v  = ann_vol.get(t, np.nan)
            s  = sharpe.get(t, np.nan)
            dd = max_dd.get(t, np.nan)
            ytd = ytd_ret.get(t, np.nan) if isinstance(ytd_ret, pd.Series) else np.nan

            # Apply hard performance filter constraints; skip non-qualifying tickers
            if not np.isnan(r) and not (min_ret/100 <= r <= max_ret/100): continue
            if not np.isnan(s) and s < min_sharpe: continue

            rows.append({
                "Ticker": t, "Name": name, "Sector": sector,
                "Ann. Return": r, "Volatility": v, "Sharpe": s, "Max Drawdown": dd, "YTD Return": ytd,
            })

        if not rows:
            st.warning("No ETFs met your strict filter criteria. Try adjusting the sliders.")
            return

        # Sort the filtered results by the user-selected column
        df_sorted = pd.DataFrame(rows).sort_values(sort_col, ascending=sort_asc, na_position="last")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("ETFs Found", len(df_sorted))
        s2.metric("Average Return", f"{df_sorted['Ann. Return'].mean():.1%}" if not df_sorted['Ann. Return'].isna().all() else "—")
        s3.metric("Average Sharpe", f"{df_sorted['Sharpe'].mean():.2f}" if not df_sorted['Sharpe'].isna().all() else "—")
        s4.metric("Average Drawdown", f"{df_sorted['Max Drawdown'].mean():.1%}" if not df_sorted['Max Drawdown'].isna().all() else "—")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        def fmt(v, pct=True):
            """Format a float as a percentage string or plain decimal; return '—' for NaN."""
            if np.isnan(v): return "—"
            return f"{v:.2%}" if pct else f"{v:.2f}"

        # Build the HTML results table manually (rather than st.dataframe) to apply
        # the custom .styled-table CSS and conditional color classes (.pos/.neg/.neu)
        table_html = """<table class="styled-table"><thead><tr>
            <th>Ticker</th><th>Asset Name</th><th>Sector</th>
            <th>Ann. Return</th><th>Volatility</th>
            <th>Sharpe Score</th><th>Max Drawdown</th><th>YTD Return</th>
        </tr></thead><tbody>"""

        for _, row in df_sorted.iterrows():
            # Conditional CSS classes for color-coding each value cell:
            # .pos = teal (positive / good), .neg = red (negative / bad), .neu = gray (neutral)
            r_cls  = "pos" if row["Ann. Return"] > 0 else "neg"
            s_cls  = "pos" if row["Sharpe"] > 1 else "neu" if row["Sharpe"] > 0 else "neg"
            y_cls  = "pos" if not np.isnan(row["YTD Return"]) and row["YTD Return"] > 0 else "neg"
            table_html += f"""<tr>
                <td>{row['Ticker']}</td>
                <td>{row['Name']}</td>
                <td style="color:#a0aec0;font-size:0.9rem;">{row['Sector']}</td>
                <td class="{r_cls}">{fmt(row['Ann. Return'])}</td>
                <td style="color:#ffffff;">{fmt(row['Volatility'])}</td>
                <td class="{s_cls}">{fmt(row['Sharpe'],False)}</td>
                <td class="neg">{fmt(row['Max Drawdown'])}</td>  <!-- Drawdown is always negative -->
                <td class="{y_cls}">{fmt(row['YTD Return'])}</td>
            </tr>"""
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        # Risk vs. Return scatter — one trace per sector for automatic legend grouping.
        # Each data point is labeled with its ticker symbol for immediate identification.
        fig_sc = go.Figure()
        for sector_name in df_sorted["Sector"].unique():
            sub = df_sorted[df_sorted["Sector"] == sector_name]
            fig_sc.add_trace(go.Scatter(
                x=sub["Volatility"], y=sub["Ann. Return"],
                mode="markers+text", name=sector_name,
                text=sub["Ticker"], textposition="top center",
                textfont=dict(size=11, color="#ffffff", family="EB Garamond, Garamond, serif"),
                marker=dict(size=12, opacity=0.85, line=dict(width=1, color="#ffffff")),
                hovertemplate="<b>%{text}</b><br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>",
            ))
        fig_sc.update_layout(**BASE_LAYOUT)
        fig_sc.update_layout(
            title="Risk vs. Return Distribution", height=480,
            margin=dict(l=60, r=24, t=80, b=60),
            xaxis=dict(title="Annualized Volatility (Risk)", tickformat=".0%"),
            yaxis=dict(title="Annualized Rate of Return", tickformat=".0%"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXPLORE ETFs — Sector Deep-Dive Analysis
# ─────────────────────────────────────────────────────────────────────────────
def page_explore():
    """
    Render the Explore Sectors page for a detailed look at a single market sector.

    Provides four analysis sections for every ETF in the selected sector:

      1. Fund information cards — grid of constituent ETF descriptions pulled
         from the ETF_UNIVERSE registry (name + one-line investment thesis).

      2. Cumulative returns chart — growth of $1 invested from the start of
         the selected period.  Lets users immediately compare compounded
         performance across sector peers.

      3. Performance stats table — ann. return, volatility, Sharpe, max drawdown,
         YTD return per fund with the same .styled-table / .pos/.neg color system.

      4. Rolling volatility chart — 30-day rolling standard deviation annualised
         via √252.  Reveals how each fund's risk profile evolves through different
         market regimes (e.g. COVID crash, rate-hike cycles).

      5. Correlation heatmap — pairwise Pearson correlations between all funds in
         the sector.  Values near +1 = highly correlated (diversification benefit low);
         values near 0 or negative = strong diversification benefit.
         Only shown when len(valid) > 1 (single-fund sectors have no pairs).

    User controls: sector selector, timeframe (1Y/2Y/3Y/5Y/Max), risk-free rate.
    """
    page_header("Explore Sectors", "Deep-dive into specific market sectors to analyze historical trends and asset correlations")

    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        sector = st.selectbox("Select Sector", ALL_SECTORS)
    with c2:
        period_map = {"1Y":365,"2Y":730,"3Y":1095,"5Y":1825,"Max":3650}
        period = st.selectbox("Timeframe", list(period_map.keys()), index=2)
    with c3:
        rf = st.slider("Risk-Free Rate (%)", 0.0, 6.0, 4.5, 0.25, format="%.2f%%") / 100

    # Fetch all tickers belonging to the selected sector
    tickers = list(ETF_UNIVERSE[sector].keys())
    start   = (datetime.today() - timedelta(days=period_map[period])).strftime("%Y-%m-%d")

    with st.spinner(f"Loading data for {sector}..."):
        data = download_data(tickers, start=start)

    if data.empty:
        st.error("Could not retrieve historical data.")
        return

    # Filter to tickers that actually returned data (handles delisted/unavailable names)
    valid   = [t for t in tickers if t in data.columns]
    returns = data[valid].pct_change().dropna()   # Daily return series
    cum_ret = (1 + returns).cumprod()             # Cumulative wealth index (starts at $1)
    ann_ret, ann_vol, sharpe, max_dd = safe_stats(returns, rf)

    # ── Section 1: Fund information cards ──────────────────────────────────────
    st.markdown("<p class='section-title' style='color:#00d4aa;'>Sector Constituent Details</p>", unsafe_allow_html=True)
    grid_cols = st.columns(3)   # 3-column card grid wraps naturally for any sector size
    for idx, t in enumerate(valid):
        name, desc = ALL_TICKERS.get(t, (t, "No summary available."))
        with grid_cols[idx % 3]:   # Modulo distributes cards evenly across columns
            st.markdown(f"""
            <div style="background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:1.2rem;margin-bottom:0.85rem;min-height:120px;">
                <div style="font-family:'EB Garamond',Georgia,serif;font-weight:600;font-size:1.15rem;color:#ffffff;margin-bottom:0.4rem;">{t} — <span style="color:#4f8ef7;">{name}</span></div>
                <div style="font-family:'EB Garamond',Georgia,serif;font-size:1.05rem;color:#e2e8f0;line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Section 2: Cumulative Returns Chart ────────────────────────────────────
    # Each line shows $1 invested growing over the selected period.
    # Crossing lines / divergence immediately reveals relative outperformers.
    fig_cum = go.Figure()
    for i, t in enumerate(valid):
        fig_cum.add_trace(go.Scatter(
            x=cum_ret.index, y=cum_ret[t], name=t, mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=2.5),
            hovertemplate=f"<b>{t}</b>  $%{{y:.3f}}<extra></extra>",
        ))
    fig_cum.update_layout(**BASE_LAYOUT)
    fig_cum.update_layout(title=f"{sector} — Historical Growth of $1", height=400,
                          margin=dict(l=24, r=24, t=80, b=40), yaxis_title="Growth Value ($)")
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── Section 3: Per-Fund Performance Stats Table ─────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    # YTD: compound from first trading day of the current calendar year
    try:
        idx       = pd.DatetimeIndex(returns.index)
        ytd_start = idx[idx.year == datetime.today().year][0]
        ytd_ret   = (1 + returns.loc[ytd_start:]).cumprod().iloc[-1] - 1
    except Exception:
        ytd_ret = pd.Series({t: np.nan for t in valid})

    def fmt(v, pct=True):
        """Format a float as a percentage or decimal; display '—' for NaN."""
        if np.isnan(v): return "—"
        return f"{v:.2%}" if pct else f"{v:.2f}"

    table_html = """<table class="styled-table"><thead><tr>
        <th>Ticker</th><th>Fund Name</th>
        <th>Ann. Return</th><th>Volatility</th>
        <th>Sharpe Score</th><th>Max Drawdown</th><th>YTD Return</th>
    </tr></thead><tbody>"""
    for t in valid:
        name  = ALL_TICKERS.get(t, (t, ""))[0]
        r     = ann_ret.get(t, np.nan)
        v     = ann_vol.get(t, np.nan)
        s     = sharpe.get(t, np.nan)
        dd    = max_dd.get(t, np.nan)
        ytd   = ytd_ret.get(t, np.nan) if isinstance(ytd_ret, pd.Series) else np.nan
        r_cls = "pos" if r > 0 else "neg"
        s_cls = "pos" if s > 1 else "neu"
        y_cls = "pos" if not np.isnan(ytd) and ytd > 0 else "neg"
        table_html += f"""<tr>
            <td>{t}</td><td>{name}</td>
            <td class="{r_cls}">{fmt(r)}</td>
            <td style="color:#ffffff;">{fmt(v)}</td>
            <td class="{s_cls}">{fmt(s,False)}</td>
            <td class="neg">{fmt(dd)}</td>
            <td class="{y_cls}">{fmt(ytd)}</td>
        </tr>"""
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    # ── Section 4: 30-Day Rolling Volatility ───────────────────────────────────
    # Rolling 30-day window captures short-term volatility regime shifts.
    # Annualised via √252 to keep the scale consistent with the stats table.
    roll_vol = returns.rolling(30).std() * np.sqrt(252)
    fig_rv   = go.Figure()
    for i, t in enumerate(valid):
        fig_rv.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol[t], name=t, mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            hovertemplate=f"<b>{t}</b>  %{{y:.2%}}<extra></extra>",
        ))
    fig_rv.update_layout(**BASE_LAYOUT)
    fig_rv.update_layout(title="30-Day Rolling Volatility Tracking", height=320,
                          margin=dict(l=24, r=24, t=80, b=40), yaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig_rv, use_container_width=True)

    # ── Section 5: Pairwise Correlation Heatmap ────────────────────────────────
    # Requires at least 2 assets; a single fund has no meaningful correlations.
    # Colorscale: red = -1 (perfect inverse), black = 0 (uncorrelated),
    #             teal = +1 (perfect positive correlation).
    # Chart height scales dynamically with the number of assets so small and
    # large sectors both render legibly.
    if len(valid) > 1:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        corr = returns.corr().round(2)   # Pearson correlation matrix
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale=[[0,"#f74b5c"], [0.5,"#111520"], [1,"#00d4aa"]],
            zmin=-1, zmax=1, text=corr.values, texttemplate="%{text:.2f}",
            textfont=dict(size=12, color="#ffffff", family="EB Garamond, Garamond, serif"),
            hovertemplate="%{x} / %{y}: %{z:.2f}<extra></extra>",
        ))
        fig_corr.update_layout(**BASE_LAYOUT)
        # Dynamic height: 55px per ticker + 100px header/margin buffer
        fig_corr.update_layout(title="Asset Correlation Matrix",
                              height=max(340, len(valid)*55+100), margin=dict(l=70, r=24, t=80, b=70))
        st.plotly_chart(fig_corr, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PORTFOLIO BUILDER — Markowitz Mean-Variance Optimization UI
# ─────────────────────────────────────────────────────────────────────────────
def page_portfolio():
    """
    Render the Portfolio Builder page.

    Allows users to assemble a custom basket of 2-15 ETFs and instantly compute
    and compare three portfolio strategies using the Markowitz framework:

      - Equal Weight    — Baseline naive diversification (1/N per asset)
      - Max Sharpe      — Tangency portfolio (highest risk-adjusted return)
      - Min Volatility  — Global Minimum Variance (lowest achievable risk)

    Output sections (rendered sequentially):

      1. Strategy Summary Metrics — 4-column grid of annualised return,
         volatility, and Sharpe ratio for each of the three strategies.

      2. Portfolio Weight Pie Charts — Three donut charts showing the target
         allocation for each strategy.  Suppressed with a Concentration Alert
         if any single asset receives >99% allocation (degenerate optimizer output
         that would make the pie chart misleading).

      3. Efficient Frontier Scatter — 800 Monte Carlo portfolios plotted as a
         color-coded cloud (red=low Sharpe → teal=high Sharpe), with the three
         strategy portfolios overlaid as diamond markers.

      4. Historical Backtest Chart — Cumulative growth of $1 for each strategy
         over the selected training period.

      5. Target Weights Table — HTML table showing the exact allocation
         percentage for every selected asset across all three strategies.
         Cells are color-coded: overweighted vs. equal weight → teal (.pos),
         at or underweighted → gray (.neu).

      6. Results Interpretation Panel — Dynamically generated plain-language
         explanation referencing the actual highest-return and lowest-volatility
         assets in the user's selection.

    User controls: asset multiselect (full TICKER_LIST), risk-free rate slider,
                   training period selector (1Y / 3Y / 5Y).
    """
    page_header("Portfolio Optimizer", "Mix assets and discover the most mathematically efficient portfolio weights")

    c1, c2 = st.columns([3, 1], gap="large")
    with c1:
        selected = st.multiselect(
            "Select Portfolio Assets (2–15 max):",
            options=TICKER_LIST,
            default=["SPY", "QQQ", "XLV", "AGG", "GLD"],
            format_func=lambda t: f"{t} — {ALL_TICKERS[t][0]}",
        )
    with c2:
        rf = st.slider("Risk-Free Rate (%)", 0.0, 6.0, 4.5, 0.25, format="%.2f%%") / 100
        period_map = {"1Y":365,"3Y":1095,"5Y":1825}
        period = st.selectbox("Training Data Period", list(period_map.keys()), index=1)

    if not selected or len(selected) < 2:
        st.info("Please select at least 2 assets above to build a portfolio.")
        return
    if len(selected) > 15:
        st.warning("Please select 15 or fewer assets to ensure clean visualization.")
        return

    start = (datetime.today() - timedelta(days=period_map[period])).strftime("%Y-%m-%d")
    with st.spinner("Calculating optimal asset allocations..."):
        data = download_data(selected, start=start)

    if data.empty:
        st.error("Failed to retrieve sufficient market data.")
        return

    # Guard: filter to assets with actual data (some tickers may have returned empty)
    valid = [t for t in selected if t in data.columns]
    if len(valid) < 2:
        st.error("Not enough valid asset data returned to compute a portfolio.")
        return

    # ── Core Optimization Inputs ────────────────────────────────────────────────
    rets = data[valid].pct_change().dropna()     # Daily return series
    ann_ret, ann_vol, sharpe, max_dd = safe_stats(rets, rf)

    mu   = rets.mean()    # Daily mean return vector  (used by optimizer + port_perf)
    cov  = rets.cov()     # Daily covariance matrix   (used by optimizer + port_perf)
    n    = len(valid)     # Number of assets in the portfolio

    # ── Compute the Three Strategy Weight Vectors ───────────────────────────────
    w_eq = np.full(n, 1/n)        # Equal Weight: trivial 1/N allocation
    w_ms = max_sharpe(mu, cov, rf) # Max Sharpe: SLSQP tangency portfolio
    w_mv = min_vol(mu, cov)        # Min Volatility: SLSQP global minimum variance

    # ── Compute Portfolio-Level Performance for Each Strategy ───────────────────
    r_eq, v_eq, s_eq = port_perf(w_eq, mu, cov, rf)
    r_ms, v_ms, s_ms = port_perf(w_ms, mu, cov, rf)
    r_mv, v_mv, s_mv = port_perf(w_mv, mu, cov, rf)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>Portfolio Strategy Comparison</p>", unsafe_allow_html=True)

    # ── Strategy Summary Metric Grid ────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Assets Used", n); m1.metric("Data Window", period)
    m2.metric("Equal Weight Return",      f"{r_eq:.2%}", delta=f"Sharpe {s_eq:.2f}")
    m2.metric("Equal Weight Volatility",  f"{v_eq:.2%}")
    m3.metric("Max Sharpe Return",        f"{r_ms:.2%}", delta=f"Sharpe {s_ms:.2f}")
    m3.metric("Max Sharpe Volatility",    f"{v_ms:.2%}")
    m4.metric("Min Volatility Return",    f"{r_mv:.2%}", delta=f"Sharpe {s_mv:.2f}")
    m4.metric("Min Volatility Volatility",f"{v_mv:.2%}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Concentration Check ─────────────────────────────────────────────────────
    # The SLSQP optimizer can produce degenerate solutions (100% in one asset)
    # when historical data strongly favors a single asset during the chosen period.
    # Detect this and suppress the misleading pie charts, showing a warning instead.
    is_ms_monopolized = np.any(w_ms > 0.99)
    is_mv_monopolized = np.any(w_mv > 0.99)

    if is_ms_monopolized or is_mv_monopolized:
        st.markdown("<div class='warn-box'><b>Concentration Alert:</b> Based on the selected data timeframe, the optimizer has allocated roughly 100% of the capital into a single asset. The standard pie charts have been disabled to keep the interface clean—check the target weights table below for exact allocations.</div>", unsafe_allow_html=True)
    else:
        # ── Weight Allocation Donut Charts (3-panel) ────────────────────────────
        # Three donut charts side-by-side in a 1×3 subplot grid.
        # hole=0.45 creates the donut effect; colors align with the COLORS palette.
        fig_alloc = make_subplots(
            rows=1, cols=3, specs=[[{"type":"pie"},{"type":"pie"},{"type":"pie"}]],
            subplot_titles=["Equal Allocation Base","Maximum Sharpe Optimizer","Minimum Volatility Model"],
        )
        for ci, (ws, lbl) in enumerate([(w_eq,"EQ"),(w_ms,"MS"),(w_mv,"MV")], 1):
            fig_alloc.add_trace(go.Pie(
                labels=valid, values=ws, name=lbl, hole=0.45,
                textfont=dict(size=13, color="#ffffff", family="EB Garamond, Garamond, serif"),
                textposition="inside", showlegend=True,
                marker=dict(colors=COLORS[:n], line=dict(color="#0b0e14", width=2)),
                hovertemplate="<b>%{label}</b><br>Target Weight: %{percent}<extra></extra>",
            ), row=1, col=ci)

        fig_alloc.update_layout(**BASE_LAYOUT)
        fig_alloc.update_layout(
            title="Portfolio Weight Distributions",
            height=430, margin=dict(l=24, r=24, t=90, b=70),
            legend=dict(orientation="h", y=-0.15, x=0.25)
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    # ── Efficient Frontier Monte Carlo Scatter ──────────────────────────────────
    # frontier() generates 800 random weight combinations.
    # Their Sharpe ratios are used as the colorscale so high-Sharpe portfolios
    # appear teal and low-Sharpe portfolios appear red — making the gradient
    # naturally guide the eye to the efficient edge of the cloud.
    with st.spinner("Simulating thousands of possible portfolios..."):
        ef_r, ef_v, ef_s = frontier(mu, cov)

    fig_ef = go.Figure()
    # Layer 1: Monte Carlo cloud — 800 random portfolios colored by Sharpe ratio
    fig_ef.add_trace(go.Scatter(
        x=ef_v, y=ef_r, mode="markers", name="Simulated Mix Distributions",
        marker=dict(color=ef_s, colorscale=[[0,"#f74b5c"],[0.5,"#f7c948"],[1,"#00d4aa"]],
                    size=6, opacity=0.6,
                    colorbar=dict(title="Sharpe Ratio", thickness=15, tickfont=dict(color="#ffffff", size=12, family="EB Garamond, Garamond, serif"))),
        hovertemplate="Annualized Risk: %{x:.2%}<br>Annualized Return: %{y:.2%}<extra></extra>",
    ))

    # Layer 2: Strategy overlays — diamond markers for the three computed portfolios
    for lbl, ret, vol, clr in [
        ("Equal Weight", r_eq, v_eq, "#a0aec0"),
        ("Max Sharpe",   r_ms, v_ms, "#00d4aa"),
        ("Min Volatility",r_mv, v_mv, "#4f8ef7"),
    ]:
        fig_ef.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers", name=lbl,
            marker=dict(color=clr, size=16, symbol="diamond", line=dict(color="#ffffff", width=2)),
            hovertemplate=f"<b>{lbl}</b><br>Risk: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>"
        ))

    fig_ef.update_layout(**BASE_LAYOUT)
    fig_ef.update_layout(
        title="The Efficient Frontier Curve",
        height=500, margin=dict(l=60, r=24, t=100, b=80),
        xaxis=dict(title="Annualized Volatility (Risk)", tickformat=".1%"),
        yaxis=dict(title="Expected Annual Return", tickformat=".1%"),
        legend=dict(orientation="h", y=-0.22, x=0.05)
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    # ── Historical Backtest Chart ───────────────────────────────────────────────
    # Applies each strategy's static weight vector to the historical daily return
    # series and compounds the result.  This is an in-sample backtest (the weights
    # were derived from the same data), so it illustrates historical behavior
    # rather than predictive performance.
    port_rets = pd.DataFrame({
        "Equal Weight":   (rets[valid] * w_eq).sum(axis=1),
        "Max Sharpe":     (rets[valid] * w_ms).sum(axis=1),
        "Min Volatility": (rets[valid] * w_mv).sum(axis=1),
    })
    cum = (1 + port_rets).cumprod()   # Cumulative wealth index starting at $1
    fig_bt = go.Figure()
    for col_name, clr in [("Equal Weight","#a0aec0"),("Max Sharpe","#00d4aa"),("Min Volatility","#4f8ef7")]:
        # Convert hex color to RGB tuple for the semi-transparent area fill
        rgb = tuple(int(clr.lstrip("#")[i:i+2], 16) for i in (0,2,4))
        fig_bt.add_trace(go.Scatter(
            x=cum.index, y=cum[col_name], name=col_name, mode="lines",
            line=dict(color=clr, width=2.5),
            fill="tozeroy", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.03)",  # Very faint fill
            hovertemplate=f"<b>{col_name}</b>  $%{{y:.3f}}<extra></extra>",
        ))
    fig_bt.update_layout(**BASE_LAYOUT)
    fig_bt.update_layout(title="Historical Portfolio Performance Backtest ($1 Initial Investment)", height=380,
                          margin=dict(l=24, r=24, t=80, b=40), yaxis_title="Portfolio Value ($)")
    st.plotly_chart(fig_bt, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>Target Portfolio Weights Table</p>", unsafe_allow_html=True)
    # ── Weights Comparison Table ────────────────────────────────────────────────
    # Color coding: teal (.pos) if the optimized weight exceeds equal weight
    # (the asset is "preferred" by the optimizer), gray (.neu) otherwise.
    table_html = """<table class="styled-table"><thead><tr>
        <th>Ticker Symbol</th><th>Asset Identity</th>
        <th>Equal Weight Profile</th><th>Max Sharpe Profile</th><th>Min Volatility Profile</th>
    </tr></thead><tbody>"""
    for i, t in enumerate(valid):
        name   = ALL_TICKERS.get(t, (t, ""))[0]
        ms_cls = "pos" if w_ms[i] > w_eq[i] else "neu"   # Max Sharpe overweight vs. equal
        mv_cls = "pos" if w_mv[i] > w_eq[i] else "neu"   # Min Vol overweight vs. equal
        table_html += f"""<tr>
            <td>{t}</td><td>{name}</td>
            <td style="color:#ffffff;">{w_eq[i]:.1%}</td>
            <td class="{ms_cls}">{w_ms[i]:.1%}</td>
            <td class="{mv_cls}">{w_mv[i]:.1%}</td>
        </tr>"""
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title' style='color:#00d4aa;'>How to Interpret Your Results</p>", unsafe_allow_html=True)

    # ── Dynamic Interpretation Panel ───────────────────────────────────────────
    # Identifies the best-performing and most-stable assets from the user's
    # actual selection, then embeds their ticker symbols in the explanation
    # for a personalized, context-aware analysis writeup.
    highest_ret_ticker = ann_ret[valid].idxmax()   # Ticker with highest annualised return
    lowest_vol_ticker  = ann_vol[valid].idxmin()   # Ticker with lowest annualised volatility

    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.8rem;line-height:1.8;font-family:'EB Garamond',Georgia,serif;font-size:1.1rem;color:#e2e8f0;">
        <div style="margin-bottom:1.4rem;">
            <b style="color:#ffffff;font-size:1.25rem;">1. Reading the Efficient Frontier</b><br>
            Every dot on the scatter plot represents a different, randomly generated mix of the assets you selected. The upper, outer edge of that cloud of dots is called the "Efficient Frontier." Any portfolio resting on this top edge is mathematically optimal. The highlighted <span style="color:#00d4aa;font-weight:600;">Max Sharpe</span> portfolio marks the absolute peak of efficiency—the exact spot where you get the most possible return for the amount of risk you take.
        </div>
        <div style="margin-bottom:1.4rem;">
            <b style="color:#ffffff;font-size:1.25rem;">2. Weighing Your Allocation Choices</b><br>
            Based on the timeline you chose, <span style="color:#f7c948;font-weight:600;">{highest_ret_ticker}</span> has been your best performer, while <span style="color:#4f8ef7;font-weight:600;">{lowest_vol_ticker}</span> has been the most stable.
            <ul style="margin-top:0.6rem;">
                <li style="margin-bottom:0.4rem;">If your primary goal is to protect capital from market crashes, look closely at the <b>Minimum Volatility Model</b>. It specifically blends assets to cancel out each other's wild swings.</li>
                <li>If you have a longer timeframe and want to grow your wealth aggressively, the <b>Max Sharpe Model</b> tells you exactly how much to allocate to capture growth without taking on unnecessary, unrewarded risk.</li>
            </ul>
        </div>
        <div>
            <b style="color:#ffffff;font-size:1.25rem;">3. Why This Matters</b><br>
            Professional asset managers use these exact models to remove emotion from investing. By looking at historical data and mathematical correlations rather than gut feelings, this dashboard lets you build structured, highly resilient portfolios tailored to exact risk tolerances.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION ROUTER — Entry Point
# ─────────────────────────────────────────────────────────────────────────────
# PAGE_MAP connects page name strings (stored in session_state) to their
# corresponding render functions.  This dict-based dispatch pattern eliminates
# if/elif chains and makes it trivial to add new pages.
#
# The final line is the application's entry point: it reads the current page
# from session_state (set by navigate() callbacks) and calls the correct
# render function.  Because Streamlit re-executes the entire script on every
# user interaction, this line always runs last, routing to whichever page is
# currently active.
# ─────────────────────────────────────────────────────────────────────────────
PAGE_MAP = {
    "Home":             page_home,
    "ETF Screener":     page_screener,
    "Explore ETFs":     page_explore,
    "Portfolio Builder": page_portfolio,
}
PAGE_MAP[st.session_state["page"]]()   # Dispatch to the active page's render function