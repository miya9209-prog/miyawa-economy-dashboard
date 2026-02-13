# app.py (V2) - 통째로 교체해서 사용하세요

import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import requests
import feedparser
from bs4 import BeautifulSoup

import plotly.graph_objects as go

import yfinance as yf
import FinanceDataReader as fdr
from pykrx import stock as krx
from pandas_datareader.data import DataReader


# =========================
# Page / Theme
# =========================
st.set_page_config(
    page_title="재테크 핵심지표 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Font */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
  font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

/* Global spacing */
.block-container {
  padding-top: 1.4rem;
  padding-bottom: 2.5rem;
  max-width: 1240px;
}

/* Titles */
h1, h2, h3 {
  letter-spacing: -0.02em;
}
h1 {
  font-weight: 700 !important;
  font-size: 1.8rem !important;
  margin-bottom: 0.2rem !important;
}
.small-muted {
  color: rgba(0,0,0,0.55);
  font-size: 0.92rem;
}

/* Section */
.section-title{
  font-weight: 700;
  font-size: 1.18rem;
  margin: 0.3rem 0 0.9rem 0;
  letter-spacing: -0.02em;
}
.section-sub{
  color: rgba(0,0,0,0.55);
  font-size: 0.92rem;
  margin-top: -0.5rem;
  margin-bottom: 1.0rem;
}

/* Cards */
.card {
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 14px 14px 12px 14px;
  background: #ffffff;
  box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}
.card-tight{
  padding: 12px 12px 10px 12px;
}
.card-title{
  font-weight: 600;
  font-size: 0.92rem;
  color: rgba(0,0,0,0.62);
  margin-bottom: 6px;
}
.kpi{
  font-size: 1.25rem;
  font-weight: 750;
  letter-spacing: -0.02em;
}
.delta{
  font-size: 0.92rem;
  font-weight: 600;
  margin-top: 2px;
}
.delta-pos{ color: #0a7b34; }
.delta-neg{ color: #b42318; }
.delta-flat{ color: rgba(0,0,0,0.55); }

.hr-soft{
  border-top: 1px solid rgba(0,0,0,0.06);
  margin: 0.9rem 0 1.1rem 0;
}

/* Plotly container spacing */
.plot-wrap{
  margin-top: 6px;
}

/* Sidebar */
section[data-testid="stSidebar"] .block-container{
  padding-top: 1.2rem;
}
.sidebar-title{
  font-weight: 800;
  font-size: 1.05rem;
  margin-bottom: 0.6rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  gap: 8px;
}
.stTabs [data-baseweb="tab"]{
  height: 40px;
  border-radius: 12px;
  padding: 0 14px;
  border: 1px solid rgba(0,0,0,0.06);
}
.stTabs [aria-selected="true"]{
  border: 1px solid rgba(0,0,0,0.12) !important;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

/* Links */
a { text-decoration: none; }
a:hover { text-decoration: underline; }

/* Reduce default st.metric padding a bit */
div[data-testid="stMetric"]{
  padding: 0.1rem 0 0.1rem 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def now_local() -> datetime:
    return datetime.now()

def days_for_freq(freq: str) -> int:
    # 그래프 보기 좋게: 일간(6개월), 주간(3년), 월간(10년)
    return {"D": 180, "W": 365 * 3, "M": 365 * 10}.get(freq, 365)

def freq_label(freq: str) -> str:
    return {"D": "일간", "W": "주간", "M": "월간"}.get(freq, freq)

def ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out

def to_close_df(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임에서 Close 컬럼을 확보해 단일 Close DF로 반환"""
    if df is None or df.empty:
        return pd.DataFrame()
    df = ensure_dt_index(df)
    if "Close" in df.columns:
        return df[["Close"]].dropna()
    # 단일 컬럼일 수 있음
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return pd.DataFrame()
    return df[[numeric_cols[0]]].rename(columns={numeric_cols[0]: "Close"}).dropna()

def resample_close(df_close: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df_close is None or df_close.empty:
        return pd.DataFrame()
    df_close = ensure_dt_index(df_close)
    if freq == "D":
        return df_close.dropna()
    rule = "W-FRI" if freq == "W" else "M"
    return df_close.resample(rule).last().dropna()

def metric_from_close(df_close: pd.DataFrame):
    if df_close is None or df_close.empty:
        return None, None, None
    s = df_close["Close"].dropna()
    if len(s) < 2:
        return float(s.iloc[-1]) if len(s) else None, None, None
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2])
    delta = last - prev
    pct = (delta / prev) * 100 if prev != 0 else None
    return last, delta, pct

def delta_class(delta: float | None):
    if delta is None:
        return "delta-flat"
    if abs(delta) < 1e-12:
        return "delta-flat"
    return "delta-pos" if delta > 0 else "delta-neg"

def normalize_100(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c].dropna()
        if len(s) == 0:
            continue
        out[c] = (out[c] / s.iloc[0]) * 100
    return out

def plot_line(df: pd.DataFrame, title: str, height: int = 280, normalized: bool = False):
    if df is None or df.empty:
        st.info(f"{title}: 데이터가 없습니다.")
        return

    if normalized:
        df = normalize_100(df.dropna(how="all"))

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=str(col)))
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=8, r=8, t=42, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

def safe_get(url: str, timeout: int = 10):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinanceDashboard/2.0; +https://streamlit.io)"}
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r


# =========================
# Data Fetchers (cached)
# =========================
@st.cache_data(ttl=60 * 30)
def fetch_fdr(symbol: str, start: str) -> pd.DataFrame:
    df = fdr.DataReader(symbol, start)
    return ensure_dt_index(df)

@st.cache_data(ttl=60 * 30)
def fetch_yf(symbol: str, start: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, progress=False, auto_adjust=False)
    return ensure_dt_index(df)

@st.cache_data(ttl=60 * 60)
def fetch_fred(series_list: list[str], start: datetime) -> pd.DataFrame:
    df = DataReader(series_list, "fred", start)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

@st.cache_data(ttl=60 * 60)
def get_krx_top10_by_mktcap():
    biz = krx.get_nearest_business_day_in_a_week()
    caps = krx.get_market_cap_by_ticker(biz).sort_values("시가총액", ascending=False).head(10)
    tickers = caps.index.tolist()
    names = [krx.get_market_ticker_name(t) for t in tickers]
    out = pd.DataFrame({"티커": tickers, "종목명": names, "시가총액": caps["시가총액"].values})
    return biz, out

@st.cache_data(ttl=60 * 10)
def fetch_rss(feed_url: str, limit: int = 20):
    d = feedparser.parse(feed_url)
    items = []
    for e in d.entries[:limit]:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        items.append({"title": title, "link": link, "published": published})
    return items

@st.cache_data(ttl=60 * 5)
def fetch_naver_finance_news(limit: int = 20):
    url = "https://finance.naver.com/news/"
    r = safe_get(url, timeout=10)
    soup = BeautifulSoup(r.text, "lxml")

    items = []
    for a in soup.select("a"):
        href = a.get("href", "")
        text = a.get_text(" ", strip=True)
        if not text or len(text) < 10:
            continue

        if "news_read.naver" in href or "news.naver.com" in href or "read.naver" in href:
            link = href
            if link.startswith("/"):
                link = "https://finance.naver.com" + link
            items.append({"title": text, "link": link, "published": ""})
        elif href.startswith("/news/"):
            link = "https://finance.naver.com" + href
            items.append({"title": text, "link": link, "published": ""})

    # dedupe
    seen = set()
    uniq = []
    for it in items:
        k = (it["title"], it["link"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)
        if len(uniq) >= limit:
            break
    return uniq


# =========================
# UI Components
# =========================
def card_kpi(title: str, last, delta, pct, suffix: str = "", precision: int = 2):
    if last is None:
        val = "-"
        dtxt = ""
        cls = "delta-flat"
    else:
        val = f"{last:,.{precision}f}{suffix}"
        if pct is None or delta is None:
            dtxt = ""
            cls = "delta-flat"
        else:
            sign = "+" if delta > 0 else ""
            dtxt = f"{sign}{delta:,.{precision}f} ({pct:+.2f}%)"
            cls = delta_class(delta)

    st.markdown(
        f"""
        <div class="card card-tight">
          <div class="card-title">{title}</div>
          <div class="kpi">{val}</div>
          <div class="delta {cls}">{dtxt}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def section(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


# =========================
# Sidebar
# =========================
st.sidebar.markdown('<div class="sidebar-title">⚙️ 설정</div>', unsafe_allow_html=True)

refresh_on = st.sidebar.toggle("자동 새로고침", value=True)
refresh_min = st.sidebar.select_slider("갱신 주기(분)", options=[1, 2, 3, 5, 10, 15], value=5)
news_limit = st.sidebar.slider("뉴스 표시 개수", 10, 60, 25, 5)

st.sidebar.markdown("—")
show_brent = st.sidebar.toggle("유가: 브렌트도 표시", value=True)
show_crypto = st.sidebar.toggle("코인(비트코인/이더) 표시", value=True)
show_metals = st.sidebar.toggle("원자재(실버/구리/천연가스) 표시", value=True)

st.sidebar.markdown("—")
keyword = st.sidebar.text_input("뉴스 키워드 필터(선택)", value="").strip()

st.sidebar.markdown("—")
if st.sidebar.button("지금 새로고침"):
    st.cache_data.clear()
    st.rerun()

if refresh_on:
    ms = int(refresh_min * 60 * 1000)
    st.components.v1.html(f"<script>setTimeout(()=>window.location.reload(), {ms});</script>", height=0)


# =========================
# Header
# =========================
st.markdown("📈 **재테크 핵심지표 대시보드 v2**")
st.markdown(
    f'<div class="small-muted">주요 지수 · 국내 시총 Top10 · 환율 · 금/유가 · 금리 · DXY/VIX/코인/원자재 · 실시간 경제뉴스 — 일간/주간/월간</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)


# =========================
# Universe (symbols)
# =========================
# 국내 지수: FDR
KR_INDICES = {
    "KOSPI": "KS11",
    "KOSDAQ": "KQ11",
}

# 미국 지수: yfinance
US_INDICES = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
}

# FX / Risk
RISK_ETC = {
    "USD/KRW": ("FDR", "USD/KRW"),
    "DXY(달러인덱스)": ("YF", "DX-Y.NYB"),
    "VIX(변동성)": ("YF", "^VIX"),
}

# Commodities
COMMODITIES = {
    "Gold": ("YF", "GC=F"),
    "WTI": ("YF", "CL=F"),
    "Brent": ("YF", "BZ=F"),
    "Silver": ("YF", "SI=F"),
    "Copper": ("YF", "HG=F"),
    "NatGas": ("YF", "NG=F"),
}

# Crypto
CRYPTO = {
    "Bitcoin": ("YF", "BTC-USD"),
    "Ethereum": ("YF", "ETH-USD"),
}

# Rates (FRED)
FRED_SERIES = {
    "US 2Y (DGS2)": "DGS2",
    "US 10Y (DGS10)": "DGS10",
    "Fed Funds": "FEDFUNDS",
    "Korea 10Y": "IRLTLT01KRM156N",
}


# =========================
# Fetch bundle
# =========================
def get_series(source: str, symbol: str, start: str) -> pd.DataFrame:
    try:
        if source == "FDR":
            return to_close_df(fetch_fdr(symbol, start))
        return to_close_df(fetch_yf(symbol, start))
    except Exception:
        return pd.DataFrame()

def render_overview(freq: str, start: str):
    section("요약 스냅샷", "오늘 시장을 빠르게 훑고, 변화가 큰 곳부터 확인하세요.")

    # 핵심 KPI 카드 8개 정도 (여백 고려)
    cols = st.columns(4)
    # KOSPI, NASDAQ, USDKRW, Gold, WTI, US10Y, DXY, VIX (+BTC optional)
    kpi_defs = [
        ("KOSPI", "FDR", "KS11", "", 2),
        ("NASDAQ", "YF", "^IXIC", "", 2),
        ("USD/KRW", "FDR", "USD/KRW", "", 2),
        ("Gold", "YF", "GC=F", "", 2),
        ("WTI", "YF", "CL=F", "", 2),
        ("US 10Y", "FRED", "DGS10", "%", 2),
        ("DXY", "YF", "DX-Y.NYB", "", 2),
        ("VIX", "YF", "^VIX", "", 2),
    ]

    # FRED는 별도 처리(데이터프레임 형태)
    # FRED 묶음 fetch
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    fred_df = pd.DataFrame()
    try:
        fred_df = fetch_fred(list(FRED_SERIES.values()), start_dt)
    except Exception:
        fred_df = pd.DataFrame()

    for i, (title, src, sym, suffix, prec) in enumerate(kpi_defs):
        with cols[i % 4]:
            if src == "FRED":
                if fred_df is not None and not fred_df.empty and sym in fred_df.columns:
                    s = fred_df[sym].dropna()
                    if freq == "W":
                        s = s.resample("W-FRI").last()
                    elif freq == "M":
                        s = s.resample("M").last()
                    dfc = pd.DataFrame({"Close": s}).dropna()
                    last, delta, pct = metric_from_close(dfc)
                    card_kpi(title, last, delta, pct, suffix=suffix, precision=prec)
                else:
                    card_kpi(title, None, None, None, suffix=suffix, precision=prec)
            else:
                df = get_series(src, sym, start)
                df = resample_close(df, freq)
                last, delta, pct = metric_from_close(df)
                card_kpi(title, last, delta, pct, suffix=suffix, precision=prec)

    if show_crypto:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            df = resample_close(get_series("YF", "BTC-USD", start), freq)
            last, delta, pct = metric_from_close(df)
            card_kpi("Bitcoin (BTC)", last, delta, pct, suffix="", precision=0)
        with c2:
            df = resample_close(get_series("YF", "ETH-USD", start), freq)
            last, delta, pct = metric_from_close(df)
            card_kpi("Ethereum (ETH)", last, delta, pct, suffix="", precision=0)

    st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)


def render_indices(freq: str, start: str):
    section("주요 주가지수", "국내(코스피/코스닥) + 미국(S&P500/나스닥/다우) 흐름 비교")

    left, right = st.columns([1.05, 1.0])

    with left:
        # 국내
        df_k = {}
        for name, sym in KR_INDICES.items():
            d = resample_close(get_series("FDR", sym, start), freq)
            if not d.empty:
                df_k[name] = d["Close"]
        df_k = pd.DataFrame(df_k).dropna(how="all")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**국내 지수(정규화 100)**")
        if not df_k.empty:
            plot_line(df_k.dropna(), "KOSPI vs KOSDAQ (Normalized=100)", height=320, normalized=True)
        else:
            st.info("국내 지수 데이터를 가져오지 못했습니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        # 미국
        df_u = {}
        for name, sym in US_INDICES.items():
            d = resample_close(get_series("YF", sym, start), freq)
            if not d.empty:
                df_u[name] = d["Close"]
        df_u = pd.DataFrame(df_u).dropna(how="all")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**미국 지수(정규화 100)**")
        if not df_u.empty:
            plot_line(df_u.dropna(), "S&P500 vs NASDAQ vs DOW (Normalized=100)", height=320, normalized=True)
        else:
            st.info("미국 지수 데이터를 가져오지 못했습니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)


def render_kr_top10(freq: str, start: str):
    section("국내 시가총액 Top10", "가장 가까운 영업일 기준 자동 추출 + 가격 흐름(정규화 100)")

    biz, top10 = get_krx_top10_by_mktcap()
    c1, c2 = st.columns([0.95, 1.55])

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**기준 영업일: {biz}**")
        # 보기 좋게 시총 단위 변환
        df_show = top10.copy()
        df_show["시가총액(조원)"] = (df_show["시가총액"] / 1e12).round(2)
        df_show = df_show.drop(columns=["시가총액"])
        st.dataframe(df_show, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Top10 가격 흐름(정규화 100)**")

        prices = {}
        for _, row in top10.iterrows():
            name = row["종목명"]
            ticker = row["티커"]
            d = resample_close(get_series("FDR", ticker, start), freq)
            if not d.empty:
                prices[name] = d["Close"]
        df_top = pd.DataFrame(prices).dropna(how="all")
        if not df_top.empty:
            plot_line(df_top.dropna(), "KR Top10 (Normalized=100)", height=360, normalized=True)
        else:
            st.info("Top10 가격 데이터를 가져오지 못했습니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)


def render_macro(freq: str, start: str):
    section("환율 · 원자재 · 리스크 지표", "원/달러, 달러 강세(DXY), 변동성(VIX), 금/유가 + 추가 원자재")

    # 1행: USDKRW / DXY / VIX
    c1, c2, c3 = st.columns(3)
    with c1:
        df = resample_close(get_series("FDR", "USD/KRW", start), freq)
        last, delta, pct = metric_from_close(df)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_kpi("USD/KRW", last, delta, pct, precision=2)
        st.markdown('<div class="plot-wrap"></div>', unsafe_allow_html=True)
        plot_line(df, "USD/KRW", height=260, normalized=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        df = resample_close(get_series("YF", "DX-Y.NYB", start), freq)
        last, delta, pct = metric_from_close(df)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_kpi("DXY (Dollar Index)", last, delta, pct, precision=2)
        st.markdown('<div class="plot-wrap"></div>', unsafe_allow_html=True)
        plot_line(df, "DXY", height=260, normalized=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        df = resample_close(get_series("YF", "^VIX", start), freq)
        last, delta, pct = metric_from_close(df)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_kpi("VIX", last, delta, pct, precision=2)
        st.markdown('<div class="plot-wrap"></div>', unsafe_allow_html=True)
        plot_line(df, "VIX", height=260, normalized=False)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # 2행: Gold / WTI / Brent(옵션)
    c1, c2, c3 = st.columns(3)
    with c1:
        df = resample_close(get_series("YF", "GC=F", start), freq)
        last, delta, pct = metric_from_close(df)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_kpi("Gold (GC=F)", last, delta, pct, precision=2)
        plot_line(df, "Gold", height=260, normalized=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        df = resample_close(get_series("YF", "CL=F", start), freq)
        last, delta, pct = metric_from_close(df)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_kpi("WTI (CL=F)", last, delta, pct, precision=2)
        plot_line(df, "WTI", height=260, normalized=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if show_brent:
            df = resample_close(get_series("YF", "BZ=F", start), freq)
            last, delta, pct = metric_from_close(df)
            card_kpi("Brent (BZ=F)", last, delta, pct, precision=2)
            plot_line(df, "Brent", height=260, normalized=False)
        else:
            st.markdown("**Brent**")
            st.caption("사이드바에서 브렌트 표시를 켜면 보입니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3행: 추가 원자재(실버/구리/천연가스)
    if show_metals:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            df = resample_close(get_series("YF", "SI=F", start), freq)
            last, delta, pct = metric_from_close(df)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            card_kpi("Silver (SI=F)", last, delta, pct, precision=2)
            plot_line(df, "Silver", height=250, normalized=False)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            df = resample_close(get_series("YF", "HG=F", start), freq)
            last, delta, pct = metric_from_close(df)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            card_kpi("Copper (HG=F)", last, delta, pct, precision=2)
            plot_line(df, "Copper", height=250, normalized=False)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            df = resample_close(get_series("YF", "NG=F", start), freq)
            last, delta, pct = metric_from_close(df)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            card_kpi("Nat Gas (NG=F)", last, delta, pct, precision=2)
            plot_line(df, "Natural Gas", height=250, normalized=False)
            st.markdown('</div>', unsafe_allow_html=True)

    # 코인(옵션)
    if show_crypto:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            df = resample_close(get_series("YF", "BTC-USD", start), freq)
            last, delta, pct = metric_from_close(df)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            card_kpi("Bitcoin (BTC-USD)", last, delta, pct, precision=0)
            plot_line(df, "Bitcoin", height=280, normalized=False)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            df = resample_close(get_series("YF", "ETH-USD", start), freq)
            last, delta, pct = metric_from_close(df)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            card_kpi("Ethereum (ETH-USD)", last, delta, pct, precision=0)
            plot_line(df, "Ethereum", height=280, normalized=False)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)


def render_rates(freq: str, start: str):
    section("금리 동향", "미국 2Y/10Y · 연준 기준금리 · 한국 10Y + 2Y-10Y 스프레드(경기 신호)")

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    try:
        fred = fetch_fred(list(FRED_SERIES.values()), start_dt)
    except Exception:
        fred = pd.DataFrame()

    if fred is None or fred.empty:
        st.info("금리 데이터를 가져오지 못했습니다.")
        st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)
        return

    # Resample
    if freq == "W":
        fred_r = fred.resample("W-FRI").last()
    elif freq == "M":
        fred_r = fred.resample("M").last()
    else:
        fred_r = fred.copy()

    # KPI row
    c1, c2, c3, c4 = st.columns(4)

    def kpi_rate(col, label, series_code):
        with col:
            s = fred_r[series_code].dropna()
            dfc = pd.DataFrame({"Close": s}).dropna()
            last, delta, pct = metric_from_close(dfc)
            # 금리는 절대변화가 중요(퍼센트포인트)
            st.markdown('<div class="card card-tight">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">{label}</div>', unsafe_allow_html=True)
            if last is None:
                st.markdown('<div class="kpi">-</div>', unsafe_allow_html=True)
                st.markdown('<div class="delta delta-flat"></div>', unsafe_allow_html=True)
            else:
                dpp = (delta) if delta is not None else None
                cls = delta_class(dpp)
                dtxt = f"{dpp:+.2f}p" if dpp is not None else ""
                st.markdown(f'<div class="kpi">{last:.2f}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="delta {cls}">{dtxt}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    kpi_rate(c1, "US 2Y", "DGS2")
    kpi_rate(c2, "US 10Y", "DGS10")
    kpi_rate(c3, "Fed Funds", "FEDFUNDS")
    kpi_rate(c4, "Korea 10Y", "IRLTLT01KRM156N")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Charts row
    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        df_plot = pd.DataFrame({
            "US 2Y": fred_r["DGS2"],
            "US 10Y": fred_r["DGS10"],
            "Fed Funds": fred_r["FEDFUNDS"],
        }).dropna(how="all")
        plot_line(df_plot, "US Rates", height=320, normalized=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # 2Y-10Y Spread
        spread = (fred_r["DGS10"] - fred_r["DGS2"]).dropna()
        df_sp = pd.DataFrame({"2Y-10Y Spread": spread}).dropna()
        plot_line(df_sp, "US 2Y-10Y Spread", height=320, normalized=False)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)


def render_news():
    section("실시간 경제 뉴스", "네이버/다음/구글 — 빠르게 훑고, 키워드로 걸러보세요.")

    # RSS
    daum_rss = "http://media.daum.net/rss/part/primary/economic/rss2.xml"
    google_rss = "http://news.google.co.kr/news?pz=1&hdlOnly=1&cf=all&ned=kr&hl=ko&topic=b&output=rss"

    colA, colB, colC = st.columns(3)

    def render_list(items, source_name: str):
        if keyword:
            k = keyword.lower()
            items = [it for it in items if k in (it["title"] or "").lower()]
        if not items:
            st.caption("표시할 뉴스가 없습니다.")
            return
        for it in items[:news_limit]:
            title = it.get("title", "").strip()
            link = it.get("link", "").strip()
            pub = it.get("published", "").strip()
            pub_txt = f" · {pub}" if pub else ""
            st.markdown(f"- [{title}]({link}){pub_txt}")

    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**NAVER (Finance)**")
        st.caption("공식 RSS 제한 → 금융 뉴스 페이지 크롤링")
        try:
            items = fetch_naver_finance_news(limit=max(news_limit, 30))
            render_list(items, "NAVER")
        except Exception as e:
            st.warning(f"네이버 수집 실패: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**DAUM (RSS)**")
        try:
            items = fetch_rss(daum_rss, limit=max(news_limit, 30))
            render_list(items, "DAUM")
        except Exception as e:
            st.warning(f"다음 RSS 실패: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with colC:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**GOOGLE NEWS (RSS)**")
        try:
            items = fetch_rss(google_rss, limit=max(news_limit, 30))
            render_list(items, "GOOGLE")
        except Exception as e:
            st.warning(f"구글 RSS 실패: {e}")
        st.markdown('</div>', unsafe_allow_html=True)


# =========================
# Main Tabs
# =========================
def render_tab(freq: str):
    days = days_for_freq(freq)
    start_dt = now_local() - timedelta(days=days)
    start = start_dt.strftime("%Y-%m-%d")

    render_overview(freq, start)
    render_indices(freq, start)
    render_kr_top10(freq, start)
    render_macro(freq, start)
    render_rates(freq, start)
    render_news()

    st.caption("※ 일부 지표는 무료 소스 특성상 간헐적으로 지연/누락될 수 있어요. 그런 경우 ‘지금 새로고침’ 버튼을 눌러주세요.")


tabs = st.tabs(["일간", "주간", "월간"])
with tabs[0]:
    render_tab("D")
with tabs[1]:
    render_tab("W")
with tabs[2]:
    render_tab("M")
