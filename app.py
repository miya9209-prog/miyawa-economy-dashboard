# app.py (v2.2) - 통째로 교체

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
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
  font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2.2rem;
  max-width: 1240px;
}

h1 {
  font-weight: 800 !important;
  font-size: 1.7rem !important;
  margin-bottom: 0.2rem !important;
  letter-spacing: -0.02em;
}
.small-muted {
  color: rgba(0,0,0,0.55);
  font-size: 0.92rem;
}

.section-title{
  font-weight: 800;
  font-size: 1.15rem;
  margin: 0.2rem 0 0.9rem 0;
  letter-spacing: -0.02em;
}
.section-sub{
  color: rgba(0,0,0,0.55);
  font-size: 0.92rem;
  margin-top: -0.55rem;
  margin-bottom: 1.0rem;
}

.card {
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 14px 14px 12px 14px;
  background: #ffffff;
  box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}
.card-tight{ padding: 12px 12px 10px 12px; }
.card-title{
  font-weight: 650;
  font-size: 0.92rem;
  color: rgba(0,0,0,0.62);
  margin-bottom: 6px;
}
.kpi{
  font-size: 1.22rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.delta{
  font-size: 0.90rem;
  font-weight: 650;
  margin-top: 2px;
}
.delta-pos{ color: #0a7b34; }
.delta-neg{ color: #b42318; }
.delta-flat{ color: rgba(0,0,0,0.55); }

.hr-soft{
  border-top: 1px solid rgba(0,0,0,0.06);
  margin: 0.9rem 0 1.1rem 0;
}

.stTabs [data-baseweb="tab-list"]{ gap: 8px; }
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

a { text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def now_local() -> datetime:
    return datetime.now()

def days_for_freq(freq: str) -> int:
    return {"D": 180, "W": 365 * 3, "M": 365 * 10}.get(freq, 365)

def ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out.sort_index()

def to_close_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = ensure_dt_index(df)
    if "Close" in df.columns:
        return df[["Close"]].dropna()
    # single numeric col fallback
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
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinanceDashboard/2.2; +https://streamlit.io)"}
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r


# =========================
# Data Fetchers (cached)
# =========================
@st.cache_data(ttl=60 * 30)
def fetch_fdr(symbol: str, start: str) -> pd.DataFrame:
    return ensure_dt_index(fdr.DataReader(symbol, start))

@st.cache_data(ttl=60 * 30)
def fetch_yf(symbol: str, start: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, progress=False, auto_adjust=False)
    return ensure_dt_index(df)

@st.cache_data(ttl=60 * 10)
def fetch_rss(feed_url: str, limit: int = 25):
    d = feedparser.parse(feed_url)
    items = []
    for e in d.entries[:limit]:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        items.append({"title": title, "link": link, "published": published})
    return items

@st.cache_data(ttl=60 * 5)
def fetch_naver_finance_news(limit: int = 25):
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

def get_series(source: str, symbol: str, start: str) -> pd.DataFrame:
    try:
        if source == "FDR":
            return to_close_df(fetch_fdr(symbol, start))
        return to_close_df(fetch_yf(symbol, start))
    except Exception:
        return pd.DataFrame()


# =========================
# Sidebar
# =========================
st.sidebar.markdown("### ⚙️ 설정")

mobile_mode = st.sidebar.toggle("모바일 보기 최적화", value=True)
refresh_on = st.sidebar.toggle("자동 새로고침", value=False)
refresh_min = st.sidebar.select_slider("갱신 주기(분)", options=[2, 3, 5, 10, 15], value=5)
news_limit = st.sidebar.slider("뉴스 표시 개수", 10, 60, 25, 5)

st.sidebar.markdown("---")
keyword = st.sidebar.text_input("뉴스 키워드 필터(선택)", value="").strip()

st.sidebar.markdown("---")
if st.sidebar.button("지금 새로고침"):
    st.cache_data.clear()
    st.rerun()

if refresh_on:
    ms = int(refresh_min * 60 * 1000)
    st.components.v1.html(f"<script>setTimeout(()=>window.location.reload(), {ms});</script>", height=0)

st.sidebar.markdown("---")
st.sidebar.caption("✍️ 10대 기업/ETF 목록은 아래에서 수정 가능")

# (기본값) 10대 기업 — 필요하면 형준님이 여기서 교체
DEFAULT_TOP10_COMP = [
    ("삼성전자", "005930"),
    ("SK하이닉스", "000660"),
    ("LG에너지솔루션", "373220"),
    ("삼성바이오로직스", "207940"),
    ("현대차", "005380"),
    ("삼성전자우", "005935"),
    ("기아", "000270"),
    ("셀트리온", "068270"),
    ("NAVER", "035420"),
    ("KB금융", "105560"),
]

# (기본값) 대표 ETF 10 — KODEX 중심(원하면 TIGER 등으로 교체)
DEFAULT_ETF10 = [
    ("KODEX 200", "069500"),
    ("KODEX 코스닥150", "229200"),
    ("KODEX 레버리지", "122630"),
    ("KODEX 인버스", "114800"),
    ("KODEX 200선물인버스2X", "252670"),
    ("KODEX 2차전지산업", "305720"),
    ("KODEX 반도체", "091160"),
    ("KODEX 은행", "091170"),
    ("KODEX 자동차", "091180"),
    ("KODEX 미국S&P500TR", "379800"),
]

top10_text = st.sidebar.text_area(
    "10대 기업 (형식: 이름,티커 / 한 줄에 하나)",
    value="\n".join([f"{n},{t}" for n, t in DEFAULT_TOP10_COMP]),
    height=180
)
etf10_text = st.sidebar.text_area(
    "대표 ETF 10 (형식: 이름,티커 / 한 줄에 하나)",
    value="\n".join([f"{n},{t}" for n, t in DEFAULT_ETF10]),
    height=180
)

def parse_list(text: str):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        name, tick = line.split(",", 1)
        name = name.strip()
        tick = tick.strip()
        if name and tick:
            out.append((name, tick))
    return out

TOP10_COMP = parse_list(top10_text)
ETF10 = parse_list(etf10_text)

# Layout params
KPI_COLS = 2 if mobile_mode else 4
CHART_H = 240 if mobile_mode else 300
NEWS_COLS = 1 if mobile_mode else 3


# =========================
# Header
# =========================
st.title("재테크 핵심지표 대시보드")
st.markdown(
    '<div class="small-muted">국내/미국 지수 · 국내 10대 기업 · 대표 ETF 10 · 환율/금/유가 · 실시간 경제뉴스</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)


# =========================
# Render Blocks
# =========================
def render_overview(freq: str, start: str):
    section("요약 스냅샷", "핵심 숫자만 빠르게 훑고, 변화 큰 곳부터 확인하세요.")
    cols = st.columns(KPI_COLS)

    # KPI: KOSPI / NASDAQ / USDKRW / GOLD / WTI / KOSDAQ
    kpi_defs = [
        ("KOSPI", "FDR", "KS11", "", 2),
        ("KOSDAQ", "FDR", "KQ11", "", 2),
        ("NASDAQ", "YF", "^IXIC", "", 2),
        ("USD/KRW", "FDR", "USD/KRW", "", 2),
        ("Gold", "YF", "GC=F", "", 2),
        ("WTI", "YF", "CL=F", "", 2),
    ]

    for i, (title, src, sym, suffix, prec) in enumerate(kpi_defs):
        with cols[i % KPI_COLS]:
            df = resample_close(get_series(src, sym, start), freq)
            last, delta, pct = metric_from_close(df)
            card_kpi(title, last, delta, pct, suffix=suffix, precision=prec)

    st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)


def render_indices(freq: str, start: str):
    section("주요 주가지수", "국내(코스피/코스닥) + 미국(S&P500/나스닥/다우) 흐름 비교")

    left, right = st.columns([1.05, 1.0]) if not mobile_mode else st.columns(1)

    # 국내 비교
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        df_k = {}
        for name, sym in [("KOSPI", "KS11"), ("KOSDAQ", "KQ11")]:
            d = resample_close(get_series("FDR", sym, start), freq)
            if not d.empty:
                df_k[name] = d["Close"]
        df_k = pd.DataFrame(df_k).dropna(how="all")
        if not df_k.empty:
            plot_line(df_k, "KOSPI vs KOSDAQ (Normalized=100)", height=CHART_H, normalized=True)
        else:
            st.info("국내 지수 데이터를 가져오지 못했습니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 미국 비교
    if not mobile_mode:
        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            df_u = {}
            for name, sym in [("S&P500", "^GSPC"), ("NASDAQ", "^IXIC"), ("DOW", "^DJI")]:
                d = resample_close(get_series("YF", sym, start), freq)
                if not d.empty:
                    df_u[name] = d["Close"]
            df_u = pd.DataFrame(df_u).dropna(how="all")
            if not df_u.empty:
                plot_line(df_u, "US Indices (Normalized=100)", height=CHART_H, normalized=True)
            else:
                st.info("미국 지수 데이터를 가져오지 못했습니다.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        df_u = {}
        for name, sym in [("S&P500", "^GSPC"), ("NASDAQ", "^IXIC"), ("DOW", "^DJI")]:
            d = resample_close(get_series("YF", sym, start), freq)
            if not d.empty:
                df_u[name] = d["Close"]
        df_u = pd.DataFrame(df_u).dropna(how="all")
        if not df_u.empty:
            plot_line(df_u, "US Indices (Normalized=100)", height=CHART_H, normalized=True)
        else:
            st.info("미국 지수 데이터를 가져오지 못했습니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)


def render_top10_companies(freq: str, start: str):
    section("국내 10대 기업", "주가 흐름을 ‘지수처럼’ 한 번에 비교 (정규화 100)")

    # 표 + 그래프 (모바일이면 세로)
    if mobile_mode:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(TOP10_COMP, columns=["기업", "티커"]), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        prices = {}
        for name, ticker in TOP10_COMP:
            d = resample_close(get_series("FDR", ticker, start), freq)
            if not d.empty:
                prices[name] = d["Close"]
        df = pd.DataFrame(prices).dropna(how="all")
        if not df.empty:
            plot_line(df, "KR Top10 Companies (Normalized=100)", height=CHART_H + 40, normalized=True)
        else:
            st.info("기업 주가 데이터를 가져오지 못했습니다. (티커/데이터소스 확인)")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        c1, c2 = st.columns([0.9, 1.6])
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(TOP10_COMP, columns=["기업", "티커"]), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            prices = {}
            for name, ticker in TOP10_COMP:
                d = resample_close(get_series("FDR", ticker, start), freq)
                if not d.empty:
                    prices[name] = d["Close"]
            df = pd.DataFrame(prices).dropna(how="all")
            if not df.empty:
                plot_line(df, "KR Top10 Companies (Normalized=100)", height=CHART_H + 40, normalized=True)
            else:
                st.info("기업 주가 데이터를 가져오지 못했습니다. (티커/데이터소스 확인)")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)


def render_etf10(freq: str, start: str):
    section("한국 대표 ETF 10", "KODEX 200 포함 — 흐름을 ‘지수처럼’ 한 번에 비교 (정규화 100)")

    if mobile_mode:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(ETF10, columns=["ETF", "티커"]), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        prices = {}
        for name, ticker in ETF10:
            d = resample_close(get_series("FDR", ticker, start), freq)
            if not d.empty:
                prices[name] = d["Close"]
        df = pd.DataFrame(prices).dropna(how="all")
        if not df.empty:
            plot_line(df, "KR 대표 ETF 10 (Normalized=100)", height=CHART_H + 40, normalized=True)
        else:
            st.info("ETF 데이터를 가져오지 못했습니다. (티커/데이터소스 확인)")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns([0.9, 1.6])
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(ETF10, columns=["ETF", "티커"]), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            prices = {}
            for name, ticker in ETF10:
                d = resample_close(get_series("FDR", ticker, start), freq)
                if not d.empty:
                    prices[name] = d["Close"]
            df = pd.DataFrame(prices).dropna(how="all")
            if not df.empty:
                plot_line(df, "KR 대표 ETF 10 (Normalized=100)", height=CHART_H + 40, normalized=True)
            else:
                st.info("ETF 데이터를 가져오지 못했습니다. (티커/데이터소스 확인)")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)


def render_fx_metals_oil(freq: str, start: str):
    section("환율 · 금 · 유가", "기본이면서 체감 큰 3가지")

    cols = st.columns(1) if mobile_mode else st.columns(3)

    # USD/KRW
    with cols[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        df = resample_close(get_series("FDR", "USD/KRW", start), freq)
        last, delta, pct = metric_from_close(df)
        card_kpi("USD/KRW", last, delta, pct, precision=2)
        plot_line(df, "USD/KRW", height=CHART_H, normalized=False)
        st.markdown('</div>', unsafe_allow_html=True)

    # Gold
    if not mobile_mode:
        with cols[1]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            df = resample_close(get_series("YF", "GC=F", start), freq)
            last, delta, pct = metric_from_close(df)
            card_kpi("Gold (GC=F)", last, delta, pct, precision=2)
            plot_line(df, "Gold", height=CHART_H, normalized=False)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        df = resample_close(get_series("YF", "GC=F", start), freq)
        last, delta, pct = metric_from_close(df)
        card_kpi("Gold (GC=F)", last, delta, pct, precision=2)
        plot_line(df, "Gold", height=CHART_H, normalized=False)
        st.markdown('</div>', unsafe_allow_html=True)

    # WTI
    if not mobile_mode:
        with cols[2]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            df = resample_close(get_series("YF", "CL=F", start), freq)
            last, delta, pct = metric_from_close(df)
            card_kpi("WTI (CL=F)", last, delta, pct, precision=2)
            plot_line(df, "WTI", height=CHART_H, normalized=False)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        df = resample_close(get_series("YF", "CL=F", start), freq)
        last, delta, pct = metric_from_close(df)
        card_kpi("WTI (CL=F)", last, delta, pct, precision=2)
        plot_line(df, "WTI", height=CHART_H, normalized=False)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)


def render_news():
    section("실시간 경제 뉴스", "네이버/다음/구글 — 키워드로 걸러서 빠르게 보기")

    daum_rss = "http://media.daum.net/rss/part/primary/economic/rss2.xml"
    google_rss = "http://news.google.co.kr/news?pz=1&hdlOnly=1&cf=all&ned=kr&hl=ko&topic=b&output=rss"

    def render_list(items):
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

    if NEWS_COLS == 1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**NAVER (Finance)**")
        try:
            render_list(fetch_naver_finance_news(limit=max(news_limit, 35)))
        except Exception as e:
            st.warning(f"네이버 수집 실패: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**DAUM (RSS)**")
        try:
            render_list(fetch_rss(daum_rss, limit=max(news_limit, 35)))
        except Exception as e:
            st.warning(f"다음 RSS 실패: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**GOOGLE NEWS (RSS)**")
        try:
            render_list(fetch_rss(google_rss, limit=max(news_limit, 35)))
        except Exception as e:
            st.warning(f"구글 RSS 실패: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**NAVER (Finance)**")
            try:
                render_list(fetch_naver_finance_news(limit=max(news_limit, 35)))
            except Exception as e:
                st.warning(f"네이버 수집 실패: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**DAUM (RSS)**")
            try:
                render_list(fetch_rss(daum_rss, limit=max(news_limit, 35)))
            except Exception as e:
                st.warning(f"다음 RSS 실패: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with colC:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**GOOGLE NEWS (RSS)**")
            try:
                render_list(fetch_rss(google_rss, limit=max(news_limit, 35)))
            except Exception as e:
                st.warning(f"구글 RSS 실패: {e}")
            st.markdown('</div>', unsafe_allow_html=True)


# =========================
# Main Tabs
# =========================
def render_tab(freq: str):
    start_dt = now_local() - timedelta(days=days_for_freq(freq))
    start = start_dt.strftime("%Y-%m-%d")

    render_overview(freq, start)
    render_indices(freq, start)
    render_top10_companies(freq, start)
    render_etf10(freq, start)
    render_fx_metals_oil(freq, start)
    render_news()

    st.caption("※ 무료 데이터 소스 특성상 간헐적 누락이 있을 수 있어요. 그럴 땐 ‘지금 새로고침’을 눌러주세요.")


tabs = st.tabs(["일간", "주간", "월간"])
with tabs[0]:
    render_tab("D")
with tabs[1]:
    render_tab("W")
with tabs[2]:
    render_tab("M")
