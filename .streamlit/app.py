import re
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


# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="재테크 핵심지표 대시보드",
    page_icon="📊",
    layout="wide"
)

st.title("📊 재테크 핵심지표 대시보드")
st.caption("주요 지수 · 국내 시총 Top10 · 환율 · 금/유가 · 금리 · 실시간 경제뉴스(네이버/다음/구글) — 일간/주간/월간")


# -----------------------------
# Helpers
# -----------------------------
def _today_kst() -> datetime:
    # Streamlit Cloud timezone may vary; treat as local now for display
    return datetime.now()

def _days_for_freq(freq: str) -> int:
    # 기본 조회 범위(탭별로 보기 편하게)
    if freq == "D":
        return 180
    if freq == "W":
        return 365 * 3
    if freq == "M":
        return 365 * 10
    return 365

def _to_freq_label(freq: str) -> str:
    return {"D": "일간", "W": "주간", "M": "월간"}.get(freq, freq)

def _resample_close(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    df index must be DatetimeIndex, must have 'Close' column (or 1 col series)
    Returns DataFrame with Close resampled to period end.
    """
    if df is None or df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    # If 'Close' not present, assume single column
    if "Close" not in df.columns:
        # pick first numeric col
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return df
        df = df[[numeric_cols[0]]].rename(columns={numeric_cols[0]: "Close"})

    if freq == "D":
        return df[["Close"]].dropna()

    rule = "W-FRI" if freq == "W" else "M"
    out = df[["Close"]].resample(rule).last()
    return out.dropna()

def _metric_delta(series: pd.Series):
    series = series.dropna()
    if len(series) < 2:
        return None, None, None
    last = float(series.iloc[-1])
    prev = float(series.iloc[-2])
    delta = last - prev
    pct = (delta / prev) * 100 if prev != 0 else None
    return last, delta, pct

def _plot_line(df: pd.DataFrame, title: str, height: int = 260):
    if df is None or df.empty:
        st.info(f"{title}: 데이터가 없습니다.")
        return

    if "Close" not in df.columns:
        # multi columns
        fig = go.Figure()
        for c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=str(c)))
        fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

def _safe_requests_get(url: str, timeout: int = 10, headers: dict | None = None):
    headers = headers or {
        "User-Agent": "Mozilla/5.0 (compatible; MISHARP-Dashboard/1.0; +https://streamlit.io)"
    }
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # 퍼포먼스 비교(기준=100)
    out = df.copy()
    for c in out.columns:
        s = out[c].dropna()
        if len(s) == 0:
            continue
        out[c] = (out[c] / s.iloc[0]) * 100
    return out


# -----------------------------
# Data Fetchers (cached)
# -----------------------------
@st.cache_data(ttl=60 * 30)  # 30분
def fetch_fdr(symbol: str, start: str):
    df = fdr.DataReader(symbol, start)
    # FinanceDataReader: Close 컬럼 보장(대부분)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(ttl=60 * 30)  # 30분
def fetch_yf(symbol: str, start: str):
    df = yf.download(symbol, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(ttl=60 * 60)  # 1시간
def fetch_fred(series_list: list[str], start: datetime):
    # FRED는 키 없이도 pandas_datareader로 다수 시리즈 조회 가능
    df = DataReader(series_list, "fred", start)
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(ttl=60 * 60)  # 1시간
def get_krx_top10_by_mktcap():
    # 가장 가까운 영업일(주) 기준 날짜 문자열 yyyyMMdd
    biz = krx.get_nearest_business_day_in_a_week()
    caps = krx.get_market_cap_by_ticker(biz)
    # '시가총액' 기준 내림차순
    caps = caps.sort_values("시가총액", ascending=False).head(10)
    # 종목명 붙이기
    tickers = caps.index.tolist()
    names = [krx.get_market_ticker_name(t) for t in tickers]
    out = pd.DataFrame({
        "티커": tickers,
        "종목명": names,
        "시가총액": caps["시가총액"].values
    })
    return biz, out

@st.cache_data(ttl=60 * 10)  # 10분
def fetch_rss(feed_url: str, limit: int = 20):
    d = feedparser.parse(feed_url)
    items = []
    for e in d.entries[:limit]:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        items.append({"title": title, "link": link, "published": published, "source": feed_url})
    return items

@st.cache_data(ttl=60 * 5)  # 5분
def fetch_naver_finance_news(limit: int = 20):
    # 네이버 금융 뉴스(메인) 페이지를 간단 크롤링
    url = "https://finance.naver.com/news/"
    r = _safe_requests_get(url, timeout=10)
    soup = BeautifulSoup(r.text, "lxml")

    items = []
    # 카드/리스트 구조가 종종 바뀌므로, 가장 흔한 a태그 패턴을 폭넓게 수집
    for a in soup.select("a"):
        href = a.get("href", "")
        text = a.get_text(" ", strip=True)
        if not text or len(text) < 8:
            continue
        # 뉴스 링크 패턴(상대/절대)
        if "news_read.naver" in href or "read.naver" in href or "news.naver.com" in href:
            link = href
            if link.startswith("/"):
                link = "https://finance.naver.com" + link
            items.append({"title": text, "link": link, "published": "", "source": "NAVER_FINANCE"})
        elif href.startswith("/news/") and "finance.naver.com" in url:
            link = "https://finance.naver.com" + href
            items.append({"title": text, "link": link, "published": "", "source": "NAVER_FINANCE"})

    # 중복 제거 + 상위 limit
    seen = set()
    uniq = []
    for it in items:
        key = (it["title"], it["link"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
        if len(uniq) >= limit:
            break
    return uniq


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ 설정")

auto_refresh = st.sidebar.toggle("자동 새로고침(5분)", value=True)
news_limit = st.sidebar.slider("뉴스 표시 개수", 10, 50, 20, 5)
show_brent = st.sidebar.toggle("유가: 브렌트(BZ=F)도 표시", value=True)

if auto_refresh:
    # Streamlit 방식의 주기적 rerun
    st.sidebar.caption("자동 새로고침: 5분마다 갱신")
    time.sleep(0.01)
    st.cache_data.clear()
    st.rerun()


# -----------------------------
# Main Rendering
# -----------------------------
def render_tab(freq: str):
    days = _days_for_freq(freq)
    start_dt = _today_kst() - timedelta(days=days)
    start = start_dt.strftime("%Y-%m-%d")

    # 1) 주요 지수
    st.subheader("1) 주요 주가지수")
    c1, c2, c3 = st.columns(3)

    with c1:
        df_ks11 = fetch_fdr("KS11", start)
        df_ks11 = _resample_close(df_ks11, freq)
        last, delta, pct = _metric_delta(df_ks11["Close"]) if not df_ks11.empty else (None, None, None)
        st.metric("KOSPI (KS11)", f"{last:,.2f}" if last is not None else "-", f"{delta:,.2f} ({pct:+.2f}%)" if pct is not None else None)
        _plot_line(df_ks11, "KOSPI")

    with c2:
        df_kq11 = fetch_fdr("KQ11", start)
        df_kq11 = _resample_close(df_kq11, freq)
        last, delta, pct = _metric_delta(df_kq11["Close"]) if not df_kq11.empty else (None, None, None)
        st.metric("KOSDAQ (KQ11)", f"{last:,.2f}" if last is not None else "-", f"{delta:,.2f} ({pct:+.2f}%)" if pct is not None else None)
        _plot_line(df_kq11, "KOSDAQ")

    with c3:
        sp = _resample_close(fetch_yf("^GSPC", start), freq)
        nas = _resample_close(fetch_yf("^IXIC", start), freq)
        dow = _resample_close(fetch_yf("^DJI", start), freq)

        # 합쳐서 하나의 비교 그래프
        df_us = pd.DataFrame({
            "S&P500": sp["Close"] if not sp.empty else pd.Series(dtype=float),
            "NASDAQ": nas["Close"] if not nas.empty else pd.Series(dtype=float),
            "DOW": dow["Close"] if not dow.empty else pd.Series(dtype=float),
        }).dropna(how="all")

        st.caption("미국 지수 비교")
        _plot_line(_normalize_cols(df_us.dropna()), "US Indices (Normalized=100)", height=260)

    st.divider()

    # 2) 국내 시총 Top10
    st.subheader("2) 국내 시가총액 Top10 (자동 추출)")
    biz, top10 = get_krx_top10_by_mktcap()
    st.caption(f"기준 영업일: {biz}")

    # Top10 가격 데이터
    tickers = top10["티커"].tolist()
    names = top10["종목명"].tolist()

    prices = {}
    for t, n in zip(tickers, names):
        try:
            dft = fetch_fdr(t, start)
            dft = _resample_close(dft, freq)
            if not dft.empty:
                prices[n] = dft["Close"]
        except Exception:
            continue

    df_top = pd.DataFrame(prices).dropna(how="all")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(top10, use_container_width=True, hide_index=True)
    with c2:
        if not df_top.empty:
            _plot_line(_normalize_cols(df_top.dropna()), "KRX Top10 (Normalized=100)", height=300)
        else:
            st.info("Top10 가격 데이터를 가져오지 못했습니다.")

    st.divider()

    # 3) 환율
    st.subheader("3) 환율 (USD/KRW)")
    df_fx = fetch_fdr("USD/KRW", start)
    df_fx = _resample_close(df_fx.rename(columns={"Close": "Close"}), freq) if "Close" in df_fx.columns else _resample_close(df_fx, freq)
    if df_fx.empty and "Close" not in df_fx.columns:
        # FDR 환율은 보통 'Close' 대신 단일 컬럼일 수도 있어 처리
        df_fx = _resample_close(df_fx, freq)

    if not df_fx.empty:
        last, delta, pct = _metric_delta(df_fx["Close"])
        st.metric("원/달러", f"{last:,.2f}", f"{delta:,.2f} ({pct:+.2f}%)")
    _plot_line(df_fx, "USD/KRW")

    st.divider()

    # 4) 금 시세
    st.subheader("4) 금 시세 (Gold Futures: GC=F)")
    df_gold = _resample_close(fetch_yf("GC=F", start), freq)
    if not df_gold.empty:
        last, delta, pct = _metric_delta(df_gold["Close"])
        st.metric("Gold (GC=F)", f"{last:,.2f}", f"{delta:,.2f} ({pct:+.2f}%)")
    _plot_line(df_gold, "Gold (GC=F)")

    st.divider()

    # 5) 유가
    st.subheader("5) 유가 (WTI / Brent)")
    wti = _resample_close(fetch_yf("CL=F", start), freq)
    brent = _resample_close(fetch_yf("BZ=F", start), freq) if show_brent else pd.DataFrame()

    c1, c2 = st.columns(2)
    with c1:
        if not wti.empty:
            last, delta, pct = _metric_delta(wti["Close"])
            st.metric("WTI (CL=F)", f"{last:,.2f}", f"{delta:,.2f} ({pct:+.2f}%)")
        _plot_line(wti, "WTI (CL=F)")
    with c2:
        if show_brent:
            if not brent.empty:
                last, delta, pct = _metric_delta(brent["Close"])
                st.metric("Brent (BZ=F)", f"{last:,.2f}", f"{delta:,.2f} ({pct:+.2f}%)")
            _plot_line(brent, "Brent (BZ=F)")
        else:
            st.info("사이드바에서 Brent 표시를 켤 수 있습니다.")

    st.divider()

    # 6) 금리 동향
    st.subheader("6) 금리 동향 (미국/국내)")
    fred = fetch_fred(
        ["DGS10", "FEDFUNDS", "IRLTLT01KRM156N"],  # US 10Y, Fed Funds, KR 10Y(OECD via FRED)
        start_dt
    )
    if fred is None or fred.empty:
        st.info("금리 데이터를 가져오지 못했습니다.")
    else:
        # resample
        if freq == "W":
            fred_r = fred.resample("W-FRI").last()
        elif freq == "M":
            fred_r = fred.resample("M").last()
        else:
            fred_r = fred.copy()

        c1, c2, c3 = st.columns(3)
        with c1:
            s = fred_r["DGS10"].dropna()
            if len(s) >= 2:
                st.metric("미국 10Y (DGS10)", f"{s.iloc[-1]:.2f}%", f"{(s.iloc[-1]-s.iloc[-2]):+.2f}p")
            _plot_line(pd.DataFrame({"Close": fred_r["DGS10"]}).dropna(), "US 10Y (DGS10)")
        with c2:
            s = fred_r["FEDFUNDS"].dropna()
            if len(s) >= 2:
                st.metric("미국 기준금리(FedFunds)", f"{s.iloc[-1]:.2f}%", f"{(s.iloc[-1]-s.iloc[-2]):+.2f}p")
            _plot_line(pd.DataFrame({"Close": fred_r["FEDFUNDS"]}).dropna(), "Fed Funds Rate")
        with c3:
            s = fred_r["IRLTLT01KRM156N"].dropna()
            if len(s) >= 2:
                st.metric("한국 10Y (FRED/OECD)", f"{s.iloc[-1]:.2f}%", f"{(s.iloc[-1]-s.iloc[-2]):+.2f}p")
            _plot_line(pd.DataFrame({"Close": fred_r["IRLTLT01KRM156N"]}).dropna(), "Korea 10Y (IRLTLT01KRM156N)")

    st.divider()

    # 7) 실시간 경제뉴스
    st.subheader("7) 실시간 경제 뉴스 (네이버/다음/구글)")
    st.caption("네이버는 공식 RSS가 제한적이라 금융 뉴스 페이지 크롤링, 다음/구글은 RSS 기반으로 수집합니다.")

    daum_rss = "http://media.daum.net/rss/part/primary/economic/rss2.xml"
    google_rss = "http://news.google.co.kr/news?pz=1&hdlOnly=1&cf=all&ned=kr&hl=ko&topic=b&output=rss"

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**NAVER (Finance)**")
        try:
            items = fetch_naver_finance_news(limit=news_limit)
            for it in items:
                st.markdown(f"- [{it['title']}]({it['link']})")
        except Exception as e:
            st.warning(f"네이버 뉴스 수집 실패: {e}")

    with colB:
        st.markdown("**DAUM (RSS)**")
        try:
            items = fetch_rss(daum_rss, limit=news_limit)
            for it in items:
                title = it["title"]
                link = it["link"]
                pub = it["published"]
                pub_txt = f" · {pub}" if pub else ""
                st.markdown(f"- [{title}]({link}){pub_txt}")
        except Exception as e:
            st.warning(f"다음 RSS 수집 실패: {e}")

    with colC:
        st.markdown("**GOOGLE NEWS (RSS)**")
        try:
            items = fetch_rss(google_rss, limit=news_limit)
            for it in items:
                title = it["title"]
                link = it["link"]
                pub = it["published"]
                pub_txt = f" · {pub}" if pub else ""
                st.markdown(f"- [{title}]({link}){pub_txt}")
        except Exception as e:
            st.warning(f"구글 RSS 수집 실패: {e}")


tabs = st.tabs(["일간", "주간", "월간"])
with tabs[0]:
    render_tab("D")
with tabs[1]:
    render_tab("W")
with tabs[2]:
    render_tab("M")

st.caption("Tip) Streamlit Cloud 무료 플랜은 외부 요청이 간헐적으로 막힐 수 있어요. 그럴 땐 캐시 TTL을 늘리거나(30분→1시간) 뉴스만 수동 새로고침으로 운영하면 안정적입니다.")
