import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go

# --- API 헬퍼 함수 ---

@st.cache_data(ttl=600)
def get_upbit_market_list():
    url = "https://api.upbit.com/v1/market/all"
    resp = requests.get(url)
    data = resp.json()
    # KRW-마켓만, 코인명만 추출 (KRW-XXX → XXX)
    return sorted([row["market"].split("-")[1] for row in data if row["market"].startswith("KRW-")])

@st.cache_data(ttl=600)
def get_okx_top30_usdt():
    """OKX SPOT USDT 마켓 중 24시간 거래대금(volCcyQuote) 기준 상위 30개 심볼만 반환"""
    url = "https://www.okx.com/api/v5/market/tickers?instType=SPOT"
    resp = requests.get(url)
    data = resp.json()["data"]
    coins = [
        {
            "symbol": row["instId"].replace("-USDT", ""),
            "vol": float(row.get("volCcyQuote", 0))
        }
        for row in data if row["instId"].endswith("-USDT")
    ]
    # 거래대금 기준 내림차순 정렬 후 30개 추출
    top30 = sorted(coins, key=lambda x: x["vol"], reverse=True)[:30]
    return [c["symbol"] for c in top30]

@st.cache_data(ttl=180)
def get_upbit_top30_by_volume():
    url = "https://api.upbit.com/v1/market/all"
    resp = requests.get(url)
    data = resp.json()
    krw_markets = [row["market"] for row in data if row["market"].startswith("KRW-")]
    tickers = requests.get("https://api.upbit.com/v1/ticker", params={"markets": ','.join(krw_markets)}).json()
    df = pd.DataFrame(tickers)
    df['acc_trade_price_24h'] = df['acc_trade_price_24h'].astype(float)
    df = df.sort_values("acc_trade_price_24h", ascending=False).head(30)
    df['symbol'] = df['market'].apply(lambda x: x.replace('KRW-', ''))
    return df['symbol'].tolist()

def get_upbit_candles(coin, interval="minutes/60", count=100):
    market = f"KRW-{coin}"
    url = f"https://api.upbit.com/v1/candles/{interval}?market={market}&count={count}"
    resp = requests.get(url)
    candles = resp.json()
    df = pd.DataFrame(candles)
    df["candle_date_time_kst"] = pd.to_datetime(df["candle_date_time_kst"])
    df = df.sort_values("candle_date_time_kst")
    df = df.rename(columns={"trade_price": "close", "candle_acc_trade_volume": "volume"})
    return df[["candle_date_time_kst", "close", "volume"]].reset_index(drop=True)

def get_okx_candles(coin, interval="1H", limit=100):
    instId = f"{coin}-USDT"
    url = f"https://www.okx.com/api/v5/market/candles?instId={instId}&bar={interval}&limit={limit}"
    resp = requests.get(url)
    candles = resp.json()["data"]
    if not candles:
        return pd.DataFrame(columns=["candle_date_time", "close", "volume"])
    df = pd.DataFrame(candles, columns=["ts","open","high","low","close","volume","volCcy","volCcyQuote","confirm"])
    df["candle_date_time"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df = df.sort_values("candle_date_time")
    return df[["candle_date_time", "close", "volume"]].reset_index(drop=True)

# --- 패턴 신호 분석 함수: 볼린저밴드 + RSI 신호 ---
def calc_signals(df):
    df["ma20"] = df["close"].rolling(20).mean()
    df["stddev"] = df["close"].rolling(20).std()
    df["upper"] = df["ma20"] + (df["stddev"] * 2)
    df["lower"] = df["ma20"] - (df["stddev"] * 2)
    # RSI
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    signals = []
    for i, row in df.iterrows():
        # 볼린저밴드 하단 돌파 + RSI<30: 매수신호
        if row["close"] < row["lower"] and row["rsi"] < 30:
            signals.append({"코인":row.get("코인"), "시그널":"매수(밴드하단+RSI<30)", "가격": row["close"], "일시": row.get("candle_date_time_kst", row.get("candle_date_time"))})
        # 볼린저밴드 상단 돌파 + RSI>70: 매도신호
        if row["close"] > row["upper"] and row["rsi"] > 70:
            signals.append({"코인":row.get("코인"), "시그널":"매도(밴드상단+RSI>70)", "가격": row["close"], "일시": row.get("candle_date_time_kst", row.get("candle_date_time"))})
    return signals

# --- 코인 필터 함수 ---
def coin_search_box(coins, label, key):
    search = st.text_input(f"{label} 검색", "", key=key+"_search")
    filtered = [c for c in coins if search.upper() in c.upper()]
    return st.selectbox(label, filtered if filtered else coins, key=key)

# --- 실시간 알림 대시보드 (코인 전체 신호 + 필터) ---
def alert_dashboard():
    st.title("실시간 알림 대시보드")

    exch = st.selectbox("거래소", ["Upbit", "OKX"])
    chart_periods = {"1시간봉": ("minutes/60", "1H"),
                     "15분봉": ("minutes/15", "15m"),
                     "4시간봉": ("minutes/240", "4H"),
                     "1일봉": ("days", "1D")}
    chart_period = st.selectbox("차트 주기", ["1시간봉", "15분봉", "4시간봉", "1일봉"], index=0)

    # 코인 리스트(필터된)
    if exch == "Upbit":
        coins = get_upbit_top30_by_volume()
        interval = chart_periods[chart_period][0]
        currency = "KRW"
    else:
        coins = get_okx_top30_usdt()
        interval = chart_periods[chart_period][1]
        currency = "USDT"

    # 전체 코인 신호 수집
    alerts = []
    progress = st.progress(0, text="코인별 신호 분석중...")
    for idx, coin in enumerate(coins):
        try:
            if exch == "Upbit":
                df = get_upbit_candles(coin, interval=interval)
            else:
                df = get_okx_candles(coin, interval=interval)
            df["코인"] = coin
            sigs = calc_signals(df)
            for s in sigs:
                s["코인"] = coin
                s["가격"] = f"{s['가격']:,.2f}" if currency=="USDT" else f"{int(s['가격']):,}"
                s["거래소"] = exch
                s["차트주기"] = chart_period
                alerts.append(s)
        except Exception:
            continue
        progress.progress((idx+1)/len(coins), text=f"{idx+1}/{len(coins)} 코인 완료")
    progress.empty()

    # 데이터프레임 변환
    if len(alerts) > 0:
        df_alert = pd.DataFrame(alerts)
    else:
        df_alert = pd.DataFrame(columns=["코인", "시그널", "가격", "일시", "거래소", "차트주기"])

    # 필터 UI
    col1, col2, col3 = st.columns(3)
    with col1:
        coin_sel = coin_search_box(["전체"] + coins, "코인", key="alert_coin")
    with col2:
        sig_types = ["전체"] + sorted(df_alert["시그널"].unique()) if not df_alert.empty else ["전체"]
        sig_sel = st.selectbox("시그널 종류", sig_types, key="alert_sig")
    with col3:
        date_sel = st.date_input("일시(날짜)", None, key="alert_date")

    # 필터 적용
    if coin_sel != "전체":
        df_alert = df_alert[df_alert["코인"] == coin_sel]
    if sig_sel != "전체":
        df_alert = df_alert[df_alert["시그널"] == sig_sel]
    if date_sel:
        date_str = pd.Timestamp(date_sel).strftime("%Y-%m-%d")
        df_alert = df_alert[df_alert["일시"].astype(str).str.startswith(date_str)]

    st.dataframe(df_alert.sort_values("일시", ascending=False), use_container_width=True)
    st.caption("업비트는 24시간 거래대금 상위 30개, OKX는 24시간 거래대금 상위 30개만. 코인/시그널/날짜별 필터 가능.")

# --- 예측 결과 및 차트 분석 ---
def prediction_screen():
    st.title("예측 결과 및 차트 분석")
    exch = st.selectbox("거래소", ["Upbit", "OKX"], key="pr_ex")
    chart_types = ["15분봉", "1시간봉", "4시간봉", "1일봉", "1주봉"]
    interval_map_upbit = {"15분봉":"minutes/15", "1시간봉":"minutes/60", "4시간봉":"minutes/240", "1일봉":"days", "1주봉":"weeks"}
    interval_map_okx = {"15분봉":"15m", "1시간봉":"1H", "4시간봉":"4H", "1일봉":"1D", "1주봉":"1W"}

    chart_type = st.selectbox("차트 주기", chart_types, key="pr_iv")
    if exch == "Upbit":
        coins = get_upbit_top30_by_volume()
        interval = interval_map_upbit[chart_type]
        currency = "KRW"
    else:
        coins = get_okx_top30_usdt()
        interval = interval_map_okx[chart_type]
        currency = "USDT"
    coin = coin_search_box(coins, "코인", key="pred_coin")

    # 캔들 데이터
    try:
        if exch == "Upbit":
            df = get_upbit_candles(coin, interval=interval)
        else:
            df = get_okx_candles(coin, interval=interval)
    except Exception:
        st.error("해당 조합에 데이터가 없습니다.")
        return

    # 분석 및 신호
    df = df.copy()
    signals = calc_signals(df)

    # 차트
    date_col = "candle_date_time_kst" if "candle_date_time_kst" in df else "candle_date_time"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df["close"], mode="lines", name="종가"))
    fig.add_trace(go.Scatter(x=df[date_col], y=df["ma20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df[date_col], y=df["upper"], mode="lines", name="Upper Band", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df[date_col], y=df["lower"], mode="lines", name="Lower Band", line=dict(dash="dot")))
    if signals:
        for s in signals:
            # 오류 수정: s["가격"]이 float 또는 str일 수 있으니, 항상 float 처리만 하도록 수정
            price_val = s["가격"]
            if isinstance(price_val, str):
                price_val = float(price_val.replace(',', ''))
            fig.add_trace(go.Scatter(
                x=[s["일시"]], y=[price_val], mode="markers+text",
                marker=dict(size=12, color="red" if "매도" in s["시그널"] else "blue"),
                text=[s["시그널"]], textposition="top center", name=s["시그널"]
            ))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("간단 예측(볼린저밴드 기반)")
    base = df["close"].iloc[-1]
    ma20 = df["ma20"].iloc[-1]
    pred = [
        {"기간":"3일", "예측값":f"{(base*0.8+ma20*0.2):,.2f}" if currency=="USDT" else f"{int(base*0.8+ma20*0.2):,}"},
        {"기간":"7일", "예측값":f"{(base*0.6+ma20*0.4):,.2f}" if currency=="USDT" else f"{int(base*0.6+ma20*0.4):,}"},
        {"기간":"15일", "예측값":f"{(base*0.4+ma20*0.6):,.2f}" if currency=="USDT" else f"{int(base*0.4+ma20*0.6):,}"},
        {"기간":"30일", "예측값":f"{(base*0.2+ma20*0.8):,.2f}" if currency=="USDT" else f"{int(base*0.2+ma20*0.8):,}"},
        {"기간":"90일", "예측값":f"{ma20:,.2f}" if currency=="USDT" else f"{int(ma20):,}"}
    ]
    st.table(pd.DataFrame(pred))
    st.caption("실제 예측은 ML/DL 기반 추가 가능. 이 코드는 볼린저밴드 기반 예시입니다.")

def main():
    st.set_page_config(page_title="코인분석 페이지", layout="wide")
    menu = st.sidebar.radio("메뉴", ["실시간 알림 대시보드", "예측 결과(차트 분석)"])

    if menu == "실시간 알림 대시보드":
        alert_dashboard()
    else:
        prediction_screen()

if __name__ == "__main__":
    main()