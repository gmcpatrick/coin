# 코인분석 페이지 (Coin Analysis Dashboard)

업비트/OKX 거래소 실시간 알림 및 차트 예측 분석 대시보드

## 주요기능
- 업비트: 거래대금 상위 30개 코인 실시간 신호 & 예측
- OKX: 24시간 거래대금 상위 30개 USDT마켓 코인 실시간 신호 & 예측
- 볼린저밴드/RSI 기반 신호
- 기간별 예측(3,7,15,30,90일)
- Streamlit 기반 간편 웹 대시보드

## 실행 방법
```bash
pip install -r requirements.txt
streamlit run coin_analysis_app.py
```

## 배포 방법 (Streamlit Cloud 권장)
1. 이 저장소를 깃허브에 push
2. https://streamlit.io/cloud 에서 "New app" → 이 저장소 지정 후 배포

---

> 본 프로젝트는 투자 참고용 예시 코드입니다.