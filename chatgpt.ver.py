# 파일: app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# 데이터 불러오기
df_india = pd.read_excel("archive/Data_File.xlsx", sheet_name="Sheet1")
df_seoul = pd.read_csv("archive (1)/AirPollutionSeoul/Measurement_summary.csv")

# 페이지 설정
st.set_page_config(page_title="실내 공기질 분석", layout="wide")

st.title("🌀 환기 방식, 에너지 소비, 실내 공기질 영향 분석")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📊 데이터 요약", "🌍 인도 지역별 분석", "🏙 서울 대기질 분석"])

with tab1:
    st.header("📁 데이터셋 개요")
    st.subheader("인도 공기질 및 건강지표 데이터")
    st.dataframe(df_india.head())
    st.subheader("서울 대기질 측정 요약")
    st.dataframe(df_seoul.head())

with tab2:
    st.header("🌏 인도 지역 PM2.5와 건강 지표 상관관계")
    fig = px.scatter(df_india[df_india["year_2020"] == 1], 
                     x="PM_2.5", 
                     y="Life_Expectancy", 
                     color="State / UT",
                     title="PM2.5 vs 생명 기대 수명 (2020 기준)")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🏙 서울 시간대별 대기오염 변화")
    df_seoul["Measurement date"] = pd.to_datetime(df_seoul["Measurement date"])
    daily_avg = df_seoul.resample("D", on="Measurement date")[["PM2.5", "PM10", "NO2", "SO2"]].mean().dropna()

    pollutant = st.selectbox("분석할 대기오염 지표 선택", daily_avg.columns)
    fig2 = px.line(daily_avg, y=pollutant, title=f"{pollutant} 일별 평균 추이")
    st.plotly_chart(fig2, use_container_width=True)
