import streamlit as st
import pandas as pd
import plotly.express as px

# 페이지 기본 설정
st.set_page_config(page_title="실내 공기질과 건강 상관관계 분석", layout="wide")
st.title("🌿 실내 공기질, 에너지 소비, 건강지표 상관관계 분석")

# 파일 업로드 또는 불러오기
air_quality = pd.read_csv("/mnt/data/AirQuality - AirQuality.csv")
energy = pd.read_csv("/mnt/data/Energy_consumption - Energy_consumption.csv")
global_air = pd.read_csv("/mnt/data/global_air_quality_data_10000 - global_air_quality_data_10000.csv")
health = pd.read_csv("/mnt/data/air_quality_health_impact_data - air_quality_health_impact_data.csv")

# 데이터 전처리 요약
air_quality = air_quality.dropna()
energy = energy.dropna()
global_air = global_air.dropna()
health = health.dropna()

# 사이드바에서 선택지 제공
view = st.sidebar.selectbox("🔍 보고 싶은 분석 항목을 선택하세요", [
    "실내 공기질 추이 분석",
    "에너지 소비 패턴",
    "건강 지표와의 상관관계",
    "전세계 공기질 비교"
])

# 1. 실내 공기질 추이 분석
if view == "실내 공기질 추이 분석":
    st.header("📈 실내 공기질 주요 지표 추이")
    selected_column = st.selectbox("분석할 항목을 선택하세요", air_quality.columns[1:])
    fig = px.line(air_quality, x=air_quality.columns[0], y=selected_column,
                  title=f"{selected_column} 변화 추이")
    st.plotly_chart(fig, use_container_width=True)

# 2. 에너지 소비 패턴
elif view == "에너지 소비 패턴":
    st.header("⚡ 에너지 소비 패턴")
    selected_type = st.selectbox("분석할 에너지 유형을 선택하세요", energy.columns[1:])
    fig = px.bar(energy, x=energy.columns[0], y=selected_type,
                 title=f"{selected_type} 소비량 추이")
    st.plotly_chart(fig, use_container_width=True)

# 3. 건강 지표와의 상관관계
elif view == "건강 지표와의 상관관계":
    st.header("💊 공기질과 건강 지표 간의 상관관계 분석")
    corr_df = pd.merge(air_quality, health, left_on=air_quality.columns[0], right_on=health.columns[0])
    selected_col = st.selectbox("분석할 공기질 항목을 선택하세요", air_quality.columns[1:])
    selected_health = st.selectbox("분석할 건강지표를 선택하세요", health.columns[1:])
    fig = px.scatter(corr_df, x=selected_col, y=selected_health,
                     trendline="ols", title=f"{selected_col} vs {selected_health}")
    st.plotly_chart(fig, use_container_width=True)

# 4. 전세계 공기질 비교
elif view == "전세계 공기질 비교":
    st.header("🌍 전세계 주요 도시의 공기질 비교")
    country = st.selectbox("국가를 선택하세요", global_air['Country'].unique())
    filtered = global_air[global_air['Country'] == country]
    fig = px.scatter_geo(filtered, lat='Latitude', lon='Longitude', color='AQI Value',
                         hover_name='City', size='AQI Value',
                         title=f"{country} 주요 도시의 AQI 분포", projection="natural earth")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("✅ 데이터 출처: 실내 공기질 측정자료, 에너지 소비 통계, WHO 건강지표 데이터")
