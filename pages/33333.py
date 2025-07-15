import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import io

# --- 1. 페이지 설정 및 기본 스타일 ---
st.set_page_config(layout="wide")
st.title('🌡️ 환기 방식, 에너지 소비 및 실내 환경 통합 분석')
st.markdown("""
이 웹 앱은 여러 데이터셋을 종합하여 환기 방식과 에너지 소비 패턴이 실내 공기질 및 거주자 건강에 미치는 영향을 분석합니다.
각 데이터셋의 개별 분석과 통합 분석 아이디어를 제공합니다.
""")

# --- 2. 데이터 로드 함수 ---
# @st.cache_data를 사용하여 데이터 로딩을 캐시합니다.
@st.cache_data
def load_data(file_path):
    """지정된 경로의 CSV 파일을 로드하고 데이터프레임으로 반환합니다."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"파일을 찾을 수 없습니다: {file_path}. GitHub 저장소에 파일이 올바르게 위치해 있는지 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

# --- 3. 데이터 파일 경로 설정 ---
# GitHub 저장소에 있는 파일 경로를 기준으로 합니다.
ENERGY_CONSUMPTION_FILE = 'Energy_consumption - Energy_consumption.csv'
GLOBAL_AIR_QUALITY_FILE = 'global_air_quality_data_10000 - global_air_quality_data_10000.csv'
HEALTH_IMPACT_FILE = 'air_quality_health_impact_data - air_quality_health_impact_data.csv'
INDOOR_AIR_QUALITY_FILE = 'AirQuality - AirQuality.csv'

# --- 4. 데이터 로드 ---
df_energy = load_data(ENERGY_CONSUMPTION_FILE)
df_global_aq = load_data(GLOBAL_AIR_QUALITY_FILE)
df_health = load_data(HEALTH_IMPACT_FILE)
df_indoor_aq = load_data(INDOOR_AIR_QUALITY_FILE)

# --- 5. 분석 모듈 함수 ---
def display_dataframe_info(df, name):
    """데이터프레임의 기본 정보를 Streamlit에 예쁘게 표시합니다."""
    st.subheader(f'"{name}" 데이터셋 개요')
    st.dataframe(df.head())
    
    st.write("**데이터 기본 정보:**")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write("**결측치 수:**")
    st.dataframe(df.isnull().sum().to_frame('결측치 수'))

    st.write("**기술 통계:**")
    st.dataframe(df.describe())


    

def global_air_quality_analysis(df):
    """글로벌 대기질 데이터 분석 및 시각화 (참고용)"""
    st.header('🌍 글로벌 대기질 데이터 분석 (참고용)')
    if df is None:
        st.warning("글로벌 대기질 데이터가 로드되지 않았습니다.")
        return

    if 'Country' in df.columns and 'PM2.5' in df.columns:
        st.subheader('국가별 PM2.5 평균 농도 (상위 15개국)')
        df['PM2.5'] = pd.to_numeric(df['PM2.5'], errors='coerce')
        avg_pm25_country = df.dropna(subset=['PM2.5']).groupby('Country')['PM2.5'].mean().nlargest(15).reset_index()
        fig = px.bar(avg_pm25_country, x='Country', y='PM2.5', title='Top 15 Countries by Average PM2.5 Concentration')
        st.plotly_chart(fig, use_container_width=True)

# --- 6. 사이드바 및 메인 화면 구성 ---
st.sidebar.header('분석 메뉴')
analysis_option = st.sidebar.radio(
    "보고 싶은 분석을 선택하세요:",
    ('데이터 개요', '실내 공기질 분석', '공기질-건강 영향 분석', '에너지 소비 분석', '통합 분석 아이디어')
)

if analysis_option == '데이터 개요':
    st.header("🔍 데이터 개요")
    st.info("각 데이터셋의 기본적인 정보를 확인합니다.")
    if df_indoor_aq is not None:
        display_dataframe_info(df_indoor_aq, "실내 공기질")
    if df_health is not None:
        display_dataframe_info(df_health, "공기질 건강 영향")
   
    if df_global_aq is not None:
        display_dataframe_info(df_global_aq, "글로벌 대기질")

elif analysis_option == '실내 공기질 분석':
    indoor_aq_analysis(df_indoor_aq)

elif analysis_option == '공기질-건강 영향 분석':
    health_impact_analysis(df_health)


    
    # 참고용 글로벌 데이터 분석도 함께 표시
    global_air_quality_analysis(df_global_aq)


st.sidebar.markdown('---')
st.sidebar.info('이 웹 앱은 Streamlit으로 구축되었으며, 실내 환경과 건강에 대한 데이터 기반 인사이트를 제공합니다.')
