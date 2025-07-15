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

def indoor_aq_analysis(df):
    """실내 공기질 데이터 분석 및 시각화"""
    st.header('🏡 실내 공기질 데이터 분석')
    if df is None:
        st.warning("실내 공기질 데이터가 로드되지 않았습니다.")
        return

    try:
        # 날짜와 시간을 합쳐 DateTime 컬럼 생성 (오류 발생 시 NaT으로 처리)
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                        format='%d/%m/%Y %H.%M.%S', errors='coerce')
        df.set_index('DateTime', inplace=True)
        
        # -200 값을 NaN으로 대체 후 선형 보간
        df.replace(-200, np.nan, inplace=True)
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].apply(lambda col: col.interpolate(method='linear'))

        # 유효하지 않은 DateTime 행 제거
        df.dropna(subset=df.index.name, inplace=True)

    except KeyError:
        st.warning("실내 공기질 데이터에 'Date' 또는 'Time' 컬럼이 없어 시계열 분석을 수행할 수 없습니다.")
        return
    except Exception as e:
        st.warning(f"데이터 전처리 중 오류 발생: {e}")
        return

    if df.empty or df.select_dtypes(include=np.number).empty:
        st.warning("처리 후 유효한 실내 공기질 데이터가 없습니다.")
        return

    air_quality_metrics = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not air_quality_metrics:
        st.info("분석 가능한 숫자형 공기질 지표가 없습니다.")
        return

    selected_aq_metric = st.sidebar.selectbox('시각화할 실내 공기질 지표', air_quality_metrics, key='indoor_aq_select')

    st.subheader(f'"{selected_aq_metric}" 시계열 변화 (시간별 평균)')
    df_resampled = df[selected_aq_metric].resample('H').mean().reset_index()
    fig = px.line(df_resampled, x='DateTime', y=selected_aq_metric, title=f'{selected_aq_metric} 시계열 변화')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f'"{selected_aq_metric}" 분포')
    fig2, ax2 = plt.subplots()
    sns.histplot(df[selected_aq_metric].dropna(), kde=True, ax=ax2)
    ax2.set_title(f'{selected_aq_metric} 분포')
    st.pyplot(fig2)

def health_impact_analysis(df):
    """공기질 건강 영향 데이터 분석 및 시각화"""
    st.header('😷 공기질 및 건강 영향 데이터 분석')
    if df is None:
        st.warning("공기질 건강 영향 데이터가 로드되지 않았습니다.")
        return

    # 대기질/환경 지표 분석
    st.subheader("대기질 및 환경 요인 분석")
    aq_env_cols = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed']
    aq_env_metrics_available = [col for col in aq_env_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if aq_env_metrics_available:
        selected_aq_env_metric = st.sidebar.selectbox('시각화할 대기질/환경 지표', aq_env_metrics_available, key='health_aq_env_select')
        fig_dist, ax_dist = plt.subplots()
        sns.histplot(df[selected_aq_env_metric].dropna(), kde=True, ax=ax_dist)
        ax_dist.set_title(f'{selected_aq_env_metric} 분포')
        st.pyplot(fig_dist)

        if len(aq_env_metrics_available) > 1:
            st.subheader('대기질 및 환경 지표 상관관계')
            corr_matrix = df[aq_env_metrics_available].corr()
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
    else:
        st.info("분석 가능한 대기질/환경 지표가 없습니다.")

    # 건강 영향 지표 분석
    st.subheader("건강 영향 지표 분석")
    health_cols = ['RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions']
    health_metrics_available = [col for col in health_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if health_metrics_available and 'AQI' in df.columns:
        selected_health_metric = st.sidebar.selectbox('시각화할 건강 영향 지표', health_metrics_available, key='health_outcome_select')
        st.subheader(f'AQI와 {selected_health_metric}의 관계')
        fig_scatter = px.scatter(df, x='AQI', y=selected_health_metric, trendline="ols",
                                 title=f'AQI vs. {selected_health_metric}')
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("분석 가능한 건강 영향 지표 또는 'AQI' 컬럼이 없습니다.")

    # 건강 영향 등급 분석
    if 'HealthImpactClass' in df.columns:
        st.subheader('건강 영향 등급 분포')
        class_counts = df['HealthImpactClass'].value_counts().reset_index()
        fig_bar = px.bar(class_counts, x='HealthImpactClass', y='count', title='건강 영향 등급 분포')
        st.plotly_chart(fig_bar, use_container_width=True)

def energy_consumption_analysis(df):
    """에너지 소비 데이터 분석 및 시각화"""
    st.header('⚡ 에너지 소비 데이터 분석')
    if df is None:
        st.warning("에너지 소비 데이터가 로드되지 않았습니다.")
        return

    # 데이터 타입 변환
    df['Energy_Consumption_kWh'] = pd.to_numeric(df['Energy_Consumption_kWh'], errors='coerce')
    df.dropna(subset=['Energy_Consumption_kWh'], inplace=True)

    if 'Building_Type' in df.columns:
        st.subheader('건물 유형별 평균 에너지 소비량')
        avg_energy_building = df.groupby('Building_Type')['Energy_Consumption_kWh'].mean().reset_index()
        fig = px.bar(avg_energy_building, x='Building_Type', y='Energy_Consumption_kWh', title='건물 유형별 평균 에너지 소비량')
        st.plotly_chart(fig, use_container_width=True)

    if 'Ventilation_System' in df.columns:
        st.subheader('환기 시스템 유형별 평균 에너지 소비량')
        avg_energy_ventilation = df.groupby('Ventilation_System')['Energy_Consumption_kWh'].mean().reset_index()
        fig2 = px.bar(avg_energy_ventilation, x='Ventilation_System', y='Energy_Consumption_kWh', title='환기 시스템 유형별 평균 에너지 소비량')
        st.plotly_chart(fig2, use_container_width=True)
    
    energy_corr_cols = ['Energy_Consumption_kWh', 'Temperature_C', 'Humidity_%', 'Occupancy_Count']
    energy_corr_available = [col for col in energy_corr_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if len(energy_corr_available) > 1:
        st.subheader('에너지 소비 관련 지표 상관관계')
        corr_matrix = df[energy_corr_available].corr()
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

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
    if df_energy is not None:
        display_dataframe_info(df_energy, "에너지 소비")
    if df_global_aq is not None:
        display_dataframe_info(df_global_aq, "글로벌 대기질")

elif analysis_option == '실내 공기질 분석':
    indoor_aq_analysis(df_indoor_aq)

elif analysis_option == '공기질-건강 영향 분석':
    health_impact_analysis(df_health)

elif analysis_option == '에너지 소비 분석':
    energy_consumption_analysis(df_energy)
    
elif analysis_option == '통합 분석 아이디어':
    st.header('💡 통합 분석 (가설 기반)')
    st.markdown("""
    **참고:** 업로드된 데이터셋 간에 직접적인 연결고리(예: 공통 ID, 정확한 시간 매칭)가 없어,
    아래 통합 분석 섹션은 일반적인 분석 아이디어를 제공합니다. 실제 실행을 위해서는 데이터 통합 전처리가 더 필요합니다.
    """)
    st.subheader('아이디어 1: 에너지 소비-환기-실내 공기질 상관관계')
    st.markdown("""
    - **가정:** `에너지 소비` 데이터와 `실내 공기질` 데이터가 건물 ID나 시간으로 연결될 수 있다면...
    - **분석:** 특정 환기 시스템(자연 vs. 기계) 사용 시 에너지 소비량과 실내 CO, NOx 등 공기질 지표의 관계를 분석할 수 있습니다.
    - **기대효과:** 에너지 효율적인 최적의 환기 전략을 도출할 수 있습니다. (예: 실내 CO2 농도가 낮을 때는 환기 강도를 줄여 에너지 절약)
    """)

    st.subheader('아이디어 2: 실내/외 공기질과 건강 영향의 관계')
    st.markdown("""
    - **가정:** `실내 공기질`과 `건강 영향` 데이터가 지역 ID나 시간으로 연결될 수 있다면...
    - **분석:** 실내 오염물질(C6H6 등) 농도와 호흡기/심혈관 질환 발병률 간의 시계열적 상관관계를 분석할 수 있습니다.
    - **기대효과:** 특정 실내 오염물질이 건강에 미치는 영향을 정량적으로 파악하고, 위험 수준을 예측하여 사전 예방 조치를 취할 수 있습니다.
    """)
    
    # 참고용 글로벌 데이터 분석도 함께 표시
    global_air_quality_analysis(df_global_aq)


st.sidebar.markdown('---')
st.sidebar.info('이 웹 앱은 Streamlit으로 구축되었으며, 실내 환경과 건강에 대한 데이터 기반 인사이트를 제공합니다.')
