import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# --- 1. 데이터 로드 함수 정의 ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"파일을 찾을 수 없습니다: {file_path}. 파일 경로를 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

# --- 데이터 파일 경로 설정 ---
ENERGY_CONSUMPTION_FILE = 'Energy_consumption - Energy_consumption.csv'
GLOBAL_AIR_QUALITY_FILE = 'global_air_quality_data_10000 - global_air_quality_data_10000.csv'
HEALTH_IMPACT_FILE = 'air_quality_health_impact_data - air_quality_health_impact_data.csv'
INDOOR_AIR_QUALITY_FILE = 'AirQuality - AirQuality.csv'

# --- 데이터 로드 ---
df_energy = load_data(ENERGY_CONSUMPTION_FILE)
df_global_aq = load_data(GLOBAL_AIR_QUALITY_FILE)
df_health = load_data(HEALTH_IMPACT_FILE)
df_indoor_aq = load_data(INDOOR_AIR_QUALITY_FILE)

# --- 2. 앱 제목 및 설명 ---
st.set_page_config(layout="wide")
st.title('🌡️ 환기 방식, 에너지 소비 및 실내 환경 통합 분석')
st.markdown("""
이 웹 앱은 환기 방식과 에너지 소비 패턴이 실내 공기질 및 거주자 건강 지표에 미치는 영향을 분석합니다.
제공된 다양한 데이터셋을 통합하여 의미 있는 인사이트를 도출합니다.
""")

# --- 3. 데이터 개요 및 결측치 확인 ---
st.sidebar.header('데이터 개요')
if st.sidebar.checkbox('데이터셋 미리보기 및 정보 확인'):
    st.subheader('데이터셋 정보')

    if df_energy is not None:
        st.write('**1. 에너지 소비 데이터 (`energy_consumption.csv`)**')
        st.dataframe(df_energy.head())
        st.write(df_energy.info())
        st.write('결측치:\n', df_energy.isnull().sum())
    if df_global_aq is not None:
        st.write('**2. 글로벌 대기질 데이터 (`global_air_quality.csv`)**')
        st.dataframe(df_global_aq.head())
        st.write(df_global_aq.info())
        st.write('결측치:\n', df_global_aq.isnull().sum())
    if df_health is not None:
        st.write('**3. 공기질 건강 영향 데이터 (`air_quality_health_impact_data.csv`)**')
        st.dataframe(df_health.head())
        st.write(df_health.info())
        st.write('결측치:\n', df_health.isnull().sum())
    if df_indoor_aq is not None:
        st.write('**4. 실내 공기질 데이터 (`indoor_air_quality.csv`)**')
        st.dataframe(df_indoor_aq.head())
        st.write(df_indoor_aq.info())
        st.write('결측치:\n', df_indoor_aq.isnull().sum())


# --- 4. 데이터 전처리 및 통합 ---
st.sidebar.header('분석 설정')

# --- 4.1 실내 공기질 (Indoor Air Quality) 분석 ---
if df_indoor_aq is not None:
    st.sidebar.subheader('실내 공기질 분석 설정')
    st.header('🏡 실내 공기질 데이터 분석')

    try:
        df_indoor_aq['DateTime'] = pd.to_datetime(df_indoor_aq['Date'] + ' ' + df_indoor_aq['Time'],
                                                  format='%d/%m/%Y %H.%M.%S', errors='coerce')
        df_indoor_aq.set_index('DateTime', inplace=True)
        
        df_indoor_aq.replace(-200, np.nan, inplace=True)
        numeric_cols = df_indoor_aq.select_dtypes(include=np.number).columns
        df_indoor_aq[numeric_cols] = df_indoor_aq[numeric_cols].apply(lambda col: col.interpolate(method='linear'))

        df_indoor_aq.dropna(subset=['DateTime'], inplace=True)

    except KeyError:
        st.warning("df_indoor_aq에 'Date' 또는 'Time' 컬럼이 없어 시계열 분석이 어렵습니다. 해당 컬럼이 있는지 확인해주세요.")
    except Exception as e:
        st.warning(f"df_indoor_aq 날짜/시간 변환 중 오류 발생: {e}. 데이터 형식(예: 10/03/2004 18.00.00)을 확인해주세요.")
        df_indoor_aq = pd.DataFrame() # 오류 발생 시 빈 DataFrame으로 처리

    if not df_indoor_aq.empty and not df_indoor_aq.select_dtypes(include=np.number).empty:
        air_quality_metrics_candidate = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'PT08.S5(O3)', 'T', 'RH']
        air_quality_metrics_available = [col for col in air_quality_metrics_candidate if col in df_indoor_aq.columns and pd.api.types.is_numeric_dtype(df_indoor_aq[col])]
        
        if air_quality_metrics_available:
            selected_aq_metric = st.sidebar.selectbox('시각화할 공기질 지표 선택', air_quality_metrics_available)

            st.subheader(f'{selected_aq_metric} 시계열 변화')
            if 'DateTime' in df_indoor_aq.index.name:
                df_resampled = df_indoor_aq.resample('H').mean(numeric_only=True).reset_index()
                fig = px.line(df_resampled, x='DateTime', y=selected_aq_metric,
                              title=f'{selected_aq_metric} Time Series (Hourly Average)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("실내 공기질 데이터에 유효한 시간 기반 인덱스가 없어 시계열 그래프를 그릴 수 없습니다.")

            st.subheader('공기질 지표 분포')
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.histplot(df_indoor_aq[selected_aq_metric].dropna(), kde=True, ax=ax2)
            ax2.set_title(f'{selected_aq_metric} Distribution')
            st.pyplot(fig2)
        else:
            st.info("실내 공기질 데이터에 분석 가능한 숫자형 공기질 지표 컬럼이 충분하지 않습니다.")
    else:
        st.warning("실내 공기질 데이터 처리 중 오류가 발생했거나, 유효한 데이터가 없어 분석할 수 없습니다.")


# --- 4.2 공기질 건강 영향 데이터 (air_quality_health_impact_data.csv) 분석 ---
if df_health is not None:
    st.sidebar.subheader('공기질 건강 영향 데이터 분석 설정')
    st.header('😷 공기질 및 건강 영향 데이터 분석')
    st.markdown("""
    이 데이터셋은 대기질 지표, 환경 요인, 그리고 호흡기/심혈관 질환 발생 및 병원 입원 데이터,
    종합 건강 영향 점수 및 등급을 포함합니다.
    """)

    # --- 대기질 및 환경 요인 분석 ---
    aq_env_cols = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed']
    aq_env_metrics_available = [col for col in aq_env_cols if col in df_health.columns and pd.api.types.is_numeric_dtype(df_health[col])]

    # 이전의 NameError 발생 지점 수정: aqi_metrics_available 대신 aq_env_metrics_available 사용
    if aq_env_metrics_available:
        selected_aqi_metric = st.sidebar.selectbox('시각화할 대기질/환경 지표 선택', aq_env_metrics_available, key='health_aq_env_select')
        
        st.subheader(f'{selected_aqi_metric} 분포')
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        sns.histplot(df_health[selected_aqi_metric].dropna(), kde=True, ax=ax_dist)
        ax_dist.set_title(f'{selected_aqi_metric} Distribution')
        st.pyplot(fig_dist)

        if len(aq_env_metrics_available) > 1:
            st.subheader('대기질 및 환경 지표 상관관계')
            corr_matrix_aq_env = df_health[aq_env_metrics_available].corr()
            fig_corr_aq_env, ax_corr_aq_env = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix_aq_env, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr_aq_env)
            ax_corr_aq_env.set_title('Correlation Matrix of Air Quality and Environmental Factors')
            st.pyplot(fig_corr_aq_env)
    else:
        st.info("이 데이터셋에 분석 가능한 숫자형 대기질/환경 지표 컬럼이 충분하지 않습니다.")

    # --- 건강 영향 지표 분석 ---
    health_outcome_cols = ['RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions']
    health_outcome_metrics_available = [col for col in health_outcome_cols if col in df_health.columns and pd.api.types.is_numeric_dtype(df_health[col])]

    if health_outcome_metrics_available:
        st.subheader('건강 영향 지표 요약')
        st.dataframe(df_health[health_outcome_metrics_available].describe().T)

        selected_health_outcome = st.sidebar.selectbox('시각화할 건강 영향 지표 선택', health_outcome_metrics_available, key='health_outcome_select')
        
        st.subheader(f'{selected_health_outcome} 분포')
        fig_health_dist, ax_health_dist = plt.subplots(figsize=(10, 6))
        sns.histplot(df_health[selected_health_outcome].dropna(), kde=True, ax=ax_health_dist)
        ax_health_dist.set_title(f'{selected_health_outcome} Distribution')
        st.pyplot(fig_health_dist)
        
        # AQI와 건강 영향 지표 간의 관계 (산점도)
        if 'AQI' in df_health.columns and pd.api.types.is_numeric_dtype(df_health['AQI']):
            st.subheader(f'AQI와 {selected_health_outcome} 관계')
            fig_aqi_health = px.scatter(df_health, x='AQI', y=selected_health_outcome,
                                        title=f'AQI vs. {selected_health_outcome}')
            st.plotly_chart(fig_aqi_health, use_container_width=True)
        else:
            st.info("'AQI' 컬럼이 없어 AQI와 건강 영향 지표 간의 관계를 분석할 수 없습니다.")
    else:
        st.info("이 데이터셋에 분석 가능한 숫자형 건강 영향 지표 컬럼이 충분하지 않습니다.")

    # --- HealthImpactScore 및 HealthImpactClass 분석 ---
    if 'HealthImpactScore' in df_health.columns and pd.api.types.is_numeric_dtype(df_health['HealthImpactScore']):
        st.subheader('건강 영향 점수 (HealthImpactScore) 분포')
        fig_score, ax_score = plt.subplots(figsize=(10, 6))
        sns.histplot(df_health['HealthImpactScore'].dropna(), kde=True, ax=ax_score)
        ax_score.set_title('HealthImpactScore Distribution')
        st.pyplot(fig_score)
    else:
        st.info("'HealthImpactScore' 컬럼이 없어 건강 영향 점수 분석을 할 수 없습니다.")

    if 'HealthImpactClass' in df_health.columns:
        st.subheader('건강 영향 등급 (HealthImpactClass) 분포')
        class_counts = df_health['HealthImpactClass'].value_counts().reset_index()
        class_counts.columns = ['HealthImpactClass', 'Count']
        fig_class = px.bar(class_counts, x='HealthImpactClass', y='Count',
                           title='HealthImpactClass Distribution')
        st.plotly_chart(fig_class, use_container_width=True)
    else:
        st.info("'HealthImpactClass' 컬럼이 없어 건강 영향 등급 분석을 할 수 없습니다.")


# --- 4.3 에너지 소비 (Energy Consumption) 데이터 분석 ---
if df_energy is not None:
    st.sidebar.subheader('에너지 소비 분석 설정')
    st.header('⚡ 에너지 소비 데이터 분석')

    energy_cols_for_corr = ['Energy_Consumption_kWh', 'Temperature_C', 'Humidity_%', 'Occupancy_Count'] 
    energy_metrics_available_for_corr = [col for col in energy_cols_for_corr if col in df_energy.columns and pd.api.types.is_numeric_dtype(df_energy[col])]

    if 'Building_Type' in df_energy.columns and 'Energy_Consumption_kWh' in df_energy.columns:
        st.subheader('건물 유형별 평균 에너지 소비량')
        df_energy['Energy_Consumption_kWh'] = pd.to_numeric(df_energy['Energy_Consumption_kWh'], errors='coerce')
        if not df_energy['Energy_Consumption_kWh'].dropna().empty:
            avg_energy_by_building = df_energy.groupby('Building_Type')['Energy_Consumption_kWh'].mean().reset_index()
            fig_energy_building = px.bar(avg_energy_by_building, x='Building_Type', y='Energy_Consumption_kWh',
                                         title='Average Energy Consumption by Building Type')
            st.plotly_chart(fig_energy_building, use_container_width=True)
        else:
            st.info("'Energy_Consumption_kWh' 컬럼에 유효한 숫자 데이터가 없어 건물 유형별 에너지 소비량 분석을 할 수 없습니다.")

    if 'Ventilation_System' in df_energy.columns and 'Energy_Consumption_kWh' in df_energy.columns:
        st.subheader('환기 시스템 유형별 평균 에너지 소비량')
        df_energy['Energy_Consumption_kWh'] = pd.to_numeric(df_energy['Energy_Consumption_kWh'], errors='coerce')
        if not df_energy['Energy_Consumption_kWh'].dropna().empty:
            avg_energy_by_ventilation = df_energy.groupby('Ventilation_System')['Energy_Consumption_kWh'].mean().reset_index()
            fig_energy_ventilation = px.bar(avg_energy_by_ventilation, x='Ventilation_System', y='Energy_Consumption_kWh',
                                            title='Average Energy Consumption by Ventilation System Type')
            st.plotly_chart(fig_energy_ventilation, use_container_width=True)
        else:
            st.info("'Energy_Consumption_kWh' 컬럼에 유효한 숫자 데이터가 없어 환기 시스템 유형별 에너지 소비량 분석을 할 수 없습니다.")

    if energy_metrics_available_for_corr and len(energy_metrics_available_for_corr) > 1:
        st.subheader('에너지 소비 관련 지표 상관관계')
        corr_matrix_energy = df_energy[energy_metrics_available_for_corr].corr()
        fig_corr_energy, ax_corr_energy = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix_energy, annot=True, cmap='viridis', fmt=".2f", ax=ax_corr_energy)
        ax_corr_energy.set_title('Correlation Matrix of Energy Consumption Factors')
        st.pyplot(fig_corr_energy)
    else:
        st.info("에너지 소비 데이터에 상관관계 분석을 위한 충분한 숫자형 컬럼이 없거나 단일 컬럼만 존재합니다.")

# --- 4.4 글로벌 대기질 (Global Air Quality) 데이터 분석 (참고용) ---
if df_global_aq is not None:
    st.sidebar.subheader('글로벌 대기질 분석 설정')
    st.header('🌍 글로벌 대기질 데이터 분석 (참고용)')

    global_aq_metrics = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    global_aq_metrics_available = [col for col in global_aq_metrics if col in df_global_aq.columns and pd.api.types.is_numeric_dtype(df_global_aq[col])]

    if 'Country' in df_global_aq.columns and 'PM2.5' in df_global_aq.columns:
        st.subheader('국가별 PM2.5 평균 농도')
        df_global_aq['PM2.5'] = pd.to_numeric(df_global_aq['PM2.5'], errors='coerce')
        if not df_global_aq['PM2.5'].dropna().empty:
            avg_pm25_by_country = df_global_aq.groupby('Country')['PM2.5'].mean().nlargest(10).reset_index()
            fig_global_pm25 = px.bar(avg_pm25_by_country, x='Country', y='PM2.5',
                                    title='Top 10 Countries by Average PM2.5 Concentration')
            st.plotly_chart(fig_global_pm25, use_container_width=True)
        else:
            st.info("'PM2.5' 컬럼에 유효한 숫자 데이터가 없어 국가별 PM2.5 분석을 할 수 없습니다.")

    if 'Pollutant' in df_global_aq.columns and 'Value' in df_global_aq.columns:
        st.subheader('전역 오염물질별 농도 분포')
        df_global_aq['Value'] = pd.to_numeric(df_global_aq['Value'], errors='coerce')
        if not df_global_aq['Value'].dropna().empty:
            fig_global_pollutant_dist = px.box(df_global_aq.dropna(subset=['Pollutant', 'Value']), x='Pollutant', y='Value',
                                               title='Global Pollutant Concentration Distribution')
            st.plotly_chart(fig_global_pollutant_dist, use_container_width=True)
        else:
            st.info("'Value' 컬럼에 유효한 숫자 데이터가 없어 전역 오염물질 농도 분포 분석을 할 수 없습니다.")
    else:
        st.info("글로벌 대기질 데이터에 분석 가능한 컬럼이 충분하지 않습니다.")

# --- 5. 통합 분석 (가설 기반) ---
st.header('💡 통합 분석 (가설 기반)')
st.markdown("""
**참고:** 업로드된 데이터셋 간에 직접적인 연결고리(예: 공통 ID, 정확한 시간 매칭)가 없어,
아래 통합 분석 섹션은 일반적인 분석 아이디어를 제공하며, 실제 실행을 위해서는 데이터 통합 전처리가 더 필요합니다.
""")

if st.checkbox('통합 분석 아이디어 보기'):
    st.subheader('아이디어 1: 에너지 소비-환기-실내 공기질 상관관계')
    st.markdown("""
    만약 `df_energy`와 `df_indoor_aq`가 `Building_ID` 및 `DateTime`으로 연결될 수 있다면:
    - 특정 환기 시스템 (예: 자연 환기 vs. 기계 환기) 사용 시 에너지 소비량과 공기질 지표 (CO2, 미세먼지 등)의 관계 분석
    - 에너지 절약을 위한 환기 전략 제안 (예: CO2 농도가 낮을 때는 환기 강도를 줄임)
    """)

    st.subheader('아이디어 2: 실내 공기질-건강 영향 지표 상관관계')
    st.markdown("""
    만약 `df_indoor_aq`와 `df_health`가 `Location_ID` 및 `DateTime`으로 연결될 수 있다면:
    - 실내 공기 오염물질 농도와 `RespiratoryCases`, `CardiovascularCases`, `HospitalAdmissions`, `HealthImpactScore` 등 건강 지표 간의 관계 분석
    - 외부 공기질(AQI, PM2.5 등)이 건강 영향 지표에 미치는 영향 분석
    """)

st.sidebar.markdown('---')
st.sidebar.info('이 웹 앱은 Streamlit으로 구축되었으며, 다양한 데이터셋을 활용하여 실내 환경과 건강에 대한 인사이트를 제공합니다.')
