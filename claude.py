import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# --- 1. 데이터 로드 함수 정의 ---
# @st.cache_data를 사용하여 데이터를 캐싱하여 앱 성능 향상
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

# --- 데이터 파일 경로 설정 (업로드된 파일명으로 변경) ---
# 실제 환경에서는 이 파일들이 앱 실행 경로에 있어야 합니다.
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
st.set_page_config(layout="wide") # 웹 앱 레이아웃을 넓게 설정
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
        st.write('**3. 공기질 건강 영향 데이터 (`health_impact.csv`)**')
        st.dataframe(df_health.head())
        st.write(df_health.info())
        st.write('결측치:\n', df_health.isnull().sum())
    if df_indoor_aq is not None:
        st.write('**4. 실내 공기질 데이터 (`indoor_air_quality.csv`)**')
        st.dataframe(df_indoor_aq.head())
        st.write(df_indoor_aq.info())
        st.write('결측치:\n', df_indoor_aq.isnull().sum())


# --- 4. 데이터 전처리 및 통합 (가장 중요한 부분) ---
# 이 부분은 데이터셋 간의 공통 컬럼이나 논리적 연결 고리를 찾아야 합니다.
# 현재 업로드된 데이터셋들은 직접적인 연결 고리가 명확하지 않습니다.
# 가정: 실내 공기질 데이터와 건강 영향 데이터는 시간/위치 등으로 연결될 수 있고,
# 에너지 소비 데이터는 특정 건물/환기 시스템과 연결될 수 있습니다.
# 여기서는 각 데이터를 분리하여 분석하고, 가능한 경우 부분적으로 통합을 시도합니다.

st.sidebar.header('분석 설정')

# --- 4.1 실내 공기질 (Indoor Air Quality) 분석 ---
if df_indoor_aq is not None:
    st.sidebar.subheader('실내 공기질 분석 설정')
    st.header('🏡 실내 공기질 데이터 분석')

    # 컬럼명 통일 (예: 'Date'를 '날짜'로, 'Time'을 '시간'으로)
    # df_indoor_aq 컬럼명: 'Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'
    # CO, NO2, O3는 주요 공기질 지표로 활용 가능

    # 날짜-시간 컬럼 결합 및 datetime 형식으로 변환
    try:
        df_indoor_aq['DateTime'] = pd.to_datetime(df_indoor_aq['Date'] + ' ' + df_indoor_aq['Time'])
        df_indoor_aq.set_index('DateTime', inplace=True)
        # 잘못된 값(-200)을 NaN으로 처리하고 보간
        df_indoor_aq.replace(-200, np.nan, inplace=True)
        # 간단한 선형 보간
        df_indoor_aq = df_indoor_aq.apply(lambda col: col.interpolate(method='linear'), axis=0)

    except KeyError:
        st.warning("df_indoor_aq에 'Date' 또는 'Time' 컬럼이 없어 시계열 분석이 어렵습니다.")
    except Exception as e:
        st.warning(f"df_indoor_aq 날짜/시간 변환 중 오류 발생: {e}")

    # air_quality_metrics = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'O3(GT)'] # O3(GT) 대신 PT08.S5(O3) 사용
    air_quality_metrics_available = [col for col in ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'PT08.S5(O3)', 'T', 'RH'] if col in df_indoor_aq.columns]
    
    if air_quality_metrics_available:
        selected_aq_metric = st.sidebar.selectbox('시각화할 공기질 지표 선택', air_quality_metrics_available)

        st.subheader(f'{selected_aq_metric} 시계열 변화')
        # 데이터가 많을 수 있으므로 시/일 단위로 리샘플링하여 표시
        if 'DateTime' in df_indoor_aq.index.name:
            df_resampled = df_indoor_aq.resample('H').mean().reset_index() # 시간 단위 평균
            fig = px.line(df_resampled, x='DateTime', y=selected_aq_metric,
                          title=f'{selected_aq_metric} Time Series (Hourly Average)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("시간 기반 인덱스가 없어 시계열 그래프를 그릴 수 없습니다.")


        st.subheader('공기질 지표 분포')
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(df_indoor_aq[selected_aq_metric].dropna(), kde=True, ax=ax2)
        ax2.set_title(f'{selected_aq_metric} Distribution')
        st.pyplot(fig2)
    else:
        st.info("실내 공기질 데이터에 분석 가능한 공기질 지표 컬럼이 충분하지 않습니다.")

# --- 4.2 건강 영향 (Health Impact) 데이터 분석 ---
if df_health is not None:
    st.sidebar.subheader('건강 영향 분석 설정')
    st.header('😷 공기질 건강 영향 데이터 분석')

    # 'Symptoms', 'Severity', 'Exposure_Level' 등의 컬럼이 있다고 가정
    # 실제 컬럼명 확인 필요
    health_symptoms = ['eye_irritation', 'throat_irritation', 'cough', 'shortness_of_breath', 'headache', 'fatigue']
    
    # Check if 'pollutant' and 'severity' columns exist before plotting
    if 'pollutant' in df_health.columns and 'severity' in df_health.columns:
        st.subheader('오염물질별 건강 증상 심각도')
        # Ensure 'severity' is numeric
        df_health['severity'] = pd.to_numeric(df_health['severity'], errors='coerce')
        avg_severity = df_health.groupby('pollutant')['severity'].mean().reset_index()
        fig = px.bar(avg_severity,
                     x='pollutant', y='severity',
                     title='Average Health Symptom Severity by Pollutant')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("건강 영향 데이터에 'pollutant' 또는 'severity' 컬럼이 없어 오염물질별 심각도 분석을 할 수 없습니다.")

    health_symptoms_available = [col for col in health_symptoms if col in df_health.columns]
    if health_symptoms_available:
        st.subheader('각 건강 증상 보고 빈도')
        # sum()을 사용하려면 해당 컬럼들이 숫자형이어야 합니다.
        # 만약 'yes/no' 또는 T/F 형태라면, 숫자로 변환 필요 (예: df[col].astype(int))
        try:
            symptom_counts = df_health[health_symptoms_available].sum().sort_values(ascending=False)
            fig_symptom, ax_symptom = plt.subplots(figsize=(10, 6))
            sns.barplot(x=symptom_counts.index, y=symptom_counts.values, ax=ax_symptom)
            ax_symptom.set_title('Frequency of Reported Health Symptoms')
            ax_symptom.set_ylabel('Number of Reports')
            ax_symptom.set_xticklabels(ax_symptom.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig_symptom)
        except TypeError:
            st.warning("건강 증상 컬럼의 데이터 타입이 숫자가 아니어서 보고 빈도 계산에 문제가 있습니다. 숫자로 변환해야 합니다.")
    else:
        st.info("건강 영향 데이터에 분석 가능한 건강 지표 컬럼이 충분하지 않습니다.")


# --- 4.3 에너지 소비 (Energy Consumption) 데이터 분석 ---
if df_energy is not None:
    st.sidebar.subheader('에너지 소비 분석 설정')
    st.header('⚡ 에너지 소비 데이터 분석')

    # 'Energy_Consumption_kWh', 'Building_Type', 'Ventilation_System', 'Temperature_C', 'Humidity_%', 'Occupancy_Count'
    # 실제 컬럼명을 확인하고 필요에 따라 수정
    # 'Energy_kWh' 대신 'Energy_Consumption_kWh' 사용
    energy_cols_for_corr = ['Energy_Consumption_kWh', 'Temperature_C', 'Humidity_%', 'Occupancy_Count'] 
    
    # 존재하지 않는 컬럼 제거
    energy_metrics_available_for_corr = [col for col in energy_cols_for_corr if col in df_energy.columns]

    if 'Building_Type' in df_energy.columns and 'Energy_Consumption_kWh' in df_energy.columns:
        st.subheader('건물 유형별 평균 에너지 소비량')
        avg_energy_by_building = df_energy.groupby('Building_Type')['Energy_Consumption_kWh'].mean().reset_index()
        fig_energy_building = px.bar(avg_energy_by_building, x='Building_Type', y='Energy_Consumption_kWh',
                                     title='Average Energy Consumption by Building Type')
        st.plotly_chart(fig_energy_building, use_container_width=True)

    if 'Ventilation_System' in df_energy.columns and 'Energy_Consumption_kWh' in df_energy.columns:
        st.subheader('환기 시스템 유형별 평균 에너지 소비량')
        avg_energy_by_ventilation = df_energy.groupby('Ventilation_System')['Energy_Consumption_kWh'].mean().reset_index()
        fig_energy_ventilation = px.bar(avg_energy_by_ventilation, x='Ventilation_System', y='Energy_Consumption_kWh',
                                        title='Average Energy Consumption by Ventilation System Type')
        st.plotly_chart(fig_energy_ventilation, use_container_width=True)

    if energy_metrics_available_for_corr:
        st.subheader('에너지 소비 관련 지표 상관관계')
        corr_matrix_energy = df_energy[energy_metrics_available_for_corr].corr()
        fig_corr_energy, ax_corr_energy = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix_energy, annot=True, cmap='viridis', fmt=".2f", ax=ax_corr_energy)
        ax_corr_energy.set_title('Correlation Matrix of Energy Consumption Factors')
        st.pyplot(fig_corr_energy)
    else:
        st.info("에너지 소비 데이터에 상관관계 분석을 위한 충분한 컬럼이 없습니다.")

# --- 4.4 글로벌 대기질 (Global Air Quality) 데이터 분석 (참고용) ---
# 이 데이터는 실내 환경 분석에 직접적으로 통합하기보다 외부 요인 참고용으로 활용될 수 있습니다.
if df_global_aq is not None:
    st.sidebar.subheader('글로벌 대기질 분석 설정')
    st.header('🌍 글로벌 대기질 데이터 분석 (참고용)')

    # 'Country', 'City', 'Pollutant', 'Value' 등의 컬럼이 있다고 가정
    global_aq_metrics = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    global_aq_metrics_available = [col for col in global_aq_metrics if col in df_global_aq.columns]

    if 'Country' in df_global_aq.columns and 'PM2.5' in df_global_aq.columns:
        st.subheader('국가별 PM2.5 평균 농도')
        # 'PM2.5' 컬럼이 숫자형인지 확인하고, 아니면 변환 시도
        df_global_aq['PM2.5'] = pd.to_numeric(df_global_aq['PM2.5'], errors='coerce')
        avg_pm25_by_country = df_global_aq.groupby('Country')['PM2.5'].mean().nlargest(10).reset_index()
        fig_global_pm25 = px.bar(avg_pm25_by_country, x='Country', y='PM2.5',
                                title='Top 10 Countries by Average PM2.5 Concentration')
        st.plotly_chart(fig_global_pm25, use_container_width=True)

    if 'Pollutant' in df_global_aq.columns and 'Value' in df_global_aq.columns:
        st.subheader('전역 오염물질별 농도 분포')
        # 'Value' 컬럼이 숫자형인지 확인하고, 아니면 변환 시도
        df_global_aq['Value'] = pd.to_numeric(df_global_aq['Value'], errors='coerce')
        fig_global_pollutant_dist = px.box(df_global_aq.dropna(subset=['Pollutant', 'Value']), x='Pollutant', y='Value',
                                           title='Global Pollutant Concentration Distribution')
        st.plotly_chart(fig_global_pollutant_dist, use_container_width=True)
    else:
        st.info("글로벌 대기질 데이터에 분석 가능한 컬럼이 충분하지 않습니다.")

# --- 5. 통합 분석 (가설 기반) ---
# 이 섹션은 앞서 로드한 데이터들이 특정 기준으로 통합될 수 있다는 가정 하에 작성됩니다.
# 예를 들어, 특정 건물 ID, 시간대 등으로 데이터 병합이 가능할 때 유용합니다.
# 현재 업로드된 파일들만으로는 직접적인 병합이 어려우므로, 이 부분은 예시 및 아이디어 제공용입니다.
# 실제 통합을 위해서는 각 데이터셋에 공통 식별자(예: building_id, timestamp)가 필요합니다.

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
    # 예시 코드 (실행하려면 데이터 통합 필요)
    # merged_df = pd.merge(df_energy, df_indoor_aq, on=['Building_ID', 'DateTime'], how='inner')
    # if not merged_df.empty:
    #     st.subheader('에너지 소비량과 CO2 농도 관계 (가정)')
    #     fig_merged = px.scatter(merged_df, x='Energy_Consumption_kWh', y='CO2(GT)', color='Ventilation_System',
    #                             title='Energy Consumption vs. CO2 by Ventilation System')
    #     st.plotly_chart(fig_merged)

    st.subheader('아이디어 2: 실내 공기질-건강 지표 상관관계')
    st.markdown("""
    만약 `df_indoor_aq`와 `df_health`가 `Location_ID` 및 `DateTime`으로 연결될 수 있다면:
    - 특정 공기 오염물질 농도 (CO, NO2, O3)와 특정 건강 증상 (두통, 호흡기 문제) 간의 상관관계 분석
    - 공기질이 악화될 때 건강 지표가 어떻게 변하는지 시계열 분석
    """)
    # 예시 코드 (실행하려면 데이터 통합 필요)
    # merged_health_aq_df = pd.merge(df_indoor_aq, df_health, on=['Location_ID', 'DateTime'], how='inner')
    # if not merged_health_aq_df.empty:
    #     st.subheader('CO2 농도와 두통 보고 빈도 관계 (가정)')
    #     fig_health = px.scatter(merged_health_aq_df, x='CO2(GT)', y='headache',
    #                             title='CO2 Concentration vs. Headache Reports')
    #     st.plotly_chart(fig_health)

st.sidebar.markdown('---')
st.sidebar.info('이 웹 앱은 Streamlit으로 구축되었으며, 다양한 데이터셋을 활용하여 실내 환경과 건강에 대한 인사이트를 제공합니다.')
