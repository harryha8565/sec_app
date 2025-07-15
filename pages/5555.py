import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import io

# --- 1. 페이지 설정 및 기본 스타일 ---
st.set_page_config(layout="wide", page_title="실내 환경 통합 분석 대시보드")
st.title('📊 실내 환경 통합 분석 대시보드')
st.markdown("""
이 대시보드는 여러 데이터셋을 종합하여 환기 방식, 에너지 소비, 실내 공기질, 건강 영향 간의 관계를 시각적으로 분석합니다.
대화형 그래프를 통해 데이터를 깊이 있게 탐색하고 의미 있는 인사이트를 발견해 보세요.
""")

# --- 2. 데이터 로드 함수 (캐시 사용) ---
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
ENERGY_CONSUMPTION_FILE = 'Energy_consumption - Energy_consumption.csv'
GLOBAL_AIR_QUALITY_FILE = 'global_air_quality_data_10000 - global_air_quality_data_10000.csv'
HEALTH_IMPACT_FILE = 'air_quality_health_impact_data - air_quality_health_impact_data.csv'
INDOOR_AIR_QUALITY_FILE = 'AirQuality - AirQuality.csv'

# --- 4. 데이터 로드 ---
df_energy = load_data(ENERGY_CONSUMPTION_FILE)
df_global_aq = load_data(GLOBAL_AIR_QUALITY_FILE)
df_health = load_data(HEALTH_IMPACT_FILE)
df_indoor_aq = load_data(INDOOR_AIR_QUALITY_FILE)

# --- 5. 분석 모듈 함수 (시각화 강화) ---

def display_dataframe_info(df, name):
    """데이터프레임의 기본 정보를 시각적으로 풍부하게 표시합니다."""
    st.subheader(f'"{name}" 데이터셋 개요')
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**데이터 기본 정보:**")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    with col2:
        st.write("**결측치 시각화:**")
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            fig = px.bar(missing_values, 
                         x=missing_values.index, 
                         y=missing_values.values,
                         labels={'x':'컬럼명', 'y':'결측치 수'},
                         title=f'{name} 데이터셋 결측치 수')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("이 데이터셋에는 결측치가 없습니다.")

    st.write("**숫자형 데이터 기술 통계:**")
    st.dataframe(df.describe().T)

def indoor_aq_analysis(df):
    """실내 공기질 데이터 분석 및 시각화 (강화 및 오류 수정)"""
    st.header('🏡 실내 공기질 상세 분석')
    if df is None:
        st.warning("실내 공기질 데이터가 로드되지 않았습니다.")
        return

    # 데이터 전처리
    try:
        df_processed = df.copy()
        
        # --- START of FIX ---
        # 1. 'Date'와 'Time' 컬럼이 존재하는지 먼저 확인
        if 'Date' not in df_processed.columns or 'Time' not in df_processed.columns:
            st.error("데이터에 'Date' 또는 'Time' 컬럼이 없습니다.")
            return
            
        # 2. 'Date'와 'Time'을 문자열로 확실하게 변환
        df_processed['Date'] = df_processed['Date'].astype(str)
        df_processed['Time'] = df_processed['Time'].astype(str)

        # 3. 시간 형식의 '.'을 ':'으로 변경하여 표준 형식으로 만듦
        df_processed['Time_formatted'] = df_processed['Time'].str.replace('.', ':', regex=False)
        
        # 4. 날짜와 포맷된 시간을 합쳐서 datetime으로 변환
        datetime_series = df_processed['Date'] + ' ' + df_processed['Time_formatted']
        df_processed['DateTime'] = pd.to_datetime(datetime_series, format='%d/%m/%Y %H:%M:%S', errors='coerce')
        # --- END of FIX ---

        df_processed.set_index('DateTime', inplace=True)
        df_processed.replace(-200, np.nan, inplace=True)
        numeric_cols = df_processed.select_dtypes(include=np.number).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].apply(lambda col: col.interpolate(method='linear'))
        df_processed.dropna(subset=df_processed.index.name, inplace=True)

    except Exception as e:
        st.error(f"데이터 전처리 중 오류가 발생했습니다. 'Date' 또는 'Time' 컬럼 형식을 확인해주세요. \n\n오류 내용: {e}")
        return

    if df_processed.empty:
        st.warning("처리 후 유효한 실내 공기질 데이터가 없습니다.")
        return

    # 시각화
    st.markdown("#### 주요 실내 오염물질 농도 변화")
    st.write("시간에 따른 주요 오염물질의 농도 변화를 확인하여 특정 시간대의 오염 패턴을 파악할 수 있습니다.")
    
    aq_metrics_options = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'PT08.S5(O3)']
    selected_metrics = st.multiselect('시계열 그래프로 볼 오염물질을 선택하세요:', 
                                      [m for m in aq_metrics_options if m in df_processed.columns], 
                                      default=[m for m in ['CO(GT)', 'NOx(GT)'] if m in df_processed.columns])

    if selected_metrics:
        df_resampled = df_processed[selected_metrics].resample('D').mean() # 일별 평균
        fig = px.line(df_resampled, x=df_resampled.index, y=selected_metrics,
                      title='주요 실내 오염물질 농도 시계열 (일별 평균)',
                      labels={'value': '농도', 'DateTime': '날짜'})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 실내 환경 요인 간의 상관관계")
    st.write("각 오염물질과 온도, 습도 간의 상관관계를 히트맵으로 확인하여 서로 어떤 영향을 주는지 파악할 수 있습니다.")
    
    corr_cols = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH']
    corr_cols_available = [col for col in corr_cols if col in df_processed.columns]
    if len(corr_cols_available) > 1:
        corr_matrix = df_processed[corr_cols_available].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        ax_corr.set_title('실내 환경 요인 상관관계 히트맵', fontsize=16)
        st.pyplot(fig_corr)

def health_impact_analysis(df):
    """공기질 건강 영향 데이터 분석 및 시각화 (강화)"""
    st.header('😷 공기질과 건강 영향 관계 분석')
    if df is None:
        st.warning("공기질 건강 영향 데이터가 로드되지 않았습니다.")
        return

    st.markdown("#### AQI(통합대기환경지수)와 건강 영향")
    st.write("AQI가 건강에 미치는 영향을 산점도와 박스 플롯으로 확인합니다. AQI가 높을수록 건강 영향 등급이 어떻게 변하는지 주목하세요.")

    col1, col2 = st.columns(2)
    with col1:
        health_outcome = st.selectbox(
            'AQI와 비교할 건강 지표를 선택하세요:',
            ['RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions'],
            key='health_outcome_select'
        )
        if health_outcome and 'AQI' in df.columns:
            fig = px.scatter(df, x='AQI', y=health_outcome, 
                             trendline="ols", trendline_color_override="red",
                             title=f'AQI와 {health_outcome} 관계',
                             labels={'AQI': '통합대기환경지수 (AQI)', health_outcome: health_outcome},
                             hover_data=['HealthImpactClass'])
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'HealthImpactClass' in df.columns and 'AQI' in df.columns:
            fig2 = px.box(df, x='HealthImpactClass', y='AQI',
                          color='HealthImpactClass',
                          title='건강 영향 등급별 AQI 분포',
                          labels={'HealthImpactClass': '건강 영향 등급', 'AQI': '통합대기환경지수 (AQI)'},
                          category_orders={"HealthImpactClass": ["Low", "Moderate", "High", "Very High"]})
            st.plotly_chart(fig2, use_container_width=True)
            
    st.markdown("#### 주요 오염물질과 건강 영향")
    st.write("PM2.5(초미세먼지)와 O3(오존)이 건강에 미치는 영향을 비교 분석합니다.")
    
    pollutants_to_check = ['PM2_5', 'O3']
    health_metrics_to_check = ['RespiratoryCases', 'CardiovascularCases']
    
    available_pollutants = [p for p in pollutants_to_check if p in df.columns]
    available_health_metrics = [h for h in health_metrics_to_check if h in df.columns]

    if available_pollutants and available_health_metrics:
        selected_pollutant = st.radio("분석할 오염물질 선택:", available_pollutants, horizontal=True)
        
        fig3 = px.scatter(df, x=selected_pollutant, y='RespiratoryCases',
                          trendline="ols",
                          title=f'{selected_pollutant} 농도와 호흡기 질환 발생 건수',
                          labels={selected_pollutant: f'{selected_pollutant} 농도', 'RespiratoryCases': '호흡기 질환 발생 건수'})
        st.plotly_chart(fig3, use_container_width=True)


def energy_consumption_analysis(df):
    """에너지 소비 데이터 분석 및 시각화 (강화)"""
    st.header('⚡ 에너지 소비 패턴 분석')
    if df is None:
        st.warning("에너지 소비 데이터가 로드되지 않았습니다.")
        return

    df['Energy_Consumption_kWh'] = pd.to_numeric(df['Energy_Consumption_kWh'], errors='coerce')
    df.dropna(subset=['Energy_Consumption_kWh'], inplace=True)

    st.markdown("#### 건물/환기 시스템 유형별 에너지 소비")
    st.write("건물 유형과 환기 시스템에 따라 에너지 소비량이 어떻게 다른지 비교합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        if 'Building_Type' in df.columns:
            avg_energy_building = df.groupby('Building_Type')['Energy_Consumption_kWh'].mean().reset_index()
            fig = px.bar(avg_energy_building, x='Building_Type', y='Energy_Consumption_kWh',
                         title='건물 유형별 평균 에너지 소비량', color='Building_Type',
                         labels={'Building_Type': '건물 유형', 'Energy_Consumption_kWh': '평균 에너지 소비량 (kWh)'})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'Ventilation_System' in df.columns:
            avg_energy_ventilation = df.groupby('Ventilation_System')['Energy_Consumption_kWh'].mean().reset_index()
            fig2 = px.bar(avg_energy_ventilation, x='Ventilation_System', y='Energy_Consumption_kWh',
                          title='환기 시스템 유형별 평균 에너지 소비량', color='Ventilation_System',
                          labels={'Ventilation_System': '환기 시스템', 'Energy_Consumption_kWh': '평균 에너지 소비량 (kWh)'})
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 환경 요인과 에너지 소비의 관계")
    st.write("온도, 습도, 재실자 수가 에너지 소비에 어떤 영향을 미치는지 산점도를 통해 확인합니다.")
    
    scatter_x_options = [col for col in ['Temperature_C', 'Humidity_%', 'Occupancy_Count'] if col in df.columns]
    if scatter_x_options:
        scatter_x = st.selectbox("X축으로 사용할 변수를 선택하세요:", scatter_x_options)
        
        if scatter_x:
            fig3 = px.scatter(df, x=scatter_x, y='Energy_Consumption_kWh',
                              color='Building_Type' if 'Building_Type' in df.columns else None,
                              title=f'{scatter_x}와 에너지 소비량 관계',
                              labels={scatter_x: scatter_x, 'Energy_Consumption_kWh': '에너지 소비량 (kWh)'})
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("에너지 소비량과 비교할 환경 요인 데이터(온도, 습도, 재실자 수)가 없습니다.")


# --- 6. 사이드바 및 메인 화면 구성 ---
st.sidebar.header('분석 메뉴')
analysis_option = st.sidebar.radio(
    "보고 싶은 분석을 선택하세요:",
    ('데이터 개요', '실내 공기질 분석', '공기질-건강 영향 분석', '에너지 소비 분석', '통합 분석 아이디어')
)

if analysis_option == '데이터 개요':
    st.header("🔍 데이터 개요")
    st.info("각 데이터셋의 기본적인 정보와 결측치를 시각적으로 확인합니다.")
    datasets = {
        "실내 공기질": df_indoor_aq,
        "공기질 건강 영향": df_health,
        "에너지 소비": df_energy,
        "글로벌 대기질": df_global_aq
    }
    selected_dataset_name = st.selectbox("정보를 확인할 데이터셋을 선택하세요:", list(datasets.keys()))
    
    if datasets[selected_dataset_name] is not None:
        display_dataframe_info(datasets[selected_dataset_name], selected_dataset_name)
    else:
        st.warning(f"{selected_dataset_name} 데이터를 불러올 수 없습니다.")

elif analysis_option == '실내 공기질 분석':
    indoor_aq_analysis(df_indoor_aq)

elif analysis_option == '공기질-건강 영향 분석':
    health_impact_analysis(df_health)

elif analysis_option == '에너지 소비 분석':
    energy_consumption_analysis(df_energy)
    
elif analysis_option == '통합 분석 아이디어':
    st.header('💡 통합 분석 가이드')
    st.markdown("""
    현재 데이터셋들은 서로 직접 연결되지 않아 개별 분석만 가능합니다. 
    만약 **시간, 위치, 건물 ID** 등 공통된 정보로 데이터들을 연결할 수 있다면, 다음과 같은 심층 분석이 가능해집니다.
    """)
    
    st.subheader('가설 1: 특정 환기 시스템은 에너지 소비를 줄이면서 실내 공기질을 효과적으로 개선할 것이다.')
    st.info("""
    - **필요 데이터:** `에너지 소비` + `실내 공기질`
    - **연결고리:** 시간, 건물 ID
    - **분석 방법:** 환기 시스템 유형별로 시간당 에너지 소비량과 실내 CO, NOx 농도를 비교 분석합니다.
    - **기대 효과:** 에너지 효율과 공기 정화 성능이 모두 뛰어난 최적의 환기 전략을 도출할 수 있습니다.
    """)

    st.subheader('가설 2: 실내 오염물질(예: 벤젠) 농도가 높은 환경은 호흡기 질환 발생률과 유의미한 상관관계를 가질 것이다.')
    st.info("""
    - **필요 데이터:** `실내 공기질` + `공기질 건강 영향`
    - **연결고리:** 시간, 지역 ID
    - **분석 방법:** C6H6(벤젠) 농도의 시계열 데이터와 호흡기 질환 발생 건수의 시계열 데이터를 비교하여 상관관계를 분석합니다.
    - **기대 효과:** 특정 실내 오염물질의 건강 위험도를 정량적으로 평가하고, 위험 예측 모델을 개발하여 사전 예방 조치를 취할 수 있습니다.
    """)

st.sidebar.markdown('---')
st.sidebar.info('이 대시보드는 데이터 시각화를 통해 복잡한 환경 데이터 속에서 명확한 인사이트를 제공하는 것을 목표로 합니다.')
