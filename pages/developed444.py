# 주요 라이브러리 임포트
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import pearsonr
import warnings

# 불필요한 경고 메시지 숨김
warnings.filterwarnings('ignore')

# --- 기본 설정 ---
st.set_page_config(page_title="실내 공기질과 건강 분석", layout="wide")
st.title("🏠 실내 공기질과 건강 지표 분석 대시보드")

# --- 데이터 로드 함수 ---
@st.cache_data
def load_data(file_path):
    """
    CSV 파일을 불러오는 함수입니다.
    오류가 발생할 경우 None을 반환합니다.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"❌ 파일을 불러오는 중 오류가 발생했습니다: {e}")
        return None

# --- 실내 공기질 전처리 함수 ---
@st.cache_data
def preprocess_indoor_air_data(df):
    """
    날짜와 시간 데이터를 처리하고, 누락값을 보정한 후 시간 인덱스로 설정합니다.
    """
    if df is None or df.empty:
        return None
    try:
        # 날짜와 시간을 결합해 DateTime 컬럼 생성
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
        elif 'datetime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'timestamp' in df.columns:
            df['DateTime'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            # 시간 정보가 없을 경우, 기본 시간 인덱스 생성
            df['DateTime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')

        df.dropna(subset=['DateTime'], inplace=True)
        df.set_index('DateTime', inplace=True)

        # -200 등 오류값을 NaN으로 처리 후 보간법으로 채움
        df.replace(-200, np.nan, inplace=True)
        df.interpolate(method='linear', inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ 전처리 과정에서 오류가 발생했습니다: {e}")
        return None

# --- 건강 지표 생성 함수 ---
@st.cache_data
def create_synthetic_health_data(df):
    """
    공기질 데이터 기반으로 인공 건강 지표 생성
    - 호흡기 증상
    - 두통 지수
    - 심혈관 지수
    - 종합 건강 지수
    """
    try:
        if df is None or df.empty:
            return None

        aq_cols = [col for col in df.columns if df[col].dtype != 'object'][:7]
        df_daily = df[aq_cols].resample('D').mean().dropna()

        health_data = pd.DataFrame(index=df_daily.index)

        if len(aq_cols) > 0:
            norm = (df_daily[aq_cols[0]] - df_daily[aq_cols[0]].min()) / (df_daily[aq_cols[0]].max() - df_daily[aq_cols[0]].min())
            health_data['Respiratory_Symptoms'] = norm * 100 + np.random.normal(0, 5, len(norm))

        if len(aq_cols) > 1:
            norm = (df_daily[aq_cols[1]] - df_daily[aq_cols[1]].min()) / (df_daily[aq_cols[1]].max() - df_daily[aq_cols[1]].min())
            health_data['Headache_Index'] = norm * 80 + np.random.normal(0, 8, len(norm))

        if len(aq_cols) > 2:
            norm = (df_daily[aq_cols[2]] - df_daily[aq_cols[2]].min()) / (df_daily[aq_cols[2]].max() - df_daily[aq_cols[2]].min())
            health_data['Cardiovascular_Index'] = norm * 90 + np.random.normal(0, 10, len(norm))

        # 종합 건강지수 = 위 세 지표의 평균
        health_data['Overall_Health_Index'] = health_data.mean(axis=1)

        return pd.concat([df_daily, health_data], axis=1)
    except Exception as e:
        st.error(f"❌ 건강 지표 생성 중 오류 발생: {e}")
        return None

# --- 파일 경로 ---
INDOOR_AIR_QUALITY_FILE = 'AirQuality - AirQuality.csv'

# --- 데이터 로딩 및 처리 ---
df_raw = load_data(INDOOR_AIR_QUALITY_FILE)
df_processed = preprocess_indoor_air_data(df_raw)
df_integrated = create_synthetic_health_data(df_processed)

# --- 사이드바 메뉴 ---
menu = st.sidebar.radio("📚 분석 메뉴를 선택하세요", ["1. 데이터 개요", "2. 시계열 변화", "3. 상관관계 분석", "4. 분포 분석 및 필터링"])

# --- 데이터 정상일 경우만 진행 ---
if df_integrated is not None:
    numeric_df = df_integrated.select_dtypes(include=np.number)
    aq_cols = [c for c in numeric_df.columns if 'Index' not in c and 'Symptoms' not in c]
    health_cols = [c for c in numeric_df.columns if c not in aq_cols]

    if menu == "1. 데이터 개요":
        st.subheader("📋 데이터 미리보기 및 요약")
        st.write("아래는 통합된 공기질 + 건강지표 데이터의 상위 5개 행입니다.")
        st.dataframe(df_integrated.head())
        st.write("데이터 요약 통계:")
        st.write(df_integrated.describe())

    elif menu == "2. 시계열 변화":
        st.subheader("📅 시간에 따른 공기질 및 건강지표 변화")
        st.write("원하는 지표를 선택해 시계열 그래프로 확인해보세요.")
        selected = st.multiselect("시계열로 볼 변수 선택", df_integrated.columns, default=['Overall_Health_Index'])
        fig = px.line(df_integrated, x=df_integrated.index, y=selected, title="시계열 변화")
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "3. 상관관계 분석":
        st.subheader("🔗 건강과 공기질 변수 간 상관관계 분석")
        st.write("전체 수치 데이터 간의 상관관계를 분석합니다.")
        corr_matrix = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.markdown("### 🔍 가장 밀접한 변수 쌍 TOP 분석")
        result = []
        for a in aq_cols:
            for h in health_cols:
                try:
                    corr, _ = pearsonr(numeric_df[a].dropna(), numeric_df[h].dropna())
                    result.append((a, h, corr))
                except:
                    continue
        top_corr = sorted(result, key=lambda x: abs(x[2]), reverse=True)
        st.dataframe(pd.DataFrame(top_corr, columns=["공기질 변수", "건강 지표", "상관계수"]).round(2))

    elif menu == "4. 분포 분석 및 필터링":
        st.subheader("📊 변수별 분포 확인 및 범위 필터링")
        col = st.selectbox("분포를 확인할 변수", numeric_df.columns)
        fig = px.histogram(numeric_df, x=col, nbins=30, title=f"{col}의 분포")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🎯 값 범위 선택하여 필터링")
        value_range = st.slider(
            f"{col} 값 범위 선택",
            float(numeric_df[col].min()),
            float(numeric_df[col].max()),
            (float(numeric_df[col].min()), float(numeric_df[col].max()))
        )
        filtered_df = df_integrated[(numeric_df[col] >= value_range[0]) & (numeric_df[col] <= value_range[1])]
        st.write(f"🔎 선택된 범위 내 데이터 개수: {len(filtered_df)}")
        st.dataframe(filtered_df.head())

else:
    st.error("❗ 데이터 로딩 또는 처리에 실패했습니다. CSV 파일 경로 또는 파일 형식을 확인해주세요.")
