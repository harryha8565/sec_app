import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="실내 공기질과 건강 분석", layout="wide")
st.title("🏠 실내 공기질과 건강 지표 분석 대시보드")

# --- 데이터 로드 함수 ---
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return None

# --- 실내 공기질 전처리 함수 ---
@st.cache_data
def preprocess_indoor_air_data(df):
    if df is None or df.empty:
        return None
    try:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
        elif 'datetime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'timestamp' in df.columns:
            df['DateTime'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            df['DateTime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
        
        df.dropna(subset=['DateTime'], inplace=True)
        df.set_index('DateTime', inplace=True)
        df.replace(-200, np.nan, inplace=True)
        df.interpolate(method='linear', inplace=True)
        return df
    except Exception as e:
        st.error(f"전처리 오류: {e}")
        return None

# --- 건강 지표 생성 함수 ---
@st.cache_data
def create_synthetic_health_data(df):
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

        health_data['Overall_Health_Index'] = health_data.mean(axis=1)
        return pd.concat([df_daily, health_data], axis=1)
    except Exception as e:
        st.error(f"건강 지표 생성 오류: {e}")
        return None

# --- 파일 경로 ---
INDOOR_AIR_QUALITY_FILE = 'AirQuality - AirQuality.csv'

# --- 데이터 처리 ---
df_raw = load_data(INDOOR_AIR_QUALITY_FILE)
df_processed = preprocess_indoor_air_data(df_raw)
df_integrated = create_synthetic_health_data(df_processed)

# --- 페이지 탐색 ---
menu = st.sidebar.radio("메뉴 선택", ["데이터 개요", "시계열 분석", "상관관계 분석", "분포 및 필터링"])

# --- 콘텐츠 영역 ---
if df_integrated is not None:
    numeric_df = df_integrated.select_dtypes(include=np.number)
    aq_cols = [c for c in numeric_df.columns if 'Index' not in c and 'Symptoms' not in c]
    health_cols = [c for c in numeric_df.columns if c not in aq_cols]

    if menu == "데이터 개요":
        st.subheader("📋 데이터 샘플과 통계 요약")
        st.write(df_integrated.head())
        st.write(df_integrated.describe())
        st.markdown("데이터는 실내 공기질 센서로부터 얻은 다양한 오염물질 농도와 이를 바탕으로 계산된 건강 지표로 구성됩니다.")

    elif menu == "시계열 분석":
        st.subheader("📅 시계열 변화 시각화")
        selected = st.multiselect("분석할 지표 선택", df_integrated.columns, default=['Overall_Health_Index'])
        fig = px.line(df_integrated, x=df_integrated.index, y=selected, title="시간에 따른 변화")
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "상관관계 분석":
        st.subheader("🔗 건강 지표와 공기질 변수 간 상관관계")
        corr_matrix = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.markdown("### 📌 상위 상관 변수 목록")
        result = []
        for a in aq_cols:
            for h in health_cols:
                try:
                    corr, p = pearsonr(numeric_df[a].dropna(), numeric_df[h].dropna())
                    result.append((a, h, corr))
                except:
                    continue
        top_corr = sorted(result, key=lambda x: abs(x[2]), reverse=True)
        st.dataframe(pd.DataFrame(top_corr, columns=["공기질 변수", "건강 지표", "상관계수"]).round(2))

    elif menu == "분포 및 필터링":
        st.subheader("📊 히스토그램 및 필터링")
        col = st.selectbox("분포를 확인할 변수", numeric_df.columns)
        fig = px.histogram(numeric_df, x=col, nbins=30, title=f"{col} 분포")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 필터링 후 값 시각화")
        value_range = st.slider(f"{col} 값 범위 선택", float(numeric_df[col].min()), float(numeric_df[col].max()), 
                                (float(numeric_df[col].min()), float(numeric_df[col].max())))
        filtered_df = df_integrated[(numeric_df[col] >= value_range[0]) & (numeric_df[col] <= value_range[1])]
        st.write(f"선택된 범위 내 데이터 개수: {len(filtered_df)}")
        st.dataframe(filtered_df.head())

else:
    st.error("❗ 데이터 로딩 또는 전처리에 실패했습니다. CSV 파일 경로를 확인해주세요.")
