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

# --- 데이터 로드 함수 ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
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

# --- 데이터 로드 및 처리 ---
df_raw = load_data(INDOOR_AIR_QUALITY_FILE)
df_processed = preprocess_indoor_air_data(df_raw)
df_integrated = create_synthetic_health_data(df_processed)

# --- Streamlit 앱 ---
st.set_page_config(page_title="실내 공기질과 건강 분석", layout="wide")
st.title("🏠 실내 공기질과 건강 지표 분석")

if df_integrated is not None:
    st.success("데이터가 성공적으로 로드 및 통합되었습니다!")
    st.dataframe(df_integrated.head())

    st.subheader("📈 상관관계 분석")
    numeric_df = df_integrated.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("📉 주요 상관관계 분석")
    health_cols = [c for c in numeric_df.columns if 'Index' in c or 'Symptoms' in c]
    aq_cols = [c for c in numeric_df.columns if c not in health_cols]
    result = []
    for a in aq_cols:
        for h in health_cols:
            try:
                corr, p = pearsonr(numeric_df[a].dropna(), numeric_df[h].dropna())
                result.append((a, h, corr))
            except:
                continue
    top_corr = sorted(result, key=lambda x: abs(x[2]), reverse=True)[:3]
    for a, h, c in top_corr:
        st.markdown(f"**{a} vs {h}** (상관계수: {c:.2f})")
        fig2 = px.scatter(df_integrated, x=a, y=h, trendline="ols")
        st.plotly_chart(fig2)
else:
    st.error("데이터 처리에 실패했습니다. 파일을 확인해주세요.")
