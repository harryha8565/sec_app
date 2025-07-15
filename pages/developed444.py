# 라이브러리 불러오기
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(page_title="실내 공기질 분석", layout="wide")
st.title("🏠 실내 공기질과 건강 지표 분석 대시보드")

# CSV 로드 함수
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"❌ 파일 로드 실패: {e}")
        return None

# 전처리 함수
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
        st.error(f"❌ 전처리 오류: {e}")
        return None

# 건강 지표 생성 함수
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
        st.error(f"❌ 건강 지표 생성 실패: {e}")
        return None

# AI 해석 함수
def interpret_correlation(corr):
    """Pearson 상관계수 해석"""
    if abs(corr) >= 0.7:
        strength = "강한"
    elif abs(corr) >= 0.4:
        strength = "중간 정도의"
    elif abs(corr) >= 0.2:
        strength = "약한"
    else:
        strength = "매우 약한"
    direction = "양의 상관관계" if corr > 0 else "음의 상관관계"
    return f"➡️ 이 변수들은 **{strength} {direction}**를 보입니다. (상관계수: {corr:.2f})"

# 데이터 불러오기
file_path = 'AirQuality - AirQuality.csv'
df_raw = load_data(file_path)
df_processed = preprocess_indoor_air_data(df_raw)
df_integrated = create_synthetic_health_data(df_processed)

# 메뉴 설정
menu = st.sidebar.radio("📚 메뉴 선택", ["📋 데이터 개요", "📈 시계열 분석", "🔗 상관관계 분석", "📊 분포 분석"])

if df_integrated is not None:
    numeric_df = df_integrated.select_dtypes(include=np.number)
    aq_cols = [c for c in numeric_df.columns if 'Index' not in c and 'Symptoms' not in c]
    health_cols = [c for c in numeric_df.columns if c not in aq_cols]

    if menu == "📋 데이터 개요":
        st.subheader("데이터 미리보기")
        st.markdown("아래는 전처리된 공기질 데이터와 인공 건강 지표 데이터입니다.")
        st.dataframe(df_integrated.head())
        st.write(df_integrated.describe())

    elif menu == "📈 시계열 분석":
        st.subheader("시간에 따른 지표 변화")
        st.markdown("선택한 변수의 시계열 변화를 확인할 수 있습니다.")
        selected = st.multiselect("시계열로 볼 변수 선택", df_integrated.columns, default=["Overall_Health_Index"])
        fig = px.line(df_integrated, x=df_integrated.index, y=selected)
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "🔗 상관관계 분석":
        st.subheader("상관관계 히트맵")
        st.markdown("모든 수치형 변수 간의 상관관계를 색으로 나타냅니다.")
        corr_matrix = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.markdown("### 공기질 변수와 건강지표 간 상관관계")
        result = []
        for a in aq_cols:
            for h in health_cols:
                try:
                    corr, _ = pearsonr(numeric_df[a].dropna(), numeric_df[h].dropna())
                    result.append((a, h, corr))
                except:
                    continue
        sorted_corr = sorted(result, key=lambda x: abs(x[2]), reverse=True)

        for a, h, c in sorted_corr[:5]:
            st.markdown(f"**{a} ↔ {h}**")
            st.write(interpret_correlation(c))
            fig2 = px.scatter(df_integrated, x=a, y=h, trendline="ols", title=f"{a} vs {h}")
            st.plotly_chart(fig2)

    elif menu == "📊 분포 분석":
        st.subheader("변수 분포 확인")
        col = st.selectbox("분포를 볼 변수", numeric_df.columns)
        st.markdown(f"선택한 변수 **{col}** 의 히스토그램입니다.")
        fig = px.histogram(numeric_df, x=col, nbins=30)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("값 범위를 지정해 필터링할 수 있습니다.")
        min_val = float(numeric_df[col].min())
        max_val = float(numeric_df[col].max())
        val_range = st.slider("값 범위", min_val, max_val, (min_val, max_val))
        filtered = df_integrated[(numeric_df[col] >= val_range[0]) & (numeric_df[col] <= val_range[1])]
        st.write(f"선택된 범위에 포함된 데이터: {len(filtered)}개")
        st.dataframe(filtered.head())
else:
    st.error("❗ 데이터 파일을 불러오지 못했습니다. 경로 또는 파일 내용을 확인해주세요.")
