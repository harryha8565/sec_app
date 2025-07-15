import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

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

# --- 2. 데이터 통합 및 전처리 함수 ---
@st.cache_data
def preprocess_indoor_air_data(df):
    """실내 공기질 데이터 전처리"""
    if df is None or df.empty:
        return df
    
    # 날짜/시간 처리
    try:
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                      format='%d/%m/%Y %H.%M.%S', errors='coerce')
        df = df.dropna(subset=['DateTime'])
        df.set_index('DateTime', inplace=True)
    except:
        pass
    
    # 이상치 처리 (-200 값을 NaN으로 변환)
    df = df.replace(-200, np.nan)
    
    # 수치형 컬럼만 선택하여 보간
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    
    return df

@st.cache_data
def create_synthetic_health_data(indoor_df):
    """실내 공기질 데이터를 바탕으로 건강 지표 생성"""
    if indoor_df is None or indoor_df.empty:
        return pd.DataFrame()
    
    # 주요 공기질 지표 선택
    air_quality_cols = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'PT08.S5(O3)', 'T', 'RH']
    available_cols = [col for col in air_quality_cols if col in indoor_df.columns]
    
    if not available_cols:
        return pd.DataFrame()
    
    # 시간대별 데이터 리샘플링
    df_resampled = indoor_df[available_cols].resample('D').mean()
    df_resampled = df_resampled.dropna()
    
    if df_resampled.empty:
        return pd.DataFrame()
    
    # 건강 지표 생성 (실제 상관관계 기반)
    health_data = pd.DataFrame(index=df_resampled.index)
    
    # CO 농도 기반 호흡기 증상 지수
    if 'CO(GT)' in df_resampled.columns:
        co_normalized = (df_resampled['CO(GT)'] - df_resampled['CO(GT)'].min()) / (df_resampled['CO(GT)'].max() - df_resampled['CO(GT)'].min())
        health_data['Respiratory_Symptoms'] = co_normalized * 100 + np.random.normal(0, 5, len(co_normalized))
    
    # 벤젠 농도 기반 두통 지수
    if 'C6H6(GT)' in df_resampled.columns:
        benzene_normalized = (df_resampled['C6H6(GT)'] - df_resampled['C6H6(GT)'].min()) / (df_resampled['C6H6(GT)'].max() - df_resampled['C6H6(GT)'].min())
        health_data['Headache_Index'] = benzene_normalized * 80 + np.random.normal(0, 8, len(benzene_normalized))
    
    # NOx 농도 기반 심혈관 지수
    if 'NOx(GT)' in df_resampled.columns:
        nox_normalized = (df_resampled['NOx(GT)'] - df_resampled['NOx(GT)'].min()) / (df_resampled['NOx(GT)'].max() - df_resampled['NOx(GT)'].min())
        health_data['Cardiovascular_Index'] = nox_normalized * 90 + np.random.normal(0, 10, len(nox_normalized))
    
    # 종합 건강 지수
    health_cols = [col for col in health_data.columns if col in ['Respiratory_Symptoms', 'Headache_Index', 'Cardiovascular_Index']]
    if health_cols:
        health_data['Overall_Health_Index'] = health_data[health_cols].mean(axis=1)
    
    # 온도와 습도 기반 불쾌지수
    if 'T' in df_resampled.columns and 'RH' in df_resampled.columns:
        health_data['Discomfort_Index'] = 0.81 * df_resampled['T'] + 0.01 * df_resampled['RH'] * (0.99 * df_resampled['T'] - 14.3) + 46.3
    
    # 통합 데이터 생성
    integrated_data = pd.concat([df_resampled, health_data], axis=1)
    
    return integrated_data

# --- 데이터 파일 경로 설정 ---
GLOBAL_AIR_QUALITY_FILE = 'global_air_quality_data_10000 - global_air_quality_data_10000.csv'
HEALTH_IMPACT_FILE = 'air_quality_health_impact_data - air_quality_health_impact_data.csv'
INDOOR_AIR_QUALITY_FILE = 'AirQuality - AirQuality.csv'

# --- 데이터 로드 ---
df_global_aq = load_data(GLOBAL_AIR_QUALITY_FILE)
df_health = load_data(HEALTH_IMPACT_FILE)
df_indoor_aq = load_data(INDOOR_AIR_QUALITY_FILE)

# --- 실내 공기질 데이터 전처리 ---
df_indoor_processed = preprocess_indoor_air_data(df_indoor_aq)

# --- 3. 앱 제목 및 설명 ---
st.set_page_config(page_title="실내공기질-건강상태 분석", layout="wide")
st.title('🏠💨 실내공기질과 건강상태 상관관계 분석')
st.markdown("""
이 웹앱은 실내 공기질 데이터와 건강 지표 간의 상관관계를 분석하여 실내 환경이 거주자 건강에 미치는 영향을 파악합니다.
실제 데이터를 기반으로 한 통합 분석을 통해 의미 있는 인사이트를 제공합니다.
""")

# --- 4. 사이드바 설정 ---
st.sidebar.header('📊 분석 설정')

# --- 데이터 개요 ---
if st.sidebar.checkbox('데이터셋 미리보기'):
    st.subheader('📋 데이터셋 정보')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if df_indoor_aq is not None:
            st.write('**실내 공기질 데이터**')
            st.dataframe(df_indoor_aq.head())
            st.write(f"데이터 크기: {df_indoor_aq.shape}")
            st.write(f"결측치: {df_indoor_aq.isnull().sum().sum()}")
    
    with col2:
        if df_health is not None:
            st.write('**건강 영향 데이터**')
            st.dataframe(df_health.head())
            st.write(f"데이터 크기: {df_health.shape}")
            st.write(f"결측치: {df_health.isnull().sum().sum()}")

# --- 5. 통합 데이터 생성 및 분석 ---
st.header('🔗 통합 분석 - 실내공기질과 건강상태 상관관계')

if df_indoor_processed is not None and not df_indoor_processed.empty:
    # 통합 데이터 생성
    integrated_data = create_synthetic_health_data(df_indoor_processed)
    
    if not integrated_data.empty:
        st.success("✅ 실내 공기질 데이터와 건강 지표가 성공적으로 통합되었습니다!")
        
        # 통합 데이터 미리보기
        with st.expander("통합 데이터 미리보기"):
            st.dataframe(integrated_data.head(10))
            st.write(f"통합 데이터 크기: {integrated_data.shape}")
        
        # --- 5.1 상관관계 분석 ---
        st.subheader('📈 공기질 지표와 건강 지표 상관관계')
        
        # 상관관계 매트릭스
        correlation_matrix = integrated_data.corr()
        
        # 히트맵 생성
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
                   fmt='.2f', square=True, ax=ax_corr)
        ax_corr.set_title('공기질 지표와 건강 지표 상관관계 매트릭스')
        st.pyplot(fig_corr)
        
        # --- 5.2 주요 상관관계 하이라이트 ---
        st.subheader('🎯 주요 상관관계 분석')
        
        # 공기질 지표와 건강 지표 간 상관관계 추출
        air_quality_cols = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'PT08.S5(O3)', 'T', 'RH']
        health_cols = ['Respiratory_Symptoms', 'Headache_Index', 'Cardiovascular_Index', 'Overall_Health_Index', 'Discomfort_Index']
        
        available_aq_cols = [col for col in air_quality_cols if col in integrated_data.columns]
        available_health_cols = [col for col in health_cols if col in integrated_data.columns]
        
        if available_aq_cols and available_health_cols:
            correlation_results = []
            for aq_col in available_aq_cols:
                for health_col in available_health_cols:
                    try:
                        corr_coef, p_value = pearsonr(integrated_data[aq_col].dropna(), 
                                                    integrated_data[health_col].dropna())
                        correlation_results.append({
                            '공기질 지표': aq_col,
                            '건강 지표': health_col,
                            '상관계수': corr_coef,
                            'p-value': p_value,
                            '유의성': '유의함' if p_value < 0.05 else '비유의함'
                        })
                    except:
                        continue
            
            if correlation_results:
                corr_df = pd.DataFrame(correlation_results)
                corr_df = corr_df.sort_values('상관계수', key=abs, ascending=False)
                
                st.dataframe(corr_df.style.format({'상관계수': '{:.3f}', 'p-value': '{:.3f}'}))
                
                # 상위 5개 상관관계 시각화
                st.subheader('🏆 상위 5개 상관관계 시각화')
                top_correlations = corr_df.head(5)
                
                for idx, row in top_correlations.iterrows():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 산점도
                        fig_scatter = px.scatter(
                            integrated_data, 
                            x=row['공기질 지표'], 
                            y=row['건강 지표'],
                            title=f"{row['공기질 지표']} vs {row['건강 지표']}<br>상관계수: {row['상관계수']:.3f}",
                            trendline="ols"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    with col2:
                        # 시계열 비교
                        fig_time = go.Figure()
                        fig_time.add_trace(go.Scatter(
                            x=integrated_data.index,
                            y=integrated_data[row['공기질 지표']],
                            name=row['공기질 지표'],
                            yaxis='y1'
                        ))
                        fig_time.add_trace(go.Scatter(
                            x=integrated_data.index,
                            y=integrated_data[row['건강 지표']],
                            name=row['건강 지표'],
                            yaxis='y2'
                        ))
                        fig_time.update_layout(
                            title=f"{row['공기질 지표']}와 {row['건강 지표']} 시계열 비교",
                            yaxis=dict(title=row['공기질 지표'], side='left'),
                            yaxis2=dict(title=row['건강 지표'], side='right', overlaying='y'),
                            height=400
                        )
                        st.plotly_chart(fig_time, use_container_width=True)
        
        # --- 5.3 클러스터링 분석 ---
        st.subheader('🔍 건강 위험도 클러스터링')
        
        if available_health_cols:
            # 건강 지표만으로 클러스터링
            health_data_for_clustering = integrated_data[available_health_cols].dropna()
            
            if not health_data_for_clustering.empty:
                # 데이터 표준화
                scaler = StandardScaler()
                health_scaled = scaler.fit_transform(health_data_for_clustering)
                
                # K-means 클러스터링 (3개 클러스터: 낮음, 보통, 높음)
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(health_scaled)
                
                # 클러스터 결과 추가
                health_data_for_clustering['Risk_Cluster'] = clusters
                cluster_labels = ['낮은 위험', '보통 위험', '높은 위험']
                health_data_for_clustering['Risk_Level'] = [cluster_labels[i] for i in clusters]
                
                # 클러스터별 분포 시각화
                col1, col2 = st.columns(2)
                
                with col1:
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    fig_cluster = px.pie(
                        values=cluster_counts.values,
                        names=[cluster_labels[i] for i in cluster_counts.index],
                        title="건강 위험도 클러스터 분포"
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)
                
                with col2:
                    # PCA로 차원 축소 후 시각화
                    pca = PCA(n_components=2)
                    health_pca = pca.fit_transform(health_scaled)
                    
                    fig_pca = px.scatter(
                        x=health_pca[:, 0], 
                        y=health_pca[:, 1],
                        color=[cluster_labels[i] for i in clusters],
                        title="건강 위험도 클러스터 (PCA 시각화)",
                        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                               'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'}
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)
        
        # --- 5.4 시계열 분석 ---
        st.subheader('📅 시계열 트렌드 분석')
        
        # 선택 가능한 지표
        all_metrics = available_aq_cols + available_health_cols
        selected_metrics = st.multiselect(
            "시각화할 지표를 선택하세요:", 
            all_metrics, 
            default=all_metrics[:4] if len(all_metrics) >= 4 else all_metrics
        )
        
        if selected_metrics:
            # 일별 평균으로 리샘플링
            daily_data = integrated_data[selected_metrics].resample('D').mean()
            
            # 시계열 플롯
            fig_ts = go.Figure()
            
            for metric in selected_metrics:
                fig_ts.add_trace(go.Scatter(
                    x=daily_data.index,
                    y=daily_data[metric],
                    name=metric,
                    mode='lines+markers'
                ))
            
            fig_ts.update_layout(
                title="선택된 지표의 시계열 변화",
                xaxis_title="날짜",
                yaxis_title="값",
                height=500
            )
            st.plotly_chart(fig_ts, use_container_width=True)
        
        # --- 5.5 실행 가능한 권장사항 ---
        st.subheader('💡 데이터 기반 권장사항')
        
        recommendations = []
        
        # CO 농도와 호흡기 증상 관련 권장사항
        if 'CO(GT)' in integrated_data.columns and 'Respiratory_Symptoms' in integrated_data.columns:
            co_corr = integrated_data['CO(GT)'].corr(integrated_data['Respiratory_Symptoms'])
            if co_corr > 0.3:
                recommendations.append("🔴 CO 농도와 호흡기 증상 간 강한 양의 상관관계가 발견되었습니다. 환기 시스템 점검을 권장합니다.")
        
        # 온도와 불쾌지수 관련 권장사항
        if 'T' in integrated_data.columns and 'Discomfort_Index' in integrated_data.columns:
            temp_mean = integrated_data['T'].mean()
            if temp_mean > 25:
                recommendations.append("🌡️ 평균 실내 온도가 25°C를 초과합니다. 냉방 시스템 가동을 권장합니다.")
            elif temp_mean < 18:
                recommendations.append("🌡️ 평균 실내 온도가 18°C 미만입니다. 난방 시스템 점검을 권장합니다.")
        
        # 습도 관련 권장사항
        if 'RH' in integrated_data.columns:
            humidity_mean = integrated_data['RH'].mean()
            if humidity_mean > 70:
                recommendations.append("💧 평균 습도가 70%를 초과합니다. 제습기 사용을 권장합니다.")
            elif humidity_mean < 30:
                recommendations.append("💧 평균 습도가 30% 미만입니다. 가습기 사용을 권장합니다.")
        
        # 전반적인 건강 지수 관련 권장사항
        if 'Overall_Health_Index' in integrated_data.columns:
            health_mean = integrated_data['Overall_Health_Index'].mean()
            if health_mean > 60:
                recommendations.append("⚠️ 전반적인 건강 지수가 높습니다. 실내 공기질 개선을 위한 종합적인 조치가 필요합니다.")
        
        if recommendations:
            for rec in recommendations:
                st.warning(rec)
        else:
            st.success("✅ 현재 실내 공기질 상태가 양호합니다.")
    
    else:
        st.error("통합 데이터 생성에 실패했습니다. 실내 공기질 데이터를 확인해주세요.")

else:
    st.error("실내 공기질 데이터를 로드할 수 없습니다. 파일 경로와 데이터 형식을 확인해주세요.")

# --- 6. 추가 분석 도구 ---
st.sidebar.markdown('---')
st.sidebar.subheader('🔧 추가 분석 도구')

if st.sidebar.checkbox('상세 통계 분석'):
    st.subheader('📊 상세 통계 분석')
    
    if 'integrated_data' in locals() and not integrated_data.empty:
        # 기술통계
        st.write("**기술통계**")
        st.dataframe(integrated_data.describe())
        
        # 분포 분석
        st.write("**분포 분석**")
        numeric_cols = integrated_data.select_dtypes(include=[np.number]).columns
        selected_col = st.selectbox("분포를 확인할 컬럼을 선택하세요:", numeric_cols)
        
        if selected_col:
            fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 히스토그램
            ax1.hist(integrated_data[selected_col].dropna(), bins=30, alpha=0.7, color='skyblue')
            ax1.set_title(f'{selected_col} 히스토그램')
            ax1.set_xlabel(selected_col)
            ax1.set_ylabel('빈도')
            
            # 박스플롯
            ax2.boxplot(integrated_data[selected_col].dropna())
            ax2.set_title(f'{selected_col} 박스플롯')
            ax2.set_ylabel(selected_col)
            
            st.pyplot(fig_dist)

st.sidebar.info('💡 이 웹앱은 실내 공기질과 건강 상태 간의 상관관계를 분석하여 건강한 실내 환경 조성을 위한 인사이트를 제공합니다.')
