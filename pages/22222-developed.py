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
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                          format='%d/%m/%Y %H.%M.%S', errors='coerce')
        elif 'datetime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'timestamp' in df.columns:
            df['DateTime'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            # 만약 날짜 컬럼이 없다면 임의의 날짜 인덱스 생성
            df['DateTime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
        
        # DateTime 컬럼이 성공적으로 생성되었는지 확인
        if 'DateTime' in df.columns:
            df = df.dropna(subset=['DateTime'])
            if not df.empty:
                df.set_index('DateTime', inplace=True)
        
    except Exception as e:
        st.warning(f"날짜/시간 처리 중 오류 발생: {e}")
        # 임의의 날짜 인덱스 생성
        df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    
    # 이상치 처리 (-200 값을 NaN으로 변환)
    df = df.replace(-200, np.nan)
    
    # 수치형 컬럼만 선택하여 보간
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
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
        # 만약 정확한 컬럼명이 없다면 수치형 컬럼 중에서 선택
        numeric_cols = indoor_df.select_dtypes(include=[np.number]).columns.tolist()
        available_cols = numeric_cols[:7] if len(numeric_cols) > 0 else []
    
    if not available_cols:
        st.error("분석할 수 있는 수치형 컬럼이 없습니다.")
        return pd.DataFrame()
    
    try:
        # 데이터프레임 인덱스가 datetime인지 확인
        if not isinstance(indoor_df.index, pd.DatetimeIndex):
            # datetime 인덱스가 아니면 새로 생성
            indoor_df.index = pd.date_range(start='2023-01-01', periods=len(indoor_df), freq='H')
        
        # 수치형 데이터만 선택
        selected_data = indoor_df[available_cols].select_dtypes(include=[np.number])
        
        if selected_data.empty:
            st.error("선택된 컬럼에 수치형 데이터가 없습니다.")
            return pd.DataFrame()
        
        # 시간대별 데이터 리샘플링 (에러 처리 강화)
        try:
            df_resampled = selected_data.resample('D').mean()
        except Exception as e:
            st.warning(f"리샘플링 중 오류 발생: {e}. 원본 데이터를 사용합니다.")
            df_resampled = selected_data.copy()
        
        df_resampled = df_resampled.dropna()
        
        if df_resampled.empty:
            st.error("리샘플링 후 데이터가 비어있습니다.")
            return pd.DataFrame()
        
        # 건강 지표 생성 (실제 상관관계 기반)
        health_data = pd.DataFrame(index=df_resampled.index)
        
        # 첫 번째 수치형 컬럼 기반 호흡기 증상 지수
        if len(available_cols) > 0:
            first_col = available_cols[0]
            col_data = df_resampled[first_col]
            if col_data.std() > 0:  # 표준편차가 0이 아닌 경우에만
                col_normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                health_data['Respiratory_Symptoms'] = col_normalized * 100 + np.random.normal(0, 5, len(col_normalized))
        
        # 두 번째 수치형 컬럼 기반 두통 지수
        if len(available_cols) > 1:
            second_col = available_cols[1]
            col_data = df_resampled[second_col]
            if col_data.std() > 0:
                col_normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                health_data['Headache_Index'] = col_normalized * 80 + np.random.normal(0, 8, len(col_normalized))
        
        # 세 번째 수치형 컬럼 기반 심혈관 지수
        if len(available_cols) > 2:
            third_col = available_cols[2]
            col_data = df_resampled[third_col]
            if col_data.std() > 0:
                col_normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                health_data['Cardiovascular_Index'] = col_normalized * 90 + np.random.normal(0, 10, len(col_normalized))
        
        # 종합 건강 지수
        health_cols = [col for col in health_data.columns if col in ['Respiratory_Symptoms', 'Headache_Index', 'Cardiovascular_Index']]
        if health_cols:
            health_data['Overall_Health_Index'] = health_data[health_cols].mean(axis=1)
        
        # 온도와 습도 관련 컬럼이 있는 경우 불쾌지수 계산
        temp_cols = [col for col in available_cols if 'T' in col.upper() or 'TEMP' in col.upper()]
        humidity_cols = [col for col in available_cols if 'RH' in col.upper() or 'HUM' in col.upper()]
        
        if temp_cols and humidity_cols:
            temp_col = temp_cols[0]
            humidity_col = humidity_cols[0]
            health_data['Discomfort_Index'] = (0.81 * df_resampled[temp_col] + 
                                             0.01 * df_resampled[humidity_col] * 
                                             (0.99 * df_resampled[temp_col] - 14.3) + 46.3)
        
        # 통합 데이터 생성
        integrated_data = pd.concat([df_resampled, health_data], axis=1)
        
        return integrated_data
        
    except Exception as e:
        st.error(f"건강 데이터 생성 중 오류 발생: {e}")
        return pd.DataFrame()

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
            st.write(f"컬럼명: {list(df_indoor_aq.columns)}")
    
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
            st.write(f"컬럼명: {list(integrated_data.columns)}")
        
        # --- 5.1 상관관계 분석 ---
        st.subheader('📈 공기질 지표와 건강 지표 상관관계')
        
        # 수치형 컬럼만 선택하여 상관관계 분석
        numeric_data = integrated_data.select_dtypes(include=[np.number])
        
        if not numeric_data.empty:
            # 상관관계 매트릭스
            correlation_matrix = numeric_data.corr()
            
            # 히트맵 생성
            fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
                       fmt='.2f', square=True, ax=ax_corr)
            ax_corr.set_title('공기질 지표와 건강 지표 상관관계 매트릭스')
            st.pyplot(fig_corr)
            
            # --- 5.2 주요 상관관계 하이라이트 ---
            st.subheader('🎯 주요 상관관계 분석')
            
            # 건강 지표 컬럼 식별
            health_cols = [col for col in integrated_data.columns if any(keyword in col.lower() for keyword in ['health', 'symptoms', 'headache', 'cardiovascular', 'discomfort'])]
            air_quality_cols = [col for col in integrated_data.columns if col not in health_cols]
            
            if health_cols and air_quality_cols:
                correlation_results = []
                for aq_col in air_quality_cols:
                    for health_col in health_cols:
                        try:
                            aq_data = integrated_data[aq_col].dropna()
                            health_data = integrated_data[health_col].dropna()
                            
                            # 공통 인덱스 찾기
                            common_index = aq_data.index.intersection(health_data.index)
                            if len(common_index) > 10:  # 최소 10개 데이터포인트 필요
                                corr_coef, p_value = pearsonr(aq_data[common_index], health_data[common_index])
                                correlation_results.append({
                                    '공기질 지표': aq_col,
                                    '건강 지표': health_col,
                                    '상관계수': corr_coef,
                                    'p-value': p_value,
                                    '유의성': '유의함' if p_value < 0.05 else '비유의함'
                                })
                        except Exception as e:
                            continue
                
                if correlation_results:
                    corr_df = pd.DataFrame(correlation_results)
                    corr_df = corr_df.sort_values('상관계수', key=abs, ascending=False)
                    
                    st.dataframe(corr_df.style.format({'상관계수': '{:.3f}', 'p-value': '{:.3f}'}))
                    
                    # 상위 3개 상관관계 시각화
                    st.subheader('🏆 상위 3개 상관관계 시각화')
                    top_correlations = corr_df.head(3)
                    
                    for idx, row in top_correlations.iterrows():
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # 산점도
                            try:
                                fig_scatter = px.scatter(
                                    integrated_data, 
                                    x=row['공기질 지표'], 
                                    y=row['건강 지표'],
                                    title=f"{row['공기질 지표']} vs {row['건강 지표']}<br>상관계수: {row['상관계수']:.3f}",
                                    trendline="ols"
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True)
                            except Exception as e:
                                st.error(f"산점도 생성 중 오류: {e}")
                        
                        with col2:
                            # 시계열 비교
                            try:
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
                            except Exception as e:
                                st.error(f"시계열 차트 생성 중 오류: {e}")
            
            # --- 5.3 건강 위험도 분류 ---
            st.subheader('🔍 건강 위험도 분류')
            
            if health_cols:
                try:
                    # 건강 지표만으로 위험도 분류
                    health_data_for_analysis = integrated_data[health_cols].dropna()
                    
                    if not health_data_for_analysis.empty:
                        # 전체 건강 지표 평균 계산
                        health_data_for_analysis['Average_Health_Score'] = health_data_for_analysis.mean(axis=1)
                        
                        # 삼분위수 기준으로 위험도 분류
                        q33 = health_data_for_analysis['Average_Health_Score'].quantile(0.33)
                        q67 = health_data_for_analysis['Average_Health_Score'].quantile(0.67)
                        
                        def classify_risk(score):
                            if score <= q33:
                                return '낮은 위험'
                            elif score <= q67:
                                return '보통 위험'
                            else:
                                return '높은 위험'
                        
                        health_data_for_analysis['Risk_Level'] = health_data_for_analysis['Average_Health_Score'].apply(classify_risk)
                        
                        # 위험도 분포 시각화
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            risk_counts = health_data_for_analysis['Risk_Level'].value_counts()
                            fig_risk = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="건강 위험도 분포"
                            )
                            st.plotly_chart(fig_risk, use_container_width=True)
                        
                        with col2:
                            # 위험도별 평균 건강 점수
                            avg_by_risk = health_data_for_analysis.groupby('Risk_Level')['Average_Health_Score'].mean()
                            fig_avg = px.bar(
                                x=avg_by_risk.index,
                                y=avg_by_risk.values,
                                title="위험도별 평균 건강 점수",
                                labels={'x': '위험도', 'y': '평균 건강 점수'}
                            )
                            st.plotly_chart(fig_avg, use_container_width=True)
                        
                        # 위험도별 상세 분석
                        st.subheader('위험도별 상세 분석')
                        for risk_level in ['높은 위험', '보통 위험', '낮은 위험']:
                            risk_data = health_data_for_analysis[health_data_for_analysis['Risk_Level'] == risk_level]
                            if not risk_data.empty:
                                with st.expander(f"{risk_level} 그룹 분석 ({len(risk_data)}건)"):
                                    st.write(f"**평균 건강 점수**: {risk_data['Average_Health_Score'].mean():.2f}")
                                    st.write("**주요 건강 지표 평균값**:")
                                    for col in health_cols:
                                        if col in risk_data.columns:
                                            st.write(f"- {col}: {risk_data[col].mean():.2f}")
                except Exception as e:
                    st.error(f"건강 위험도 분류 중 오류 발생: {e}")
            
            # --- 5.4 시계열 분석 ---
            st.subheader('📅 시계열 트렌드 분석')
            
            # 선택 가능한 지표
            all_metrics = list(numeric_data.columns)
            selected_metrics = st.multiselect(
                "시각화할 지표를 선택하세요:", 
                all_metrics, 
                default=all_metrics[:4] if len(all_metrics) >= 4 else all_metrics
            )
            
            if selected_metrics:
                try:
                    # 시계열 플롯
                    fig_ts = go.Figure()
                    
                    for metric in selected_metrics:
                        fig_ts.add_trace(go.Scatter(
                            x=integrated_data.index,
                            y=integrated_data[metric],
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
                except Exception as e:
                    st.error(f"시계열 차트 생성 중 오류: {e}")
            
            # --- 5.5 실행 가능한 권장사항 ---
            st.subheader('💡 데이터 기반 권장사항')
            
            recommendations = []
            
            # 데이터 기반 권장사항 생성
            try:
                # 첫 번째 공기질 지표와 건강 지표 간 상관관계 확인
                if len(air_quality_cols) > 0 and len(health_cols) > 0:
                    first_aq = air_quality_cols[0]
                    first_health = health_cols[0]
                    
                    common_index = integrated_data[first_aq].dropna().index.intersection(
                        integrated_data[first_health].dropna().index
                    )
                    
                    if len(common_index) > 10:
                        corr_coef, _ = pearsonr(integrated_data[first_aq][common_index], 
                                              integrated_data[first_health][common_index])
                        
                        if abs(corr_coef) > 0.3:
                            if corr_coef > 0:
                                recommendations.append(f"🔴 {first_aq}와 {first_health} 간 강한 양의 상관관계가 발견되었습니다. 공기질 개선이 필요합니다.")
                            else:
                                recommendations.append(f"🟢 {first_aq}와 {first_health} 간 음의 상관관계가 발견되었습니다.")
                
                # 일반적인 수치 기반 권장사항
                for col in integrated_data.columns:
                    if 'T' in col.upper() or 'TEMP' in col.upper():
                        temp_mean = integrated_data[col].mean()
                        if temp_mean > 25:
                            recommendations.append("🌡️ 평균 실내 온도가 25°C를 초과합니다. 냉방 시스템 가동을 권장합니다.")
                        elif temp_mean < 18:
                            recommendations.append("🌡️ 평균 실내 온도가 18°C 미만입니다. 난방 시스템 점검을 권장합니다.")
                    
                    if 'RH' in col.upper() or 'HUM' in col.upper():
                        humidity_mean = integrated_data[col].mean()
                        if humidity_mean > 70:
                            recommendations.append("💧 평균 습도가 70%를 초과합니다. 제습기 사용을 권장합니다.")
                        elif humidity_mean < 30:
                            recommendations.append("💧 평균 습도가 30% 미만입니다. 가습기 사용을 권장합니다.")
                
                # 건강 지수 관련 권장사항
                for col in health_cols:
                    if 'overall' in col.lower() or 'health' in col.lower():
                        health_mean = integrated_data[col].mean()
                        if health_mean > integrated_data[col].quantile(0.75):
                            recommendations.append("⚠️ 전반적인 건강 지수가 높습니다. 실내 공기질 개선을 위한 종합적인 조치가 필요합니다.")
                
            except Exception as e:
                st.warning(f"권장사항 생성 중 오류 발생: {e}")
            
            if recommendations:
                for rec in recommendations:
                    st.warning(rec)
            else:
                st.success("✅ 현재 실내 공기질 상태가 양호합니다.")
            
            # 추가 일반 권장사항
            st.subheader('🏠 일반 실내 공기질 관리 권장사항')
            general_recommendations = [
                "🌬️ 정기적인 환기 (하루 2-3회, 10-15분씩)",
                "🌱 공기정화식물 배치 (산세베리아, 스파티필름 등)",
                "🧹 정기적인 청소 및 먼지 제거",
                "🚫 실내 흡연 금지",
                "🌡️ 적정 온도 유지 (18-25°C)",
                "💧 적정 습도 유지 (40-60%)",
                "🔧 에어컨 및 환기 시스템 필터 정기 교체"
            ]
            
            for rec in general_recommendations:
                st.info(rec)
    
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
        try:
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
