import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(
    page_title="실내 공기질 및 환기 효율성 분석",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 메인 타이틀
st.title("🌬️ 실내 공기질 및 환기 효율성 분석 시스템")
st.markdown("---")

# 사이드바 설정
st.sidebar.header("분석 옵션")
analysis_type = st.sidebar.selectbox(
    "분석 유형 선택",
    ["개요", "환기 방식 비교", "에너지 소비 분석", "건강 영향 분석", "종합 리포트"]
)

# 샘플 데이터 생성 함수
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    
    # 환기 방식별 데이터
    ventilation_types = ["자연환기", "기계환기", "혼합환기"]
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    
    data = []
    for date in dates:
        for vent_type in ventilation_types:
            # 계절별 효과 반영
            season_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            # 환기 방식별 기본 성능
            base_efficiency = {"자연환기": 0.6, "기계환기": 0.8, "혼합환기": 0.9}
            
            record = {
                "date": date,
                "ventilation_type": vent_type,
                "pm25": np.random.normal(25, 5) * (2 - base_efficiency[vent_type]),
                "pm10": np.random.normal(40, 8) * (2 - base_efficiency[vent_type]),
                "co2": np.random.normal(800, 100) * (2 - base_efficiency[vent_type]),
                "vocs": np.random.normal(200, 50) * (2 - base_efficiency[vent_type]),
                "humidity": np.random.normal(50, 10),
                "temperature": np.random.normal(22, 3),
                "energy_consumption": np.random.normal(100, 20) * (1.5 - base_efficiency[vent_type]) * season_factor,
                "health_score": np.random.normal(80, 10) * base_efficiency[vent_type],
                "respiratory_issues": np.random.poisson(3) * (2 - base_efficiency[vent_type]),
                "productivity_index": np.random.normal(85, 15) * base_efficiency[vent_type]
            }
            data.append(record)
    
    return pd.DataFrame(data)

# 데이터 로드
df = generate_sample_data()

if analysis_type == "개요":
    st.header("📊 프로젝트 개요")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("총 데이터 포인트", f"{len(df):,}")
    
    with col2:
        st.metric("분석 기간", "2024년 전체")
    
    with col3:
        st.metric("환기 방식", "3종류")
    
    st.subheader("🎯 분석 목표")
    st.write("""
    - 환기 방식별 실내 공기질 개선 효과 분석
    - 에너지 소비 패턴과 환기 효율성 상관관계 파악
    - 공기질이 건강 지표에 미치는 영향 정량화
    - 최적의 환기 전략 수립을 위한 데이터 기반 인사이트 제공
    """)
    
    st.subheader("📈 주요 지표")
    metrics_df = df.groupby('ventilation_type').agg({
        'pm25': 'mean',
        'co2': 'mean',
        'energy_consumption': 'mean',
        'health_score': 'mean'
    }).round(2)
    
    st.dataframe(metrics_df, use_container_width=True)

elif analysis_type == "환기 방식 비교":
    st.header("🔄 환기 방식별 성능 비교")
    
    # 필터링 옵션
    col1, col2 = st.columns(2)
    with col1:
        selected_months = st.multiselect(
            "월 선택",
            options=list(range(1, 13)),
            default=[6, 7, 8, 12, 1, 2],
            format_func=lambda x: f"{x}월"
        )
    
    with col2:
        selected_metrics = st.multiselect(
            "비교 지표 선택",
            options=['pm25', 'pm10', 'co2', 'vocs'],
            default=['pm25', 'co2'],
            format_func=lambda x: {'pm25': 'PM2.5', 'pm10': 'PM10', 'co2': 'CO2', 'vocs': 'VOCs'}[x]
        )
    
    # 데이터 필터링
    filtered_df = df[df['date'].dt.month.isin(selected_months)]
    
    # 환기 방식별 공기질 비교
    fig_comparison = go.Figure()
    
    for metric in selected_metrics:
        for vent_type in filtered_df['ventilation_type'].unique():
            data = filtered_df[filtered_df['ventilation_type'] == vent_type][metric]
            fig_comparison.add_trace(go.Box(
                y=data,
                name=f"{vent_type} - {metric.upper()}",
                boxpoints='outliers'
            ))
    
    fig_comparison.update_layout(
        title="환기 방식별 공기질 지표 비교",
        yaxis_title="농도 (μg/m³ 또는 ppm)",
        height=500
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # 효율성 순위
    st.subheader("📊 환기 방식별 효율성 순위")
    
    efficiency_scores = []
    for vent_type in df['ventilation_type'].unique():
        vent_data = df[df['ventilation_type'] == vent_type]
        
        # 점수 계산 (낮을수록 좋은 지표는 역수 적용)
        pm25_score = 100 - (vent_data['pm25'].mean() / df['pm25'].max() * 100)
        co2_score = 100 - (vent_data['co2'].mean() / df['co2'].max() * 100)
        health_score = vent_data['health_score'].mean()
        
        overall_score = (pm25_score + co2_score + health_score) / 3
        
        efficiency_scores.append({
            '환기방식': vent_type,
            'PM2.5 개선도': f"{pm25_score:.1f}",
            'CO2 개선도': f"{co2_score:.1f}",
            '건강점수': f"{health_score:.1f}",
            '종합점수': f"{overall_score:.1f}"
        })
    
    efficiency_df = pd.DataFrame(efficiency_scores)
    efficiency_df = efficiency_df.sort_values('종합점수', ascending=False).reset_index(drop=True)
    efficiency_df.index += 1
    
    st.dataframe(efficiency_df, use_container_width=True)

elif analysis_type == "에너지 소비 분석":
    st.header("⚡ 에너지 소비 패턴 분석")
    
    # 월별 에너지 소비 패턴
    monthly_energy = df.groupby([df['date'].dt.month, 'ventilation_type'])['energy_consumption'].mean().reset_index()
    monthly_energy['month'] = monthly_energy['date'].map(lambda x: f"{x}월")
    
    fig_energy = px.line(
        monthly_energy,
        x='month',
        y='energy_consumption',
        color='ventilation_type',
        title='월별 환기 방식별 에너지 소비량',
        labels={'energy_consumption': '에너지 소비량 (kWh)', 'ventilation_type': '환기 방식'}
    )
    
    st.plotly_chart(fig_energy, use_container_width=True)
    
    # 에너지 효율성 분석
    st.subheader("🔋 에너지 효율성 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 에너지 대비 공기질 개선 효과
        efficiency_df = df.groupby('ventilation_type').agg({
            'energy_consumption': 'mean',
            'pm25': 'mean',
            'co2': 'mean',
            'health_score': 'mean'
        }).reset_index()
        
        # 효율성 지수 계산 (건강점수 / 에너지소비량)
        efficiency_df['efficiency_index'] = efficiency_df['health_score'] / efficiency_df['energy_consumption']
        
        fig_efficiency = px.bar(
            efficiency_df,
            x='ventilation_type',
            y='efficiency_index',
            title='에너지 효율성 지수',
            labels={'efficiency_index': '효율성 지수', 'ventilation_type': '환기 방식'}
        )
        
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    with col2:
        # 에너지 소비 vs 건강 점수 산점도
        fig_scatter = px.scatter(
            df,
            x='energy_consumption',
            y='health_score',
            color='ventilation_type',
            title='에너지 소비량 vs 건강 점수',
            labels={'energy_consumption': '에너지 소비량 (kWh)', 'health_score': '건강 점수'}
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # 비용 분석
    st.subheader("💰 비용 분석")
    
    cost_per_kwh = st.slider("전력 단가 (원/kWh)", 100, 300, 150)
    
    cost_analysis = df.groupby('ventilation_type').agg({
        'energy_consumption': 'mean',
        'health_score': 'mean'
    }).reset_index()
    
    cost_analysis['daily_cost'] = cost_analysis['energy_consumption'] * cost_per_kwh
    cost_analysis['annual_cost'] = cost_analysis['daily_cost'] * 365
    cost_analysis['cost_per_health_point'] = cost_analysis['annual_cost'] / cost_analysis['health_score']
    
    st.dataframe(cost_analysis, use_container_width=True)

elif analysis_type == "건강 영향 분석":
    st.header("🏥 건강 영향 분석")
    
    # 공기질 지표별 건강 영향
    st.subheader("🔬 공기질 지표와 건강 점수 상관관계")
    
    air_quality_metrics = ['pm25', 'pm10', 'co2', 'vocs']
    
    fig_correlation = plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(air_quality_metrics, 1):
        plt.subplot(2, 2, i)
        plt.scatter(df[metric], df['health_score'], alpha=0.5)
        plt.xlabel(metric.upper())
        plt.ylabel('Health Score')
        plt.title(f'{metric.upper()} vs Health Score')
        
        # 추세선 추가
        z = np.polyfit(df[metric], df['health_score'], 1)
        p = np.poly1d(z)
        plt.plot(df[metric], p(df[metric]), "r--", alpha=0.8)
    
    plt.tight_layout()
    st.pyplot(fig_correlation)
    
    # 상관관계 히트맵
    st.subheader("🔥 상관관계 히트맵")
    
    correlation_data = df[['pm25', 'pm10', 'co2', 'vocs', 'health_score', 'respiratory_issues', 'productivity_index']].corr()
    
    fig_heatmap = plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_data, annot=True, cmap='RdYlBu_r', center=0)
    plt.title('공기질 지표와 건강 지표 간 상관관계')
    st.pyplot(fig_heatmap)
    
    # 환기 방식별 건강 영향
    st.subheader("🎯 환기 방식별 건강 영향 비교")
    
    health_comparison = df.groupby('ventilation_type').agg({
        'health_score': ['mean', 'std'],
        'respiratory_issues': ['mean', 'std'],
        'productivity_index': ['mean', 'std']
    }).round(2)
    
    st.dataframe(health_comparison, use_container_width=True)

elif analysis_type == "종합 리포트":
    st.header("📋 종합 분석 리포트")
    
    # 요약 통계
    st.subheader("📊 핵심 지표 요약")
    
    summary_stats = df.groupby('ventilation_type').agg({
        'pm25': 'mean',
        'co2': 'mean',
        'energy_consumption': 'mean',
        'health_score': 'mean',
        'respiratory_issues': 'mean'
    }).round(2)
    
    st.dataframe(summary_stats, use_container_width=True)
    
    # 권장사항
    st.subheader("💡 권장사항")
    
    best_ventilation = summary_stats.loc[summary_stats['health_score'].idxmax()]
    most_efficient = summary_stats.loc[summary_stats['energy_consumption'].idxmin()]
    
    st.write(f"""
    ### 분석 결과 요약
    
    **최고 건강 점수**: {best_ventilation.name} ({best_ventilation['health_score']:.1f}점)
    **최저 에너지 소비**: {most_efficient.name} ({most_efficient['energy_consumption']:.1f}kWh)
    
    ### 권장사항
    1. **건강 우선**: {best_ventilation.name}를 권장합니다.
    2. **에너지 효율성 우선**: {most_efficient.name}를 고려해보세요.
    3. **균형적 접근**: 혼합환기 시스템이 대부분의 상황에서 최적의 균형을 제공합니다.
    
    ### 추가 고려사항
    - 계절별 환기 전략 수립 필요
    - 건물 특성에 따른 맞춤형 환기 설계
    - 정기적인 공기질 모니터링 시스템 구축
    """)
    
    # 다운로드 버튼
    if st.button("📥 리포트 다운로드"):
        st.success("리포트가 생성되었습니다!")

# 푸터
st.markdown("---")
st.markdown("**데이터 출처**: 시뮬레이션 데이터 | **개발**: Indoor Air Quality Analysis System")
