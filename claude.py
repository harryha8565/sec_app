import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 함수 (위에서 정의한 내용)
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 2. 앱 제목 설정
st.title('환기 방식, 에너지 소비 패턴과 실내 공기질 및 건강 지표 분석')

# 3. 사이드바 구성
st.sidebar.header('분석 설정')

# 4. 데이터 로드 (실제 파일 경로로 변경)
data_path = 'your_data.csv'
df = load_data(data_path)

if df is not None:
    st.sidebar.subheader('데이터 미리보기')
    st.sidebar.write(df.head())

    # 사용자 입력 예시: 환기 방식 선택
    ventilation_modes = df['환기방식'].unique().tolist() # '환기방식' 컬럼이 있다고 가정
    selected_ventilation = st.sidebar.multiselect(
        '환기 방식 선택',
        options=ventilation_modes,
        default=ventilation_modes
    )

    # 사용자 입력 예시: 에너지 소비량 범위 선택
    min_energy = df['에너지소비량'].min() # '에너지소비량' 컬럼이 있다고 가정
    max_energy = df['에너지소비량'].max()
    energy_range = st.sidebar.slider(
        '에너지 소비량 범위 선택',
        min_value=float(min_energy),
        max_value=float(max_energy),
        value=(float(min_energy), float(max_energy))
    )

    # 5. 메인 화면에 분석 결과 표시
    st.subheader('분석 결과')

    # 필터링된 데이터
    filtered_df = df[
        (df['환기방식'].isin(selected_ventilation)) &
        (df['에너지소비량'] >= energy_range[0]) &
        (df['에너지소비량'] <= energy_range[1])
    ]

    if not filtered_df.empty:
        st.write(f"선택된 조건으로 필터링된 데이터 ({len(filtered_df)}개):")
        st.dataframe(filtered_df.head())

        # 예시: 공기질 지표와 건강 지표의 상관관계 분석
        st.subheader('공기질 지표와 건강 지표 상관관계')
        # 'CO2', '미세먼지', 'VOCs', '두통', '피로도' 등 관련 컬럼이 있다고 가정
        # 실제 데이터의 컬럼명에 맞게 변경하세요.
        air_quality_cols = ['CO2', '미세먼지', 'VOCs']
        health_cols = ['두통', '피로도', '집중력']

        # 데이터에 해당 컬럼이 있는지 확인
        available_cols = [col for col in air_quality_cols + health_cols if col in filtered_df.columns]
        if len(available_cols) > 1:
            corr_matrix = filtered_df[available_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("선택된 데이터에 상관관계 분석을 위한 충분한 공기질 또는 건강 지표 컬럼이 없습니다.")


        # 예시: 환기 방식별 공기질 지표 평균 비교
        st.subheader('환기 방식별 주요 공기질 지표 평균')
        if '환기방식' in filtered_df.columns and 'CO2' in filtered_df.columns:
            avg_co2_by_ventilation = filtered_df.groupby('환기방식')['CO2'].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.barplot(x='환기방식', y='CO2', data=avg_co2_by_ventilation, ax=ax2)
            ax2.set_ylabel('평균 CO2 농도')
            st.pyplot(fig2)
        else:
            st.warning("선택된 데이터에 '환기방식' 또는 'CO2' 컬럼이 없어 환기 방식별 공기질 지표 비교를 할 수 없습니다.")

        # 추가 분석 및 시각화 코드 작성...

    else:
        st.warning("선택된 조건에 해당하는 데이터가 없습니다. 필터를 조정해 주세요.")
else:
    st.error("데이터를 불러오는 데 실패했습니다. 파일 경로를 확인해 주세요.")
