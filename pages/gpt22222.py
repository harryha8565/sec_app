# air_quality_health_app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("실내 공기질과 건강 영향 분석")

# 데이터 로드
df = pd.read_csv("air_quality_health_impact_data - air_quality_health_impact_data.csv")
st.subheader("데이터 미리보기")
st.dataframe(df.head())

# 컬럼 선택
st.sidebar.header("상관관계 분석 설정")
air_quality_col = st.sidebar.selectbox("공기질 지표 선택", df.columns)
health_col = st.sidebar.selectbox("건강 지표 선택", df.columns)

# 두 지표가 다를 때만 분석 실행
if air_quality_col != health_col:
    st.subheader(f"{air_quality_col}과(와) {health_col}의 상관관계 분석")

    fig, ax = plt.subplots()
    sns.regplot(x=df[air_quality_col], y=df[health_col], ax=ax)
    st.pyplot(fig)

    corr = df[air_quality_col].corr(df[health_col])
    st.write(f"**상관계수 (Pearson r):** {corr:.3f}")

    if abs(corr) > 0.7:
        result = "강한 상관관계"
    elif abs(corr) > 0.3:
        result = "중간 정도의 상관관계"
    else:
        result = "약한 또는 거의 없는 상관관계"
    st.write(f"➡️ 해석: 두 변수 사이에는 **{result}**가 있습니다.")

else:
    st.warning("서로 다른 두 지표를 선택해 주세요.")
