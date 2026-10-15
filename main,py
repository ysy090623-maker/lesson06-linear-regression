import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업의 기준 기간 끝
MIN_DAYS = 300        # 한 해에 필요한 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")
st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온으로 회귀 직선을 만들고, 고른 연도의 기온을 예상합니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_annual() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜나 평균기온이 비어 있는 행은 관측일로 세지 않음
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year

    # 기준 기간(2025년까지)만 사용
    df = df[df["연도"] <= LAST_YEAR]

    annual = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    # 관측일이 300일 미만인 해 제외
    annual = annual[annual["관측일수"] >= MIN_DAYS].reset_index(drop=True)
    return annual


try:
    annual = load_annual()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(annual) < 2:
    st.error("회귀 직선을 만들 데이터가 부족합니다.")
    st.stop()

years = annual["연도"].to_numpy(dtype=float)
temps = annual["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(years, temps, 1)
r = float(np.corrcoef(years, temps)[0, 1])

n_years = len(annual)
start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())

# 최근 20년 (끝 연도 포함해서 20년 구간) 기울기
RECENT_SPAN = 20
recent_start = end_year - RECENT_SPAN + 1
recent = annual[annual["연도"] >= recent_start]
if len(recent) >= 2:
    slope_recent, intercept_recent = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1,
    )
else:
    slope_recent, intercept_recent = None, None

# ---- 슬라이더 & 예상 기온 ----
target_year = st.slider("예상할 연도", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * target_year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem; border-radius:16px;
                background:rgba(128,128,128,0.12); margin:0.5rem 0 1rem 0;">
        <div style="font-size:1.1rem; opacity:0.8;">{target_year}년 서울의 예상 연평균기온</div>
        <div style="font-size:4rem; font-weight:800; line-height:1.2;">{pred:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- 요약 정보 ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{r:.3f}")

def per_century_card(title: str, sub: str, value) -> str:
    if value is None:
        big = "계산 불가"
    else:
        big = f"{value * 100:+.2f} °C"
    return f"""
    <div style="text-align:center; padding:1.2rem; border-radius:16px;
                background:rgba(128,128,128,0.12); height:100%;">
        <div style="font-size:1.05rem; font-weight:600;">{title}</div>
        <div style="font-size:0.85rem; opacity:0.7; margin-bottom:0.3rem;">{sub}</div>
        <div style="font-size:3rem; font-weight:800; line-height:1.2;">{big}</div>
        <div style="font-size:0.9rem; opacity:0.8;">100년당 기온 변화</div>
    </div>
    """


st.subheader("100년에 몇 도 오를까? (기울기 × 100)")
col_all, col_recent = st.columns(2)
with col_all:
    st.markdown(
        per_century_card("전체 기간", f"{start_year}~{end_year}년 · {n_years}개 해", slope),
        unsafe_allow_html=True,
    )
with col_recent:
    st.markdown(
        per_century_card(
            f"최근 {RECENT_SPAN}년",
            f"{recent_start}~{end_year}년 · {len(recent)}개 해",
            slope_recent,
        ),
        unsafe_allow_html=True,
    )

if slope_recent is not None and slope != 0:
    ratio = slope_recent / slope
    st.write(
        f"최근 {RECENT_SPAN}년의 기울기는 전체 기간의 약 **{ratio:.1f}배**입니다. "
        f"(최근 {RECENT_SPAN}년은 해의 수가 적어 값이 크게 흔들릴 수 있어요.)"
    )

st.write(
    f"전체 기간 회귀 직선: **기온 = {slope:.4f} × 연도 + ({intercept:.2f})**  "
    f"(결정계수 R² = {r ** 2:.3f}). 위의 예상 기온은 이 전체 기간 직선으로 계산합니다."
)
st.caption(
    f"{LAST_YEAR}년까지의 데이터 중 관측일이 {MIN_DAYS}일 이상인 해만 사용했습니다."
)

# ---- 그래프 ----
x_min = min(start_year, target_year)
x_max = max(end_year, target_year)
line_x = np.array([x_min, x_max], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선(전체 기간)",
        line=dict(width=3, dash="dash"),
    )
)
if slope_recent is not None:
    rx = np.array([recent_start, end_year], dtype=float)
    fig.add_trace(
        go.Scatter(
            x=rx,
            y=slope_recent * rx + intercept_recent,
            mode="lines",
            name=f"회귀 직선(최근 {RECENT_SPAN}년)",
            line=dict(width=4, dash="dot"),
        )
    )
fig.add_trace(
    go.Scatter(
        x=[target_year],
        y=[pred],
        mode="markers",
        name=f"{target_year}년 예상",
        marker=dict(size=16, symbol="star", line=dict(width=1, color="white")),
        hovertemplate="%{x}년 예상<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온과 회귀 직선 (r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=520,
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("연도별 평균기온 표 보기"):
    st.dataframe(
        annual.rename(columns={"연평균기온": "연평균기온(°C)"}).round(2),
        use_container_width=True,
        hide_index=True,
    )
