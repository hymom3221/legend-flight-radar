import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
import numpy as np

# 페이지 기본 설정
st.set_page_config(page_title="한반도 실시간 비행기 추적", layout="wide")

st.title("✈️ 한반도 상공 실시간 비행기 이상 탐지 웹앱")
st.write("OpenSky API 데이터를 직접 호출하여 Z-score 통계 기법으로 급강하 중인 비행기를 자동으로 감지합니다.")

# -----------------------------------------------------------
# 1. 사이드바 UI 및 Secrets 로드
# -----------------------------------------------------------
st.sidebar.header("⚙️ 컨트롤 타워")
refresh_button = st.sidebar.button("🔄 실시간 데이터 새로고침")

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 이상 탐지(Anomaly Detection) 설정")

# Z-score 기준값 슬라이더 (기본값 -3.0)
z_threshold = st.sidebar.slider(
    "급강하 감지 Z-score 기준값",
    min_value=-5.0,
    max_value=5.0,
    value=-3.0,
    step=0.1
)

# -----------------------------------------------------------
# 2. OpenSky API 데이터 수집 함수
# -----------------------------------------------------------
def get_flight_data():
    # 이전에 발생했던 HTML 엔티티(&quot;) 버그 완전 해결
    url = "https://opensky-network.org/api/states/all"
    
    # 대한민국 영공을 커버하는 위경도 바운딩 박스
    params = {"lamin": 33.0, "lamax": 39.0, "lomin": 124.0, "lomax": 132.0}
    
    # Streamlit Secrets에서 안전하게 계정 정보 로드
    try:
        api_user = st.secrets["OPENSKY_USERNAME"]
        api_password = st.secrets["OPENSKY_PASSWORD"]
        auth = (api_user, api_password)
    except KeyError:
        # 혹시 Secrets 설정이 안 되어 있으면 인증 없이 시도 (제한이 엄격함)
        auth = None
        st.sidebar.warning("⚠️ Streamlit Secrets에 인증 정보가 설정되지 않아 비인증 모드로 접근합니다.")

    try:
        # 타임아웃을 20초로 늘려 서버 무응답 에러 방지
        response = requests.get(url, params=params, timeout=20, auth=auth)
        
        # API 응답 상태 코드 확인
        if response.status_code == 429:
            st.error("🚨 OpenSky API 요청 횟수가 초과되었습니다. 잠시 후 다시 시도해주세요.")
            return []
        elif response.status_code != 200:
            st.error(f"🚨 API 서버 에러 (상태 코드: {response.status_code})")
            return []
            
        data = response.json()
        if data is not None and data.get("states") is not None:
            return data["states"]
        return []
        
    except requests.exceptions.Timeout:
        st.error("🚨 OpenSky API 서버 응답 시간이 초과되었습니다. (Timeout)")
        return []
    except Exception as e:
        st.error(f"🚨 데이터를 가져오는 중 오류가 발생했습니다: {e}")
        return []

# 데이터 호출
with st.spinner("OpenSky API로부터 실시간 항공 데이터를 불러오는 중..."):
    raw_data = get_flight_data()

# -----------------------------------------------------------
# 3. 데이터 전처리 및 Z-score 계산 (Pandas)
# -----------------------------------------------------------
if len(raw_data) > 0:
    columns = [
        'icao24', 'callsign', 'origin_country', 'time_position', 'last_contact',
        'longitude', 'latitude', 'baro_altitude', 'on_ground', 'velocity',
        'true_track', 'vertical_rate', 'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source'
    ]
    df = pd.DataFrame(raw_data, columns=columns)
    
    # 필요한 컬럼만 추출
    df = df[['callsign', 'longitude', 'latitude', 'baro_altitude', 'velocity', 'vertical_rate']]
    
    # 위치 정보 및 수직 속도가 누락된 데이터 제거
    df = df.dropna(subset=['longitude', 'latitude', 'vertical_rate'])
    df['callsign'] = df['callsign'].astype(str).str.strip().replace('', '알 수 없음')

    # --- [Z-score 통계 계산] ---
    mean_vr = df['vertical_rate'].mean()
    std_vr = df['vertical_rate'].std()
    
    # 표준편차가 0이거나 데이터가 너무 적을 때 분모가 0이 되는 오류 방지
    if std_vr > 0 and len(df) > 1:
        df['z_score'] = (df['vertical_rate'] - mean_vr) / std_vr
    else:
        df['z_score'] = 0.0

    # 상태 분류 (슬라이더 값 이하를 위험으로 간주)
    df['status'] = df['z_score'].apply(lambda z: '위험(급강하)' if z <= z_threshold else '정상')

    # 상태에 따른 색상 정의 (위험: 빨간색, 정상: 노란색)
    def assign_color(status):
        if status == '위험(급강하)':
            return [255, 0, 0, 255]
        return [255, 200, 0, 180]
        
    df['color'] = df['status'].apply(assign_color)

    # 사이드바 대시보드 요약 정보 업데이트
    diving_count = len(df[df['status'] == '위험(급강하)'])
    st.sidebar.success(f"현재 추적 비행기: {len(df)}대")
    if diving_count > 0:
        st.sidebar.error(f"⚠️ 급강하 감지: {diving_count}대!!")
    else:
        st.sidebar.info("✅ 현재 특이 이상 징후 없음")

    # -----------------------------------------------------------
    # 4. Pydeck 3D 지도 시각화
    # -----------------------------------------------------------
    view_state = pdk.ViewState(latitude=36.0, longitude=128.0, zoom=6, pitch=45)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_radius=6000,
        get_fill_color="color",  # 위에서 계산한 컬럼 연동
        pickable=True
    )

    tooltip = {
        "html": """
        <b>콜사인:</b> {callsign} <br/>
        <b>상태:</b> {status} <br/>
        <b>수직 속도:</b> {vertical_rate} m/s <br/>
        <b>Z-score:</b> {z_score:.2f} <br/>
        <b>현재 고도:</b> {baro_altitude} m
        """,
        "style": {"backgroundColor": "black", "color": "white"}
    }

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="dark"
    )

    st.pydeck_chart(r)
    
    # -----------------------------------------------------------
    # 5. 데이터 테이블 출력
    # -----------------------------------------------------------
    st.subheader("📊 실시간 항공 통계 및 데이터")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="평균 수직 속도", value=f"{mean_vr:.2f} m/s")
    with col2:
        st.metric(label="수직 속도 표준편차", value=f"{std_vr:.2f}")
        
    # 소수점 포맷팅을 적용해 데이터프레임 깔끔하게 출력
    st.dataframe(df[['callsign', 'status', 'z_score', 'vertical_rate', 'baro_altitude', 'velocity']])

else:
    st.warning("현재 한반도 상공에서 수집된 유효한 OpenSky 비행 데이터가 없습니다. 잠시 후 새로고침을 눌러보세요.")
