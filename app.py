import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
import numpy as np

# 페이지 기본 설정
st.set_page_config(page_title="한반도 실시간 비행기 추적", layout="wide")

st.title("✈️ 한반도 상공 실시간 비행기 관제 및 이상 탐지 시스템")
st.write("30초마다 자동으로 데이터를 갱신하며, 진행 방향이 반영된 비행기 아이콘으로 실시간 위치를 추적합니다.")

# -----------------------------------------------------------
# 1. 사이드바 UI 설정
# -----------------------------------------------------------
st.sidebar.header("⚙️ 컨트롤 타워")

# Z-score 기준값 슬라이더 (사용자가 직접 조절 가능)
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
    url = "https://opensky-network.org/api/states/all"
    # 대한민국 전역을 포함하는 위경도 바운딩 박스
    params = {"lamin": 33.0, "lamax": 43.0, "lomin": 124.0, "lomax": 132.0}
    
    try:
        api_user = st.secrets["OPENSKY_USERNAME"]
        api_password = st.secrets["OPENSKY_PASSWORD"]
        auth = (api_user, api_password)
    except KeyError:
        auth = None
        st.sidebar.warning("⚠️ Streamlit Secrets 인증 정보 없음 (비인증 모드)")

    try:
        response = requests.get(url, params=params, timeout=20, auth=auth)
        if response.status_code != 200:
            return []
        data = response.json()
        return data.get("states", []) if data else []
    except Exception:
        return []

# -----------------------------------------------------------
# 3. 데이터 전처리 및 Z-score 계산 (Fragment 활용하여 30초마다 갱신)
# -----------------------------------------------------------
@st.fragment(run_every="30s")
def render_dashboard():
    # 데이터 로드 안내
    raw_data = get_flight_data()

    if len(raw_data) > 0:
        columns = [
            'icao24', 'callsign', 'origin_country', 'time_position', 'last_contact',
            'longitude', 'latitude', 'baro_altitude', 'on_ground', 'velocity',
            'true_track', 'vertical_rate', 'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source'
        ]
        df = pd.DataFrame(raw_data, columns=columns)
        
        # 방향(true_track)을 시각화에 써야 하므로 포함시킵니다.
        df = df[['callsign', 'longitude', 'latitude', 'baro_altitude', 'velocity', 'vertical_rate', 'true_track']]
        df = df.dropna(subset=['longitude', 'latitude', 'vertical_rate'])
        df['callsign'] = df['callsign'].astype(str).str.strip().replace('', '알 수 없음')
        df['true_track'] = df['true_track'].fillna(0) # 방향 정보 없으면 0도(북쪽) 고정

        # Z-score 계산
        mean_vr = df['vertical_rate'].mean()
        std_vr = df['vertical_rate'].std()
        
        if std_vr > 0 and len(df) > 1:
            df['z_score'] = (df['vertical_rate'] - mean_vr) / std_vr
        else:
            df['z_score'] = 0.0

        # 상태 분류 (슬라이더 값 이하를 위험으로 간주)
        df['status'] = df['z_score'].apply(lambda z: '위험(급강하)' if z <= z_threshold else '정상')

        # [3번 기능 개선] 상태에 따른 색상 데이터 지정 (빨간색 투명도 조정, 크기 조정)
        def assign_color_data(status):
            if status == '위험(급강하)':
                # 빨간색 비행기 아이콘 매핑 (오픈소스 이미지 주소 활용, 투명도 추가)
                return {"url": "https://img.icons8.com/isometric/100/fa3131/airplane-mode-on.png", "width": 128, "height": 128, "anchorY": 64, "color": [255, 0, 0, 200]}
            # 노란색 비행기 아이콘 매핑
            return {"url": "https://img.icons8.com/isometric/100/fab005/airplane-mode-on.png", "width": 128, "height": 128, "anchorY": 64, "color": [255, 176, 5, 200]}

        df['icon_data'] = df['status'].apply(assign_color_data)

        # 사이드바 대시보드 정보 업데이트
        diving_count = len(df[df['status'] == '위험(급강하)'])
        st.sidebar.metric(label="🛰️ 현재 추적 항공기", value=f"{len(df)} 대")
        if diving_count > 0:
            st.sidebar.error(f"🚨 급강하 위험 항공기: {diving_count}대!!")
        else:
            st.sidebar.success("✅ 영공 내 특이 이상 징후 없음")

        # -----------------------------------------------------------
        # [3번 기능 개선] Pydeck IconLayer 적용 (방향 회전 포함)
        # -----------------------------------------------------------
        # 한국 전체가 보이도록 위도와 줌 레벨 조정
        view_state = pdk.ViewState(latitude=37.5, longitude=128.0, zoom=6.0, pitch=40)

        layer = pdk.Layer(
            "IconLayer",
            data=df,
            get_icon="icon_data",
            get_position="[longitude, latitude]",
            get_size=40,                 # 아이콘 크기 (약간 줄임)
            pickable=True,
            get_angle="-true_track",     # 항공기 진행 방향 각도만큼 아이콘 회전
        )

        tooltip = {
            "html": """
            <div style='font-family: sans-serif; padding: 5px;'>
                <b>✈️ 편명(Callsign):</b> {callsign} <br/>
                <b>상태:</b> {status} <br/>
                <b>수직 속도:</b> {vertical_rate} m/s <br/>
                <b>현재 고도:</b> {baro_altitude} m <br/>
                <b>비행 방향:</b> {true_track}° <br/>
                <b>Z-score:</b> {z_score:.2f}
            </div>
            """,
            "style": {"backgroundColor": "#1e1e1e", "color": "white", "borderRadius": "5px"}
        }

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json" # CartoDB 다크 모드 지도
        )

        st.pydeck_chart(r)
        
        # -----------------------------------------------------------
        # 4. 데이터 테이블 출력
        # -----------------------------------------------------------
        st.subheader("📊 실시간 항공 세부 데이터 (30초마다 자동 갱신)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="평균 수직 속도", value=f"{mean_vr:.2f} m/s")
        with col2:
            st.metric(label="수직 속도 표준편차", value=f"{std_vr:.2f}")
            
        st.dataframe(
            df[['callsign', 'status', 'z_score', 'vertical_rate', 'baro_altitude', 'velocity', 'true_track']],
            use_container_width=True
        )
    else:
        st.warning("현재 한반도 상공에서 수집된 유효한 OpenSky 비행 데이터가 없습니다. (30초 후 재시도합니다)")

# 대시보드 실행
render_dashboard()
