import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from CoolProp.CoolProp import PropsSI


def process_jeonju_data():
    """
    Process Jeonju weather data and calculate qt values for leap year (8784 hours)
    """
    # Read the Excel file
    df = pd.read_excel("Jeonju_data.xlsx")

    # Generate timestamp for each hour of the leap year
    start_date = pd.Timestamp('2024-01-01')  # 2024년은 윤년입니다
    df['timestamp'] = [start_date + pd.Timedelta(hours=i) for i in range(8784)]
    df['hour'] = range(1, 8785)

    # 기온(섭씨)을 켈빈으로 변환: Toutdoor
    Toutdoor = df['기온'].values + 273  # 섭씨에 273을 더해 켈빈으로 변환

    # 일사량을 W/m2로 변환하고 결측값을 0으로 채움: radSolar
    df['일사'] = df['일사'].fillna(0)  # 결측값을 0으로 채움
    radSolar = df['일사'].values * 277.778  # MJ/m2/hr에서 W/m2로 정확한 변환

    # 지중온도 설정
    Tground = 283  # 10℃ + 273 = 283K

    # Basic constants
    P_atm = 101325
    T_mean = 300
    rhoAir = PropsSI("D", "T", T_mean, "P", P_atm, "air")
    cAir = PropsSI("C", "T", T_mean, "P", P_atm, "air") / 1000

    # Building dimensions
    lengthHouse = 40.05
    widthHouse = 16
    heightHouse = 4
    fracSolarWindow = 0.55
    transGlass = 0.896

    # Calculate areas
    areaHouse = lengthHouse * widthHouse
    surfaceHouse = (widthHouse * heightHouse + lengthHouse * heightHouse) * 2
    volumeHouse = areaHouse * heightHouse
    mHouse = volumeHouse * rhoAir

    # Other parameters
    ACH = 1
    agh = 0.15
    Yt = 5.2  # Heat transfer coefficient of PO film

    # Process data and calculate qt
    Troom = np.ones(8784) * (15 + 273)  # Initialize at 288K (15°C) for 8784 hours
    qt = np.zeros(8784)

    for i in range(8784):
        # Calculate Tsolair2
        Tsolair2 = Toutdoor[i] + (agh * radSolar[i]) / 17

        # Calculate components
        qRad = (fracSolarWindow * transGlass * areaHouse * radSolar[i]) / 1000
        qRoof = (Tsolair2 - Troom[i]) * Yt * areaHouse / 1000
        qFloor = (Tground - Troom[i]) * Yt * areaHouse / 1000
        qSideWall = (Tsolair2 - Troom[i]) * Yt * surfaceHouse / 1000
        qvent = ACH * (Toutdoor[i] - Troom[i]) * mHouse * cAir / 3600 / 1000

        # Calculate total qt
        qt[i] = qRad + qRoof + qFloor + qSideWall + qvent

    return df, qt, Toutdoor, radSolar, Troom, {
        'rhoAir': rhoAir, 'cAir': cAir, 'areaHouse': areaHouse,
        'surfaceHouse': surfaceHouse, 'mHouse': mHouse,
        'fracSolarWindow': fracSolarWindow, 'transGlass': transGlass,
        'Tground': Tground, 'ACH': ACH, 'agh': agh, 'Yt': Yt
    }

def show_detailed_calculation(hour_index, df, params, Toutdoor, radSolar, Troom):
    """
    특정 시간의 qt 계산 과정을 상세히 보여주는 함수
    """
    # Get timestamp for the selected hour
    timestamp = df['timestamp'].iloc[hour_index]
    date_str = timestamp.strftime("%Y년 %m월 %d일 %H시")

    # 해당 시간의 기본 데이터
    outdoor_temp = Toutdoor[hour_index]
    solar_rad = radSolar[hour_index]
    room_temp = Troom[hour_index]

    # Tsolair2 계산
    tsolair2 = outdoor_temp + (params['agh'] * solar_rad) / 17

    # 각 컴포넌트 계산
    q_rad = (params['fracSolarWindow'] * params['transGlass'] * params['areaHouse'] * solar_rad) / 1000
    q_roof = (tsolair2 - room_temp) * params['Yt'] * params['areaHouse'] / 1000
    q_floor = (params['Tground'] - room_temp) * params['Yt'] * params['areaHouse'] / 1000
    q_sidewall = (tsolair2 - room_temp) * params['Yt'] * params['surfaceHouse'] / 1000
    q_vent = params['ACH'] * (outdoor_temp - room_temp) * params['mHouse'] * params['cAir'] / 3600 / 1000

    # 총 qt 계산
    qt_total = q_rad + q_roof + q_floor + q_sidewall + q_vent

    calculations = {
        "날짜 및 시간": date_str,
        "기본 데이터": {
            "외기온도": f"{outdoor_temp:.2f} K ({outdoor_temp - 273.15:.2f}°C)",
            "일사량": f"{solar_rad:.2f} W/m² (원본: {df['일사'].iloc[hour_index]:.2f} MJ/m²/hr)",
            "실내온도": f"{room_temp:.2f} K ({room_temp - 273.15:.2f}°C)",
            "지중온도": f"{params['Tground']:.2f} K ({params['Tground'] - 273.15:.2f}°C)"
        },
        "Tsolair2 계산": {
            "공식": "Tsolair2 = Toutdoor + (agh * radSolar) / 17",
            "계산": f"{outdoor_temp:.2f} + ({params['agh']:.2f} * {solar_rad:.2f}) / 17",
            "결과": f"{tsolair2:.2f} K"
        },
        "에너지 부하 계산": {
            "태양 복사 에너지 부하 (qRad)": {
                "공식": "qRad = (fracSolarWindow * transGlass * areaHouse * radSolar) / 1000",
                "계산": f"({params['fracSolarWindow']:.3f} * {params['transGlass']:.3f} * {params['areaHouse']:.2f} * {solar_rad:.2f}) / 1000",
                "결과": f"{q_rad:.2f} kW"
            },
            "지붕 관류열 부하 (qRoof)": {
                "공식": "qRoof = (Tsolair2 - Troom) * Yt * areaHouse / 1000",
                "계산": f"({tsolair2:.2f} - {room_temp:.2f}) * {params['Yt']:.2f} * {params['areaHouse']:.2f} / 1000",
                "결과": f"{q_roof:.2f} kW"
            },
            "지중 전열 부하 (qFloor)": {
                "공식": "qFloor = (Tground - Troom) * Yt * areaHouse / 1000",
                "계산": f"({params['Tground']:.2f} - {room_temp:.2f}) * {params['Yt']:.2f} * {params['areaHouse']:.2f} / 1000",
                "결과": f"{q_floor:.2f} kW"
            },
            "측벽 관류열 부하 (qSideWall)": {
                "공식": "qSideWall = (Tsolair2 - Troom) * Yt * surfaceHouse / 1000",
                "계산": f"({tsolair2:.2f} - {room_temp:.2f}) * {params['Yt']:.2f} * {params['surfaceHouse']:.2f} / 1000",
                "결과": f"{q_sidewall:.2f} kW"
            },
            "틈새 환기 전열 부하 (qVent)": {
                "공식": "qVent = ACH * (Toutdoor - Troom) * mHouse * cAir / 3600 / 1000",
                "계산": f"{params['ACH']:.2f} * ({outdoor_temp:.2f} - {room_temp:.2f}) * {params['mHouse']:.2f} * {params['cAir']:.4f} / 3600 / 1000",
                "결과": f"{q_vent:.2f} kW"
            }
        },
        "총 에너지 부하 (qt)": {
            "공식": "qt = qRad + qRoof + qFloor + qSideWall + qVent",
            "계산": f"{q_rad:.2f} + {q_roof:.2f} + {q_floor:.2f} + {q_sidewall:.2f} + {q_vent:.2f}",
            "결과": f"{qt_total:.2f} kW"
        }
    }

    return calculations

def main():
    st.markdown("<h2 style='text-align: center; color: black;'>2024년 전북대 온실 에너지 부하</h2>", unsafe_allow_html=True)

    try:
        df, qt, Toutdoor, radSolar, Troom, params = process_jeonju_data()

        # 시간 입력
        selected_hour = st.number_input('시간을 입력하세요 (1-8784)', min_value=1, max_value=8784, value=1)

        # 선택된 시간의 상세 계산 표시
        calculations = show_detailed_calculation(selected_hour - 1, df, params, Toutdoor, radSolar, Troom)

        # 계산 과정 표시
        st.header(f"{calculations['날짜 및 시간']}의 상세 계산 과정")

        # 기본 데이터 표시
        st.subheader('1. 기본 데이터')
        for key, value in calculations['기본 데이터'].items():
            st.write(f"- {key}: {value}")

        # Tsolair2 계산 표시
        st.subheader('2. Tsolair2 계산')
        st.write(f"공식: {calculations['Tsolair2 계산']['공식']}")
        st.write(f"계산: {calculations['Tsolair2 계산']['계산']}")
        st.write(f"결과: {calculations['Tsolair2 계산']['결과']}")

        # 열부하 계산 표시
        st.subheader('3. 각 부분별 에너지 부하 계산')
        for component, details in calculations['에너지 부하 계산'].items():
            st.write(f"\n**{component}**")
            st.write(f"공식: {details['공식']}")
            st.write(f"계산: {details['계산']}")
            st.write(f"결과: {details['결과']}")

        # 총 열부하 표시
        st.subheader('4. 총 에너지 부하 (qt)')
        st.write(f"공식: {calculations['총 에너지 부하 (qt)']['공식']}")
        st.write(f"계산: {calculations['총 에너지 부하 (qt)']['계산']}")
        st.write(f"결과: {calculations['총 에너지 부하 (qt)']['결과']}")

        # 연간 열부하 그래프 표시
        st.header('연간 에너지 부하 그래프')

        # Create figure
        fig = go.Figure()

        # 에너지 손실 (음수 값)
        losses = [q if q < 0 else 0 for q in qt]
        fig.add_trace(go.Scatter(
            x=list(range(1, 8785)),
            y=losses,
            mode='lines',
            name='에너지 손실',
            line=dict(color='red'),
            fill='tozeroy'
        ))

        # 에너지 유입 (양수 값)
        gains = [q if q > 0 else 0 for q in qt]
        fig.add_trace(go.Scatter(
            x=list(range(1, 8785)),
            y=gains,
            mode='lines',
            name='에너지 유입',
            line=dict(color='blue'),
            fill='tozeroy'
        ))

        # Update layout
        fig.update_layout(
            title='Annual Energy Load (qt)',
            xaxis_title='Hour of Year (h)',
            yaxis_title='Energy Load (kW)',
            showlegend=True,
            hovermode='x unified'
        )

        # Add a horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

        st.plotly_chart(fig)

    except FileNotFoundError:
        st.error("'Jeonju_data.xlsx' 파일을 찾을 수 없습니다. 파일이 올바른 위치에 있는지 확인해주세요.")
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")


if __name__ == '__main__':
    main()
