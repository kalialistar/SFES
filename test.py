import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from CoolProp.CoolProp import PropsSI


def calculate_covering_combination(selected_covers):
    """
    Calculate thermal transmittance and solar window factor for different covering combinations

    Args:
        selected_covers (list): List of covering types to include

    Returns:
        tuple: (thermal transmittance value, solar window factor)
    """
    # Define thermal transmittance values for each material
    material_values = {
        'PO필름': 6.2,  # PO film 0.1mm
        'AL스크린': 5.5,  # AL screen
        '보온커튼': 4.5  # Thermal curtain
    }

    # Calculate Rt (total thermal resistance)
    Rt = sum(1 / material_values[cover] for cover in selected_covers)

    # Calculate Ti (initial thermal transmittance)
    Ti = 1 / Rt

    # Apply correction equation (4): Yi = 1.2944Ti - 0.4205
    Yi = 1.2944 * Ti - 0.4205

    # Determine fracfracSolarWindow based on presence of AL스크린
    fracfracSolarWindow = 0.55 if 'AL스크린' in selected_covers else 1.0

    return Yi, fracfracSolarWindow  # Return corrected thermal transmittance


def process_jeonju_data():
    """
    Process Jeonju weather data and calculate qt values for leap year (8784 hours)
    """

    # Read the Excel file
    df = pd.read_excel("Jeonju_data.xlsx")

    # UI에서 피복 선택을 위한 옵션 정의
    covering_options = {
        '1중피복(PO필름)': ['PO필름'],
        '2중피복(PO필름 + AL스크린(55%))': ['PO필름', 'AL스크린'],
        '3중피복(PO필름 + AL스크린(55%) + 3겹보온커튼)': ['PO필름', 'AL스크린', '보온커튼']
    }

    selected_covering = st.selectbox('피복 방식을 선택하세요:', list(covering_options.keys()))

    # 선택된 피복에 따른 보정된 열관류율과 fracSolarWindow 계산
    Yt, fracSolarWindow = calculate_covering_combination(covering_options[selected_covering])

    # Generate timestamp for each hour of the leap year
    start_date = pd.Timestamp('2024-01-01')
    df['timestamp'] = [start_date + pd.Timedelta(hours=i) for i in range(8784)]
    df['hour'] = range(1, 8785)

    # 기존 계산 로직 유지
    Toutdoor = df['기온'].values + 273
    df['일사'] = df['일사'].fillna(0)
    radSolar = df['일사'].values * 278
    Tground = 283

    # Basic constants
    P_atm = 101325
    T_mean = 300
    rhoAir = PropsSI("D", "T", T_mean, "P", P_atm, "air")
    cAir = PropsSI("C", "T", T_mean, "P", P_atm, "air") / 1000

    # Building dimensions
    lengthHouse = 40.05
    widthHouse = 16
    heightHouse = 4
    transGlass = 0.896

    # Calculate areas
    areaHouse = lengthHouse * widthHouse
    surfaceHouse = (widthHouse * heightHouse + lengthHouse * heightHouse) * 2
    volumeHouse = areaHouse * heightHouse
    mHouse = volumeHouse * rhoAir

    # Other parameters
    ACH = 1
    agh = 0.15

    # Initialize Troom array with initial temperature
    Troom = np.zeros(8784)
    Troom[0] = 20 + 273  # Initial room temperature
    qt = np.zeros(8784)
    qt[0] = 0

    # Calculate qt and Troom iteratively
    for i in range(0, 8784):
        Tsolair2 = Toutdoor[i] + (agh * radSolar[i]) / 17
        qRad = (fracSolarWindow * transGlass * areaHouse * radSolar[i]) / 1000
        qRoof = (Tsolair2 - Troom[i]) * Yt * areaHouse / 1000
        qFloor = (Tground - Troom[i]) * Yt * areaHouse / 1000
        qSideWall = (Tsolair2 - Troom[i]) * Yt * surfaceHouse / 1000
        qvent = ACH * (Toutdoor[i] - Troom[i]) * mHouse * cAir / 3600
        qt[i] = qRad + qRoof + qFloor + qSideWall + qvent

        # Calculate next hour's room temperature
        if i < 8783:  # Prevent index out of bounds
            Troom[i + 1] = Troom[i] + (qt[i] / (mHouse * cAir))

    return df, qt, Toutdoor, radSolar, Troom, {
        'rhoAir': rhoAir, 'cAir': cAir, 'areaHouse': areaHouse,
        'surfaceHouse': surfaceHouse, 'mHouse': mHouse,
        'fracSolarWindow': fracSolarWindow, 'transGlass': transGlass,
        'Tground': Tground, 'ACH': ACH, 'agh': agh, 'Yt': Yt
    }

def show_detailed_calculation(hour_index, df, params, Toutdoor, radSolar, Troom, qt_array):
    """
    특정 시간의 qt와 Troom 계산 과정을 상세히 보여주는 함수
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
    q_vent = params['ACH'] * (outdoor_temp - room_temp) * params['mHouse'] * params['cAir'] / 3600

    # 총 qt 계산
    qt = q_rad + q_roof + q_floor + q_sidewall + q_vent

    if hour_index < 8783:
        next_troom = Troom[hour_index + 1]
        troom_calculation = {
            "실내온도 (Troom) 계산": {
                "공식": "Troom[i+1] = Troom[i] + (qt[i] / (mHouse * cAir))",
                "계산": f"{room_temp:.2f} + ({qt_array[hour_index]:.2f} / ({params['mHouse']:.2f} * {params['cAir']:.4f}))",
                "중간과정": f"{room_temp:.2f} + ({qt_array[hour_index]:.2f} / {params['mHouse'] * params['cAir']:.4f})",
                "최종계산": f"{room_temp:.2f} + {(qt_array[hour_index] / (params['mHouse'] * params['cAir'])):.4f}",
                "결과": f"{next_troom:.2f} K ({next_troom - 273.15:.2f}°C)"
            }
        }
    else:
        troom_calculation = {
            "다음 시간의 실내온도": "마지막 시간대입니다."
        }

    calculations = {
        "날짜 및 시간": date_str,
        "기본 데이터": {
            "외기온도": f"{outdoor_temp:.2f} K ({outdoor_temp - 273.15:.2f}°C)",
            "일사량": f"{solar_rad:.2f} W/m² ({df['일사'].iloc[hour_index]:.2f} MJ/m²/hr)",
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
                "공식": "qVent = ACH * (Toutdoor - Troom) * mHouse * cAir / 3600",
                "계산": f"{params['ACH']:.2f} * ({outdoor_temp:.2f} - {room_temp:.2f}) * {params['mHouse']:.2f} * {params['cAir']:.4f} / 3600",
                "결과": f"{q_vent:.2f} kW"
            }
        },
        "총 에너지 부하 (qt)": {
            "공식": "qt = qRad + qRoof + qFloor + qSideWall + qVent",
            "계산": f"{q_rad:.2f} + {q_roof:.2f} + {q_floor:.2f} + {q_sidewall:.2f} + {q_vent:.2f}",
            "결과": f"{qt:.2f} kW"
        },
        "실내온도 계산": troom_calculation
    }

    return calculations


def main():
    st.markdown("<h2 style='text-align: center; color: black;'>2024년 전북대 온실 에너지 부하</h2>", unsafe_allow_html=True)

    try:
        df, qt, Toutdoor, radSolar, Troom, params = process_jeonju_data()

        # 시간 입력
        selected_hour = st.number_input('시간을 입력하세요 (1-8784)', min_value=1, max_value=8784, value=1)

        # 선택된 시간의 날짜 확인
        selected_date = df['timestamp'].iloc[selected_hour - 1].date()

        # 해당 날짜의 시작 시간 인덱스 찾기
        start_hour_index = df[df['timestamp'].dt.date == selected_date].index[0]
        end_hour_index = start_hour_index + 24

        # 선택된 시간의 상세 계산 표시
        calculations = show_detailed_calculation(selected_hour - 1, df, params, Toutdoor, radSolar, Troom, qt)

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

        # Add Troom calculation display
        st.subheader('5. 다음 시간의 실내온도 계산')
        if isinstance(calculations['실내온도 계산'], dict):
            for key, details in calculations['실내온도 계산'].items():
                st.write(f"\n**{key}**")
                if isinstance(details, dict):
                    st.write(f"공식: {details['공식']}")
                    st.write(f"계산: {details['계산']}")
                    st.write(f"결과: {details['결과']}")
                else:
                    st.write(details)

        # 1. 에너지 부하 그래프
        st.header('연간 에너지 부하 그래프')

        fig_energy = go.Figure()

        # 에너지 손실 (음수 값)
        losses = [q if q < 0 else 0 for q in qt]
        fig_energy.add_trace(go.Scatter(
            x=list(range(1, 8785)),
            y=losses,
            mode='lines',
            name='에너지 손실',
            line=dict(color='red'),
            fill='tozeroy'
        ))

        # 에너지 유입 (양수 값)
        gains = [q if q > 0 else 0 for q in qt]
        fig_energy.add_trace(go.Scatter(
            x=list(range(1, 8785)),
            y=gains,
            mode='lines',
            name='에너지 유입',
            line=dict(color='blue'),
            fill='tozeroy'
        ))

        fig_energy.update_layout(
            title='Annual Energy Load (qt)',
            xaxis_title='Hour of Year (h)',
            yaxis_title='Energy Load (kW)',
            showlegend=True,
            hovermode='x unified'
        )

        fig_energy.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

        st.plotly_chart(fig_energy)

        # 2. 연간 온도 변화 그래프
        st.header('연간 온도 변화 그래프')

        fig_temp = go.Figure()

        fig_temp.add_trace(go.Scatter(
            x=list(range(1, 8785)),
            y=[t - 273.15 for t in Troom],
            mode='lines',
            name='실내온도 (°C)',
            line=dict(color='green')
        ))

        fig_temp.add_trace(go.Scatter(
            x=list(range(1, 8785)),
            y=[t - 273.15 for t in Toutdoor],
            mode='lines',
            name='외기온도 (°C)',
            line=dict(color='orange', dash='dash')
        ))

        fig_temp.update_layout(
            title='Annual Temperature Variation',
            xaxis_title='Hour of Year (h)',
            yaxis_title='Temperature (°C)',
            showlegend=True,
            hovermode='x unified'
        )

        st.plotly_chart(fig_temp)

        # 선택된 날짜의 24시간 온도 변화 그래프
        st.header(f'{selected_date.strftime("%Y년 %m월 %d일")} 실내온도 변화 그래프')

        hours = list(range(24))

        fig_selected_day = go.Figure()

        # 실내온도
        fig_selected_day.add_trace(go.Scatter(
            x=hours,
            y=[t - 273.15 for t in Troom[start_hour_index:end_hour_index]],
            mode='lines+markers',
            name='실내온도 (°C)',
            line=dict(color='green'),
            marker=dict(size=8)
        ))

        # 외기온도
        fig_selected_day.add_trace(go.Scatter(
            x=hours,
            y=[t - 273.15 for t in Toutdoor[start_hour_index:end_hour_index]],
            mode='lines+markers',
            name='외기온도 (°C)',
            line=dict(color='orange', dash='dash'),
            marker=dict(size=8)
        ))

        # 선택된 시간 표시를 위한 수직선
        selected_hour_of_day = (selected_hour - 1) % 24
        fig_selected_day.add_vline(
            x=selected_hour_of_day,
            line_dash="dash",
            line_color="red",
            annotation_text="선택된 시간",
            annotation_position="top"
        )

        # Update layout for selected day temperature graph
        fig_selected_day.update_layout(
            title=f'Temperature Variation on {selected_date.strftime("%Y-%m-%d")}',
            xaxis=dict(
                title='Hour of Day',
                tickmode='array',
                ticktext=[f'{h:02d}:00' for h in hours],
                tickvals=hours
            ),
            yaxis_title='Temperature (°C)',
            showlegend=True,
            hovermode='x unified',
            xaxis_gridcolor='lightgray',
            yaxis_gridcolor='lightgray',
            plot_bgcolor='white'
        )

        st.plotly_chart(fig_selected_day)

    except FileNotFoundError:
        st.error("'Jeonju_data.xlsx' 파일을 찾을 수 없습니다. 파일이 올바른 위치에 있는지 확인해주세요.")
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")


if __name__ == '__main__':
    main()
