import pandas as pd
import numpy as np
import plotly.graph_objects as go
from CoolProp.CoolProp import PropsSI
import datetime

##############################
# 1) 바닥 열관류율 계산 함수 #
##############################
def calculate_floor_u_value():
    """
    3개 층(콘크리트, 자갈, 모래)의 열전도율, 두께를 이용해
    전체 바닥 열관류율(Yg)을 계산하여 반환합니다.
    열전도율 단위 변환(kJ/h·m·K -> W/m·K)에 주의하십시오.
    """
    # 열전도율 (kJ/h·m·K -> W/m·K)
    k_concrete = 6.23 * (1000/3600)  # 6.23 kJ/h·m·K
    k_gravel   = 7.20 * (1000/3600)  # 7.20 kJ/h·m·K
    k_sand     = 6.29 * (1000/3600)  # 6.29 kJ/h·m·K

    # 두께(m)
    d_concrete = 0.35
    d_gravel   = 0.20
    d_sand     = 0.35

    # 각 층의 열저항(m^2·K/W)
    R_concrete = d_concrete / k_concrete
    R_gravel   = d_gravel   / k_gravel
    R_sand     = d_sand     / k_sand

    # 총 열저항
    R_total = R_concrete + R_gravel + R_sand

    # 전체 열관류율 (W/m^2·K)
    Yg = 1.0 / R_total
    return Yg

#################################
# 2) 피복재 열관류율 계산 함수  #
#################################
def calculate_covering_properties(radSolar, Troom, Ttarget=288.15):
    covers = ['PO필름']
    transGlass = 0.896  # PO필름 광투과율

    # 조건에 따른 AL스크린 및 보온커튼 적용
    if radSolar > 0:
        if radSolar >= 600:
            covers.append('AL스크린')
            transGlass *= 0.55  # AL스크린 광투과율
        if Troom < Ttarget:
            covers.append('보온커튼')
            transGlass *= 0.5   # 보온커튼 광투과율
    else:
        # 밤에는 스크린과 커튼 모두 적용 (3중 피복)
        covers.append('AL스크린')
        covers.append('보온커튼')
        transGlass *= 0.55 * 0.5  # AL스크린 및 보온커튼 광투과율

    # 열관류율 계산 (기존 식 유지)
    material_values = {
        'PO필름': 6.2,
        'AL스크린': 5.5,
        '보온커튼': 4.5
    }
    Rt = sum(1 / material_values[cover] for cover in covers)
    Ti = 1 / Rt
    Yt = 1.2944 * Ti - 0.4205

    return Yt, transGlass, covers

#####################################
# 3) 엑셀 데이터 읽고 시뮬레이션 함수 #
#####################################
def process_jeonju_data():
    # 데이터 불러오기
    df = pd.read_excel("Jeonju_data.xlsx")

    start_date = pd.Timestamp('2024-01-01')
    df['timestamp'] = [start_date + pd.Timedelta(hours=i) for i in range(8784)]
    df['hour'] = range(1, 8785)

    # 외기온도 (K), 일사량 보정
    Toutdoor = df['기온'].values + 273.15
    df['일사'] = df['일사'].fillna(0)
    radSolar = df['일사'].values * 278

    # 상수 및 초기값 설정
    Tground = 15 + 273.15
    P_atm = 101325
    lengthHouse = 40.05
    widthHouse = 16
    heightHouse = 4
    areaHouse = lengthHouse * widthHouse
    surfaceHouse = (widthHouse * heightHouse + lengthHouse * heightHouse) * 2
    volumeHouse = areaHouse * heightHouse
    ACH = 1
    agh = 0.15
    Chouse = 50000
    Ttarget = 288.15  # 15°C

    # 바닥 열관류율 계산
    Yg = calculate_floor_u_value()

    # 배열 초기화
    Troom = np.zeros(8784)
    rhoAir = np.zeros(8784)
    cAir = np.zeros(8784)
    mHouse = np.zeros(8784)
    Yt_values = np.zeros(8784)
    transGlass_values = np.zeros(8784)
    active_covers = [None] * 8784
    qt = np.zeros(8784)

    # 초기값 설정
    Troom[0] = 15 + 273.15
    rhoAir[0] = PropsSI("D", "T", Troom[0], "P", P_atm, "air")
    cAir[0] = PropsSI("C", "T", Troom[0], "P", P_atm, "air") / 1000
    mHouse[0] = volumeHouse * rhoAir[0]
    qt[0] = 0

    emissivity = 0.9
    stefan_boltzmann = 5.67e-8
    Tsky = 0.0552 * (Toutdoor ** 1.5)

    # 시간 루프
    for i in range(8784):
        if i > 0:
            rhoAir[i] = PropsSI("D", "T", Troom[i], "P", P_atm, "air")
            cAir[i] = PropsSI("C", "T", Troom[i], "P", P_atm, "air") / 1000
            mHouse[i] = volumeHouse * rhoAir[i]

        # 피복재 속성 계산
        Yt_values[i], transGlass_values[i], active_covers[i] = calculate_covering_properties(
            radSolar[i], Troom[i], Ttarget
        )

        # 에너지 부하 계산
        Tsolair2 = Toutdoor[i] + (agh * radSolar[i]) / 17
        qRad = (transGlass_values[i] * areaHouse * radSolar[i]) / 1000
        qRoof = (Tsolair2 - Troom[i]) * Yt_values[i] * areaHouse / 1000
        qFloor = (Tground - Troom[i]) * Yg * areaHouse / 1000
        qSideWall = (Tsolair2 - Troom[i]) * Yt_values[i] * surfaceHouse / 1000
        qvent = ACH * (Toutdoor[i] - Troom[i]) * mHouse[i] * cAir[i] / 3600
        qsky = emissivity * stefan_boltzmann * areaHouse * (Troom[i] ** 4 - Tsky[i] ** 4) / 1000

        qt[i] = qRad + qRoof + qFloor + qSideWall + qvent - qsky

        # 다음 시간대 온도 업데이트
        if i < 8783:
            Troom[i + 1] = Troom[i] + (qt[i] / (mHouse[i] * cAir[i] + Chouse)) * 3600

    # 결과를 df에 저장
    df['Troom'] = Troom
    df['rhoAir'] = rhoAir
    df['cAir'] = cAir
    df['Yt'] = Yt_values
    df['transGlass'] = transGlass_values
    df['active_covers'] = [','.join(covers) for covers in active_covers]
    df['qt'] = qt

    # 파라미터 묶음
    params = {
        'areaHouse': areaHouse,
        'surfaceHouse': surfaceHouse,
        'volumeHouse': volumeHouse,
        'Tground': Tground,
        'ACH': ACH,
        'agh': agh,
        'Chouse': Chouse,
        'mHouse': mHouse[-1],
        'cAir': cAir[-1],
        'Ttarget': Ttarget,
        'Yg': Yg
    }

    return df, qt, Toutdoor, radSolar, Troom, params, rhoAir, cAir, mHouse, Yt_values, transGlass_values, active_covers

#################################################
# 4) 시간별 상세 계산(에너지 부하, 다음 시간 온도) #
#################################################
def show_detailed_calculation(hour_index, df, params, Toutdoor, radSolar, Troom, qt_array,
                              rhoAir, cAir, mHouse, Yt_values, transGlass_values, active_covers):
    """
    특정 hour_index(0~8783)에 대한 상세 계산 내역을
    딕셔너리로 반환. main()에서 가독성 있게 print 처리.
    """
    timestamp = df['timestamp'].iloc[hour_index]
    date_str = timestamp.strftime("%Y년 %m월 %d일 %H시")
    outdoor_temp = Toutdoor[hour_index]
    solar_rad = radSolar[hour_index]
    room_temp = Troom[hour_index]
    current_rhoAir = rhoAir[hour_index]
    current_cAir = cAir[hour_index]
    current_mHouse = mHouse[hour_index]
    current_Yt = Yt_values[hour_index]
    current_transGlass = transGlass_values[hour_index]
    current_covers = active_covers[hour_index]

    Tsky_local = 0.0552 * (outdoor_temp ** 1.5)
    tsolair2 = outdoor_temp + (params['agh'] * solar_rad) / 17

    q_rad = (current_transGlass * params['areaHouse'] * solar_rad) / 1000
    q_roof = (tsolair2 - room_temp) * current_Yt * params['areaHouse'] / 1000
    Yg_floor = params['Yg']
    q_floor = (params['Tground'] - room_temp) * Yg_floor * params['areaHouse'] / 1000
    q_sidewall = (tsolair2 - room_temp) * current_Yt * params['surfaceHouse'] / 1000
    q_vent = params['ACH'] * (outdoor_temp - room_temp) * current_mHouse * current_cAir / 3600

    emissivity = 0.9
    stefan_boltzmann = 5.67e-8
    q_sky = emissivity * stefan_boltzmann * params['areaHouse'] * (room_temp ** 4 - Tsky_local ** 4) / 1000

    qt_value = q_rad + q_roof + q_floor + q_sidewall + q_vent - q_sky

    qt_calculation = {
        "공식": "qt = qRad + qRoof + qFloor + qSideWall + qVent - qSky",
        "상세 계산": [
            f"태양 복사 에너지 부하 (qRad): {q_rad:.2f} kW",
            f"지붕 관류열 부하 (qRoof): {q_roof:.2f} kW",
            f"지중 전열 부하 (qFloor): {q_floor:.2f} kW",
            f"측벽 관류열 부하 (qSideWall): {q_sidewall:.2f} kW",
            f"틈새 환기 전열 부하 (qVent): {q_vent:.2f} kW",
            f"복사 냉각 에너지 부하 (qSky): {q_sky:.2f} kW"
        ],
        "최종 결과": f"{qt_value:.2f} kW"
    }

    if hour_index < 8783:
        delta_t = (qt_array[hour_index] / (current_mHouse * current_cAir + params['Chouse'])) * 3600
        next_troom = room_temp + delta_t
        troom_calc_detail = [
            f"1. mHouse[i] * cAir[i] + Chouse = {current_mHouse:.2f} * {current_cAir:.4f} + {params['Chouse']:.2f} = {current_mHouse * current_cAir + params['Chouse']:.2f}",
            f"2. qt[i] / (...) * 3600 = {qt_array[hour_index]:.2f} / {current_mHouse * current_cAir + params['Chouse']:.2f} * 3600",
            f"3. 최종 계산: {room_temp:.2f} + {delta_t:.4f} = {next_troom:.2f} K"
        ]
        troom_calculation = {
            "공식": "Troom[i+1] = Troom[i] + (qt[i] / (mHouse[i] * cAir[i] + Chouse)) * 3600",
            "단계별 계산": troom_calc_detail,
            "최종 결과": f"{next_troom:.2f} K ({next_troom - 273.15:.2f}°C)"
        }
    else:
        troom_calculation = {"마지막 시간대": "다음 시간 온도 계산 불가"}

    calculations = {
        "시간정보": f"Hour {hour_index+1} / {date_str}",
        "기본 데이터": {
            "외기온도": f"{outdoor_temp:.2f} K ({outdoor_temp - 273.15:.2f}°C)",
            "일사량": f"{solar_rad:.2f} W/m² ({df['일사'].iloc[hour_index]:.2f} MJ/m²/hr)",
            "실내온도": f"{room_temp:.2f} K ({room_temp - 273.15:.2f}°C)",
            "지중온도": f"{params['Tground']:.2f} K ({params['Tground'] - 273.15:.2f}°C)",
            "활성 피복재": f"{current_covers}",
            "광투과율": f"{current_transGlass:.4f}",
            "열관류율(Yt)": f"{current_Yt:.4f}",
            "바닥 열관류율(Yg)": f"{Yg_floor:.4f}"
        },
        "Tsolair2 계산": {
            "공식": "Tsolair2 = Toutdoor + (agh * radSolar) / 17",
            "계산": f"{outdoor_temp:.2f} + ({params['agh']:.2f} * {solar_rad:.2f}) / 17",
            "결과": f"{tsolair2:.2f} K"
        },
        "에너지 부하 계산": {
            "qRad": {
                "이름": "태양 복사 에너지 부하 (qRad)",
                "공식": "qRad = (transGlass * areaHouse * radSolar) / 1000",
                "계산": f"({current_transGlass:.4f} * {params['areaHouse']:.2f} * {solar_rad:.2f}) / 1000",
                "결과": f"{q_rad:.2f} kW"
            },
            "qRoof": {
                "이름": "지붕 관류열 부하 (qRoof)",
                "공식": "qRoof = (Tsolair2 - Troom) * Yt * areaHouse / 1000",
                "계산": f"({tsolair2:.2f} - {room_temp:.2f}) * {current_Yt:.4f} * {params['areaHouse']:.2f} / 1000",
                "결과": f"{q_roof:.2f} kW"
            },
            "qFloor": {
                "이름": "지중 전열 부하 (qFloor)",
                "공식": "qFloor = (Tground - Troom) * Yg * areaHouse / 1000",
                "계산": f"({params['Tground']:.2f} - {room_temp:.2f}) * {Yg_floor:.2f} * {params['areaHouse']:.2f} / 1000",
                "결과": f"{q_floor:.2f} kW"
            },
            "qSideWall": {
                "이름": "측벽 관류열 부하 (qSideWall)",
                "공식": "qSideWall = (Tsolair2 - Troom) * Yt * surfaceHouse / 1000",
                "계산": f"({tsolair2:.2f} - {room_temp:.2f}) * {current_Yt:.4f} * {params['surfaceHouse']:.2f} / 1000",
                "결과": f"{q_sidewall:.2f} kW"
            },
            "qVent": {
                "이름": "틈새 환기 전열 부하 (qVent)",
                "공식": "qVent = ACH * (Toutdoor - Troom) * mHouse * cAir / 3600",
                "계산": f"{params['ACH']:.2f} * ({outdoor_temp:.2f} - {room_temp:.2f}) * "
                         f"{current_mHouse:.2f} * {current_cAir:.4f} / 3600",
                "결과": f"{q_vent:.2f} kW"
            },
            "qSky": {
                "이름": "복사 냉각 에너지 부하 (qSky)",
                "공식": "qSky = emissivity * stefan_boltzmann * areaHouse * (Troom^4 - Tsky^4) / 1000",
                "계산": f"{emissivity:.2f} * {stefan_boltzmann:.2e} * {params['areaHouse']:.2f} * "
                         f"({room_temp:.2f}^4 - {Tsky_local:.2f}^4) / 1000",
                "결과": f"{q_sky:.2f} kW"
            },
        },
        "총 에너지 부하 상세 계산": qt_calculation,
        "실내온도 상세 계산": troom_calculation
    }

    return calculations

##################################
# 5) 연간 그래프(필요 시 사용 가능)
##################################
def display_annual_graphs(qt, Troom, Toutdoor):
    # 1) 연간 에너지 부하 그래프
    fig_annual_energy = go.Figure()
    fig_annual_energy.add_trace(go.Scatter(
        x=list(range(1, 8785)),
        y=qt,
        mode='lines',
        name='에너지 부하',
        line=dict(color='blue', width=1)
    ))
    fig_annual_energy.update_layout(
        title='Annual Energy Load',
        xaxis_title='Hour of Year',
        yaxis_title='Energy Load (kW)',
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white'
    )
    fig_annual_energy.write_image("annual_energy_load.png")
    fig_annual_energy.show()

    # 2) 연간 온도 변화 그래프
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
    fig_temp.write_image("annual_temperature_variation.png")
    fig_temp.show()

##################################
# 6) 일간 그래프(피복재 그래프 제거)
##################################
def display_daily_graphs(selected_date, start_hour_index, Troom, Toutdoor, qt, active_covers):
    hours = list(range(24))
    daily_qt = qt[start_hour_index:start_hour_index + 24]
    daily_Troom = Troom[start_hour_index:start_hour_index + 24]
    daily_Toutdoor = Toutdoor[start_hour_index:start_hour_index + 24]
    daily_covers = active_covers[start_hour_index:start_hour_index + 24]

    # 1) 일간 온도 그래프
    fig_daily_temp = go.Figure()
    fig_daily_temp.add_trace(go.Scatter(
        x=hours,
        y=[t - 273.15 for t in daily_Troom],
        mode='lines',
        name='실내온도 (°C)'
    ))
    fig_daily_temp.add_trace(go.Scatter(
        x=hours,
        y=[t - 273.15 for t in daily_Toutdoor],
        mode='lines',
        name='외기온도 (°C)',
        line=dict(dash='dash')
    ))
    fig_daily_temp.update_layout(
        title=f'Daily Temperature Variation ({selected_date})',
        xaxis=dict(
            title='Hour of Day',
            tickmode='array',
            ticktext=[f'{h:02d}:00' for h in hours],
            tickvals=hours,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Temperature (°C)',
            gridcolor='lightgray'
        ),
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white'
    )
    fig_daily_temp.write_image("daily_temperature_variation.png")
    fig_daily_temp.show()

    # 2) 일간 에너지 부하 그래프
    fig_daily_energy = go.Figure()
    fig_daily_energy.add_trace(go.Scatter(
        x=hours,
        y=daily_qt,
        mode='lines',
        name='에너지 부하 (kW)'
    ))
    fig_daily_energy.update_layout(
        title=f'Daily Energy Load ({selected_date})',
        xaxis=dict(
            title='Hour of Day',
            tickmode='array',
            ticktext=[f'{h:02d}:00' for h in hours],
            tickvals=hours,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Energy Load (kW)',
            gridcolor='lightgray'
        ),
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white'
    )
    fig_daily_energy.write_image("daily_energy_load.png")
    fig_daily_energy.show()

    # 시간별 활성 피복재 데이터프레임 출력
    cover_df = pd.DataFrame({
        'Hour': [f'{h:02d}:00' for h in hours],
        'Active Covering': daily_covers
    })
    print("\n[일간 활성 피복재 정보]")
    print(cover_df)

##########################################
# 7) 에너지 부하 결과 DataFrame 구성 함수 #
##########################################
def display_qt_dataframe(df):
    qt_df = df[['hour', 'timestamp', 'qt', 'Troom', 'Yt', 'transGlass', 'active_covers']].copy()
    qt_df['date'] = qt_df['timestamp'].dt.date
    qt_df['time'] = qt_df['timestamp'].dt.strftime('%H:%M')
    qt_df = qt_df[['hour', 'date', 'time', 'qt', 'Troom', 'Yt', 'transGlass', 'active_covers']]
    qt_df.columns = ['시간', '날짜', '시각', '에너지 부하(kW)', '실내온도(K)', '열관류율', '광투과율', '활성 피복재']
    return qt_df

######################
# 8) 메인 실행 함수  #
######################
def main():
    try:
        # 1) 시뮬레이션 계산
        (df, qt, Toutdoor, radSolar, Troom, params,
         rhoAir, cAir, mHouse, Yt_values, transGlass_values, active_covers
        ) = process_jeonju_data()

        # 2) 콘솔에서 단일 명령 입력
        print("\n[사용 예시]")
        print("  graph 010102  => 1월 1일 02시 그래프 표시")
        print("  data 010102   => 1월 1일 02시 데이터(계산 상세, 일간 DF) 표시")
        print("  graph data 010102 => 그래프와 데이터 둘 다 표시")
        print("  graph 3003    => 1년 중 3003번째 시간 그래프 표시")
        print("  data 3003     => 3003번째 시간 데이터 표시")
        print("  graph data 3003 => 둘 다 표시\n")

        cmd = input("명령을 입력하세요: ").strip()
        parts = cmd.split()
        if len(parts) < 2:
            print("형식에 맞지 않는 입력입니다. 예) graph 010102 혹은 data 3003 등")
            return

        # 마지막 토큰이 시간 정보(6자리 or 4자리)
        time_str = parts[-1]
        # 나머지 토큰에서 graph / data 여부 확인
        mode_graph = ("graph" in parts[:-1]) or ("graph" in parts)
        mode_data = ("data" in parts[:-1]) or ("data" in parts)

        # 시간 파싱
        if time_str.isdigit():
            if len(time_str) == 6:
                # 예: 010102 => 1월1일 02시
                month = int(time_str[0:2])
                day = int(time_str[2:4])
                hour_of_day = int(time_str[4:6])
                try:
                    dt_obj = datetime.datetime(2024, month, day, hour_of_day)
                except ValueError:
                    print("유효하지 않은 날짜/시간입니다.")
                    return
                # 해당 시점의 hour_index 찾기
                row_idx = df.index[df['timestamp'] == dt_obj]
                if len(row_idx) == 0:
                    print("해당 날짜/시간에 해당하는 시뮬레이션 데이터가 없습니다.")
                    return
                hour_index = row_idx[0]
                # 일간 그래프를 위해 해당 날짜의 시작 인덱스
                start_hour_index = df[df['timestamp'].dt.date == dt_obj.date()].index[0]

            else:
                # 예: 3003 => hour=3003
                hour_index = int(time_str) - 1
                if not (0 <= hour_index < 8784):
                    print("시간(hour) 범위는 1~8784입니다.")
                    return
                dt_obj = df.loc[hour_index, 'timestamp']
                start_hour_index = df[df['timestamp'].dt.date == dt_obj.date()].index[0]
                hour_of_day = dt_obj.hour
            # 이제 graph/data 여부에 따라 작업
            if mode_graph:
                display_daily_graphs(
                    selected_date=dt_obj.date(),
                    start_hour_index=start_hour_index,
                    Troom=Troom,
                    Toutdoor=Toutdoor,
                    qt=qt,
                    active_covers=active_covers
                )
            if mode_data:
                # 상세 계산(해당 hour_index)
                calc = show_detailed_calculation(
                    hour_index=hour_index,
                    df=df,
                    params=params,
                    Toutdoor=Toutdoor,
                    radSolar=radSolar,
                    Troom=Troom,
                    qt_array=qt,
                    rhoAir=rhoAir,
                    cAir=cAir,
                    mHouse=mHouse,
                    Yt_values=Yt_values,
                    transGlass_values=transGlass_values,
                    active_covers=active_covers
                )
                # 콘솔에 보기 좋게 출력
                print(f"\n=== [시간 {hour_index+1} 상세 계산] ===")
                print(f"{calc['시간정보']}")
                print("\n[기본 데이터]")
                for k, v in calc['기본 데이터'].items():
                    print(f" - {k}: {v}")

                print("\n[Tsolair2 계산]")
                print(f"  [공식]: {calc['Tsolair2 계산']['공식']}")
                print(f"  [계산]: {calc['Tsolair2 계산']['계산']}")
                print(f"  [결과]: {calc['Tsolair2 계산']['결과']}")

                print("\n[에너지 부하 계산]")
                for comp_key, comp_val in calc['에너지 부하 계산'].items():
                    print(f"{comp_val['이름']}")
                    print(f"  [공식]: {comp_val['공식']}")
                    print(f"  [계산]: {comp_val['계산']}")
                    print(f"  [결과]: {comp_val['결과']}\n")

                print("[총 에너지 부하 상세 계산]")
                print(f"  [공식]: {calc['총 에너지 부하 상세 계산']['공식']}")
                print("  [상세 계산]:")
                for step_line in calc['총 에너지 부하 상세 계산']['상세 계산']:
                    print(f"    - {step_line}")
                print(f"  [최종 결과]: {calc['총 에너지 부하 상세 계산']['최종 결과']}")

                print("\n[다음 시간의 실내온도 계산]")
                if isinstance(calc['실내온도 상세 계산'], dict) and '공식' in calc['실내온도 상세 계산']:
                    troom_dict = calc['실내온도 상세 계산']
                    print(f"  [공식]: {troom_dict['공식']}")
                    print("  [단계별 계산]:")
                    for step in troom_dict['단계별 계산']:
                        print(f"    - {step}")
                    print(f"  [최종 결과]: {troom_dict['최종 결과']}")
                else:
                    print(f"  {calc['실내온도 상세 계산']}")

                # 해당 날짜의 일간 DF 출력
                daily_mask = df['timestamp'].dt.date == dt_obj.date()
                daily_df = df.loc[daily_mask, ['hour', 'timestamp', 'qt', 'Troom', 'Yt', 'transGlass', 'active_covers']]
                print("\n[일간 데이터프레임]")
                print(daily_df.head(24))
        else:
            print("시간 정보를 숫자로 입력해야 합니다. 예) 010102 혹은 3003")

    except FileNotFoundError:
        print("'Jeonju_data.xlsx' 파일을 찾을 수 없습니다. 경로 또는 파일명을 확인해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")

if __name__ == '__main__':
    main()
