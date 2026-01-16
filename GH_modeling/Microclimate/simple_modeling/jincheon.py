import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def calculate_floor_u_value():  # 온실 바닥면의 열관류율 계수 산정 (Ug)
    k_concrete = 6.23 * (1000 / 3600)
    k_gravel   = 7.20 * (1000 / 3600)
    k_sand     = 6.29 * (1000 / 3600)

    d_concrete = 0.30
    d_gravel   = 0.20
    d_sand     = 0.35

    R_concrete = d_concrete / k_concrete
    R_gravel   = d_gravel   / k_gravel
    R_sand     = d_sand     / k_sand

    R_total = R_concrete + R_gravel + R_sand
    Ug = 1.0 / R_total

    return Ug

def calculate_covering_properties(radSolar, Troom, Ttarget=20+273.15):

    covers = ['PO필름']
    transGlass = 0.71  # Old plastic 기준, New plastic = 0.86

    if radSolar > 0:
        if radSolar > 500:
            covers.append('AL스크린')
            transGlass *= 0.45  # AL 스크린 55% 차광
        if Troom < Ttarget:
            covers.append('보온커튼')
            transGlass *= 0.5   # 보온커튼(오래된 PP 스크린 가정)
    else:
        covers.append('AL스크린')
        covers.append('보온커튼')
        transGlass *= 0.45 * 0.5

    material_values = {
        'PO필름': 5.2,
        'AL스크린': 5.5,
        '보온커튼': 2.7
    }

    if len(covers) >= 2:
        Rt = sum(1 / material_values[cover] for cover in covers)
        Ti = 1 / Rt
        Ur = 1.2944 * Ti - 0.4205
    else:
        Ur = material_values[covers[0]]

    Uw = material_values['PO필름']

    return Ur, Uw, transGlass, covers

def process_greenhouse_data():

    df = pd.read_excel("climate/jincheon_greenhouse.xlsx")

    if '내부 습도' in df.columns and 'Humi' not in df.columns:
        df.rename(columns={'내부 습도': 'Humi'}, inplace=True)

    for col in ['외부 온도', '일사', '외부 습도', '내부 온도', '내부 습도']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['외부 온도', '일사', '외부 습도', '내부 온도', '내부 습도']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    df['날짜&시간'] = pd.to_datetime(df['날짜&시간'])
    n_hours = len(df)
    df['hour'] = range(n_hours)

    Toutdoor = df['외부 온도'].values + 273.15
    radSolar = df['일사'].values # MJ/m2 * 277.78 = W/m2

    Tground = 20 + 273.15
    P_atm = 101325

    lengthHouse = 100
    widthHouse  = 46
    heightHouse = 4
    areaHouse   = lengthHouse * widthHouse
    surfaceHouse = (widthHouse * heightHouse + lengthHouse * heightHouse) * 2
    volumeHouse  = areaHouse * heightHouse

    Chouse = 500000  # 온실의 열용량(kJ/k)
    CWhouse = 500000 # 온실의 수분용량(kg)

    agh = 0.21  # 태양 단파 복사 흡수율(방사율) - Kirchhoff's laws of thermal radiation
    Ttarget = 20 + 273.15

    Ug = calculate_floor_u_value()

    Troom = np.zeros(n_hours)
    rhoAir = np.zeros(n_hours)
    cAir = np.zeros(n_hours)
    mHouse = np.zeros(n_hours)

    Ur_values = np.zeros(n_hours)
    Uw_values = np.zeros(n_hours)
    transGlass_values = np.zeros(n_hours)
    active_covers = [None] * n_hours

    qt = np.zeros(n_hours)
    ach_array = np.zeros(n_hours)

    RHin = np.zeros(n_hours)
    RHin[0] = 0.8
    Wpl_values = np.zeros(n_hours)
    Wvt_values = np.zeros(n_hours)
    Wr_values  = np.zeros(n_hours)

    # Development of a strawberry transpiration model based on a simplified Penman–Monteith model under different irrigation regimes
    a_coeff = 0.0461232 # 회귀 계수
    b_coeff = 0.0166169 # 회귀 계수
    LAI     = 2.0
    K_val   = 0.86 # 광 감쇠 계수

    Troom[0] = 20 + 273.15
    rhoAir[0] = PropsSI("D", "T", Troom[0], "P", P_atm, "air")
    cAir[0]   = PropsSI("C", "T", Troom[0], "P", P_atm, "air") / 1000
    mHouse[0] = volumeHouse * rhoAir[0]
    qt[0]     = 0

    for i in range(n_hours):
        if i > 0:
            rhoAir[i] = PropsSI("D", "T", Troom[i], "P", P_atm, "air")
            cAir[i]   = PropsSI("C", "T", Troom[i], "P", P_atm, "air") / 1000
            mHouse[i] = volumeHouse * rhoAir[i]

        Ur, Uw, transGlass, cov = calculate_covering_properties(radSolar[i], Troom[i])
        Ur_values[i] = Ur
        Uw_values[i] = Uw
        transGlass_values[i] = transGlass
        active_covers[i] = cov

        T_c    = Troom[i] - 273.15
        RHin_p = RHin[i] * 100.0

        if (T_c >= 25.0) or (RHin_p >= 70):     # ACH는 시간당 공기 교환량으로 자연환기 및 강제환기 모두 고려한 값임
            current_ACH = 30                    # 온도가 높거나 습도가 높은 낮에 창을 계속 열어둠(ACH값이 높은 이유)
        elif (20 <= T_c < 25) or (50.0 <= RHin_p < 70.0):
            current_ACH = 20
        elif (T_c < 20.0) or (RHin_p < 50.0):   # 온도가 낮은 밤에 창을 닫음(밤에는 온도가 20도 밑으로 떨어짐)
            current_ACH = 1                     # 밤에는 온실의 창을 닫지만 고습을 피하기 위해 환기팬을 가동함(ACH가 커지는 이유)

        ach_array[i] = current_ACH

        Tsolair2 = Toutdoor[i] + (agh * radSolar[i]) / 17

        qRad = (transGlass_values[i] * areaHouse * radSolar[i]) / 1000

        qRoof = (Tsolair2 - Troom[i]) * Ur_values[i] * areaHouse / 1000        # Ur
        qFloor = (Tground - Troom[i]) * Ug * areaHouse / 1000                   # Ug
        qSideWall = (Tsolair2 - Troom[i]) * Uw_values[i] * surfaceHouse / 1000            # Uw
        qVent = current_ACH * (Toutdoor[i] - Troom[i]) * mHouse[i] * cAir[i] / 3600

        qt[i] = qRad + qRoof + qFloor + qSideWall + qVent

        if i < n_hours - 1:
            Troom[i + 1] = Troom[i] + (qt[i] / (mHouse[i] * cAir[i] + Chouse)) * 3600

        # ===== 수분 쪽 기존 로직 유지 =====
        T_for_sat_in = max(Troom[i], 273.15)
        p_sat_in = PropsSI("P", "T", T_for_sat_in, "Q", 0, "Water")
        Win = 0.622 * (RHin[i] * p_sat_in) / (P_atm - (RHin[i] * p_sat_in))

        T_for_sat_out = max(Toutdoor[i], 273.15)
        p_sat_out = PropsSI("P", "T", T_for_sat_out, "Q", 0, "Water")
        RHout = df['외부 습도'].iloc[i] / 100
        Wout = 0.622 * (RHout * p_sat_out) / (P_atm - (RHout * p_sat_out))

        VPD = (p_sat_in / 1000.0) * (1 - RHin[i])

        qsol_wm2 = radSolar[i] * transGlass_values[i]
        qsol_j = qsol_wm2 * 3600 * areaHouse
        h_fg = PropsSI("H", "T", T_for_sat_in, "Q", 1, "Water") - PropsSI("H", "T", T_for_sat_in, "Q", 0, "Water")
        qsol_kg = qsol_j / h_fg if h_fg > 0 else 0
        Aterm = (a_coeff * (1 - np.exp(-K_val * LAI)))
        Bterm = (b_coeff * LAI * VPD * areaHouse)
        Wpl = Aterm * qsol_kg + Bterm

        Wvt = current_ACH * mHouse[i] * (Win - Wout)
        Wr  = (Wpl - Wvt) / (mHouse[i] + CWhouse)

        Wpl_values[i] = Wpl
        Wvt_values[i] = Wvt
        Wr_values[i]  = Wr

        if i < n_hours - 1:
            new_W = Win + Wr
            T_next = max(Troom[i + 1], 273.15)
            p_sat_in_next = PropsSI("P", "T", T_next, "Q", 0, "Water")
            RHin[i + 1] = max(0.0, min(new_W * P_atm / (p_sat_in_next * (new_W + 0.622)), 1.0))

    # ===== 결과 df 컬럼 정리 =====
    df['Troom'] = Troom
    df['rhoAir'] = rhoAir
    df['cAir'] = cAir

    df['Ur'] = Ur_values                # 천장 U값
    df['transGlass'] = transGlass_values
    df['active_covers'] = [','.join(c) for c in active_covers]

    df['qt'] = qt
    df['Temp_K'] = df['내부 온도'] + 273.15

    df['RHin'] = RHin * 100
    df['Wpl'] = Wpl_values
    df['Wvt'] = Wvt_values
    df['Wr']  = Wr_values
    df['RHout'] = df['외부 습도']

    # 파라미터 묶음 (열관류율 관련 키도 새로 정의)
    params = {
        'areaHouse': areaHouse,
        'surfaceHouse': surfaceHouse,
        'volumeHouse': volumeHouse,
        'Tground': Tground,
        'ACH': ach_array,
        'agh': agh,
        'Chouse': Chouse,
        'CWhouse': CWhouse,
        'Ttarget': Ttarget,
        'Ug': Ug,
        'a_coeff': a_coeff,
        'b_coeff': b_coeff,
        'K_val': K_val,
        'LAI': LAI
    }

    return (df, qt, Toutdoor, radSolar, Troom, params,
        rhoAir, cAir, mHouse,
        Ur_values, Uw_values,
        transGlass_values, active_covers,
        ach_array)

def prepare_export_dataframe(df):
    export_df = df.copy()
    export_df['일시'] = export_df.apply(
        lambda row: f"{row['날짜&시간'].year}년 {row['날짜&시간'].month}월 {row['날짜&시간'].day}일 {row['날짜&시간'].hour:02d}시({int(row['hour'])})",
        axis=1
    )
    if '외부 온도' in export_df.columns:
        export_df['외부 온도'] = export_df['외부 온도'].apply(lambda x: f"{x:.1f}°C")
    if '내부 온도' in export_df.columns:
        export_df['내부 온도'] = export_df['내부 온도'].apply(lambda x: f"{x:.1f}°C")
    if 'Troom' in export_df.columns:
        export_df['Troom(°C)'] = export_df['Troom'].apply(lambda x: f"{(x - 273.15):.2f}°C")
    if '일사' in export_df.columns:
        export_df['일사(W/m^2)'] = export_df['일사']

    drop_cols = ['rhoAir', 'cAir', 'hour', 'Temp_K', '외부 습도']
    export_df.drop(columns=[c for c in drop_cols if c in export_df.columns], inplace=True, errors='ignore')
    return export_df

def plot_troom_vs_temp_regression(troom_c, temp_c, title="", save_graph=False, filename=None):
    if len(troom_c) < 2:
        print("데이터 부족 -> 외부 온도 회귀분석 불가.")
        return
    slope, intercept = np.polyfit(troom_c, temp_c, 1)
    predicted = slope * troom_c + intercept
    r2 = r2_score(temp_c, predicted)
    rmse = np.sqrt(mean_squared_error(temp_c, predicted))
    plt.figure()
    plt.scatter(troom_c, temp_c, color='green', label='Data', alpha=0.5)
    x_line = np.linspace(min(troom_c), max(troom_c), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='black', label=f'Regression line\nSlope={slope:.3f}\nR²={r2:.3f}\nRMSE={rmse:.3f}')
    plt.xlabel('Measured inside temperature(°C)')
    plt.ylabel('predicted inside temperature(°C)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_graph and filename:
        plt.savefig(filename)
        print(f"[외부 온도] 회귀분석 그래프 -> {filename} 저장.")
    plt.show()

def plot_rhin_vs_humi_regression(rhin, humi, title="", save_graph=False, filename=None):
    mask = ~np.isnan(rhin) & ~np.isnan(humi)
    if mask.sum() < 2:
        print("데이터 부족 -> [외부 습도] 회귀분석 불가.")
        return
    x_data = rhin[mask]
    y_data = humi[mask]
    slope, intercept = np.polyfit(x_data, y_data, 1)
    predicted = slope * x_data + intercept
    r2 = r2_score(y_data, predicted)
    rmse = np.sqrt(mean_squared_error(y_data, predicted))
    plt.figure()
    plt.scatter(x_data, y_data, alpha=0.5, color='green', label='Data')
    x_line = np.linspace(min(x_data), max(x_data), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='black', label=f'Regression line\nSlope={slope:.3f}\nR²={r2:.3f}\nRMSE={rmse:.3f}')
    plt.xlabel('Measured inside relative humidity(%)')
    plt.ylabel('predicted inside relative humidity(%)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_graph and filename:
        plt.savefig(filename)
        print(f"[외부 습도] 회귀분석 그래프 -> {filename} 저장.")
    plt.show()

def display_daily_temperature_graphs(selected_date, start_hour_index, df, Troom, Toutdoor, qt, save_graph=False):
    daily_slice = slice(start_hour_index, start_hour_index + 24)
    daily_Troom = Troom[daily_slice]
    daily_Toutdoor = Toutdoor[daily_slice]
    daily_Temp_K = df['Temp_K'].values[daily_slice]
    daily_qt = qt[daily_slice]

    hours_full = np.arange(24)
    n = len(daily_Troom)
    hours = hours_full[:n]

    plt.figure()
    plt.plot(hours, daily_qt[:n], color='orange', label='Thermal Energy Load(kWh)')
    plt.title(f"Thermal Energy Load")
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy Load(kWh)')
    plt.xlim(0, 23)
    plt.xticks(hours_full)
    plt.grid(True); plt.legend(); plt.tight_layout()
    if save_graph:
        fn = f"energy_load_{selected_date.strftime('%Y%m%d')}.png"
        plt.savefig(fn); print(f"Thermal Energy Load Graph -> {fn} 저장.")
    plt.show()

    plt.figure()
    plt.plot(hours, (daily_Troom[:n] - 273.15),   color='black', label='Predicted inside temperature(°C)')
    plt.plot(hours, (daily_Temp_K[:n] - 273.15),  color='blue',  label='Measured inside temperature(°C)')
    plt.plot(hours, (daily_Toutdoor[:n] - 273.15),color='red',   label='Measured outside temperature(°C)')
    plt.title(f"Predicted & Measured (Inside/Outside) Temperature")
    plt.xlabel('Hour of Day')
    plt.ylabel('Temperature(°C)')
    plt.xlim(0, 23)
    plt.xticks(hours_full)
    plt.grid(True); plt.legend(); plt.tight_layout()
    if save_graph:
        fn = f"temp_all_{selected_date.strftime('%Y%m%d')}.png"
        plt.savefig(fn); print(f"Temperature Combined Graph -> {fn} 저장.")
    plt.show()

    troom_c = daily_Troom - 273.15
    temp_c = daily_Temp_K - 273.15
    plot_troom_vs_temp_regression(
        troom_c, temp_c,
        title=f"Measured vs Predicted inside temperature Regression",
        save_graph=save_graph,
        filename=f"regression_{selected_date.strftime('%Y%m%d')}.png" if save_graph else None
    )

def display_daily_moisture_graphs(selected_date, start_hour_index, df, save_graph=False):
    daily_slice = slice(start_hour_index, start_hour_index + 24)
    daily_Wpl = df['Wpl'].values[daily_slice]
    daily_RHin = df['RHin'].values[daily_slice]
    daily_Humi = df['Humi'].values[daily_slice]
    daily_RHout = df['RHout'].values[daily_slice]

    hours_full = np.arange(24)
    n = len(daily_RHin)
    hours = hours_full[:n]

    plt.figure()
    plt.plot(hours, daily_Wpl[:n], color='orange', label='Transpiration(kg/h)')
    plt.title(f"Transpiration")
    plt.xlabel('Hour of Day')
    plt.ylabel('Transpiration(kg/h)')
    plt.xlim(0, 23)
    plt.xticks(hours_full)
    plt.grid(True); plt.legend(); plt.tight_layout()
    if save_graph:
        fn = f"Transpiration_{selected_date.strftime('%Y%m%d')}.png"
        plt.savefig(fn); print(f"Transpiration Graph -> {fn} 저장.")
    plt.show()

    plt.figure()
    plt.plot(hours, daily_RHin[:n],  color='black', label='Predicted inside relative humidity(%)')
    plt.plot(hours, daily_Humi[:n],  color='blue',  label='Measured inside relative humidity(%)')
    plt.plot(hours, daily_RHout[:n], color='red',   label='Measured outside relative humidity(%)')
    plt.title(f"Predicted & Measured (Inside/Outside) Relative Humidity")
    plt.xlabel('Hour of Day')
    plt.ylabel('Relative Humidity(%)')
    plt.xlim(0, 23)
    plt.xticks(hours_full)
    plt.grid(True); plt.legend(); plt.tight_layout()
    if save_graph:
        fn = f"rh_all_{selected_date.strftime('%Y%m%d')}.png"
        plt.savefig(fn); print(f"RH Combined Graph -> {fn} 저장.")
    plt.show()

    plot_rhin_vs_humi_regression(
        daily_RHin, daily_Humi,
        title=f"Measured vs Predicted inside relative humidity Regression",
        save_graph=save_graph,
        filename=f"regression_{selected_date.strftime('%Y%m%d')}.png" if save_graph else None
    )

def display_daily_graphs(selected_date, start_hour_index, df, Troom, Toutdoor, qt, save_graph=False):
    display_daily_temperature_graphs(selected_date, start_hour_index, df, Troom, Toutdoor, qt, save_graph)
    display_daily_moisture_graphs(selected_date, start_hour_index, df, save_graph)

def parse_time_str(input_str, df, allow_nearest=False):
    s = input_str.strip()
    if not s.isdigit():
        print(f"숫자형식 아님: {s}")
        return None

    try:
        val = int(s)
        if len(s) <= 5 and 1 <= val <= len(df):
            return val - 1
    except:
        pass

    dt_series = df['날짜&시간']

    if len(s) == 10:
        fmt_candidates = ['%Y%m%d%H']
    elif len(s) == 8:
        fmt_candidates = ['%y%m%d%H']
    elif len(s) == 6:
        base_year = int(pd.to_datetime(dt_series.iloc[0]).strftime('%Y'))
        s = f"{base_year}{s}"
        fmt_candidates = ['%Y%m%d%H']
    else:
        print(f"지원하지 않는 길이: {s} (len={len(s)})")
        return None

    target_dt = None
    for fmt in fmt_candidates:
        try:
            target_dt = pd.to_datetime(datetime.strptime(s, fmt))
            break
        except Exception:
            continue
    if target_dt is None:
        print("입력 날짜/시간 파싱 실패.")
        return None

    min_dt = dt_series.min()
    max_dt = dt_series.max()
    target_key = target_dt.strftime('%Y%m%d%H')
    key = dt_series.dt.strftime('%Y%m%d%H')

    exact_idx = df.index[key == target_key]
    if len(exact_idx) > 0:
        return int(exact_idx[0])

    if allow_nearest:
        deltas = (dt_series - target_dt).abs()
        near_idx = int(deltas.idxmin())
        near_dt = dt_series.iloc[near_idx]
        print(f"[주의] 정확히 일치 없음 → 가장 가까운 시간으로 이동: {near_dt:%Y-%m-%d %H} (index {near_idx})")
        return near_idx
    else:
        if target_dt < min_dt or target_dt > max_dt:
            print(f"[오류] 입력 {target_dt:%Y%m%d%H} 는 데이터 범위({min_dt:%Y%m%d%H} ~ {max_dt:%Y%m%d%H}) 밖입니다.")
        else:
            print(f"[오류] 입력 {target_dt:%Y%m%d%H} 시각이 데이터에 ‘정확히’ 존재하지 않습니다.")
        return None

def show_detailed_calculation(hour_index, df, params, Toutdoor, radSolar, Troom, qt_array,
                                rhoAir, cAir, mHouse,
                                Ur_values, Uw_values,
                                transGlass_values, active_covers,
                                ach_array):

    try:
        timestamp = df['날짜&시간'].iloc[hour_index]
        date_str = timestamp.strftime("%Y년 %m월 %d일 %H시")
        outdoor_temp = Toutdoor[hour_index]
        solar_rad = radSolar[hour_index]
        room_temp = Troom[hour_index]
        current_rhoAir = rhoAir[hour_index]
        current_cAir = cAir[hour_index]
        current_mHouse = mHouse[hour_index]

        current_Ur = Ur_values[hour_index]
        current_Uw = Uw_values[hour_index]
        current_transGlass = transGlass_values[hour_index]
        current_covers = active_covers[hour_index]
        current_ACH = ach_array[hour_index]

        Ug = params['Ug']

        tsolair2 = outdoor_temp + (params['agh'] * solar_rad) / 17

        q_rad = (current_transGlass * params['areaHouse'] * solar_rad) / 1000
        q_roof = (tsolair2 - room_temp) * current_Ur * params['areaHouse'] / 1000
        q_floor = (params['Tground'] - room_temp) * Ug * params['areaHouse'] / 1000
        q_sidewall = (tsolair2 - room_temp) * current_Uw * params['surfaceHouse'] / 1000
        q_vent = current_ACH * (outdoor_temp - room_temp) * current_mHouse * current_cAir / 3600

        qt_value = qt_array[hour_index]

        qvent_calc_str = f"{current_ACH:.2f} * ({outdoor_temp:.2f}-{room_temp:.2f})* {current_mHouse:.2f}* {current_cAir:.4f}/3600"

        qt_calculation = {
            "공식": "qt= qRad+ qRoof+ qFloor+ qSideWall+ qVent",
            "상세 계산": [
                f"태양 복사열 입사량(qRad)= {q_rad:.2f} kWh",
                f"지붕면 관류 열량(qRoof)= {q_roof:.2f} kWh",
                f"바닥면 관류 열량(qFloor)= {q_floor:.2f} kWh",
                f"측벽 관류 열량(qSideWall)= {q_sidewall:.2f} kWh",
                f"틈새 환기 전열량(qVent)= {q_vent:.2f} kWh"
            ],
            "qVent 계산식": f"qVent= {qvent_calc_str} = {q_vent:.2f} kWh",
            "최종 결과": f"{qt_value:.2f} kWh"
        }

        if hour_index < len(df) - 1:
            denominator = current_mHouse * current_cAir + params['Chouse']
            numer = qt_array[hour_index]
            delta_t = numer / denominator * 3600
            next_troom = room_temp + delta_t
            troom_calc_detail = [
                f"1) mHouse*cAir+Chouse= {current_mHouse:.2f}*{current_cAir:.4f}+{params['Chouse']:.2f}= {denominator:.2f}",
                f"2) qt/(mHouse*cAir+Chouse)*3600= {numer:.2f}/{denominator:.2f}*3600= {delta_t:.4f}",
                f"3) 최종: {room_temp:.2f}+{delta_t:.4f}= {next_troom:.2f}K"
            ]
            troom_calculation = {
                "공식": "Troom[i+1]= Troom[i]+ (qt[i]/(mHouse*cAir+Chouse))*3600",
                "단계별 계산": troom_calc_detail,
                "최종 결과": f"{next_troom:.2f}K ({next_troom - 273.15:.2f}°C)"
            }
        else:
            troom_calculation = {"마지막 시간대": "다음 시간 외부 온도 계산 불가"}

        calculations = {
            "시간정보": f"Hour {hour_index + 1}/ {date_str}",
            "기본 데이터": {
                "외기외부 온도": f"{outdoor_temp:.2f}K({outdoor_temp - 273.15:.2f}°C)",
                "일사량(solar_rad)": f"{solar_rad:.2f}W/m²",
                "실내외부 온도(Troom)": f"{room_temp:.2f}K({room_temp - 273.15:.2f}°C)",
                "지중외부 온도(Tground)": f"{params['Tground']:.2f}K({params['Tground'] - 273.15:.2f}°C)",
                "유효 피복재(지붕)": f"{current_covers}",
                "광투과율(transGlass)": f"{current_transGlass:.4f}",
                "천장 열관류율(Ur)": f"{current_Ur:.4f}",
                "측벽 열관류율(Uw)": f"{current_Uw:.4f}",
                "바닥 열관류율(Ug)": f"{params['Ug']:.4f}",
                "현재 ACH": f"{current_ACH:.2f}"
            },
            "Tsolair2 계산": {
                "공식": "Tsolair2= Tout+(agh*radSolar)/17",
                "계산": f"{outdoor_temp:.2f}+({params['agh']:.2f}*{solar_rad:.2f})/17",
                "결과": f"{tsolair2:.2f}K"
            },
            "열 에너지 부하 계산": {
                "qRad": {
                    "공식": "qRad= (transGlass*areaHouse*radSolar)/1000",
                    "계산": f"({current_transGlass:.4f}*{params['areaHouse']:.2f}*{solar_rad:.2f})/1000",
                    "결과": f"{q_rad:.2f} kWh"
                },
                "qRoof": {
                    "공식": "qRoof= (Tsolair2- Troom)*Ur*areaHouse/1000",
                    "계산": f"({tsolair2:.2f}-{room_temp:.2f})*{current_Ur:.4f}*{params['areaHouse']:.2f}/1000",
                    "결과": f"{q_roof:.2f} kWh"
                },
                "qFloor": {
                    "공식": "qFloor= (Tground- Troom)*Ug*areaHouse/1000",
                    "계산": f"({params['Tground']:.2f}-{room_temp:.2f})*{Ug:.4f}*{params['areaHouse']:.2f}/1000",
                    "결과": f"{q_floor:.2f} kWh"
                },
                "qSideWall": {
                    "공식": "qSideWall= (Tsolair2- Troom)*Uw*surfaceHouse/1000",
                    "계산": f"({tsolair2:.2f}-{room_temp:.2f})*{current_Uw:.4f}*{params['surfaceHouse']:.2f}/1000",
                    "결과": f"{q_sidewall:.2f} kWh"
                },
                "qVent": {
                    "공식": "qVent= ACH*(Tout- Troom)* mHouse*cAir/3600",
                    "계산": qvent_calc_str,
                    "결과": f"{q_vent:.2f} kWh"
                }
            },
            "총 열 에너지 부하 상세 계산": qt_calculation,
            "실내외부 온도 상세 계산": troom_calculation
        }

        return calculations

    except Exception as e:
        print(f"show_detailed_calculation 오류: {e}")
        return {}

def show_detailed_moisture_calculation(hour_index, df, params, Troom, mHouse, ach_array):
    try:
        P_atm = 101325

        timestamp = df['날짜&시간'].iloc[hour_index]
        date_str = timestamp.strftime("%Y년 %m월 %d일 %H시")
        RHin_val = df['RHin'].iloc[hour_index] / 100
        Humi_val = df['Humi'].iloc[hour_index]
        RHout_val = df['RHout'].iloc[hour_index] / 100
        Wpl_val = df['Wpl'].iloc[hour_index]
        Wvt_val = df['Wvt'].iloc[hour_index]
        Wr_val  = df['Wr'].iloc[hour_index]

        T_in = max(Troom[hour_index], 273.15)
        p_sat_in = PropsSI("P", "T", T_in, "Q", 0, "Water")
        Win = 0.622 * (RHin_val * p_sat_in) / (P_atm - (RHin_val * p_sat_in))

        T_out = df['외부 온도'].iloc[hour_index] + 273.15
        if T_out < 273.15:
            T_out = 273.15
        p_sat_out = PropsSI("P", "T", T_out, "Q", 0, "Water")
        Wout = 0.622 * (RHout_val * p_sat_out) / (P_atm - (RHout_val * p_sat_out))

        VPD = (p_sat_in / 1000.0) * (1 - RHin_val)
        current_ACH = ach_array[hour_index]
        areaHouse = params['areaHouse']
        qsol_wm2 = df['일사'].iloc[hour_index] * df['transGlass'].iloc[hour_index]
        qsol_j = qsol_wm2 * 3600
        h_fg = PropsSI("H", "T", T_in, "Q", 1, "Water") - PropsSI("H", "T", T_in, "Q", 0, "Water")
        qsol_kg = qsol_j / h_fg if h_fg > 0 else 0.0

        a_val = params['a_coeff']
        b_val = params['b_coeff']
        K_val = params['K_val']
        LAI   = params['LAI']
        Wpl_calc_str = f"[({a_val:.6f}*(1-exp(-{K_val:.3f}*{LAI})))]*{qsol_kg:.6f} + [({b_val:.6f}*{LAI}*({VPD:.5f}*1000*{areaHouse:.2f}/9.81))]"
        Wvt_calc_str = f"{current_ACH:.2f}*{mHouse[hour_index]:.2f}*({Win:.6f}-{Wout:.6f})"
        Wr_calc_str  = f"({Wpl_val:.6f}-{Wvt_val:.6f})/({mHouse[hour_index]:.6f}+{params['CWhouse']:.2f})"

        if hour_index < len(df) - 1:
            new_W = Win + Wr_val
            T_next = max(Troom[hour_index + 1], 273.15)
            p_sat_in_next = PropsSI("P", "T", T_next, "Q", 0, "Water")
            new_RHin_val = (new_W * P_atm) / (p_sat_in_next * (new_W + 0.622))
            next_rh_str = f"""
1) new_W = {Win:.6f}+{Wr_val:.6f} = {new_W:.6f}
2) p_sat_in_next = {p_sat_in_next:.2f}
3) new_RHin = ({new_W:.6f}*101325)/({p_sat_in_next:.2f}*({new_W:.6f}+0.622)) = {new_RHin_val:.6f}
=> {new_RHin_val * 100:.2f}%"""
        else:
            next_rh_str = "다음 시간 외부 습도 계산 불가"

        moisture_calc = {
            "시간정보": f"Hour {hour_index + 1}/{date_str}",
            "기본 외부 습도 데이터": {
                "예측 내부외부 습도(RHin)": f"{RHin_val * 100:.2f}%",
                "실측 내부외부 습도(Humi)": f"{Humi_val:.2f}%",
                "외부 외부 습도(RHout)": f"{RHout_val * 100:.2f}%"
            },
            "혼합비(Win,Wout)": {
                "Win": {
                    "공식": "Win = 0.622*(RHin*p_sat_in)/(P_atm-(RHin*p_sat_in))",
                    "계산": f"0.622*({RHin_val:.4f}*{p_sat_in:.2f})/(101325-{RHin_val:.4f}*{p_sat_in:.2f})",
                    "결과": f"{Win:.6f} kg/kg"
                },
                "Wout": {
                    "공식": "Wout = 0.622*(RHout*p_sat_out)/(P_atm-(RHout*p_sat_out))",
                    "계산": f"0.622*({RHout_val:.4f}*{p_sat_out:.2f})/(101325-{RHout_val:.4f}*{p_sat_out:.2f})",
                    "결과": f"{Wout:.6f} kg/kg"
                }
            },
            "VPD 계산": {
                "공식": "VPD = (p_sat_in/1000)*(1-RHin)",
                "계산": f"({p_sat_in:.2f}/1000)*(1-{RHin_val:.4f})",
                "결과": f"{VPD:.5f} kPa"
            },
            "qsol_kg 계산": {
                "공식": "qsol_kg = (qsol_wm2*3600)/h_fg",
                "계산": f"({qsol_wm2:.4f}*3600)/{h_fg:.2f}",
                "결과": f"{qsol_kg:.6f} kg/m²·h"
            },
            "Wpl (작물 증산량)": {
                "공식": "Wpl = [(a*(1-exp(-K*LAI)))*qsol_kg] + [(b*LAI*(VPD*1000*areaHouse/9.81))]",
                "계산": Wpl_calc_str,
                "결과": f"{Wpl_val:.6f} kg/h"
            },
            "Wvt (환기 수분 교환량)": {
                "공식": "Wvt = ACH*mHouse*(Win-Wout)",
                "계산": Wvt_calc_str,
                "결과": f"{Wvt_val:.6f} kg/h"
            },
            "Wr (총 혼합비 변화량)": {
                "공식": "Wr = (Wpl - Wvt)/(mHouse+CWhouse)",
                "계산": Wr_calc_str,
                "결과": f"{Wr_val:.8f} kg/kg·h"
            },
            "다음 시간의 실내 외부 습도(RHin)": next_rh_str
        }
        return moisture_calc

    except Exception as e:
        print(f"show_detailed_moisture_calculation 오류: {e}")
        return {}

def show_range_8graphs(start_index, end_index, df, Troom, Toutdoor, qt, save_graph=False):
    x_range = range(start_index, end_index + 1)
    sub_df = df.iloc[start_index:end_index + 1].copy()
    sub_Troom = Troom[start_index:end_index + 1]
    sub_Toutdoor = Toutdoor[start_index:end_index + 1]
    sub_qt = qt[start_index:end_index + 1]
    sub_Temp_K = sub_df['Temp_K'].values
    sub_Wpl = sub_df['Wpl'].values
    sub_RHin = sub_df['RHin'].values
    sub_Humi = sub_df['Humi'].values
    sub_RHout = sub_df['RHout'].values

    plt.figure()
    plt.plot(x_range, sub_qt, color='orange', label='Thermal Energy Load(kWh)')
    plt.title(f"Thermal Energy Load", fontsize=15)
    plt.xlabel('Hour Index', fontsize=15)
    plt.ylabel('Energy Load(kWh)', fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_graph:
        fn = f"range_energy_{start_index + 1}_{end_index + 1}.png"
        plt.savefig(fn)
        print(f"Thermal Energy Load Graph -> {fn} 저장.")
    plt.show()

    plt.figure()
    plt.plot(x_range, sub_Troom - 273.15,   color='red', label='Predicted inside temperature(°C)')
    plt.plot(x_range, sub_Temp_K - 273.15,  color='blue',  label='Measured inside temprature(°C)')
    plt.plot(x_range, sub_Toutdoor - 273.15,color='black',   label='Measured outside temperature(°C)')
    # plt.title(f"Predicted & Measured (Inside/Outside) Temperature", fontsize=15)
    plt.xlabel('Hour Index', fontsize=12); plt.ylabel('Temperature(°C)', fontsize=12)
    plt.grid(True); plt.legend(fontsize=10); plt.tight_layout()
    if save_graph:
        fn = f"range_temp_all_{start_index + 1}_{end_index + 1}.png"
        plt.savefig(fn); print(f"Temperature Combined Graph -> {fn} 저장.")
    plt.show()

    troom_c = sub_Troom - 273.15
    temp_c = sub_Temp_K - 273.15

    if len(troom_c) > 2:
        slope, intercept = np.polyfit(troom_c, temp_c, 1)
        predicted = slope * troom_c + intercept
        r2 = r2_score(temp_c, predicted)
        rmse = np.sqrt(mean_squared_error(temp_c, predicted))
        plt.figure()
        plt.scatter(troom_c, temp_c, alpha=0.5, color='green', label='Data')
        x_line = np.linspace(min(troom_c), max(troom_c), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='black', label=f"Regression line\nSlope={slope:.3f}\nR²={r2:.3f}\nRMSE={rmse:.3f}")
        plt.xlabel('Measured inside temperature(°C)', fontsize=12)
        plt.ylabel('Predicted inside temperature(°C)', fontsize=12)
        # plt.title(f"Measured & Predicted inside temprature Regression", fontsize=15)
        plt.legend(fontsize=10)
        plt.grid(True)
        if save_graph:
            fn = f"range_regression_{start_index + 1}_{end_index + 1}.png"
            plt.savefig(fn)
            print(f"Linear Regression Graph -> {fn} 저장.")
        plt.show()
    else:
        print("Data is not enough")

    plt.figure()
    plt.plot(x_range, sub_Wpl, color='orange', label='Transpiration(kg/h)')
    plt.title(f"Transpiration", fontsize=15)
    plt.xlabel('Hour Index', fontsize=15)
    plt.ylabel('Transpiration(kg/h)', fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_graph:
        fn = f"range_Transpiration_{start_index + 1}_{end_index + 1}.png"
        plt.savefig(fn)
        print(f"Transpiration Graph -> {fn} 저장.")
    plt.show()

    plt.figure()
    plt.plot(x_range, sub_RHin,  color='red', label='Predicted inside relative humidity(%)')
    plt.plot(x_range, sub_Humi,  color='blue',  label='Measured inside relative humidity(%)')
    plt.plot(x_range, sub_RHout, color='black',   label='Measured outside relative humidity(%)')
    # plt.title(f"Predicted & Measured (Inside/Outside) Relative Humidity", fontsize=15)
    plt.xlabel('Hour Index', fontsize=12)
    plt.ylabel('Relative Humidity(%)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    if save_graph:
        fn = f"range_rh_all_{start_index + 1}_{end_index + 1}.png"
        plt.savefig(fn)
        print(f"RH Combined Graph -> {fn} 저장.")
    plt.show()

    mask = ~np.isnan(sub_RHin) & ~np.isnan(sub_Humi)
    if mask.sum() > 2:
        x_data = sub_RHin[mask]
        y_data = sub_Humi[mask]
        slope, intercept = np.polyfit(x_data, y_data, 1)
        predicted = slope * x_data + intercept
        r2 = r2_score(y_data, predicted)
        rmse = np.sqrt(mean_squared_error(y_data, predicted))
        plt.figure()
        plt.scatter(x_data, y_data, alpha=0.5, color='green', label='Data')
        x_line = np.linspace(min(x_data), max(x_data), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='black', label=f"Regression line\nSlope={slope:.3f}\nR²={r2:.3f}\nRMSE={rmse:.3f}")
        plt.xlabel('Measured inside relative humidity(%)', fontsize=12)
        plt.ylabel('Predicted inside relative humidity(%)', fontsize=12)
        # plt.title(f"Measured & Predicted inside relative humidity Regression", fontsize=15)
        plt.legend(fontsize=10)
        plt.grid(True)
        if save_graph:
            fn = f"range_regression_{start_index + 1}_{end_index + 1}.png"
            plt.savefig(fn)
            print(f"Linear Regression Graph -> {fn} 저장.")
        plt.show()
    else:
        print("Data is not enough")

def main():
    try:
        (df, qt, Toutdoor, radSolar, Troom, params,
         rhoAir, cAir, mHouse, Ur_values, Uw_values, transGlass_values, active_covers,
         ach_array) = process_greenhouse_data()

        print("\n[사용 방법]")
        print(" - 특정 시점(ex: 100 or 23051200)")
        print(" - 특정 기간(ex: 1~24 or 23051200~24051123)")
        user_input = input("시점(기간) 입력:").strip()
        save_choice = input("그래프 & DF 저장(y/n): ").strip().lower()
        save_flag = (save_choice == 'y')

        if "~" in user_input:
            left_str, right_str = user_input.split("~")
            left_str = left_str.strip()
            right_str = right_str.strip()
            start_index = parse_time_str(left_str, df, allow_nearest=False)
            end_index = parse_time_str(right_str, df, allow_nearest=False)
            if start_index is None or end_index is None:
                return
            if start_index > end_index:
                start_index, end_index = end_index, start_index

            sub = df.iloc[start_index:end_index + 1]
            x = sub['RHin'].to_numpy()
            y = sub['Humi'].to_numpy()

            mask = ~np.isnan(x) & ~np.isnan(y)
            x_m = x[mask]; y_m = y[mask]

            if len(y_m) >= 2:
                slope, intercept = np.polyfit(x_m, y_m, 1)
                y_hat = slope * x_m + intercept
                r2 = r2_score(y_m, y_hat)
                rmse = np.sqrt(mean_squared_error(y_m, y_hat))
                print(f"\n[검증] RH(in) 예측 vs 실측  points={len(y_m)}")
                print(f"[검증] slope={slope:.3f}  intercept={intercept:.3f}")
                print(f"[검증] R²={r2:.3f}  RMSE={rmse:.3f}")
            else:
                print("\n[검증] 유효한 데이터 포인트가 부족합니다.")

            start_dt = df.loc[start_index, '날짜&시간']
            end_dt = df.loc[end_index, '날짜&시간']

            print("\n[입력 기간 인덱스 → 날짜/시간]")
            print(f" - 시작 {start_index + 1} → {start_dt:%Y-%m-%d %H시}")
            print(f" - 종료 {end_index + 1} → {end_dt:%Y-%m-%d %H시}")

            show_range_8graphs(start_index, end_index, df, Troom, Toutdoor, qt, save_graph=save_flag)

            sub_df = df.iloc[start_index:end_index + 1].copy()
            export_sub_df = prepare_export_dataframe(sub_df)
            if save_flag:
                chunk_size = 5000
                total_rows = len(export_sub_df)
                if total_rows > chunk_size:
                    num_parts = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)
                    for i in range(num_parts):
                        part_df = export_sub_df.iloc[i * chunk_size:(i + 1) * chunk_size]
                        fn = f"range_data_part{i + 1}.xlsx"
                        part_df.to_excel(fn, index=False)
                        print(f"기간 데이터 part{i + 1} -> {fn} 저장.")
                else:
                    export_sub_df.to_excel("range_data.xlsx", index=False)
                    print("기간 range_data.xlsx 저장.")

        else:
            hour_index = parse_time_str(user_input, df, allow_nearest=True)
            if hour_index is None:
                return

            detail_calc = show_detailed_calculation(hour_index, df, params, Toutdoor, radSolar,
                                                    Troom, qt, rhoAir, cAir, mHouse,
                                                    Ur_values, Uw_values, transGlass_values,
                                                    active_covers, ach_array
                                                    )
            if detail_calc:
                print(f"\n=== [시간 {hour_index + 1} 상세 계산 - 열 에너지] ===")
                print(detail_calc['시간정보'])
                print("\n[기본 데이터]")
                for k, v in detail_calc['기본 데이터'].items():
                    print(f" - {k}: {v}")
                print("\n[Tsolair2 계산]")
                print(f"  [공식]: {detail_calc['Tsolair2 계산']['공식']}")
                print(f"  [계산]: {detail_calc['Tsolair2 계산']['계산']}")
                print(f"  [결과]: {detail_calc['Tsolair2 계산']['결과']}")
                print("\n[열 에너지 부하 계산]")
                for comp_key, comp_val in detail_calc['열 에너지 부하 계산'].items():
                    if isinstance(comp_val, dict):
                        print(f"  [공식]: {comp_val['공식']}")
                        print(f"  [계산]: {comp_val['계산']}")
                        print(f"  [결과]: {comp_val['결과']}")
                        print()
                    else:
                        print(f"  {comp_key}: {comp_val}")
                print("[총 열 에너지 부하 상세 계산]")
                print(f"  [공식]: {detail_calc['총 열 에너지 부하 상세 계산']['공식']}")
                print("  [상세 계산]:")
                for step_line in detail_calc['총 열 에너지 부하 상세 계산']['상세 계산']:
                    print(f"    - {step_line}")
                print(f"  [최종 결과]: {detail_calc['총 열 에너지 부하 상세 계산']['최종 결과']}")

            detail_moist = show_detailed_moisture_calculation(hour_index, df, params, Troom, mHouse, ach_array)
            if detail_moist:
                print(f"\n=== [시간 {hour_index + 1} 상세 계산 - 수분] ===")
                print(detail_moist['시간정보'])
                print("\n[기본 외부 습도 데이터]")
                for kk, vv in detail_moist['기본 외부 습도 데이터'].items():
                    print(f" - {kk}: {vv}")
                print("\n[혼합비(Win,Wout) 계산]")
                for mix_key, mix_val in detail_moist['혼합비(Win,Wout)'].items():
                    print(f"  ({mix_key})")
                    print(f"    [공식]: {mix_val['공식']}")
                    print(f"    [계산]: {mix_val['계산']}")
                    print(f"    [결과]: {mix_val['결과']}")
                    print()
                print("[VPD 계산]")
                print(f"  [공식]: {detail_moist['VPD 계산']['공식']}")
                print(f"  [계산]: {detail_moist['VPD 계산']['계산']}")
                print(f"  [결과]: {detail_moist['VPD 계산']['결과']}")
                print("\n[Wpl (작물 증산량)]")
                print(f"  [공식]: {detail_moist['Wpl (작물 증산량)']['공식']}")
                print(f"  [계산]: {detail_moist['Wpl (작물 증산량)']['계산']}")
                print(f"  [결과]: {detail_moist['Wpl (작물 증산량)']['결과']}")
                print("\n[Wvt (환기 수분 교환량)]")
                print(f"  [공식]: {detail_moist['Wvt (환기 수분 교환량)']['공식']}")
                print(f"  [계산]: {detail_moist['Wvt (환기 수분 교환량)']['계산']}")
                print(f"  [결과]: {detail_moist['Wvt (환기 수분 교환량)']['결과']}")
                print("\n[Wr (총 혼합비 변화량)]")
                print(f"  [공식]: {detail_moist['Wr (총 혼합비 변화량)']['공식']}")
                print(f"  [계산]: {detail_moist['Wr (총 혼합비 변화량)']['계산']}")
                print(f"  [결과]: {detail_moist['Wr (총 혼합비 변화량)']['결과']}")
                print(f"\n[다음 시간의 실내 외부 습도(RHin)]: {detail_moist['다음 시간의 실내 외부 습도(RHin)']}")

            dt_obj = df.loc[hour_index, '날짜&시간']
            selected_date = dt_obj.date()
            daily_mask = df['날짜&시간'].dt.date == selected_date
            if not daily_mask.any():
                print(f"{selected_date} 날짜 데이터 없음.")
                return
            start_hour_index = df.index[daily_mask][0]
            display_daily_graphs(selected_date, start_hour_index, df, Troom, Toutdoor, qt, save_graph=save_flag)
            daily_df = df.loc[daily_mask].copy()
            export_daily_df = prepare_export_dataframe(daily_df)
            print("\n[해당 일자 데이터프레임]")
            print(daily_df[
                      ['hour', '날짜&시간', 'qt', 'Wpl', 'Wvt', 'Wr', 'Troom',
                       'Ur', 'transGlass', 'active_covers', 'Temp_K', 'RHin', 'Humi', 'RHout']])
            if save_flag:
                export_daily_df.to_excel("daily_data.xlsx", index=False)
                print("일간 데이터프레임 daily_data.xlsx 저장.")

    except FileNotFoundError:
        print("파일이 없습니다.")
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    main()

# 해당 데이터는 충청북도 진천군의 한 딸기 농가의 온실 환경 데이터로 농림축산식품부 공공데이터에서 가져 옴
# Tground(지온) 고정(영향이 미미함)
# LAI(엽면적) 고정(작물 및 작기를 알 수 없음)
# a와 b 계수는 작물 마다 다름(딸기와 파프리카 논문 참고하여 유의미한 증산량에 맞게 조정함)
# ACH는 자연환기 + 강제환기를 모두 고려한 값을 간단한 온도 범위를 기준으로 하나의 값으로 사용함
# 스크린 및 보온 커튼은 임시 로직을 사용함(제어 데이터를 알 수 없음)
# 기타 물성치 값은 임시로 지정(시설 정보를 알 수 없음) - 피복재는 비닐이지만, 두께 광투과율 등을 알 수 없음(기존 PO필름 사용)
# 온실의 열용량 및 수분 용량은 단위 면적 당 150Kg,air 정도로 서로 같음(계산 상 단위 일치, 실제로 다를 수 있음)
# 온실에서 천창 부분만 차광 스크린 및 보온 커튼 로직이 작동함(측창 부분은 해당 안됨)
# ACH가 높을수록 거의 1시간 내내 창을 열어 환기한다는 의미(+팬 등을 통한 강제 환기도 실시함)
# 비와 눈 등의 다른 기상은 고려하지 않음
