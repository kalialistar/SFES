import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from CoolProp.CoolProp import PropsSI
from datetime import datetime

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def calc_u(Rsi, Rse, layers):
    R_layers = 0.0
    for (thickness, k) in layers:
        R_layers += thickness / k
    R_total = Rsi + R_layers + Rse
    return 1.0 / R_total


def process_greenhouse_data(wall_thickness=0.05, ACH_value=1.0, PPFD_value=200.0):
    df = pd.read_excel("jeonju_greenhouse.xlsx")

    if '내부 습도' in df.columns and 'Humi' not in df.columns:
        df.rename(columns={'내부 습도': 'Humi'}, inplace=True)

    for col in ['외부 온도', '일사']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['외부 온도', '일사']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    df['날짜&시간'] = pd.to_datetime(df['날짜&시간'])
    n_hours = len(df)
    df['hour'] = range(n_hours)

    Toutdoor = df['외부 온도'].values + 273.15
    radSolar = df['일사'].values * 277.78

    lengthHouse = 4
    widthHouse = 2
    heightHouse = 3

    areaHouse = lengthHouse * widthHouse
    surfaceHouse = (widthHouse * heightHouse + lengthHouse * heightHouse) * 2
    volumeHouse = areaHouse * heightHouse

    Chouse = 1000

    P_ATM = 101325
    Tfloor = 20 + 273.15
    agh = 0.15

    Uw = calc_u(
        Rsi=0.110,
        Rse=0.043,
        layers=[(wall_thickness, 0.025)]
    )
    Ur = calc_u(
        Rsi=0.110,
        Rse=0.043,
        layers=[(0.05, 0.025)]
    )
    Ug = calc_u(
        Rsi=0.086,
        Rse=0.043,
        layers=[(0.1, 0.025)]
    )

    PPE = 2.5           # µmol/J
    PPFD = PPFD_value   # µmol/m²/s
    alamp = 1.0

    Ilamp = PPFD / PPE  # W/m²
    photoperiod_hours = 14
    lamp_on_start_hour = 8
    hour_of_day = df['날짜&시간'].dt.hour.values

    lamp_on = (((hour_of_day - lamp_on_start_hour) % 24) < photoperiod_hours).astype(int)

    lamp_kWh_per_h = alamp * Ilamp * areaHouse / 1000.0
    lamp_kWh_arr = lamp_kWh_per_h * lamp_on  # kWh

    Troom = np.zeros(n_hours)
    rhoAir = np.zeros(n_hours)
    cAir = np.zeros(n_hours)
    mHouse = np.zeros(n_hours)
    qt = np.zeros(n_hours)
    heat_kWh = np.zeros(n_hours)
    cool_kWh = np.zeros(n_hours)
    ach_array = np.zeros(n_hours)

    qLED_arr = np.zeros(n_hours)
    qRoof_arr = np.zeros(n_hours)
    qFloor_arr = np.zeros(n_hours)
    qSideWall_arr = np.zeros(n_hours)
    qVent_arr = np.zeros(n_hours)
    Tsolair_arr = np.zeros(n_hours)

    Troom[0] = 20 + 273.15

    rhoAir[0] = PropsSI("D", "T", Troom[0], "P", P_ATM, "air")
    cAir[0]  = PropsSI("C", "T", Troom[0], "P", P_ATM, "air") / 1000.0  # kJ/kgK
    mHouse[0] = volumeHouse * rhoAir[0]
    qt[0] = 0.0

    Th_set = 20 + 273.15
    Tc_set = 22 + 273.15

    for i in range(n_hours):

        rhoAir[i] = PropsSI("D", "T", Troom[i], "P", P_ATM, "air")
        cAir[i]  = PropsSI("C", "T", Troom[i], "P", P_ATM, "air") / 1000.0
        mHouse[i] = volumeHouse * rhoAir[i]

        current_ACH = ACH_value
        ach_array[i] = current_ACH

        Tsolair = Toutdoor[i] + (agh * radSolar[i]) / 17.0
        Tsolair_arr[i] = Tsolair

        qLED = lamp_kWh_per_h * lamp_on[i]  # kWh
        qRoof = (Tsolair - Troom[i]) * Ur * areaHouse / 1000.0
        qFloor = (Tfloor   - Troom[i]) * Ug * areaHouse / 1000.0
        qSideWall = (Tsolair - Troom[i]) * Uw * surfaceHouse / 1000.0
        qVent = current_ACH * (Toutdoor[i] - Troom[i]) * mHouse[i] * cAir[i] / 3600.0

        qLED_arr[i]      = qLED
        qRoof_arr[i]     = qRoof
        qFloor_arr[i]    = qFloor
        qSideWall_arr[i] = qSideWall
        qVent_arr[i]     = qVent

        qt_current = qLED + qRoof + qFloor + qSideWall + qVent

        Troom_next = Troom[i] + (qt_current / (mHouse[i] * cAir[i] + Chouse)) * 3600.0

        if Troom_next < Th_set:
            required_energy = (mHouse[i] * cAir[i] + Chouse) * (Th_set - Troom[i]) / 3600.0
            qHeat = max(0.0, required_energy - qt_current)
            qCool = 0.0
            qt[i] = qt_current + qHeat

        elif Troom_next > Tc_set:
            required_energy = (mHouse[i] * cAir[i] + Chouse) * (Tc_set - Troom[i]) / 3600.0
            qCool = max(0.0, qt_current - required_energy)
            qHeat = 0.0
            qt[i] = qt_current - qCool

        else:
            qHeat = 0.0
            qCool = 0.0
            qt[i] = qt_current

        heat_kWh[i] = qHeat
        cool_kWh[i] = qCool

        if i < n_hours - 1:
            Troom[i + 1] = Troom[i] + (qt[i] / (mHouse[i] * cAir[i] + Chouse)) * 3600.0

    df['Troom']      = Troom
    df['rhoAir']     = rhoAir
    df['cAir']       = cAir
    df['qt']         = qt
    df['lamp_on']    = lamp_on
    df['lamp_kWh']   = lamp_kWh_arr
    df['heat_kWh']   = heat_kWh
    df['cool_kWh']   = cool_kWh
    df['qLED']       = qLED_arr
    df['qRoof']      = qRoof_arr
    df['qFloor']     = qFloor_arr
    df['qSideWall']  = qSideWall_arr
    df['qVent']      = qVent_arr
    df['Tsolair']    = Tsolair_arr

    params = {
        'areaHouse': areaHouse,
        'surfaceHouse': surfaceHouse,
        'volumeHouse': volumeHouse,
        'lengthHouse': lengthHouse,
        'widthHouse': widthHouse,
        'heightHouse': heightHouse,
        'Tfloor': Tfloor,
        'ACH': ach_array,
        'agh': agh,
        'Chouse': Chouse,
        'Uw': Uw,
        'Ur': Ur,
        'Ug': Ug,
        'Th_set': Th_set,
        'Tc_set': Tc_set,
        'PPE': PPE,
        'PPFD': PPFD,
        'alamp': alamp,
        'Ilamp': Ilamp,
        'lamp_kWh_per_h': lamp_kWh_per_h,
        'photoperiod_hours': photoperiod_hours,
        'lamp_on_start_hour': lamp_on_start_hour,
        'wall_thickness': wall_thickness,
    }

    return df, Toutdoor, radSolar, Troom, params


def parse_time_str(input_str, df, allow_nearest=False):
    s = input_str.strip()
    if not s:
        return None

    if "~" in s:
        left_str, right_str = s.split("~")
        sL = left_str.strip()
        sR = right_str.strip()
    else:
        sL = sR = s

    def _to_index(token):
        token = token.strip()
        if token.isdigit():
            try:
                val = int(token)
                if len(token) <= 5 and 1 <= val <= len(df):
                    return val - 1
            except:
                pass

        dt_series = df['날짜&시간']
        if len(token) == 10:
            fmts = ['%Y%m%d%H']
        elif len(token) == 8:
            fmts = ['%y%m%d%H']
        elif len(token) == 6:
            base_year = int(pd.to_datetime(dt_series.iloc[0]).strftime('%Y'))
            token = f"{base_year}{token}"
            fmts = ['%Y%m%d%H']
        else:
            return None

        target_dt = None
        for fmt in fmts:
            try:
                target_dt = pd.to_datetime(datetime.strptime(token, fmt))
                break
            except:
                continue
        if target_dt is None:
            return None

        key = dt_series.dt.strftime('%Y%m%d%H')
        target_key = target_dt.strftime('%Y%m%d%H')
        exact_idx = df.index[key == target_key]
        if len(exact_idx) > 0:
            return int(exact_idx[0])

        if allow_nearest:
            deltas = (dt_series - target_dt).abs()
            near_idx = int(deltas.idxmin())
            return near_idx
        return None

    iL = _to_index(sL)
    iR = _to_index(sR)

    if iL is None and iR is None:
        return None
    if iL is None:
        iL = iR
    if iR is None:
        iR = iL
    if iL > iR:
        iL, iR = iR, iL
    return (iL, iR)


def print_detailed_calculation(df, idx, params, radSolar):
    """특정 시점의 상세 계산 과정 출력 (터미널 설명용)"""

    print("\n" + "=" * 80)
    print(f"{'온실 에너지 부하 상세 계산 결과':^80}")
    print("=" * 80)

    print(f"\n[시간 정보]")
    print(f"  날짜 및 시간: {df.loc[idx, '날짜&시간']}")
    print(f"  데이터 인덱스: {idx + 1}")

    print(f"\n[온실 구조 정보]")
    print(f"  길이 × 폭 × 높이: {params['lengthHouse']} m × {params['widthHouse']} m × {params['heightHouse']} m")
    print(f"  바닥 면적: {params['areaHouse']:.2f} m²")
    print(f"  측벽 면적: {params['surfaceHouse']:.2f} m²")
    print(f"  체적: {params['volumeHouse']:.2f} m³")

    Tout_C = df.loc[idx, '외부 온도']
    Troom_C = df.loc[idx, 'Troom'] - 273.15
    Tsolair_C = df.loc[idx, 'Tsolair'] - 273.15
    Tfloor_C = params['Tfloor'] - 273.15
    solar_W = radSolar[idx]

    print(f"\n[환경 조건]")
    print(f"  외기 온도 (Tout): {Tout_C:.2f} °C")
    print(f"  실내 온도 (Troom): {Troom_C:.2f} °C")
    print(f"  바닥 온도 (Tfloor): {Tfloor_C:.2f} °C")
    print(f"  일사량: {solar_W:.2f} W/m²")
    print(f"  Sol-air 온도 (Tsolair): {Tsolair_C:.2f} °C")
    print(f"    → 계산: Tout + (agh × 일사) / 17")
    print(f"    → 계산: {Tout_C:.2f} + ({params['agh']:.2f} × {solar_W:.2f}) / 17 = {Tsolair_C:.2f} °C")

    print(f"\n[열관류율 (U-value)]")
    print(f"  측벽 두께: {params['wall_thickness']:.3f} m")
    print(f"  측벽 (Uw): {params['Uw']:.4f} W/(m²·K)")
    print(f"  지붕 (Ur): {params['Ur']:.4f} W/(m²·K)")
    print(f"  바닥 (Ug): {params['Ug']:.4f} W/(m²·K)")

    print(f"\n[공기 물성치]")
    print(f"  밀도 (ρ): {df.loc[idx, 'rhoAir']:.4f} kg/m³")
    print(f"  비열 (Cp): {df.loc[idx, 'cAir']:.4f} kJ/(kg·K)")
    print(f"  공기 질량 (m): {df.loc[idx, 'rhoAir'] * params['volumeHouse']:.2f} kg")
    print(
        f"    → 계산: ρ × V = {df.loc[idx, 'rhoAir']:.4f} × {params['volumeHouse']:.2f} = {df.loc[idx, 'rhoAir'] * params['volumeHouse']:.2f} kg")
    print(f"  환기 횟수 (ACH): {params['ACH'][idx]:.2f} 회/h")

    print(f"\n[보광 시스템]")
    print(f"  조명 ON/OFF: {'ON' if df.loc[idx, 'lamp_on'] == 1 else 'OFF'}")
    print(f"  PPE (광양자 효율): {params['PPE']:.2f} µmol/J")
    print(f"  PPFD (목표 광량): {params['PPFD']:.2f} µmol/(m²·s)")
    print(f"  조명 강도 (Ilamp): {params['Ilamp']:.2f} W/m²")
    print(f"    → 계산: PPFD / PPE = {params['PPFD']:.2f} / {params['PPE']:.2f} = {params['Ilamp']:.2f} W/m²")
    print(f"  조명 전력 (kWh/h): {params['lamp_kWh_per_h']:.4f} kWh/h")
    print(
        f"    → 계산: Ilamp × 면적 / 1000 = {params['Ilamp']:.2f} × {params['areaHouse']:.2f} / 1000 = {params['lamp_kWh_per_h']:.4f} kWh/h")

    print(f"\n[에너지 부하 계산 (kWh/h)]")
    qLED      = df.loc[idx, 'qLED']
    qRoof     = df.loc[idx, 'qRoof']
    qFloor    = df.loc[idx, 'qFloor']
    qSideWall = df.loc[idx, 'qSideWall']
    qVent     = df.loc[idx, 'qVent']

    print(f"\n  1) 보광 부하 (QLED): {qLED:.4f} kWh/h")
    print(f"\n  2) 지붕 열전달 (QRoof): {qRoof:.4f} kWh/h")
    print(f"  3) 바닥 열전달 (QFloor): {df.loc[idx, 'qFloor']:.4f} kWh/h")
    print(f"  4) 측벽 열전달 (QSideWall): {df.loc[idx, 'qSideWall']:.4f} kWh/h")
    print(f"  5) 환기 열전달 (QVent): {df.loc[idx, 'qVent']:.4f} kWh/h")

    qt_current = qLED + qRoof + qFloor + qSideWall + qVent
    print(f"\n  총 열부하 (Qtotal) = {qt_current:.4f} kWh/h")

    print(f"\n[실내 온도 (Troom) 계산 과정]")

    if idx > 0:
        Troom_prev_C = df.loc[idx - 1, 'Troom'] - 273.15
        qt_prev      = df.loc[idx - 1, 'qt']
        mAir_prev    = df.loc[idx - 1, 'rhoAir'] * params['volumeHouse']
        cAir_prev    = df.loc[idx - 1, 'cAir']

        print(f"  이전 시간 (t-1)의 실내 온도: {Troom_prev_C:.2f} °C")
        print(f"  이전 시간 (t-1)의 순 열부하: {qt_prev:.4f} kWh/h")
        print(f"\n  ΔTroom = (Qt / (m × Cp + Chouse)) × 3600")
        print(f"           = ({qt_prev:.4f} / ({mAir_prev:.2f} × {cAir_prev:.4f} + {params['Chouse']:.0f})) × 3600")

        delta_T = (qt_prev / (mAir_prev * cAir_prev + params['Chouse'])) * 3600.0
        print(f"  ΔTroom = {delta_T:.4f} K")
        print(f"\n  현재 시간 (t)의 실내 온도: {Troom_C:.2f} °C")
    else:
        print(f"  초기 시간: Troom = {Troom_C:.2f} °C (초기값)")

    qHeat = df.loc[idx, 'heat_kWh']
    qCool = df.loc[idx, 'cool_kWh']

    mAir_current = df.loc[idx, 'rhoAir'] * params['volumeHouse']
    cAir_current = df.loc[idx, 'cAir']
    Troom_next_no_control = Troom_C + (qt_current / (mAir_current * cAir_current + params['Chouse'])) * 3600.0

    print(f"\n[냉난방 제어]")
    print(f"  설정 온도 - 난방: {params['Th_set'] - 273.15:.1f} °C")
    print(f"  설정 온도 - 냉방: {params['Tc_set'] - 273.15:.1f} °C")
    print(f"  제어 없을 경우 다음 시간 예상 온도: {Troom_next_no_control:.2f} °C")

    if qHeat > 0:
        print(f"  → 난방 가동 중")
        print(f"    난방 부하 (QHeat): {qHeat:.4f} kWh/h")
        print(f"    냉방 부하 (QCool): 0.0000 kWh/h")
    elif qCool > 0:
        print(f"  → 냉방 가동 중")
        print(f"    난방 부하 (QHeat): 0.0000 kWh/h")
        print(f"    냉방 부하 (QCool): {qCool:.4f} kWh/h")
    else:
        print(f"  → 자연 온도 유지 범위")

    print(f"\n[에너지 수지 요약]")
    print(f"  보광:      {qLED:.4f} kWh/h")
    print(f"  난방 부하: {qHeat:.4f} kWh/h")
    print(f"  냉방 부하: {qCool:.4f} kWh/h")
    print(f"  환기 손실: {qVent:.4f} kWh/h")
    print(f"  순 열부하: {df.loc[idx, 'qt']:.4f} kWh/h")

    print("\n" + "=" * 80 + "\n")


def _make_bar_and_heatmap(
    x_labels,
    results_dict,
    x_title,
    fig_title,
    hm_title
):
    heating_arr  = np.array(results_dict['heating'])
    cooling_arr  = np.array(results_dict['cooling'])
    lighting_arr = np.array(results_dict['lighting'])

    heatmap_data = np.vstack([
        heating_arr,
        cooling_arr,
        lighting_arr
    ])

    load_labels = ['난방(kWh)', '냉방(kWh)', '보광(kWh)']

    cmap_colors = [
        (1.0, 1.0, 1.0),   # white
        (1.0, 1.0, 0.6),   # yellow
        (1.0, 0.7, 0.2),   # orange
        (0.8, 0.0, 0.0),   # red
    ]
    cmap_custom = LinearSegmentedColormap.from_list("white_yellow_orange_red", cmap_colors)

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[2, 1.2], hspace=0.4)

    ax_bar = fig.add_subplot(gs[0, 0])

    x = np.arange(len(x_labels))
    width = 0.25

    bars1 = ax_bar.bar(x - width, heating_arr,  width, label='난방', color='red')
    bars2 = ax_bar.bar(x,         cooling_arr,  width, label='냉방', color='blue')
    bars3 = ax_bar.bar(x + width, lighting_arr, width, label='보광', color='#DAA520')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.,
                h,
                f'{h:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    ax_bar.set_xlabel(x_title)
    ax_bar.set_ylabel('에너지 부하(kWh)')
    ax_bar.set_title(fig_title)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(x_labels)
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.3)

    ax_hm = fig.add_subplot(gs[1, 0])

    im = ax_hm.imshow(
        heatmap_data,
        aspect='auto',
        cmap=cmap_custom
    )

    ax_hm.set_yticks(np.arange(len(load_labels)))
    ax_hm.set_yticklabels(load_labels)
    ax_hm.set_xticks(np.arange(len(x_labels)))
    ax_hm.set_xticklabels(x_labels)

    ax_hm.set_xlabel(x_title)
    ax_hm.set_title(hm_title)

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data[i, j]
            ax_hm.text(
                j, i,
                f"{val:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9
            )

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label('kWh', rotation=90)

    plt.tight_layout()
    plt.show()


def plot_parameter_sensitivity(df, start_idx, end_idx):
    print("\n파라미터 민감도 분석 중...")

    wall_thicknesses = [0.05, 0.075, 0.1]
    wall_results = {'heating': [], 'cooling': [], 'lighting': []}

    for thickness in wall_thicknesses:
        df_temp, _, _, _, _ = process_greenhouse_data(
            wall_thickness=thickness,
            ACH_value=1.0,
            PPFD_value=200.0
        )
        wall_results['heating'].append(df_temp.loc[start_idx:end_idx, 'heat_kWh'].sum())
        wall_results['cooling'].append(df_temp.loc[start_idx:end_idx, 'cool_kWh'].sum())
        wall_results['lighting'].append(df_temp.loc[start_idx:end_idx, 'lamp_kWh'].sum())

    ACH_values = [0.1, 0.5, 1.0]
    ach_results = {'heating': [], 'cooling': [], 'lighting': []}

    for ach in ACH_values:
        df_temp, _, _, _, _ = process_greenhouse_data(
            wall_thickness=0.05,
            ACH_value=ach,
            PPFD_value=200.0
        )
        ach_results['heating'].append(df_temp.loc[start_idx:end_idx, 'heat_kWh'].sum())
        ach_results['cooling'].append(df_temp.loc[start_idx:end_idx, 'cool_kWh'].sum())
        ach_results['lighting'].append(df_temp.loc[start_idx:end_idx, 'lamp_kWh'].sum())

    PPFD_values = [150, 200, 250]
    ppfd_results = {'heating': [], 'cooling': [], 'lighting': []}

    for ppfd in PPFD_values:
        df_temp, _, _, _, _ = process_greenhouse_data(
            wall_thickness=0.05,
            ACH_value=1.0,
            PPFD_value=ppfd
        )
        ppfd_results['heating'].append(df_temp.loc[start_idx:end_idx, 'heat_kWh'].sum())
        ppfd_results['cooling'].append(df_temp.loc[start_idx:end_idx, 'cool_kWh'].sum())
        ppfd_results['lighting'].append(df_temp.loc[start_idx:end_idx, 'lamp_kWh'].sum())

    print("파라미터 민감도 분석 완료!\n")

    wall_labels = [f"{t:.3f}" for t in wall_thicknesses]
    _make_bar_and_heatmap(
        x_labels = wall_labels,
        results_dict = wall_results,
        x_title = '측벽 두께(m)',
        fig_title = '측벽 두께에 따른 에너지 부하',
        hm_title = '측벽 두께에 따른 민감도 히트맵'
    )

    ach_labels = [f"{a:.1f}" for a in ACH_values]
    _make_bar_and_heatmap(
        x_labels = ach_labels,
        results_dict = ach_results,
        x_title = 'ACH',
        fig_title = 'ACH에 따른 에너지 부하',
        hm_title = 'ACH에 따른 민감도 히트맵'
    )

    ppfd_labels = [f"{p:.0f}" for p in PPFD_values]
    _make_bar_and_heatmap(
        x_labels = ppfd_labels,
        results_dict = ppfd_results,
        x_title = 'LED PPFD(µmol/m²/s)',
        fig_title = 'LED PPFD에 따른 에너지 부하',
        hm_title='LED PPFD에 따른 민감도 히트맵'
    )


def plot_graphs(df, Troom, Toutdoor, heat_kWh, cool_kWh, lamp_kWh, start_idx, end_idx, areaHouse):
    sl = slice(start_idx, end_idx + 1)
    hours = np.arange(end_idx - start_idx + 1)

    plt.figure()
    plt.plot(hours, (Troom[sl] - 273.15), color='purple', linestyle='-', label='실내온도 (°C)', linewidth=1.8)
    plt.plot(hours, (Toutdoor[sl] - 273.15), color='black', linestyle='--', label='외기온도 (°C)', linewidth=1.4)
    plt.title("온도 비교")
    plt.xlabel("시간")
    plt.ylabel("온도(°C)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(hours, heat_kWh[sl], color='red', linestyle='-', label='난방 부하 (kWh/h)', linewidth=1.6)
    plt.plot(hours, cool_kWh[sl], color='blue', linestyle='--', label='냉방 부하 (kWh/h)', linewidth=1.6)
    plt.title("에너지 부하 비교")
    plt.xlabel("시간")
    plt.ylabel("에너지 부하(kWh)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    lamp_sum = float(np.sum(lamp_kWh[sl]))
    heat_sum = float(np.sum(heat_kWh[sl]))
    cool_sum = float(np.sum(cool_kWh[sl]))

    total_measured = lamp_sum + heat_sum + cool_sum
    total_energy = total_measured / 0.8 if total_measured > 0 else 0
    etc_sum = total_energy * 0.2

    labels = ['보광', '난방', '냉방']
    values = [lamp_sum, heat_sum, cool_sum]
    colors_bar = ['#DAA520', 'red', 'blue']

    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, values, width=0.5, color=colors_bar)
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.1f}", ha='center', va='bottom', fontsize=10)
    plt.xticks(x, labels)
    plt.ylabel("총 에너지 부하(kWh)")
    plt.title("총 에너지 부하")
    plt.tight_layout()
    plt.show()

    values_per_m2 = [v / areaHouse for v in values]
    plt.figure()
    plt.bar(x, values_per_m2, width=0.5, color=colors_bar)
    for i, v in enumerate(values_per_m2):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
    plt.xticks(x, labels)
    plt.ylabel("단위면적당 총 에너지 부하(kWh/m²)")
    plt.title("단위면적당 총 에너지 부하")
    plt.tight_layout()
    plt.show()

    total_core = sum(values)
    if total_core > 0:
        total_with_etc = total_core / 0.8
        etc_value = total_with_etc * 0.2

        values_with_etc = values + [etc_value]
        labels_with_etc = labels + ['기타']
        colors_with_etc = colors_bar + ['green']

        percentages = [(v / total_with_etc) * 100 for v in values_with_etc]

        plt.figure()
        wedges, texts, autotexts = plt.pie(
            percentages,
            labels=labels_with_etc,
            colors=colors_with_etc,
            autopct='%1.1f%%',
            startangle=90
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_weight('bold')
        plt.title("에너지 부하 비율")
        plt.tight_layout()
        plt.show()


def main():
    try:
        df, Toutdoor, radSolar, Troom, params = process_greenhouse_data()

        try:
            user_input = input("시점(기간) 입력(예: 100/24051200/24051200~24060123): ").strip()
        except Exception:
            user_input = ""

        if user_input:
            rng = parse_time_str(user_input, df, allow_nearest=True)
        else:
            rng = None

        n = len(df)
        if rng is None:
            start_idx, end_idx = 0, n - 1
        else:
            start_idx, end_idx = rng

        if start_idx == end_idx:
            print_detailed_calculation(df, start_idx, params, radSolar)

        else:
            print(f"\n분석 기간: {df.loc[start_idx, '날짜&시간']} ~ {df.loc[end_idx, '날짜&시간']}")
            print(f"총 시간: {end_idx - start_idx + 1} 시간")

            lamp_sum = float(np.sum(df.loc[start_idx:end_idx, 'lamp_kWh']))
            heat_sum = float(np.sum(df.loc[start_idx:end_idx, 'heat_kWh']))
            cool_sum = float(np.sum(df.loc[start_idx:end_idx, 'cool_kWh']))

            total_measured = lamp_sum + heat_sum + cool_sum
            total_energy = total_measured / 0.8 if total_measured > 0 else 0
            etc_sum = total_energy * 0.2

            print(f"\n[에너지 사용량 합계]")
            print(f"  보광: {lamp_sum:.2f} kWh ({(lamp_sum / total_energy * 100):.1f}%)")
            print(f"  난방: {heat_sum:.2f} kWh ({(heat_sum / total_energy * 100):.1f}%)")
            print(f"  냉방: {cool_sum:.2f} kWh ({(cool_sum / total_energy * 100):.1f}%)")
            print(f"  기타: {etc_sum:.2f} kWh (20.0%)")
            print(f"  ─────────────────────────────")
            print(f"  전체: {total_energy:.2f} kWh (100.0%)")
            print(f"\n  * 보광/난방/냉방은 전체 에너지의 80%로 계산됨")
            print(f"  * 기타(가습, 제습 등)는 전체 에너지의 20%로 추정됨\n")

        plot_graphs(
            df=df,
            Troom=df['Troom'].values,
            Toutdoor=Toutdoor,
            heat_kWh=df['heat_kWh'].values,
            cool_kWh=df['cool_kWh'].values,
            lamp_kWh=df['lamp_kWh'].values,
            start_idx=start_idx,
            end_idx=end_idx,
            areaHouse=params['areaHouse']
        )

        plot_parameter_sensitivity(df, start_idx, end_idx)

    except FileNotFoundError:
        print("'jeonju_greenhouse.xlsx' 파일이 없습니다.")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# 전북대학교 생물산업기계공학과 양명균 교수님의 수직농장 실험실에 대한 에너지 부하 모델링
# 온도만 고려함(습도는 고려하지 않음)
# 작물이 없다는 가정 하에 계산함
# 민감도 분석 추가 수행
