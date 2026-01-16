import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

hourly_gsw = {
    0: 0.017, 1: 0.017, 2: 0.017, 3: 0.017, 4: 0.017, 5: 0.017,
    6: 0.080,
    7: 0.104,
    8: 0.123,
    9: 0.115,
    10: 0.107,
    11: 0.101,
    12: 0.096,
    13: 0.101,
    14: 0.088,
    15: 0.085,
    16: 0.080,
    17: 0.075,
    18: 0.069,
    19: 0.017, 20: 0.017, 21: 0.017, 22: 0.017, 23: 0.017
}

Patm = 101325.0     # 대기압 [Pa]
Sigma = 5.670374419e-8  # 스테판 볼츠만 상수

a_sw_can = 0.56     # 캐노피 단파 흡수율
k_sw = 0.86         # Beer–Lambert 광감쇠계수
LAI = 2.0           # 엽면적지수

Ld = 0.11           # 평균 수확기 엽장 [m]
Lw = 0.09           # 평균 수확기 엽폭 [m]

ga = 9.81           # 중력 가속도 [m/s2]
leaf_le = 0.97      # 캐노피 장파 방사율

hc_cover = 0.3      # 피복재 열전도율 [W/m/K]
th_cover = 0.0001   # 피복재 두께 [m]
ss_sp = 1.0         # 온실 바닥면적 = 피복면적

cloud_default = 0.5  # 0~1
Cout_ppm_default = 420.0 # 외부 CO2 농도가 없을 경우 사용

deltaT_dyn = 2.0 # 온실 내부 기온과 엽온의 온도 차

lengthHouse = 39    # 온실 길이 [m]
widthHouse = 100      # 온실 폭 [m]
heightHouse = 4      # 온실 높이 [m]

areaHouse = lengthHouse * widthHouse  # 온실 지면 면적 [m²]
surfaceHouse = (widthHouse * heightHouse + lengthHouse * heightHouse) * 2  # 온실 측면 면적 [m²]
volumeHouse = areaHouse * heightHouse  # 온실 부피 [m³]

Chouse = 350000   # 온실 열 용량 관련 계수(kJ/K)
CWhouse = 350000  # 온실 수분 용량 관련 계수(Kg,water)

agh = 0.21  # 필름의 단파 복사 흡수(방사)율
Ttarget = 15 + 273.15  # 보온 커튼 가동 기준 온도 [K]
Tgroundi = 18 + 273.15  # 내부 지면 온도(고정) [K]
Tgrounde = 15.0 + 273.15  # 외부 지면 온도(고정) [K]
trns_hours = 3  # ACH 전환 시간 (시간 동안 선형적으로 변화)

COVER_IR_PROPS = {
    "PO필름": {"tau_ir": 0.35, "eps_lw": 0.60},
    "AL스크린": {"tau_ir": 0.10, "eps_lw": 0.20},
    "보온커튼": {"tau_ir": 0.05, "eps_lw": 0.80},
}

def get_hourly_gsw(ts):
    if isinstance(ts, (pd.Timestamp, datetime)):
        h = ts.hour
    else:
        h = int(ts) % 24
    return float(hourly_gsw.get(int(h), 0.017))


def build_gsw_timeseries_from_timestamp(df_time_series):
    if '날짜&시간' in df_time_series.columns:
        ts = pd.to_datetime(df_time_series['날짜&시간'])
        return np.array([get_hourly_gsw(t) for t in ts], dtype=float)
    return np.array([get_hourly_gsw(i) for i in range(len(df_time_series))], dtype=float)

def _nanstd(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.nanstd(x)) if x.size else np.nan


def _nunique(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0
    return int(np.unique(x).size)

def warn_if_unintended_constant(arr, name, ref_arrays=None, eps_rel=1e-6, eps_abs=1e-9):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]

    if a.size == 0:
        print(f"[WARN] {name}: all NaN (계산 실패 가능)")
        return

    a_std = float(np.nanstd(a))
    a_mean = float(np.nanmean(a))
    is_const = (a_std <= eps_abs) or (abs(a_mean) > 0 and (a_std / max(abs(a_mean), eps_abs)) <= eps_rel) or (_nunique(a) <= 1)

    if not is_const:
        return

    if not ref_arrays:
        return

    ref_varies = False
    for rn, rv in ref_arrays.items():
        r = np.asarray(rv, dtype=float)
        r = r[np.isfinite(r)]
        if r.size == 0:
            continue
        if _nunique(r) >= 2 and _nanstd(r) > eps_abs:
            ref_varies = True
            break

    if ref_varies:
        nan_ratio = 1.0 - (np.isfinite(np.asarray(arr, dtype=float)).sum() / max(len(arr), 1))
        if nan_ratio > 0.0:
            print(f"[WARN] {name}: appears constant ({a_mean:.6g}) while inputs vary (NaN ratio={nan_ratio:.2%}).")
        else:
            print(f"[WARN] {name}: appears constant ({a_mean:.6g}) while inputs vary.")

def air_properties(T_air_K, P=Patm):
    T = float(max(T_air_K, 200.0))
    rho = PropsSI("D", "T", T, "P", P, "Air")
    cp  = PropsSI("C", "T", T, "P", P, "Air")
    k   = PropsSI("L", "T", T, "P", P, "Air")
    mu  = PropsSI("V", "T", T, "P", P, "Air")
    return rho, cp, k, mu

def latent_heat_vaporization(T_K):
    T = float(max(T_K, 273.15))
    return PropsSI("H", "T", T, "Q", 1, "Water") - PropsSI("H", "T", T, "Q", 0, "Water")

def saturation_vapor_pressure_Pa(T_K):
    T = float(max(T_K, 273.15))
    return PropsSI("P", "T", T, "Q", 0, "Water")

def determine_ach_and_internal_wind(T_air_K, RH_in_frac, radSolar):
    T_c = T_air_K - 273.15
    RH_p = float(np.clip(RH_in_frac, 0.0, 1.0)) * 100.0

    if radSolar > 0:
        if T_c >= 25 or RH_p >= 70:
            return 20, 0.5
        elif T_c >= 20 or RH_p >= 60:
            return 15, 0.3
        else:
            return 7, 0.1
    else:
        if RH_p >= 80:
            return 1, 0.1
        else:
            return 0.2, 0.05

    # if radSolar > 0:
    #     if T_c >= 25 or RH_p >= 70:
    #         return 20, 0.5
    #     elif T_c >= 20 or RH_p >= 60:
    #         return 10, 0.3
    #     else:
    #         return 1.0, 0.1
    # else:
    #     if RH_p >= 80:
    #         return 1.0, 0.1
    #     else:
    #         return 0.2, 0.05

def stomatal_and_canopy_resistance(gsw, LAI_val=LAI):
    """
    rs = 1 / (0.0224 * gsw)
    rc = 2 * rs / LAI
    """
    gsw_eff = max(float(gsw), 1e-9)
    rs = 1.0 / (0.0224 * gsw_eff)  # [s/m]
    LAI_eff = max(float(LAI_val), 1e-9)
    rc = 2.0 * rs / LAI_eff       # [s/m]
    return rs, rc

def aerodynamic_resistance_SM(
    T_air_K,
    u_canopy,
    Ld=Ld,
    Lw=Lw,
    LAI_val=LAI,
    deltaT=0.0,
    P=Patm
):
    """
    d = 2 / (1/L + 1/W)
    Re = rho * V * d / mu
    Gr = g * beta * ΔT * d^3 * rho^2 / mu^2
    Nu = 0.68 * (Re^1.5 + Gr^0.75)^(1/3)
    hs = Nu * k / d

    ra = rho * cp / (LAI * hs)
    """
    rho, cp, k_air, mu = air_properties(T_air_K, P=P)
    V = max(float(u_canopy), 1e-4)

    L = max(float(Ld), 1e-6)
    W = max(float(Lw), 1e-6)
    d = 2.0 / (1.0 / L + 1.0 / W)

    Re = (rho * V * d) / max(mu, 1e-12)

    beta = 1.0 / max(float(T_air_K), 1.0)
    Gr = (ga * beta * float(deltaT) * (d ** 3) * (rho ** 2)) / max(mu ** 2, 1e-24)

    Re_term = max(Re, 0.0) ** 1.5
    Gr_term = max(Gr, 0.0) ** 0.75
    Nu = 0.68 * (Re_term + Gr_term) ** (1.0 / 3.0)

    hs = (Nu * k_air) / max(d, 1e-12)  # [W/m2/K]

    LAI_eff = max(float(LAI_val), 1e-9)
    ra = (rho * cp) / max(LAI_eff * hs, 1e-12)  # [s/m]

    return ra, hs, d, Re, Gr, Nu

def compute_Rns_paper_form(
    radSolar_out,
    transGlass,
    alpha_sw=a_sw_can,
    k_sw=k_sw,
    LAI_val=LAI
):
    """
    Rn = Rns

    Rns = a_sw_can * (1 - exp(-k_sw * LAI)) * Is
    Is  = radSolar_out * transGlass
    """
    Is = float(radSolar_out) * float(transGlass)
    LAI_eff = max(float(LAI_val), 1e-9)

    f_int = 1.0 - np.exp(-float(k_sw) * LAI_eff)
    Rns = float(alpha_sw) * f_int * Is

    return Rns, Is, f_int

def compute_latent_flux_SM(
    T_air_K,
    RH_in_frac,
    Rn,
    ra,
    rc,
    areaHouse,
    LAI_val=LAI
):
    """
    Stanghellini Model(PM in greenhouse) 형태

    λE = [Δ(Rn - G) + (2*LAI*ρ*cp*VPD)/ra] / [Δ + γ(1 + rc/ra)]
    G=0,
    Wet = (λE / h_fg) * A * 3600    [kg/h]
    """
    T_air_C = T_air_K - 273.15
    RH = float(np.clip(RH_in_frac, 0.0, 1.0))

    p_sat = saturation_vapor_pressure_Pa(T_air_K)  # [Pa]
    VPD_Pa = max(p_sat * (1.0 - RH), 0.0)

    # Δ (Pa/°C)
    e_s_kPa = p_sat / 1000.0
    Delta_kPa = 4098.0 * e_s_kPa / ((T_air_C + 237.3) ** 2)
    Delta = Delta_kPa * 1000.0

    # γ (Pa/°C)
    gamma = 66

    rho, cp, _, _ = air_properties(T_air_K)

    ra_eff = max(float(ra), 1e-9)
    rc_eff = max(float(rc), 0.0)
    LAI_eff = max(float(LAI_val), 1e-9)

    numerator = Delta * (Rn) + rho * cp * VPD_Pa / ra_eff
    denominator = Delta + gamma * (1.0 + rc_eff / ra_eff)

    lambdaE = numerator / max(denominator, 1e-12)

    h_fg = latent_heat_vaporization(T_air_K)
    mass_flux = lambdaE / max(h_fg, 1e-12)
    Wet = mass_flux * float(areaHouse) * 3600.0    # W1. 작물의 증발산량

    return lambdaE, Wet, h_fg, VPD_Pa, Delta, gamma

def calculate_floor_u_value():
    k_concrete = 6.23 * (1000 / 3600)
    k_gravel   = 7.20 * (1000 / 3600)
    k_sand     = 6.29 * (1000 / 3600)

    d_concrete = 0.30
    d_gravel   = 0.15
    d_sand     = 0.25

    R_concrete = d_concrete / k_concrete
    R_gravel   = d_gravel   / k_gravel
    R_sand     = d_sand     / k_sand

    R_total = R_concrete + R_gravel + R_sand
    Ug = 1.0 / R_total
    return Ug

def calculate_covering_properties(radSolar, Troom, Ttarget=15 + 273.15):
    covers = ['PO필름']
    transGlass = 0.71  # Old plastic 기준 필름 단파 투과율

    if radSolar > 0:
        if radSolar > 500:
            covers.append('AL스크린')
            transGlass *= 0.45  # 55% AL 차광 스크린
        if Troom < Ttarget:
            covers.append('보온커튼')
            transGlass *= 0.5  # Old 보온 커튼
    else:
        covers.append('보온커튼')
        covers.append('AL스크린')
        transGlass *= 0.45 * 0.5

    material_values = {
        'PO필름': 5.2,
        'AL스크린': 5.5,
        '보온커튼': 4.5  # 3겹 보온커튼
    }

    if len(covers) >= 2:
        Rt = sum(1 / material_values[c] for c in covers)
        Ti = 1 / Rt
        Ur = 1.2944 * Ti - 0.4205
    else:
        Ur = material_values[covers[0]]

    Uw = material_values['PO필름']
    return Ur, Uw, transGlass, covers

def combine_eps_iterative(eps_list):
    """
    등가 레이어:
      1/ε_eq = 1/ε1 + 1/ε2 - 1
      ε_eq   = 1 / (1/ε1 + 1/ε2 - 1)
    """
    if len(eps_list) == 0:
        return 0.0

    eps_eq = float(eps_list[0])

    for eps in eps_list[1:]:
        e1 = float(eps_eq)
        e2 = float(eps)

        e1 = min(max(e1, 1e-4), 0.9999)
        e2 = min(max(e2, 1e-4), 0.9999)

        denom = (1.0 / e1) + (1.0 / e2) - 1.0
        if abs(denom) < 1e-6:
            eps_eq = 0.9999
        else:
            eps_eq_new = 1.0 / denom
            eps_eq = max(min(eps_eq_new, 0.9999), 0.0)

    return eps_eq

def compute_Tsky_ClarkAllen_cloud_single(
    Tair_for_dew_K,
    RH_frac,
    Toutdoor_K,
    cloud_fraction,
    Patm_Pa=Patm
):
    """
    Clark & Allen 계열 맑은 하늘 복사율 + Walton 전운량 보정으로 유효 하늘온도 계산
    이슬점 CoolProp(HumidAirProp.HAPropsSI)사용, 없으면 Magnus 근사
    """
    Tair_for_dew_K = float(Tair_for_dew_K)
    Toutdoor_K = float(Toutdoor_K)

    RH = float(RH_frac)
    RH = float(np.clip(RH, 0.01, 1.0))

    cloud_frac = float(np.clip(cloud_fraction, 0.0, 1.0))

    use_coolprop = False
    try:
        from CoolProp.HumidAirProp import HAPropsSI
        use_coolprop = True
    except Exception:
        use_coolprop = False

    if use_coolprop:
        try:
            T_dew_K = float(HAPropsSI("Tdp", "T", float(Tair_for_dew_K), "P", float(Patm_Pa), "R", float(RH)))
        except Exception:
            use_coolprop = False

    if not use_coolprop:
        Ta_C = Tair_for_dew_K - 273.15
        a = 17.27
        b = 237.7
        gamma_m = np.log(RH) + (a * Ta_C) / (b + Ta_C)
        T_dew_C = (b * gamma_m) / (a - gamma_m)
        T_dew_K = float(T_dew_C + 273.15)

    eps_sky_clear = 0.787 + 0.764 * np.log(max(T_dew_K, 1.0) / 273.15)
    eps_sky_clear = float(np.clip(eps_sky_clear, 0.3, 1.0))

    N10 = float(np.clip(cloud_frac * 10.0, 0.0, 10.0))
    cloud_factor = 1.0 + 0.0224 * N10 - 0.0035 * (N10 ** 2) + 0.00028 * (N10 ** 3)

    eps_sky = float(np.clip(eps_sky_clear * cloud_factor, 0.0, 1.0))

    T_sky_eff = (eps_sky ** 0.25) * Toutdoor_K
    return float(T_sky_eff)

def get_tau_eps_from_covers(covers):
    """
    - tau_eff = 레이어 투과율 곱
    - eps_eff = 등가 방사율(레이어 결합)
    """
    if isinstance(covers, (list, tuple, np.ndarray)):
        names = [str(c).strip() for c in covers if str(c).strip()]
    else:
        names = [c.strip() for c in str(covers).split(",") if c.strip()]

    if not names:
        names = ["PO필름"]

    tau_prod = 1.0
    eps_list = []

    for name in names:
        props = COVER_IR_PROPS.get(name)
        if props is None:
            continue
        tau_prod *= float(props["tau_ir"])
        eps_list.append(float(props["eps_lw"]))

    eps_eff = combine_eps_iterative(eps_list) if len(eps_list) else 0.0
    return float(tau_prod), float(eps_eff)

def compute_cover_surface_temperature_single(
    Ti_K,
    Te_K,
    Tsi_K,
    Tse_K,
    u_out,
    tau_ir,
    Tsky_eff_K,
    radSolar_out,
    agh_sw,
    hc_cover=hc_cover,
    thickness=th_cover,
    ss_sp=ss_sp,
    max_iter=40,
    tol=1e-3
):
    """
    피복 외측 표면온도 Tp 계산(표면 온도는 야간에 대한 장파 방사를 주로 고려해 주간에 단파 복사는 고려하지 않음)
    """
    Ti = float(Ti_K)
    Te = float(Te_K)
    Tsi = float(Tsi_K)
    Tse = float(Tse_K)

    u = max(float(u_out), 0.0)
    tau_ir = float(tau_ir)
    TA = float(Tsky_eff_K)

    SsSp = float(ss_sp)
    fpsi = SsSp
    fpA = 0.5 * (1.0 + SsSp)
    fpp = 1.0 - fpsi
    fpse = 1.0 - fpA

    Tp = 0.5 * (Ti + Te)

    for _ in range(max_iter):
        # 대류열전달계수(Kittas)
        dTiTp = max(abs(Ti - Tp), 0.1)
        dTpTe = max(abs(Tp - Te), 0.1)

        hci = 4.3 * (dTiTp ** 0.25)
        hce = 1.22 * (dTpTe ** 0.25) + 3.12 * (u ** 0.8)

        # 복사열전달계수
        denom_in = (Ti - Tp)
        if abs(denom_in) < 1e-6:
            denom_in = 1e-6 if Ti >= Tp else -1e-6

        num_in = (
            -(1.0 - tau_ir) * fpsi * Sigma * (Tp ** 4)
            - fpA * fpsi * tau_ir * Sigma * (TA ** 4)
            + fpsi * Sigma * (Ti ** 4)
        )
        hri = num_in / denom_in

        denom_out = (Tp - Te)
        if abs(denom_out) < 1e-6:
            denom_out = 1e-6 if Tp >= Te else -1e-6

        num_out = (
            (1.0 - tau_ir) * Sigma * (Tp ** 4)
            + fpp * tau_ir * (1.0 - tau_ir) * Sigma * (Tp ** 4)
            + fpsi * tau_ir * Sigma * (Tsi ** 4)
            - fpA * Sigma * (TA ** 4)
            - fpse * Sigma * (Tse ** 4)
            + fpA * fpp * (tau_ir ** 2) * Sigma * (TA ** 4)
            + fpse * (tau_ir ** 2) * Sigma * (Tse ** 4)
        )
        hre = num_out / denom_out

        hci_eff = max(float(hci), 1e-3)
        hce_eff = max(float(hce), 1e-3)
        hri_eff = max(float(hri), 1e-3)
        hre_eff = max(float(hre), 1e-3)

        R_in = 1.0 / (hri_eff + hci_eff)
        R_out = 1.0 / (hre_eff + hce_eff)
        R_cond = float(thickness) / max(float(hc_cover), 1e-6)

        K = 1.0 / (R_in + R_out + R_cond)

        denom_K = (hre_eff + hce_eff)
        Tp_new = Te + (K * (Ti - Te)) / max(denom_K, 1e-9)

        if abs(Tp_new - Tp) < tol:
            Tp = Tp_new
            break
        Tp = Tp_new

    return float(Tp)

def compute_qFIR_kW_internal(
    Ti_K,
    Te_K,
    RH_in_frac,
    radSolar_out,
    covers,
    areaHouse,
    u_out=1.0,  # default
    cloud_frac=cloud_default,
    Tsi_K=None,
    Tse_K=None,
):
    """
    qFIR_kW = eps_eff * Sigma * A * (Tp^4 - Tsky^4) / 1000
    """

    tau_ir, eps_eff = get_tau_eps_from_covers(covers)

    Tsky_eff = compute_Tsky_ClarkAllen_cloud_single(
        Tair_for_dew_K=Ti_K,
        RH_frac=RH_in_frac,
        Toutdoor_K=Te_K,
        cloud_fraction=cloud_frac,
        Patm_Pa=Patm
    )

    if Tsi_K is None:
        Tsi_K = Tgroundi
    if Tse_K is None:
        Tse_K = Tgrounde

    Tp = compute_cover_surface_temperature_single(
        Ti_K=Ti_K,
        Te_K=Te_K,
        Tsi_K=Tsi_K,
        Tse_K=Tse_K,
        u_out=u_out,
        tau_ir=tau_ir,
        Tsky_eff_K=Tsky_eff,
        radSolar_out=radSolar_out,
        agh_sw=agh,
        hc_cover=hc_cover,
        thickness=th_cover,
        ss_sp=ss_sp
    )

    qFIR_kW = float(eps_eff) * Sigma * float(areaHouse) * (Tp ** 4 - Tsky_eff ** 4) / 1000.0
    return float(qFIR_kW), float(Tp), float(Tsky_eff), float(eps_eff)

def process_greenhouse_data():
    df = pd.read_excel("climate/goheung_greenhouse.xlsx")

    if '내부 습도' in df.columns and 'Humi' not in df.columns:
        df.rename(columns={'내부 습도': 'Humi'}, inplace=True)

    if '내부 CO2' in df.columns and 'CO2_in' not in df.columns:
        df.rename(columns={'내부 CO2': 'CO2_in'}, inplace=True)

    has_external_co2 = '외부 CO2' in df.columns
    if has_external_co2:
        df['외부 CO2'] = pd.to_numeric(df['외부 CO2'], errors='coerce')
        df['외부 CO2'] = df['외부 CO2'].interpolate(method='linear').ffill().bfill()
        print(f"[INFO] '외부 CO2' 열을 데이터에서 읽어 사용합니다.")
    else:
        df['외부 CO2'] = Cout_ppm_default
        print(f"[INFO] '외부 CO2' 열이 없어 고정값({Cout_ppm_default} ppm)을 사용합니다.")

    num_cols_base = ['외부 온도', '일사', '외부 습도', '내부 온도', '내부 습도', '내부 CO2']
    for col in num_cols_base:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 결측 보간
    for col in num_cols_base:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')
            df[col] = df[col].ffill().bfill()

    df['날짜&시간'] = pd.to_datetime(df['날짜&시간'])
    n_hours = len(df)
    df['hour'] = range(n_hours)

    gsw_series = build_gsw_timeseries_from_timestamp(df)

    Toutdoor = df['외부 온도'].values + 273.15
    radSolar = df['일사'].values

    if '외부 풍속' in df.columns:
        u_out_series = pd.to_numeric(df['외부 풍속'], errors='coerce').interpolate(limit_direction='both').ffill().bfill().to_numpy(dtype=float)
    elif '풍속' in df.columns:
        u_out_series = pd.to_numeric(df['풍속'], errors='coerce').interpolate(limit_direction='both').ffill().bfill().to_numpy(dtype=float)
    else:
        u_out_series = np.full(n_hours, 1.0, dtype=float)

    if '전운량' in df.columns:
        cloud_raw = pd.to_numeric(df['전운량'], errors='coerce').interpolate(limit_direction='both').ffill().bfill().to_numpy(dtype=float)
        cloud_frac_series = np.clip(cloud_raw / 10.0, 0.0, 1.0)
    elif '운량' in df.columns:
        cloud_raw = pd.to_numeric(df['운량'], errors='coerce').interpolate(limit_direction='both').ffill().bfill().to_numpy(dtype=float)
        cloud_frac_series = np.clip(cloud_raw / 10.0, 0.0, 1.0)
    else:
        cloud_frac_series = np.full(n_hours, cloud_default, dtype=float)

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
    qLatent_kW = np.zeros(n_hours)
    qFIR = np.zeros(n_hours)

    Tp_lw_K = np.full(n_hours, np.nan, dtype=float)
    Tsky_lw_K = np.full(n_hours, np.nan, dtype=float)
    eps_lw = np.full(n_hours, np.nan, dtype=float)

    ach_array = np.zeros(n_hours)
    u_int_array = np.zeros(n_hours)

    RHin = np.zeros(n_hours)

    Wet_values = np.zeros(n_hours)
    Wvt_values = np.zeros(n_hours)
    Wr_values = np.zeros(n_hours)

    ra_values = np.zeros(n_hours)
    rc_values = np.zeros(n_hours)
    rs_values = np.zeros(n_hours)
    hs_values = np.zeros(n_hours)

    Rns_values = np.zeros(n_hours)
    Rnl_values = np.zeros(n_hours)
    Rn_values = np.zeros(n_hours)
    Is_values = np.zeros(n_hours)
    f_int_values = np.zeros(n_hours)

    lambdaE_values = np.zeros(n_hours)

    gsw_used_array = np.zeros(n_hours)

    Troom[0] = 20 + 273.15
    rhoAir[0] = PropsSI("D", "T", Troom[0], "P", Patm, "Air")
    cAir[0] = PropsSI("C", "T", Troom[0], "P", Patm, "Air") / 1000.0
    mHouse[0] = volumeHouse * rhoAir[0]
    qt[0] = 0
    RHin[0] = 0.7

    ramp_active = False
    ramp_start_i = 0
    ramp_len = trns_hours
    ramp_start_ach = 0.0
    ramp_target_ach = 0.0

    ramp_start_u = 0.0
    ramp_target_u = 0.0

    prev_is_day = None

    for i in range(n_hours):
        if i > 0:
            rhoAir[i] = PropsSI("D", "T", Troom[i], "P", Patm, "Air")
            cAir[i] = PropsSI("C", "T", Troom[i], "P", Patm, "Air") / 1000.0
            mHouse[i] = volumeHouse * rhoAir[i]

        Ur, Uw, transGlass, cov = calculate_covering_properties(radSolar[i], Troom[i], Ttarget=Ttarget)
        Ur_values[i] = Ur
        Uw_values[i] = Uw
        transGlass_values[i] = transGlass
        active_covers[i] = cov

        raw_ACH, raw_u = determine_ach_and_internal_wind(Troom[i], RHin[i], radSolar[i])

        is_day = radSolar[i] > 0

        if prev_is_day is None:
            prev_is_day = is_day
            current_ACH = raw_ACH
            u_int = raw_u

        else:
            if is_day != prev_is_day:
                ramp_active = True
                ramp_start_i = max(i - 1, 0)

                if i > 0:
                    ramp_start_ach = ach_array[i - 1]
                    ramp_start_u = u_int_array[i - 1]
                else:
                    ramp_start_ach = raw_ACH
                    ramp_start_u = raw_u

                ramp_target_ach = raw_ACH
                ramp_target_u = raw_u

                prev_is_day = is_day

            if ramp_active:
                t = i - ramp_start_i
                if t <= ramp_len and ramp_len > 0:
                    frac = t / ramp_len
                    current_ACH = (1 - frac) * ramp_start_ach + frac * ramp_target_ach
                    u_int = (1 - frac) * ramp_start_u + frac * ramp_target_u
                else:
                    current_ACH = ramp_target_ach
                    u_int = ramp_target_u
                    ramp_active = False
            else:
                current_ACH = raw_ACH
                u_int = raw_u

        ach_array[i] = current_ACH
        u_int_array[i] = u_int

        RHout = df['외부 습도'].iloc[i] / 100.0

        Tsolair2 = Toutdoor[i] + (agh * radSolar[i]) / 17.0

        # Q1. 단파 복사
        qRad = (transGlass_values[i] * areaHouse * radSolar[i]) / 1000.0

        # Q2. 전도/대류
        qRoof = (Tsolair2 - Troom[i]) * Ur_values[i] * areaHouse / 1000.0
        qFloor = (Tgroundi - Troom[i]) * Ug * areaHouse / 1000.0
        qSideWall = (Tsolair2 - Troom[i]) * Uw_values[i] * surfaceHouse / 1000.0

        # Q3. 환기 전열
        qVent = current_ACH * (Toutdoor[i] - Troom[i]) * mHouse[i] * cAir[i] / 3600.0

        # 실내 포화수증기압 & 혼합비
        T_for_sat_in = max(Troom[i], 273.15)
        p_sat_in = saturation_vapor_pressure_Pa(T_for_sat_in)
        Win = 0.622 * (RHin[i] * p_sat_in) / (Patm - (RHin[i] * p_sat_in))

        # 외기 혼합비
        T_for_sat_out = max(Toutdoor[i], 273.15)
        p_sat_out = saturation_vapor_pressure_Pa(T_for_sat_out)
        Wout = 0.622 * (RHout * p_sat_out) / (Patm - (RHout * p_sat_out))

        Rns, Is, f_int = compute_Rns_paper_form(
            radSolar_out=radSolar[i],
            transGlass=transGlass_values[i],
            alpha_sw=a_sw_can,
            k_sw=k_sw,
            LAI_val=LAI
        )

        Rnl = 0.0  # 무시함 (엽온이 필요하기도 하고 온실에서는 방사된 장파 복사가 주변 구조에 다시 흡수돼서 순 장파복사 손실이 작아 무시 가능함)
        Rn = Rns

        Rns_values[i] = Rns
        Rnl_values[i] = Rnl
        Rn_values[i] = Rn
        Is_values[i] = Is
        f_int_values[i] = f_int

        gsw = float(gsw_series[i]) if i < len(gsw_series) else 0.017
        gsw_used_array[i] = gsw

        rs, rc = stomatal_and_canopy_resistance(gsw=gsw, LAI_val=LAI)

        ra, hs, d_eq, Re, Gr, Nu = aerodynamic_resistance_SM(
            T_air_K=Troom[i],
            u_canopy=u_int,
            Ld=Ld,
            Lw=Lw,
            LAI_val=LAI,
            deltaT=deltaT_dyn
        )

        lambdaE, Wet, h_fg, VPD_Pa, Delta, gamma = compute_latent_flux_SM(
            T_air_K=Troom[i],
            RH_in_frac=RHin[i],
            Rn=Rn,
            ra=ra,
            rc=rc,
            areaHouse=areaHouse,
            LAI_val=LAI
        )

        lambdaE_values[i] = lambdaE

        rs_values[i] = rs
        rc_values[i] = rc
        ra_values[i] = ra
        hs_values[i] = hs

        # W2. 환기 수분 교환량
        Wvt = current_ACH * mHouse[i] * (Win - Wout)

        # Q4. 잠열
        qLatent = (h_fg * Wet) / 3600.0 / 1000.0
        qLatent_kW[i] = qLatent

        # Q5. 장파복사
        qFIR_kW_i, Tp_i, Tsky_i, eps_i = compute_qFIR_kW_internal(
            Ti_K=Troom[i],
            Te_K=Toutdoor[i],
            RH_in_frac=RHin[i],
            radSolar_out=radSolar[i],
            covers=active_covers[i],
            areaHouse=areaHouse,
            u_out=u_out_series[i],
            cloud_frac=cloud_frac_series[i],
        )
        qFIR[i] = qFIR_kW_i
        Tp_lw_K[i] = Tp_i
        Tsky_lw_K[i] = Tsky_i
        eps_lw[i] = eps_i

        # 실내 수분 혼합비 변화량
        Wr = (Wet - Wvt) / (mHouse[i] + CWhouse)

        # 총 열 에너지 부하
        qt[i] = qRad + qRoof + qFloor + qSideWall + qVent - qLatent - qFIR[i]

        if i < n_hours - 1:
            cap = mHouse[i] * cAir[i] + Chouse
            if cap > 0:
                Troom[i + 1] = Troom[i] + (qt[i] / cap) * 3600.0
            else:
                Troom[i + 1] = Troom[i]

            new_W = Win + Wr
            T_next = max(Troom[i + 1], 273.15)
            p_sat_in_next = saturation_vapor_pressure_Pa(T_next)
            RH_next = (new_W * Patm) / (p_sat_in_next * (new_W + 0.622))
            RHin[i + 1] = max(0.0, min(RH_next, 1.0))

        Wet_values[i] = Wet
        Wvt_values[i] = Wvt
        Wr_values[i] = Wr

    df['Troom'] = Troom
    df['rhoAir'] = rhoAir
    df['cAir'] = cAir

    df['Ur'] = Ur_values
    df['Uw'] = Uw_values
    df['transGlass'] = transGlass_values
    df['active_covers'] = [','.join(c) for c in active_covers]

    df['qt'] = qt
    df['qLatent_kW'] = qLatent_kW
    df['qFIR_kW'] = qFIR

    df['Tp_lw_K'] = Tp_lw_K
    df['Tsky_lw_K'] = Tsky_lw_K
    df['eps_lw'] = eps_lw

    df['Temp_K'] = df['내부 온도'] + 273.15
    df['RHin'] = RHin * 100.0

    df['Wet'] = Wet_values
    df['Wvt'] = Wvt_values
    df['Wr'] = Wr_values

    df['RHout'] = df['외부 습도']

    df['ra'] = ra_values
    df['rc'] = rc_values
    df['rs'] = rs_values
    df['hs'] = hs_values
    df['u_int'] = u_int_array
    df['ACH'] = ach_array

    df['Is'] = Is_values
    df['f_int'] = f_int_values
    df['Rns'] = Rns_values
    df['Rnl'] = Rnl_values
    df['Rn'] = Rn_values

    df['lambdaE'] = lambdaE_values

    df['gsw'] = gsw_used_array
    df['gsw_used'] = gsw_used_array

    params = {
        'areaHouse': areaHouse,
        'surfaceHouse': surfaceHouse,
        'volumeHouse': volumeHouse,
        'Tgroundi': Tgroundi,
        'ACH': ach_array,
        'agh': agh,
        'Chouse': Chouse,
        'CWhouse': CWhouse,
        'Ttarget': Ttarget,
        'Ug': Ug,
        'Ld': Ld,
        'Lw': Lw,
        'a_sw_can': a_sw_can,
        'k_sw': k_sw,
        'LAI': LAI
    }

    refs_env = {
        "radSolar": radSolar,
        "Toutdoor": Toutdoor,
        "RHout": df['RHout'].to_numpy(dtype=float) if 'RHout' in df.columns else None,
    }
    refs_env = {k: v for k, v in refs_env.items() if v is not None}

    ref_hour = np.array([t.hour for t in df['날짜&시간']])
    warn_if_unintended_constant(df['gsw'].to_numpy(), "gsw", ref_arrays={"hour": ref_hour})

    warn_if_unintended_constant(df['ACH'].to_numpy(), "ACH", ref_arrays=refs_env)
    warn_if_unintended_constant(df['u_int'].to_numpy(), "u_int", ref_arrays=refs_env)
    warn_if_unintended_constant(df['transGlass'].to_numpy(), "transGlass", ref_arrays=refs_env)
    warn_if_unintended_constant(df['Ur'].to_numpy(), "Ur", ref_arrays=refs_env)
    warn_if_unintended_constant(df['Uw'].to_numpy(), "Uw", ref_arrays=refs_env)

    warn_if_unintended_constant(df['Rns'].to_numpy(), "Rns", ref_arrays=refs_env)
    warn_if_unintended_constant(df['Rn'].to_numpy(), "Rn", ref_arrays=refs_env)

    warn_if_unintended_constant(df['rs'].to_numpy(), "rs", ref_arrays={"gsw": df['gsw'].to_numpy()})
    warn_if_unintended_constant(df['rc'].to_numpy(), "rc", ref_arrays={"gsw": df['gsw'].to_numpy()})
    warn_if_unintended_constant(df['ra'].to_numpy(), "ra", ref_arrays={"u_int": df['u_int'].to_numpy(), "Troom": df['Troom'].to_numpy()})
    warn_if_unintended_constant(df['hs'].to_numpy(), "hs", ref_arrays={"u_int": df['u_int'].to_numpy(), "Troom": df['Troom'].to_numpy()})

    warn_if_unintended_constant(df['Wet'].to_numpy(), "Wet", ref_arrays={"Rn": df['Rn'].to_numpy(), "RHin": df['RHin'].to_numpy(), "gsw": df['gsw'].to_numpy()})
    warn_if_unintended_constant(df['Wvt'].to_numpy(), "Wvt", ref_arrays={"ACH": df['ACH'].to_numpy(), "RHout": df['RHout'].to_numpy(), "RHin": df['RHin'].to_numpy()})
    warn_if_unintended_constant(df['Wr'].to_numpy(), "Wr", ref_arrays={"Wet": df['Wet'].to_numpy(), "Wvt": df['Wvt'].to_numpy()})

    warn_if_unintended_constant(df['qLatent_kW'].to_numpy(), "qLatent_kW", ref_arrays={"Wet": df['Wet'].to_numpy()})
    warn_if_unintended_constant(df['qFIR_kW'].to_numpy(), "qFIR_kW", ref_arrays=refs_env)
    warn_if_unintended_constant(df['qt'].to_numpy(), "qt", ref_arrays=refs_env)

    warn_if_unintended_constant(df['Troom'].to_numpy(), "Troom", ref_arrays=refs_env)
    warn_if_unintended_constant(df['RHin'].to_numpy(), "RHin", ref_arrays=refs_env)

    return (
        df, qt, Toutdoor, radSolar, Troom,
        params, rhoAir, cAir, mHouse,
        Ur_values, Uw_values, transGlass_values,
        active_covers, ach_array
    )

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
        print("데이터 부족 -> 온도 회귀분석 불가.")
        return

    slope, intercept = np.polyfit(troom_c, temp_c, 1)
    predicted = slope * troom_c + intercept
    r2 = r2_score(temp_c, predicted)
    rmse = np.sqrt(mean_squared_error(temp_c, predicted))

    plt.figure()
    plt.scatter(troom_c, temp_c, color='green', label='Data', alpha=0.5)

    x_line = np.linspace(min(troom_c), max(troom_c), 100)
    y_line = slope * x_line + intercept
    plt.plot(
        x_line, y_line,
        color='black',
        label=f'Regression line\nSlope={slope:.3f}\nR²={r2:.3f}\nRMSE={rmse:.3f}'
    )

    plt.xlabel('Measured inside temperature(°C)')
    plt.ylabel('Predicted inside temperature(°C)')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_graph and filename:
        plt.savefig(filename)
        print(f"[온도 회귀] 그래프 -> {filename} 저장.")
    plt.show()

def plot_rhin_vs_humi_regression(rhin, humi, title="", save_graph=False, filename=None):
    mask = ~np.isnan(rhin) & ~np.isnan(humi)
    if mask.sum() < 2:
        print("데이터 부족 -> 습도 회귀분석 불가.")
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
    plt.plot(
        x_line, y_line,
        color='black',
        label=f'Regression line\nSlope={slope:.3f}\nR²={r2:.3f}\nRMSE={rmse:.3f}'
    )

    plt.xlabel('Measured inside relative humidity(%)')
    plt.ylabel('Predicted inside relative humidity(%)')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_graph and filename:
        plt.savefig(filename)
        print(f"[습도 회귀] 그래프 -> {filename} 저장.")
    plt.show()

def display_daily_graphs(selected_date, start_hour_index, df, Troom, Toutdoor, qt, save_graph=False):
    daily_slice = slice(start_hour_index, start_hour_index + 24)

    daily_Troom = Troom[daily_slice]
    daily_Toutdoor = Toutdoor[daily_slice]
    daily_Temp_K = df['Temp_K'].values[daily_slice]
    daily_qt = qt[daily_slice]

    daily_Wet = df['Wet'].values[daily_slice]
    daily_RHin = df['RHin'].values[daily_slice]
    daily_Humi = df['Humi'].values[daily_slice]
    daily_RHout = df['RHout'].values[daily_slice]

    hours_full = np.arange(24)
    n = len(daily_Troom)
    hours = hours_full[:n]

    # 1. Thermal Energy Load
    plt.figure()
    plt.plot(hours, daily_qt[:n], label='Thermal Energy Load(kW)', color='orange')
    plt.title("Thermal Energy Load")
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy Load(kW)')
    plt.xlim(0, 23)
    plt.xticks(hours_full)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_graph:
        fn = f"energy_load_{selected_date.strftime('%Y%m%d')}.png"
        plt.savefig(fn)
        print(f"Thermal Energy Load Graph -> {fn} 저장.")
    plt.show()

    # 2. Temperature Combined
    plt.figure()
    plt.plot(hours, (daily_Troom[:n] - 273.15), label='Predicted inside temperature(°C)', color='red')
    plt.plot(hours, (daily_Temp_K[:n] - 273.15), label='Measured inside temperature(°C)', color='blue')
    plt.plot(hours, (daily_Toutdoor[:n] - 273.15), label='Measured outside temperature(°C)', color='black')
    plt.title("Predicted & Measured (Inside/Outside) Temperature")
    plt.xlabel('Hour of Day')
    plt.ylabel('Temperature(°C)')
    plt.xlim(0, 23)
    plt.xticks(hours_full)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_graph:
        fn = f"temp_all_{selected_date.strftime('%Y%m%d')}.png"
        plt.savefig(fn)
        print(f"Temperature Combined Graph -> {fn} 저장.")
    plt.show()

    # 3. Temperature Regression
    troom_c = daily_Troom - 273.15
    temp_c = daily_Temp_K - 273.15
    plot_troom_vs_temp_regression(
        troom_c, temp_c,
        title="Measured vs Predicted inside temperature Regression",
        save_graph=save_graph,
        filename=f"regression_temp_{selected_date.strftime('%Y%m%d')}.png" if save_graph else None
    )

    # 4. Evapotranspiration
    plt.figure()
    plt.plot(hours, daily_Wet[:n], label='Evapotranspiration(kg/h)', color='orange')
    plt.title("Evapotranspiration")
    plt.xlabel('Hour of Day')
    plt.ylabel('Water flux(kg/h)')
    plt.xlim(0, 23)
    plt.xticks(hours_full)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_graph:
        fn = f"Evapotranspiration_{selected_date.strftime('%Y%m%d')}.png"
        plt.savefig(fn)
        print(f"Evapotranspiration Graph -> {fn} 저장.")
    plt.show()

    # 5. Relative Humidity Combined
    plt.figure()
    plt.plot(hours, daily_RHin[:n], label='Predicted inside relative humidity(%)', color='red')
    plt.plot(hours, daily_Humi[:n], label='Measured inside relative humidity(%)', color='blue')
    plt.plot(hours, daily_RHout[:n], label='Measured outside relative humidity(%)', color='black')
    plt.title("Predicted & Measured (Inside/Outside) Relative Humidity")
    plt.xlabel('Hour of Day')
    plt.ylabel('Relative Humidity(%)')
    plt.xlim(0, 23)
    plt.xticks(hours_full)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_graph:
        fn = f"rh_all_{selected_date.strftime('%Y%m%d')}.png"
        plt.savefig(fn)
        print(f"RH Combined Graph -> {fn} 저장.")
    plt.show()

    # 6. Humidity Regression
    plot_rhin_vs_humi_regression(
        daily_RHin, daily_Humi,
        title="Measured vs Predicted inside relative humidity Regression",
        save_graph=save_graph,
        filename=f"regression_rh_{selected_date.strftime('%Y%m%d')}.png" if save_graph else None
    )

def parse_time_str(input_str, df, allow_nearest=False):
    s = input_str.strip()

    if not s.isdigit():
        print(f"숫자형식 아님: {s}")
        return None

    try:
        val = int(s)
        if len(s) <= 5 and 1 <= val <= len(df):
            return val - 1
    except Exception:
        pass

    dt_series = df['날짜&시간']

    if len(s) == 8:
        fmt = '%y%m%d%H'
    else:
        print(f"지원하지 않는 길이: {s} (len={len(s)})")
        return None

    try:
        target_dt = datetime.strptime(s, fmt)
        target_dt = pd.to_datetime(target_dt)
    except Exception:
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

"""
CO2 Balance Modeling (직각쌍곡선 + Beer-Lambert + 환기)
"""

patm = 101325.0  # 대기압 [Pa]
gas_constant = 8.314  # 기체상수 [J/mol/K]
m_CO2 = 44.01e-3  # CO2 분자량 [kg/mol]

Cout_ppm = 420.0  # 외기 CO2 농도 [ppm]
Cinit_ppm = 500.0  # 초기 내부 CO2 농도 [ppm]

CO2_meas = '내부 CO2'

# 직각쌍곡선 고정 파라미터
Pmax = 14.66  # [µmol m-2 s-1]
alpha = 0.09216  # 양자효율
Rd = 0.9736  # [µmol m-2 s-1]

PARAMS = {
    "rect_hyperbola": {"Pmax": Pmax, "alpha": alpha, "Rd": Rd}
}

# PPFD 변환 파라미터
frac_PAR = 0.45  # PAR 비율
J_to_mol = 4.57  # W/m2 -> µmol/m2/s 변환

def radiation_Wm2_to_PPFD(radiation_Wm2, frac_PAR=frac_PAR, J_to_mol=J_to_mol):
    rad = np.asarray(radiation_Wm2, dtype=float)
    rad = np.clip(rad, 0.0, None)
    par_Wm2 = frac_PAR * rad
    ppfd = par_Wm2 * J_to_mol
    return ppfd

def beer_lambert_fint(LAI=LAI, k=k_sw):
    return 1.0 - np.exp(-k * LAI)

def rectangular_hyperbola_A(P, Pmax, alpha, Rd=0.0):
    P = np.asarray(P, dtype=float)
    P = np.clip(P, 0.0, None)

    denom = (alpha * P + Pmax)
    denom = np.where(denom == 0.0, np.nan, denom)

    A_gross = (alpha * P * Pmax) / denom
    A_gross = np.nan_to_num(A_gross, nan=0.0, posinf=0.0, neginf=0.0)

    A_net = A_gross - Rd
    return A_net

def _get_inside_radiation_for_co2(df, i):
    if 'Is' in df.columns:
        v = pd.to_numeric(df['Is'].iloc[i], errors='coerce')
        if np.isfinite(v):
            return max(float(v), 0.0)

    if 'G_inside' in df.columns:
        v = pd.to_numeric(df['G_inside'].iloc[i], errors='coerce')
        if np.isfinite(v):
            return max(float(v), 0.0)

    rad_out = pd.to_numeric(df['일사'].iloc[i], errors='coerce') if '일사' in df.columns else 0.0
    rad_out = max(float(rad_out) if np.isfinite(rad_out) else 0.0, 0.0)

    if 'transGlass' in df.columns:
        tg = pd.to_numeric(df['transGlass'].iloc[i], errors='coerce')
        if np.isfinite(tg):
            return max(rad_out * float(tg), 0.0)

    return rad_out * 0.71

def _get_Tair_K(df, i):
    if 'Troom' in df.columns:
        return float(df['Troom'].iloc[i])
    raise KeyError("df에 'Troom' 컬럼이 없습니다. energy_modeling 결과 df를 사용해야 합니다.")

def _get_RHin_frac(df, i):
    if 'RHin' in df.columns:
        return float(df['RHin'].iloc[i]) / 100.0
    if '내부 습도' in df.columns:
        return float(df['내부 습도'].iloc[i]) / 100.0
    if 'Humi' in df.columns:
        return float(df['Humi'].iloc[i]) / 100.0
    raise KeyError("df에 'RHin' (또는 내부 습도/Humi) 컬럼이 없습니다.")

def run_photosynthesis_and_co2(df):
    n = len(df)

    time = df['날짜&시간']
    dt_sec_raw = time.diff().dt.total_seconds().to_numpy()

    if n > 1:
        valid = dt_sec_raw > 0
        dt_rep = np.nanmedian(dt_sec_raw[valid])
        dt_sec = dt_sec_raw.copy()
        dt_sec[~valid] = dt_rep
        dt_sec[dt_sec > 2.0 * dt_rep] = dt_rep
    else:
        dt_sec = np.array([3600.0])

    f_int = beer_lambert_fint(LAI=LAI, k=k_sw)

    PPFD_above = np.zeros(n)
    PPFD_canopy = np.zeros(n)

    A_leaf = np.zeros(n)
    A_canopy = np.zeros(n)
    Cin_ppm = np.zeros(n)
    Ci_store = np.zeros(n)

    Cphoto_ppm_per_h = np.zeros(n)
    Cvent_ppm_per_h = np.zeros(n)
    Ctotal_ppm_per_h = np.zeros(n)

    Cphoto_kg_per_h = np.zeros(n)
    Cvent_kg_per_h = np.zeros(n)
    Cr_kg_per_h = np.zeros(n)

    Cin = Cinit_ppm

    for i in range(n):
        Tk_air = _get_Tair_K(df, i)
        T_used = max(Tk_air, 273.15)

        rad_in_Wm2 = _get_inside_radiation_for_co2(df, i)
        rad_out = max(float(df['일사'].iloc[i]) if '일사' in df.columns else 0.0, 0.0)

        this_dt_sec = dt_sec[i]
        if not np.isfinite(this_dt_sec) or this_dt_sec <= 0:
            this_dt_sec = 3600.0

        base_sub_sec = 60.0
        n_sub = max(1, int(np.round(this_dt_sec / base_sub_sec)))
        sub_dt_sec = this_dt_sec / n_sub
        sub_dt_h = sub_dt_sec / 3600.0

        if '외부 CO2' in df.columns:
            Cout_ppm_i = float(df['외부 CO2'].iloc[i])
        else:
            Cout_ppm_i = Cout_ppm_default

        ppfd_above = float(radiation_Wm2_to_PPFD([rad_in_Wm2], frac_PAR, J_to_mol)[0])
        ppfd_canopy = ppfd_above * f_int

        PPFD_above[i] = ppfd_above
        PPFD_canopy[i] = ppfd_canopy

        A_leaf_val = float(rectangular_hyperbola_A(
            P=np.array([ppfd_canopy]),
            Pmax=Pmax,
            alpha=alpha,
            Rd=Rd
        )[0])

        A_can = A_leaf_val * LAI

        A_leaf[i] = A_leaf_val
        A_canopy[i] = A_can

        n_air = patm * volumeHouse / (gas_constant * T_used)
        if n_air <= 0:
            n_air = 1.0

        coeff_kg_to_ppm_per_h = 1e6 / (n_air * m_CO2)

        sum_dCin_photo_h = 0.0
        sum_dCin_vent_h = 0.0
        sum_Cr_kg_h = 0.0
        sum_F_photo_kg_h = 0.0
        sum_F_vent_kg_h = 0.0

        last_dCin_dt_total_h = 0.0
        last_Ci_ppm = Cin

        F_photo_umol_s = A_can * areaHouse
        F_photo_mol_s = F_photo_umol_s * 1e-6
        F_photo_kg_h_internal_const = -F_photo_mol_s * 3600.0 * m_CO2 # C1. 광합성 및 호흡을 통한 CO2 변화량

        RH_in_frac_now = _get_RHin_frac(df, i)
        if 'ACH' in df.columns:
            ACH_now = float(df['ACH'].iloc[i])
        else:
            ACH_now = None

        for _ in range(n_sub):
            chi_in = Cin / 1e6
            chi_out = Cout_ppm_i / 1e6

            dnCO2_dt_vent_mol_h = ACH_now * n_air * (chi_out - chi_in) # C2. 환기를 통한 CO2 변화량
            F_vent_kg_h_internal = dnCO2_dt_vent_mol_h * m_CO2

            F_photo_kg_h_internal = F_photo_kg_h_internal_const
            Cr_kg_h_internal = F_photo_kg_h_internal + F_vent_kg_h_internal

            dCin_dt_photo_h = F_photo_kg_h_internal * coeff_kg_to_ppm_per_h
            dCin_dt_vent_h = F_vent_kg_h_internal * coeff_kg_to_ppm_per_h
            dCin_dt_total_h = Cr_kg_h_internal * coeff_kg_to_ppm_per_h

            last_dCin_dt_total_h = dCin_dt_total_h
            sum_dCin_photo_h += dCin_dt_photo_h
            sum_dCin_vent_h += dCin_dt_vent_h
            sum_Cr_kg_h += Cr_kg_h_internal
            sum_F_photo_kg_h += F_photo_kg_h_internal
            sum_F_vent_kg_h += F_vent_kg_h_internal

            Cin = Cin + dCin_dt_total_h * sub_dt_h
            last_Ci_ppm = Cin

        Cin_ppm[i] = Cin
        Ci_store[i] = last_Ci_ppm

        Cphoto_ppm_per_h[i] = sum_dCin_photo_h / n_sub
        Cvent_ppm_per_h[i] = sum_dCin_vent_h / n_sub
        Ctotal_ppm_per_h[i] = last_dCin_dt_total_h

        Cphoto_kg_per_h[i] = sum_F_photo_kg_h / n_sub
        Cvent_kg_per_h[i] = sum_F_vent_kg_h / n_sub
        Cr_kg_per_h[i] = sum_Cr_kg_h / n_sub

    df['PPFD_above'] = PPFD_above
    df['PPFD_canopy'] = PPFD_canopy
    df['A_leaf_umol_m2_s'] = A_leaf
    df['A_canopy_umol_m2_s'] = A_canopy
    df['Cin_ppm'] = Cin_ppm
    df['Ci_ppm'] = Ci_store
    df['Cphoto_ppm_per_h'] = Cphoto_ppm_per_h
    df['Cvent_ppm_per_h'] = Cvent_ppm_per_h
    df['Ctotal_ppm_per_h'] = Ctotal_ppm_per_h

    df['Cphoto_kg_per_h'] = Cphoto_kg_per_h
    df['Cvent_kg_per_h'] = Cvent_kg_per_h
    df['Cr_kg_per_h'] = Cr_kg_per_h

    df['f_int'] = f_int
    df['LAI'] = LAI
    df['k_sw'] = k_sw
    df['Pmax'] = Pmax
    df['alpha'] = alpha
    df['Rd'] = Rd

    refs_co2 = {"radSolar": df['일사'].to_numpy(dtype=float) if '일사' in df.columns else None,
                "ACH": df['ACH'].to_numpy(dtype=float) if 'ACH' in df.columns else None}
    refs_co2 = {k: v for k, v in refs_co2.items() if v is not None}

    warn_if_unintended_constant(df['Cin_ppm'].to_numpy(), "Cin_ppm", ref_arrays=refs_co2)
    warn_if_unintended_constant(df['A_leaf_umol_m2_s'].to_numpy(), "A_leaf_umol_m2_s", ref_arrays={"radSolar": refs_co2.get("radSolar", np.array([]))})

    return df

def plot_co2_regression(meas, pred, title="Measured vs Predicted CO2 Regression", save_graph=False, filename=None):
    mask = ~np.isnan(meas) & ~np.isnan(pred)
    if mask.sum() < 2:
        print("데이터 부족 -> CO2 회귀분석 불가.")
        return

    x = meas[mask]
    y = pred[mask]

    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    r2 = r2_score(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))

    plt.figure()
    plt.scatter(x, y, alpha=0.5, color='green', label='Data')

    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept
    plt.plot(
        x_line, y_line,
        color='black',
        label=f'Regression line\nSlope={slope:.3f}\nR²={r2:.3f}\nRMSE={rmse:.3f}'
    )

    plt.xlabel('Measured inside CO2 (ppm)')
    plt.ylabel('Predicted inside CO2 (ppm)')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_graph and filename:
        plt.savefig(filename)
        print(f"[CO2 회귀] 그래프 -> {filename} 저장.")
    plt.show()

def display_range_graphs(start_index, end_index, df, Troom, Toutdoor, qt, save_graph=False):
    x_range = range(start_index, end_index + 1)
    sub_df = df.iloc[start_index:end_index + 1].copy()

    sub_Troom = Troom[start_index:end_index + 1]
    sub_Toutdoor = Toutdoor[start_index:end_index + 1]
    sub_qt = qt[start_index:end_index + 1]
    sub_Temp_K = sub_df['Temp_K'].values

    sub_Wet = sub_df['Wet'].values
    sub_RHin = sub_df['RHin'].values
    sub_Humi = sub_df['Humi'].values
    sub_RHout = sub_df['RHout'].values

    # 1. Thermal Energy Load
    plt.figure()
    plt.plot(x_range, sub_qt, label='Thermal Energy Load(kW)', color='orange')
    plt.title("Thermal Energy Load")
    plt.xlabel('Hour Index')
    plt.ylabel('Energy Load(kW)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_graph:
        fn = f"range_energy_{start_index + 1}_{end_index + 1}.png"
        plt.savefig(fn)
        print(f"Thermal Energy Load Graph -> {fn} 저장.")
    plt.show()

    # 2. Temperature Combined
    plt.figure()
    plt.plot(x_range, sub_Troom - 273.15, label='Predicted inside temperature(°C)', color='red')
    plt.plot(x_range, sub_Temp_K - 273.15, label='Measured inside temperature(°C)', color='blue')
    plt.plot(x_range, sub_Toutdoor - 273.15, label='Measured outside temperature(°C)', color='black')
    plt.xlabel('Hour Index')
    plt.ylabel('Temperature(°C)')
    plt.title("Predicted & Measured (Inside/Outside) Temperature")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_graph:
        fn = f"range_temp_all_{start_index + 1}_{end_index + 1}.png"
        plt.savefig(fn)
        print(f"Temperature Combined Graph -> {fn} 저장.")
    plt.show()

    # 3. Temperature Regression
    troom_c = sub_Troom - 273.15
    temp_c = sub_Temp_K - 273.15
    if len(troom_c) > 2:
        slope, intercept = np.polyfit(troom_c, temp_c, 1)
        predicted = slope * troom_c + intercept
        r2 = r2_score(temp_c, predicted)
        rmse = np.sqrt(mean_squared_error(temp_c, predicted))

        plt.figure()
        plt.scatter(troom_c, temp_c, alpha=0.5, label='Data', color='green')
        x_line = np.linspace(min(troom_c), max(troom_c), 100)
        y_line = slope * x_line + intercept
        plt.plot(
            x_line, y_line,
            color='black',
            label=f"slope={slope:.3f}\nR²={r2:.3f}\nRMSE={rmse:.3f}"
        )
        plt.xlabel('Measured inside temperature(°C)')
        plt.ylabel('Predicted inside temperature(°C)')
        plt.title("Temperature Regression")
        plt.legend()
        plt.grid(True)
        if save_graph:
            fn = f"range_regression_temp_{start_index + 1}_{end_index + 1}.png"
            plt.savefig(fn)
            print(f"Temperature Regression Graph -> {fn} 저장.")
        plt.show()
    else:
        print("온도 회귀용 데이터 부족")

    # 4. Evapotranspiration
    plt.figure()
    plt.plot(x_range, sub_Wet, label='Evapotranspiration(kg/h)', color='orange')
    plt.title("Evapotranspiration")
    plt.xlabel('Hour Index')
    plt.ylabel('Water flux(kg/h)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_graph:
        fn = f"range_Evapotranspiration_{start_index + 1}_{end_index + 1}.png"
        plt.savefig(fn)
        print(f"Evapotranspiration Graph -> {fn} 저장.")
    plt.show()

    # 5. Relative Humidity Combined
    plt.figure()
    plt.plot(x_range, sub_RHin, label='Predicted inside relative humidity(%)', color='red')
    plt.plot(x_range, sub_Humi, label='Measured inside relative humidity(%)', color='blue')
    plt.plot(x_range, sub_RHout, label='Measured outside relative humidity(%)', color='black')
    plt.xlabel('Hour Index')
    plt.ylabel('Relative Humidity(%)')
    plt.title("Predicted & Measured (Inside/Outside) Relative Humidity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_graph:
        fn = f"range_rh_all_{start_index + 1}_{end_index + 1}.png"
        plt.savefig(fn)
        print(f"RH Combined Graph -> {fn} 저장.")
    plt.show()

    # 6. Humidity Regression
    mask = ~np.isnan(sub_RHin) & ~np.isnan(sub_Humi)
    if mask.sum() > 2:
        x_data = sub_RHin[mask]
        y_data = sub_Humi[mask]
        slope, intercept = np.polyfit(x_data, y_data, 1)
        predicted = slope * x_data + intercept
        r2 = r2_score(y_data, predicted)
        rmse = np.sqrt(mean_squared_error(y_data, predicted))

        plt.figure()
        plt.scatter(x_data, y_data, alpha=0.5, label='Data', color='green')
        x_line = np.linspace(min(x_data), max(x_data), 100)
        y_line = slope * x_line + intercept
        plt.plot(
            x_line, y_line,
            label=f"slope={slope:.3f}\nR²={r2:.3f}\nRMSE={rmse:.3f}",
            color='black'
        )
        plt.xlabel('Measured inside relative humidity(%)')
        plt.ylabel('Predicted inside relative humidity(%)')
        plt.title("Humidity Regression")
        plt.legend()
        plt.grid(True)
        if save_graph:
            fn = f"range_regression_rh_{start_index + 1}_{end_index + 1}.png"
            plt.savefig(fn)
            print(f"RH Regression Graph -> {fn} 저장.")
        plt.show()
    else:
        print("습도 회귀용 데이터 부족")

def show_detailed_energy_calculation(hour_index, df, params, Toutdoor, radSolar, Troom,
                              qt_array, rhoAir, cAir, mHouse,
                              Ur_values, Uw_values, transGlass_values,
                              active_covers, ach_array):
    try:
        ts = df['날짜&시간'].iloc[hour_index]
        date_str = ts.strftime("%Y년 %m월 %d일 %H시")

        if '전운량' in df.columns:
            cloud_amount = df['전운량'].iloc[hour_index]
        else:
            cloud_amount = np.nan

        Tout = Toutdoor[hour_index]
        T_in = Troom[hour_index]
        solar = radSolar[hour_index]
        rho = rhoAir[hour_index]
        c = cAir[hour_index]
        m = mHouse[hour_index]
        Ur = Ur_values[hour_index]
        Uw = Uw_values[hour_index]
        tg = transGlass_values[hour_index]
        covers = active_covers[hour_index]
        ACH = ach_array[hour_index]
        Ug = params['Ug']

        if 'gsw_used' in df.columns:
            gsw_now = df['gsw_used'].iloc[hour_index]
        else:
            gsw_now = None

        Tsolair2 = Tout + (params['agh'] * solar) / 17.0

        A = params['areaHouse']
        Aside = params['surfaceHouse']

        qRad = (tg * A * solar) / 1000.0
        qRoof = (Tsolair2 - T_in) * Ur * A / 1000.0
        qFloor = (params['Tgroundi'] - T_in) * Ug * A / 1000.0
        qSideWall = (Tsolair2 - T_in) * Uw * Aside / 1000.0
        qVent = ACH * (Tout - T_in) * m * c / 3600.0
        qLatent = df['qLatent_kW'].iloc[hour_index] if 'qLatent_kW' in df.columns else 0.0

        Tp_lw = df['Tp_lw_K'].iloc[hour_index] if 'Tp_lw_K' in df.columns else np.nan
        Tsky_lw = df['Tsky_lw_K'].iloc[hour_index] if 'Tsky_lw_K' in df.columns else np.nan
        eps_lw_val = df['eps_lw'].iloc[hour_index] if 'eps_lw' in df.columns else np.nan

        qFIR_calc = np.nan
        if np.isfinite(Tp_lw) and np.isfinite(Tsky_lw) and np.isfinite(eps_lw_val):
            qFIR_calc = float(eps_lw_val) * Sigma * A * (float(Tp_lw) ** 4 - float(Tsky_lw) ** 4) / 1000.0
        elif 'qFIR_kW' in df.columns and not pd.isna(df['qFIR_kW'].iloc[hour_index]):
            qFIR_calc = float(df['qFIR_kW'].iloc[hour_index])
        else:
            qFIR_calc = 0.0

        qt_val = qt_array[hour_index]

        Wet_val = df['Wet'].iloc[hour_index] if 'Wet' in df.columns else 0.0

        try:
            h_fg_loc = latent_heat_vaporization(T_in)
        except Exception:
            h_fg_loc = np.nan

        energy_terms = {
            "qRad": {
                "공식": "qRad = (transGlass * A * G_solar) / 1000",
                "계산": f"({tg:.4f} * {A:.2f} * {solar:.2f}) / 1000",
                "결과": f"{qRad:.3f} kW"
            },
            "qRoof": {
                "공식": "qRoof = (Tsolair2 - Troom) * Ur * A / 1000",
                "계산": f"({Tsolair2:.2f} - {T_in:.2f}) * {Ur:.4f} * {A:.2f} / 1000",
                "결과": f"{qRoof:.3f} kW"
            },
            "qFloor": {
                "공식": "qFloor = (Tgroundi - Troom) * Ug * A / 1000",
                "계산": f"({params['Tgroundi']:.2f} - {T_in:.2f}) * {Ug:.4f} * {A:.2f} / 1000",
                "결과": f"{qFloor:.3f} kW"
            },
            "qSideWall": {
                "공식": "qSideWall = (Tsolair2 - Troom) * Uw * Aside / 1000",
                "계산": f"({Tsolair2:.2f} - {T_in:.2f}) * {Uw:.4f} * {Aside:.2f} / 1000",
                "결과": f"{qSideWall:.3f} kW"
            },
            "qVent": {
                "공식": "qVent = ACH * (Tout - Troom) * mHouse * cAir / 3600",
                "계산": f"{ACH:.2f} * ({Tout:.2f} - {T_in:.2f}) * {m:.2f} * {c:.4f} / 3600",
                "결과": f"{qVent:.3f} kW"
            },
            "qLatent": {
                "공식": "qLatent = h_fg * Wet / 3600 / 1000",
                "계산": f"{h_fg_loc:.2f} * {Wet_val:.6f} / 3600 / 1000",
                "결과": f"{qLatent:.3f} kW"
            },
            "qFIR": {
                "공식": "qFIR = ε * σ * A * (Tp^4 - Tsky^4) / 1000",
                "계산": (
                    f"{eps_lw_val:.3f} * {Sigma:.3e} * {A:.2f} * ({Tp_lw:.2f}^4 - {Tsky_lw:.2f}^4) / 1000"
                    if np.isfinite(Tp_lw) and np.isfinite(Tsky_lw) and np.isfinite(eps_lw_val)
                    else "Tp_lw_K/Tsky_lw_K/eps_lw 값이 유효하지 않아 대입식을 표시할 수 없습니다."
                ),
                "결과": f"{qFIR_calc:.3f} kW"
            }
        }

        qt_detail = {
            "공식": "qt = qRad + qRoof + qFloor + qSideWall + qVent - qLatent - qFIR",
            "계산": (
                f"{qRad:.3f} + {qRoof:.3f} + {qFloor:.3f} + "
                f"{qSideWall:.3f} + {qVent:.3f} - {qLatent:.3f} - {qFIR_calc:.3f}"
            ),
            "결과": f"{qt_val:.3f} kW"
        }

        if hour_index < len(df) - 1:
            cap = m * c + params['Chouse']
            dT = qt_val / cap * 3600.0 if cap > 0 else 0.0
            T_next = T_in + dT
            troom_calc = {
                "공식": "Troom[i+1] = Troom[i] + (qt[i]/(m*c+Chouse)) * 3600",
                "계산": (
                    f"cap = {m:.2f} * {c:.4f} + {params['Chouse']:.2f} = {cap:.2f}\n"
                    f"ΔT = {qt_val:.3f} / {cap:.2f} * 3600 = {dT:.4f} K"
                ),
                "결과": f"{T_next:.2f} K ({T_next - 273.15:.2f}°C)"
            }
        else:
            troom_calc = {"마지막 시간대": "다음 시간 실내 온도 계산 불가"}

        result = {
            "시간정보": f"Hour {hour_index + 1} / {date_str}",
            "기본 데이터": {
                "온실 바닥면적": f"{A:.2f} m²",
                "외기온도": f"{Tout:.2f} K ({Tout - 273.15:.2f}°C)",
                "실내온도": f"{T_in:.2f} K ({T_in - 273.15:.2f}°C)",
                "지중온도": f"{params['Tgroundi']:.2f} K ({params['Tgroundi'] - 273.15:.2f}°C)",
                "기공전도도": f"{gsw_now:.6f} mol m⁻² s⁻¹",
                "일사량": f"{solar:.2f} W/m²",
                "전운량": f"{cloud_amount:.1f} (0~10)",
                "피복조합": f"{covers}",
                "transGlass": f"{tg:.4f}",
                "Ur": f"{Ur:.4f}",
                "Uw": f"{Uw:.4f}",
                "Ug": f"{Ug:.4f}",
                "ACH": f"{ACH:.2f}",
                "rhoAir": f"{rho:.4f} kg/m³",
                "cAir": f"{c:.4f} kJ/kgK",
                "mHouse": f"{m:.2f} kg"
            },
            "Tsolair2": {
                "공식": "Tsolair2 = Tout + (agh * G_solar) / 17",
                "계산": f"{Tout:.2f} + ({params['agh']:.2f} * {solar:.2f}) / 17",
                "결과": f"{Tsolair2:.2f} K"
            },
            "열 에너지 항목": energy_terms,
            "Qt": qt_detail,
            "Troom[next]": troom_calc
        }

        return result

    except Exception as e:
        print(f"show_detailed_energy_calculation 오류: {e}")
        return {}

def show_detailed_moisture_calculation(hour_index, df, params, Troom, mHouse, ach_array):
    try:
        ts = df['날짜&시간'].iloc[hour_index]
        date_str = ts.strftime("%Y년 %m월 %d일 %H시")

        RHin_val  = df['RHin'].iloc[hour_index] / 100.0
        Humi_val  = df['Humi'].iloc[hour_index]
        RHout_val = df['RHout'].iloc[hour_index] / 100.0

        Wet_val   = df['Wet'].iloc[hour_index]
        Wvt_val   = df['Wvt'].iloc[hour_index]
        Wr_val    = df['Wr'].iloc[hour_index]

        T_in = max(Troom[hour_index], 273.15)
        p_sat_in = saturation_vapor_pressure_Pa(T_in)
        Win = 0.622 * (RHin_val * p_sat_in) / (Patm - (RHin_val * p_sat_in))

        T_out = df['외부 온도'].iloc[hour_index] + 273.15
        T_out = max(T_out, 273.15)
        p_sat_out = saturation_vapor_pressure_Pa(T_out)
        Wout = 0.622 * (RHout_val * p_sat_out) / (Patm - (RHout_val * p_sat_out))

        VPD_kPa = (p_sat_in / 1000.0) * (1.0 - RHin_val)

        ACH = ach_array[hour_index]
        m = mHouse[hour_index]

        A = params['areaHouse']

        try:
            h_fg_loc = latent_heat_vaporization(T_in)
        except Exception:
            h_fg_loc = np.nan

        if np.isfinite(h_fg_loc) and A > 0:
            lambdaE_est = (Wet_val * h_fg_loc) / (A * 3600.0)
        else:
            lambdaE_est = np.nan

        ra = df['ra'].iloc[hour_index] if 'ra' in df.columns else np.nan
        rc = df['rc'].iloc[hour_index] if 'rc' in df.columns else np.nan
        rs = df['rs'].iloc[hour_index] if 'rs' in df.columns else np.nan

        if 'gsw_used' in df.columns:
            gsw_now = df['gsw_used'].iloc[hour_index]
        else:
            gsw_now = None

        Win_info = {
            "공식": "Win = 0.622*(RHin*p_sat_in)/(Patm - RHin*p_sat_in)",
            "계산": f"0.622*({RHin_val:.4f}*{p_sat_in:.2f})/(101325 - {RHin_val:.4f}*{p_sat_in:.2f})",
            "결과": f"{Win:.6f} kg/kg"
        }

        Wout_info = {
            "공식": "Wout = 0.622*(RHout*p_sat_out)/(Patm - RHout*p_sat_out)",
            "계산": f"0.622*({RHout_val:.4f}*{p_sat_out:.2f})/(101325 - {RHout_val:.4f}*{p_sat_out:.2f})",
            "결과": f"{Wout:.6f} kg/kg"
        }

        VPD_info = {
            "공식": "VPD = (p_sat_in/1000)*(1-RHin)",
            "계산": f"({p_sat_in:.2f}/1000)*(1-{RHin_val:.4f})",
            "결과": f"{VPD_kPa:.5f} kPa"
        }

        Wet_info = {
            "공식": "Wet = (λE / h_fg) * A * 3600",
            "계산": f"({lambdaE_est:.3f} / {h_fg_loc:.2f}) * {A:.2f} * 3600",
            "결과": f"{Wet_val:.6f} kg/h"
        }

        Wvt_calc = {
            "공식": "Wvt = ACH * mHouse * (Win - Wout)",
            "계산": f"{ACH:.2f} * {m:.2f} * ({Win:.6f} - {Wout:.6f})",
            "결과": f"{Wvt_val:.6f} kg/h"
        }

        Wr_calc = {
            "공식": "Wr = (Wet - Wvt) / (mHouse + CWhouse)",
            "계산": (
                f"({Wet_val:.6f} - {Wvt_val:.6f})"
                f" / ({m:.2f} + {params['CWhouse']:.2f})"
            ),
            "결과": f"{Wr_val:.8f} kg/kg·h"
        }

        result = {
            "시간정보": f"Hour {hour_index + 1} / {date_str}",
            "기본 데이터": {
                "예측 내부 습도(RHin)": f"{RHin_val * 100:.2f}%",
                "실측 내부 습도(Humi)": f"{Humi_val:.2f}%",
                "외부 습도(RHout)": f"{RHout_val * 100:.2f}%"
            },
            "혼합비": {"Win": Win_info, "Wout": Wout_info},
            "VPD": VPD_info,
            "Stanghellini Model Parameter": {
                "rs(기공저항)": f"{rs:.3f} s/m",
                "rc(캐노피저항)": f"{rc:.3f} s/m",
                "ra(기류저항)": f"{ra:.3f} s/m",
                "gsw(딸기의 기공전도도)": f"{gsw_now:.6f} mol m⁻² s⁻¹",
                "gamma(사이크로매트릭 상수)": "66",
                "G(토양 열 유동)": "0",
            },
            "Wet": Wet_info,
            "Wvt": Wvt_calc,
            "Wr": Wr_calc
        }
        return result

    except Exception as e:
        print(f"show_detailed_moisture_calculation 오류: {e}")
        return {}

def show_detailed_CO2_calculation(hour_index, df):
    try:
        ts = df['날짜&시간'].iloc[hour_index]
        date_str = ts.strftime("%Y년 %m월 %d일 %H시")

        Tout_C = float(df['외부 온도'].iloc[hour_index]) if '외부 온도' in df.columns else np.nan
        Tout_K = Tout_C + 273.15 if np.isfinite(Tout_C) else np.nan

        rad_out = float(df['일사'].iloc[hour_index]) if '일사' in df.columns else 0.0
        rad_out = max(rad_out, 0.0)

        if 'transGlass' in df.columns and np.isfinite(df['transGlass'].iloc[hour_index]):
            transGlass = float(df['transGlass'].iloc[hour_index])
        else:
            transGlass = np.nan

        if 'Is' in df.columns:
            Is = float(df['Is'].iloc[hour_index])
        else:
            Is = rad_out * transGlass

        CO2_meas = '내부 CO2'

        if '외부 CO2' in df.columns:
            Cout_ppm_i = float(df['외부 CO2'].iloc[hour_index])
        else:
            Cout_ppm_i = Cout_ppm_default

        T_in_K = _get_Tair_K(df, hour_index)
        RH_in_frac = _get_RHin_frac(df, hour_index)

        Cin_now = float(df['Cin_ppm'].iloc[hour_index]) if 'Cin_ppm' in df.columns else np.nan
        CO2_meas = float(df[CO2_meas].iloc[hour_index]) if CO2_meas in df.columns else np.nan

        Cphoto_kg = float(df['Cphoto_kg_per_h'].iloc[hour_index]) if 'Cphoto_kg_per_h' in df.columns else np.nan
        Cvent_kg = float(df['Cvent_kg_per_h'].iloc[hour_index]) if 'Cvent_kg_per_h' in df.columns else np.nan
        Cr_kg = float(df['Cr_kg_per_h'].iloc[hour_index]) if 'Cr_kg_per_h' in df.columns else np.nan

        A_leaf = float(df['A_leaf_umol_m2_s'].iloc[hour_index]) if 'A_leaf_umol_m2_s' in df.columns else np.nan
        A_can = float(df['A_canopy_umol_m2_s'].iloc[hour_index]) if 'A_canopy_umol_m2_s' in df.columns else np.nan

        ppfd_above = float(df['PPFD_above'].iloc[hour_index]) if 'PPFD_above' in df.columns else np.nan
        ppfd_canopy = float(df['PPFD_canopy'].iloc[hour_index]) if 'PPFD_canopy' in df.columns else np.nan

        if hour_index < len(df) - 1:
            ts_next = df['날짜&시간'].iloc[hour_index + 1]
            dt_h = (ts_next - ts).total_seconds() / 3600.0
            if not np.isfinite(dt_h) or dt_h <= 0:
                dt_h = 1.0
        else:
            dt_h = 1.0

        T_used = max(T_in_K, 273.15)
        n_air = patm * volumeHouse / (gas_constant * T_used)
        if n_air <= 0:
            n_air = 1.0

        coeff_kg_to_ppm_per_h = 1e6 / (n_air * m_CO2)

        Cphoto_ppm = Cphoto_kg * coeff_kg_to_ppm_per_h if np.isfinite(Cphoto_kg) else np.nan
        Cvent_ppm = Cvent_kg * coeff_kg_to_ppm_per_h if np.isfinite(Cvent_kg) else np.nan
        Ctotal_ppm = Cr_kg * coeff_kg_to_ppm_per_h if np.isfinite(Cr_kg) else np.nan

        Cin_next = np.nan
        if np.isfinite(Cin_now) and np.isfinite(Cr_kg):
            Cin_next = Cin_now + Cr_kg * coeff_kg_to_ppm_per_h * dt_h

        if 'ACH' in df.columns:
            ACH_now = float(df['ACH'].iloc[hour_index])
        else:
            ACH_now = np.nan

        chi_in = Cin_now / 1e6 if np.isfinite(Cin_now) else np.nan
        chi_out = Cout_ppm_i / 1e6

        f_int = beer_lambert_fint(LAI=LAI, k=k_sw)

        basic_data = {
            "Hour 정보": f"Hour {hour_index + 1} / {date_str}",
            "외기 온도": f"{Tout_K:.2f} K ({Tout_C:.2f}°C)" if np.isfinite(Tout_K) else "nan",
            "외부 일사": f"{rad_out:.2f} W/m²",
            "투과율": f"{transGlass:.4f}" if np.isfinite(transGlass) else "nan",
            "내부 유효일사": f"{Is:.2f} W/m²",
            "예측 내부 온도": f"{T_in_K:.2f} K ({T_in_K - 273.15:.2f}°C)",
            "예측 내부 습도": f"{RH_in_frac * 100.0:.2f}%",
            "외부 CO2": f"{Cout_ppm_i:.2f} ppm",
            "현재 내부 CO2(예측)": f"{Cin_now:.2f} ppm" if np.isfinite(Cin_now) else "nan",
            "현재 내부 CO2(실측)": f"{CO2_meas:.2f} ppm" if np.isfinite(CO2_meas) else "nan",
            "Δt": f"{dt_h:.3f} h",
            "공기 몰수 n_air": f"{n_air:.3e} mol",
            "kg→ppm/h 계수": f"{coeff_kg_to_ppm_per_h:.3e} (ppm/h)/(kg/h)"
        }

        if 'active_covers' in df.columns:
            basic_data["커튼/스크린 상태(active_covers)"] = str(
                df['active_covers'].iloc[hour_index]
            )

        if np.isfinite(A_leaf):
            F_photo_umol_s = A_can * areaHouse
            F_photo_mol_s = F_photo_umol_s * 1e-6
            Cphoto_kg_from_A = -F_photo_mol_s * 3600.0 * m_CO2
        else:
            F_photo_umol_s = np.nan
            F_photo_mol_s = np.nan
            Cphoto_kg_from_A = np.nan

        cphoto_block = {
            "공식": ("\n"
                   f"  [직각쌍곡선 + Beer-Lambert 모델]\n"
                   f"  PPFD_above = (내부일사*{frac_PAR})*{J_to_mol} µmol m⁻² s⁻¹\n"
                   f"  f_int = 1-exp(-k*LAI) = 1-exp(-{k_sw}*{LAI}) = {f_int:.4f}\n"
                   f"  PPFD_canopy = PPFD_above * f_int\n"
                   f"  A_leaf = (alpha*P*Pmax)/(alpha*P + Pmax) - Rd\n"
                   f"  A_canopy = A_leaf * LAI\n"
                   f"  F_photo_umol_s = A_canopy * areaHouse\n"
                   f"  Cphoto_kg = -(F_photo_umol_s*1e-6) * 3600 * m_CO2"
                   ),
            "계산": ("\n"
                   f"  (params) Pmax={Pmax:.2f}, alpha={alpha:.5f}, Rd={Rd:.4f}\n"
                   f"  PPFD_above = {ppfd_above:.3f} µmol m⁻² s⁻¹\n"
                   f"  PPFD_canopy = {ppfd_above:.3f} * {f_int:.4f} = {ppfd_canopy:.3f} µmol m⁻² s⁻¹\n"
                   f"  A_leaf = {A_leaf:.3f} µmol m⁻²(leaf) s⁻¹\n"
                   f"  A_canopy = {A_leaf:.3f} * {LAI:.3f} = {A_can:.3f} µmol m⁻²(ground) s⁻¹\n"
                   f"  F_photo_umol_s = {A_can:.3f} * {areaHouse:.1f} = {F_photo_umol_s:.3f} µmol/s\n"
                   f"  F_photo_mol_s  = {F_photo_umol_s:.3f} * 1e-6 = {F_photo_mol_s:.6e} mol/s\n"
                   f"  Cphoto_kg(from A) = -{(F_photo_mol_s):.6e} * 3600 * {m_CO2:.5f} "
                   f"= {Cphoto_kg_from_A:.6f} kg/h\n"
                   f"  Cphoto_kg_per_h = {Cphoto_kg:.6f} kg/h\n"
                   f"  Cphoto_ppm = Cphoto_kg * coeff = {Cphoto_ppm:.6f} ppm/h"
                   ),
            "결과": "\n"
                  f"  {Cphoto_kg:.6f} kg/h  ({Cphoto_ppm:.6f} ppm/h)"
        }

        cvent_block = {
            "공식": ("\n"
                   "  χ_in = Cin_now / 10⁶,  χ_out = C_out / 10⁶\n"
                   "  dnCO₂/dt_vent = ACH * n_air * (χ_out - χ_in) mol/h\n"
                   "  Cvent_kg = dnCO₂/dt_vent * m_CO2 kg/h\n"
                   ),
            "계산": ("\n"
                   f"  ACH = {ACH_now:.3f} h⁻¹,  n_air = {n_air:.3e} mol\n"
                   f"  Cin_now = {Cin_now:.3f} ppm → χ_in = {chi_in:.6e}\n"
                   f"  C_out   = {Cout_ppm_i:.1f} ppm → χ_out = {chi_out:.6e}\n"
                   f"  dnCO₂/dt_vent_mol_h = {ACH_now:.3f} * {n_air:.3e} * "
                   f"({chi_out:.6e} - {chi_in:.6e})\n"
                   f"  Cvent_kg = dnCO₂/dt_vent_mol_h * {m_CO2:.5f} ≈ {Cvent_kg:.6f} kg/h\n"
                   f"  Cvent_ppm = Cvent_kg * coeff = {Cvent_ppm:.6f} ppm/h"
                   ),
            "결과": ("\n"
                   f"  {Cvent_kg:.6f} kg/h  ({Cvent_ppm:.6f} ppm/h)"
                   )
        }

        cin_update_block = {
            "공식": "Cin_next = Cin_now + Cr_kg_per_h * coeff_kg_to_ppm_per_h * Δt",
            "계산": (
                f"{Cin_now:.3f} ppm + ({Cr_kg:.6f} kg/h * "
                f"  {coeff_kg_to_ppm_per_h} (ppm/h)/(kg/h)) * {dt_h:.3f} h"
                f"  = {Cin_next:.3f} ppm (Ctotal_ppm/h = {Ctotal_ppm:.6f})"
                if np.isfinite(Cin_now) and np.isfinite(Cr_kg) else "nan"
            ),
            "결과": f"{Cin_next:.3f} ppm" if np.isfinite(Cin_next) else "nan"
        }

        result = {
            "시간정보": f"Hour {hour_index + 1} / {date_str}",
            "기본 데이터": basic_data,
            "CO2 항목": {
                "Cphoto": cphoto_block,
                "Cvent": cvent_block
            },
            "Cin 업데이트": cin_update_block
        }

        return result

    except Exception as e:
        print(f"show_detailed_CO2_calculation 오류: {e}")
        return {}

def main():
    try:
        (
            df, qt, Toutdoor, radSolar, Troom,
            params, rhoAir, cAir, mHouse,
            Ur_values, Uw_values, transGlass_values,
            active_covers, ach_array
        ) = process_greenhouse_data()

        try:
            df = run_photosynthesis_and_co2(df.copy())
        except Exception as e:
            print(f"[CO2] CO2 계산 실패: {e}")

        if '내부 CO2' not in df.columns and 'CO2_in' in df.columns:
            df['내부 CO2'] = df['CO2_in']

        print("\n[사용 방법]")
        print(" - 특정 시점: 예) 100 또는 23051200")
        print(" - 특정 기간: 예) 1~24 또는 23051200~23051323\n")

        user_input = input("시점(또는 기간) 입력: ").strip()
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
            x_m = x[mask]
            y_m = y[mask]

            if len(y_m) >= 2:
                slope, intercept = np.polyfit(x_m, y_m, 1)
                y_hat = slope * x_m + intercept
                r2 = r2_score(y_m, y_hat)
                rmse = np.sqrt(mean_squared_error(y_m, y_hat))

                print(f"\n[검증] RH(in) 예측 vs 실측 points={len(y_m)}")
                print(f" slope={slope:.3f}, intercept={intercept:.3f}")
                print(f" R²={r2:.3f}, RMSE={rmse:.3f}")
            else:
                print("\n[검증] 유효한 데이터 포인트 부족.")

            start_dt = df.loc[start_index, '날짜&시간']
            end_dt = df.loc[end_index, '날짜&시간']

            print("\n[입력 기간 인덱스 → 날짜/시간]")
            print(f" - 시작 {start_index + 1} → {start_dt:%Y-%m-%d %H시}")
            print(f" - 종료 {end_index + 1} → {end_dt:%Y-%m-%d %H시}")

            display_range_graphs(start_index, end_index, df, Troom, Toutdoor, qt, save_graph=save_flag)

            if 'Cin_ppm' in sub.columns:
                meas_col = None
                if CO2_meas in sub.columns:
                    meas_col = CO2_meas
                elif '내부 CO2' in sub.columns:
                    meas_col = '내부 CO2'
                elif 'CO2_in' in sub.columns:
                    meas_col = 'CO2_in'

                df_sel = sub.copy()

                # 1. Cphoto 그래프
                if 'Cphoto_ppm_per_h' in df_sel.columns:
                    plt.figure()
                    plt.plot(
                        df_sel['날짜&시간'],
                        df_sel['Cphoto_ppm_per_h'],
                        color='orange', linestyle='-', label='Cphoto (ppm/h)'
                    )
                    plt.ylabel("Cphoto (ppm/h)")
                    plt.xlabel("Time")
                    plt.title("Cphoto timeseries")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    if save_flag:
                        fn = f"cphoto_timeseries_{start_index + 1}_{end_index + 1}.png"
                        plt.savefig(fn)
                        print(f"Cphoto 시계열 그래프 -> {fn} 저장.")
                    plt.show()

                # 2. CO2 농도 비교 그래프
                if meas_col is not None:
                    plt.figure()
                    plt.plot(
                        df_sel['날짜&시간'],
                        df_sel['Cin_ppm'],
                        color='red', linestyle='-', label='Predicted inside CO2 (ppm)'
                    )
                    plt.plot(
                        df_sel['날짜&시간'],
                        df_sel[meas_col],
                        color='blue', linestyle='-', label='Measured inside CO2 (ppm)'
                    )
                    plt.ylabel("CO2 Concentration (ppm)")
                    plt.xlabel("Time")
                    plt.title("Predicted & Measured Inside CO2")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    if save_flag:
                        fn = f"co2_timeseries_{start_index + 1}_{end_index + 1}.png"
                        plt.savefig(fn)
                        print(f"CO2 시계열 그래프 -> {fn} 저장.")
                    plt.show()

                # 3. CO2 농도 회귀 그래프
                if meas_col is not None:
                    plot_co2_regression(
                        df_sel[meas_col].to_numpy(),
                        df_sel['Cin_ppm'].to_numpy(),
                        title="Measured vs Predicted inside CO2 Regression",
                        save_graph=save_flag,
                        filename=f"co2_regression_{start_index + 1}_{end_index + 1}.png" if save_flag else None
                    )

            return

        idx_single = parse_time_str(user_input, df, allow_nearest=False)
        if idx_single is None:
            return

        dt_obj = df.loc[idx_single, '날짜&시간']
        selected_date = dt_obj.date()
        daily_mask = df['날짜&시간'].dt.date == selected_date
        if not daily_mask.any():
            print(f"{selected_date} 날짜 데이터 없음.")
            return

        start_hour_index = df.index[daily_mask][0]
        display_daily_graphs(selected_date, start_hour_index, df, Troom, Toutdoor, qt, save_graph=save_flag)

        daily_df = df.loc[daily_mask].copy()
        export_daily_df = prepare_export_dataframe(daily_df)

        cols = [
            'hour', '날짜&시간', 'qt',
            'Wet', 'Wvt', 'Wr', 'gsw_used',
            'Troom', 'Ur', 'Uw', 'transGlass',
            'active_covers',
            'Temp_K', 'RHin', 'Humi', 'RHout',
            'qLatent_kW',
            'qFIR_kW', 'Tp_lw_K', 'Tsky_lw_K', 'eps_lw',
            'u_int', 'ACH', 'ra', 'rc', 'rs', 'hs',
            'Is', 'f_int', 'Rns', 'Rn',
            # CO2
            'Cin_ppm', 'A_leaf_umol_m2_s', 'A_canopy_umol_m2_s',
            'Cphoto_ppm_per_h', 'Cvent_ppm_per_h', 'Ctotal_ppm_per_h',
            'Cphoto_kg_per_h', 'Cvent_kg_per_h', 'Cr_kg_per_h'
        ]
        cols_exist = [c for c in cols if c in daily_df.columns]

        print("\n[해당 일자 주요 데이터 (상위 40행)]")
        print(daily_df[cols_exist].head(40))

        if save_flag:
            export_daily_df.to_excel("daily_data.xlsx", index=False)
            print("일간 데이터프레임 daily_data.xlsx 저장.")

        detail_energy = show_detailed_energy_calculation(
            idx_single, df, params, Toutdoor, radSolar, Troom,
            qt, rhoAir, cAir, mHouse,
            Ur_values, Uw_values, transGlass_values,
            active_covers, ach_array
        )

        if detail_energy:
            print("\n" + "=" * 60)
            print("=== [에너지(온도) 상세 계산 결과] ===")
            print("=" * 60)
            print(detail_energy["시간정보"])

            print("\n[기본 데이터]")
            for k, v in detail_energy["기본 데이터"].items():
                print(f" - {k}: {v}")

            print("\n[Tsolair2 (Sol-air 온도)]")
            ts = detail_energy["Tsolair2"]
            print(f" [공식] {ts['공식']}")
            print(f" [계산] {ts['계산']}")
            print(f" [결과] {ts['결과']}")

            print("\n[열 에너지 항목]")
            for term_name, term_data in detail_energy["열 에너지 항목"].items():
                print(f"\n  [{term_name}]")
                print(f"  공식: {term_data['공식']}")
                print(f"  계산: {term_data['계산']}")
                print(f"  결과: {term_data['결과']}")

            print("\n[Qt (총 열 부하)]")
            qt_d = detail_energy["Qt"]
            print(f" [공식] {qt_d['공식']}")
            print(f" [계산] {qt_d['계산']}")
            print(f" [결과] {qt_d['결과']}")

            print("\n[다음 시간 실내 온도]")
            troom_d = detail_energy["Troom[next]"]
            for k, v in troom_d.items():
                print(f" [{k}] {v}")

        detail_moisture = show_detailed_moisture_calculation(
            idx_single, df, params, Troom, mHouse, ach_array
        )

        if detail_moisture:
            print("\n" + "=" * 60)
            print("=== [수분(습도) 상세 계산 결과] ===")
            print("=" * 60)
            print(detail_moisture["시간정보"])

            print("\n[기본 데이터]")
            for k, v in detail_moisture["기본 데이터"].items():
                print(f" - {k}: {v}")

            print("\n[혼합비]")
            for k, v in detail_moisture["혼합비"].items():
                print(f"\n  [{k}]")
                print(f"  공식: {v['공식']}")
                print(f"  계산: {v['계산']}")
                print(f"  결과: {v['결과']}")

            print("\n[VPD (증기압 차이)]")
            vpd = detail_moisture["VPD"]
            print(f" [공식] {vpd['공식']}")
            print(f" [계산] {vpd['계산']}")
            print(f" [결과] {vpd['결과']}")

            print("\n[Stanghellini Model Parameter]")
            for k, v in detail_moisture["Stanghellini Model Parameter"].items():
                print(f" - {k}: {v}")

            print("\n[Wet (증발산량)]")
            wet = detail_moisture["Wet"]
            print(f" [공식] {wet['공식']}")
            print(f" [계산] {wet['계산']}")
            print(f" [결과] {wet['결과']}")

            print("\n[Wvt (환기 수분 교환)]")
            wvt = detail_moisture["Wvt"]
            print(f" [공식] {wvt['공식']}")
            print(f" [계산] {wvt['계산']}")
            print(f" [결과] {wvt['결과']}")

            print("\n[Wr (실내 수분 변화율)]")
            wr = detail_moisture["Wr"]
            print(f" [공식] {wr['공식']}")
            print(f" [계산] {wr['계산']}")
            print(f" [결과] {wr['결과']}")

        if 'Cin_ppm' in df.columns:
            meas_col = None
            if CO2_meas in df.columns:
                meas_col = CO2_meas
            elif '내부 CO2' in df.columns:
                meas_col = '내부 CO2'
            elif 'CO2_in' in df.columns:
                meas_col = 'CO2_in'

            df_sel = daily_df.copy()

            # 1. Cphoto 그래프
            if 'Cphoto_ppm_per_h' in df_sel.columns:
                plt.figure()
                plt.plot(
                    df_sel['날짜&시간'],
                    df_sel['Cphoto_ppm_per_h'],
                    color='orange', linestyle='-', label='Cphoto (ppm/h)'
                )
                plt.ylabel("Cphoto (ppm/h)")
                plt.xlabel("Time")
                plt.title("Cphoto timeseries")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                if save_flag:
                    fn = f"cphoto_timeseries_{selected_date.strftime('%Y%m%d')}.png"
                    plt.savefig(fn)
                    print(f"Cphoto 시계열 그래프 -> {fn} 저장.")
                plt.show()

            # 2. CO2 농도 비교 그래프
            if meas_col is not None:
                plt.figure()
                plt.plot(
                    df_sel['날짜&시간'],
                    df_sel['Cin_ppm'],
                    color='red', linestyle='-', label='Predicted inside CO2 (ppm)'
                )
                plt.plot(
                    df_sel['날짜&시간'],
                    df_sel[meas_col],
                    color='blue', linestyle='-', label='Measured inside CO2 (ppm)'
                )
                plt.ylabel("CO2 Concentration (ppm)")
                plt.xlabel("Time")
                plt.title("Predicted & Measured Inside CO2")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                if save_flag:
                    fn = f"co2_timeseries_{selected_date.strftime('%Y%m%d')}.png"
                    plt.savefig(fn)
                    print(f"CO2 시계열 그래프 -> {fn} 저장.")
                plt.show()

            # 3. CO2 농도 회귀 그래프
            if meas_col is not None:
                plot_co2_regression(
                    df_sel[meas_col].to_numpy(),
                    df_sel['Cin_ppm'].to_numpy(),
                    title="Measured vs Predicted inside CO2 Regression",
                    save_graph=save_flag,
                    filename=f"co2_regression_{selected_date.strftime('%Y%m%d')}.png" if save_flag else None
                )

            detail_co2 = show_detailed_CO2_calculation(idx_single, df)
            if detail_co2:
                print("\n" + "=" * 60)
                print("=== [가스(CO2 농도) 상세 계산 결과] ===")
                print("=" * 60)
                print(detail_co2["시간정보"])

                print("\n[기본 데이터]")
                for k, v in detail_co2["기본 데이터"].items():
                    print(f" - {k}: {v}")

                print("\n[Cphoto (작물 광합성·호흡) - 직각쌍곡선 + Beer-Lambert]")
                cp = detail_co2["CO2 항목"]["Cphoto"]
                print(f" [공식] {cp['공식']}")
                print(f" [계산] {cp['계산']}")
                print(f" [결과] {cp['결과']}")

                print("\n[Cvent (환기 기체 교환)]")
                cv = detail_co2["CO2 항목"]["Cvent"]
                print(f" [공식] {cv['공식']}")
                print(f" [계산] {cv['계산']}")
                print(f" [결과] {cv['결과']}")

                print("\n[다음 시점 내부 CO2(Cin) 업데이트]")
                cu = detail_co2["Cin 업데이트"]
                print(f" [공식] {cu['공식']}")
                print(f" [계산] {cu['계산']}")
                print(f" [결과] {cu['결과']}")

    except FileNotFoundError:
        print("파일이 없습니다. 경로/파일명 확인하세요.")
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
