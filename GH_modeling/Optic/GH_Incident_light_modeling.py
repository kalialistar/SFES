import sys
import pandas as pd
import numpy as np
from pathlib import Path
import re
import math

file = "data/jeonju_climate_data.xlsx"  # 2024년 1월 1일부터 12월 31일까지의 전주시의 기상 데이터(기상청 기상자료포털)
GHI_column  = "일사"
IRRAD_UNIT = "auto"
Spectrum_data = "data/AM_1.5.xlsx" # NREL에서 제공하는 Air Mass 1.5 Spectra 파일(Global Tilt) / 파장별 GHI 값을 제공함

# 온실 규격(스마트 딸기 온실) -> 지붕면 = 수평 가정
lengthHouse = 97
widthHouse  = 7.6
heightHouse = 2.7

IAM_b0   = 0.17 # 복층 피복재(단층 피복재 = 0.10)

Tau, Rho, Alpha = 0.71, 0.08, 0.21   # Tau + Rho + Alpha = 1, old plastic film
# 반사율(Rho)는 사용하지 않음

# 온실 및 작물 유효 파장대 영역
Wavelength_min = 280 # [nm]
Wavelength_max = 2500 # [nm]

Output = Path("result_file")
decimals = 6
decimals_sum = 3

Lat = 35.82 # 전주시 위도
Lon_kst = 135 # 대한민국 표준시 경도
Lon = 127.15 # 전주시 경도

Screen_on = True
shading_rate = 0.55                   # 차광률(55%)
trns_Beam = 1 - shading_rate          # 직달광 투과율
trns_Diffuse = 1 - shading_rate       # 산란광 투과율

shading_threshold = 400

beta_roof = 0.0  # 지붕 = 수평 (β=0)
AOI_model = "ASHRAE"

def parse_yymmddhh(s: str) -> pd.Timestamp:
    s = str(s).strip()
    if not re.fullmatch(r"\d{8}", s):
        raise ValueError("입력은 yymmddhh로 해야 함.")
    yy = int(s[0:2]); mm = int(s[2:4]); dd = int(s[4:6]); hh = int(s[6:8])
    year = 2000 + yy
    return pd.Timestamp(year=year, month=mm, day=dd, hour=hh)

def _normalize_col_name(s):
    s = str(s)
    s = s.replace("×", "*").replace("·", "*").replace("−", "-")
    s = re.sub(r"\s+", "", s)
    return s.lower()

def guess_datetime_col(df: pd.DataFrame):
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    raise ValueError("datetime 열을 찾지 못함.")

def convert_irradiance_unit(x: pd.Series, mode="auto"):
    if mode == "Wm2":
        return pd.to_numeric(x, errors="coerce")
    if mode == "MJm2h":
        return pd.to_numeric(x, errors="coerce") * 1e6 / 3600
    v = pd.to_numeric(x, errors="coerce")
    vmax = np.nanmax(v.to_numpy())
    if vmax < 50:
        return v * 1e6 / 3600
    return v

def trapz_on_grid(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))

def _select_global_tilt_numeric(df: pd.DataFrame):
    if df.shape[1] < 2:
        raise ValueError("스펙트럼 열 부족")
    df_num = df.apply(pd.to_numeric, errors="coerce").dropna(how="all", axis=1)

    wl = df_num.iloc[:, 0].to_numpy()
    wl = wl[np.isfinite(wl)]
    if len(wl) < 10:
        raise ValueError("첫번째 열 데이터 부족.")
    vmin, vmax = float(np.nanmin(wl)), float(np.nanmax(wl))
    if not (150 <= vmin < vmax <= 5000):
        raise ValueError(f"첫번째 열 데이터 해석 불가 (min={vmin}, max={vmax})")

    df_num = df_num.loc[np.isfinite(df_num.iloc[:, 0])]
    df_num = df_num.sort_values(df_num.columns[0])
    wl = df_num.iloc[:, 0].to_numpy()

    candidates = []
    for j in range(1, df_num.shape[1]):
        Ecol = df_num.iloc[:, j].to_numpy()
        if np.nanmax(Ecol) <= 0:
            continue
        G = trapz_on_grid(wl, Ecol)
        if np.isfinite(G) and G > 0:
            candidates.append((j, G))
    if not candidates:
        raise ValueError("Global Tilt 열 값을 찾지 못함.")

    target = 1000
    in_window = [(j, G) for (j, G) in candidates if 800 <= G <= 1200]
    if in_window:
        j_sel, _ = min(in_window, key=lambda x: abs(x[1]-target))
    else:
        j_sel, _ = max(candidates, key=lambda x: x[1])

    wl_col = df_num.columns[0]
    gt_col = df_num.columns[j_sel]
    out = df_num[[wl_col, gt_col]].copy()
    out.columns = ["wavelength_nm", "E_Wm2nm"]
    out = out.dropna()
    return out

def _coerce_spectrum_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    header_is_numeric_like = True
    for c in df_raw.columns:
        try:
            float(str(c).strip().replace(",", ""))
        except Exception:
            header_is_numeric_like = False
            break
    if header_is_numeric_like:
        return _select_global_tilt_numeric(df_raw)

    df = df_raw.copy()
    cols_norm = {c: _normalize_col_name(c) for c in df.columns}

    wl_col, gt_col = None, None
    for c, key in cols_norm.items():
        if ("wavelength" in key) or ("wvlgth" in key) or (key.startswith("nm")) or (key == "lambda") or ("wavel" in key):
            wl_col = c; break
    if wl_col is None:
        for c in df.columns:
            series = pd.to_numeric(df[c], errors="coerce")
            if series.notna().sum() > 10:
                vmin, vmax = float(series.min()), float(series.max())
                if 150 <= vmin < vmax <= 5000:
                    wl_col = c; break

    for c, key in cols_norm.items():
        if ("global" in key) and ("tilt" in key):
            gt_col = c; break
    if gt_col is None:
        for c, key in cols_norm.items():
            if ("global" in key) and ("nm" in key):
                gt_col = c; break

    if wl_col is not None and gt_col is not None:
        out = df[[wl_col, gt_col]].copy()
        out.columns = ["wavelength_nm", "E_Wm2nm"]
        out = out.apply(pd.to_numeric, errors="coerce").dropna()
        return out.sort_values("wavelength_nm").reset_index(drop=True)

    return _select_global_tilt_numeric(df_raw)

def _read_am15_two_line_header_xlsx(path: Path) -> pd.DataFrame:
    try:
        df_try = pd.read_excel(path, header=1)
        df_try = df_try.dropna(how="all")
        return _coerce_spectrum_frame(df_try)
    except Exception:
        pass
    df_raw = pd.read_excel(path, header=None)
    if df_raw.shape[0] < 3:
        raise RuntimeError("데이터 행이 3줄 미만")
    header = df_raw.iloc[1].astype(str).tolist()
    df = df_raw.iloc[2:].copy()
    df.columns = header
    df = df.dropna(how="all")
    return _coerce_spectrum_frame(df)

def load_am15_bn_and_gt() -> pd.DataFrame:
    p = Path(Spectrum_data)
    if not p.exists():
        raise FileNotFoundError(f"스펙트럼 파일을 찾을 수 없음: {Spectrum_data}")
    try:
        df = pd.read_excel(p, header=1).dropna(how="all")
    except Exception:
        df = pd.read_excel(p, header=None)

    cols = {c: _normalize_col_name(c) for c in df.columns}
    wl_col = None
    for c,k in cols.items():
        if ("wavelength" in k) or ("wvlgth" in k) or (k.startswith("nm")) or (k=="lambda") or ("wavel" in k):
            wl_col = c; break
    if wl_col is None:
        for c in df.columns:
            series = pd.to_numeric(df[c], errors="coerce")
            if series.notna().sum() > 10:
                vmin, vmax = float(series.min()), float(series.max())
                if 150 <= vmin < vmax <= 5000:
                    wl_col = c; break
    if wl_col is None:
        raise RuntimeError("AM1.5 파일에서 파장 열을 찾지 못했습니다.")

    def is_bn(name):
        k = _normalize_col_name(name)
        return ("direct" in k and ("normal" in k or "circumsolar" in k)) or ("dni" in k) or ("beam" in k)
    def is_gt(name):
        k = _normalize_col_name(name)
        return (("global" in k) and ("tilt" in k)) or ("poa" in k and "global" in k)

    bn_cols = [c for c in df.columns if c != wl_col and is_bn(c)]
    gt_cols = [c for c in df.columns if c != wl_col and is_gt(c)]
    if not bn_cols and not gt_cols:
        raise RuntimeError("AM1.5에서 직달/글로벌 틸트 열을 찾지 못했습니다.")

    df_num = df[[wl_col]+bn_cols+gt_cols].copy()
    df_num = df_num.apply(pd.to_numeric, errors="coerce").dropna(subset=[wl_col])
    df_num = df_num.sort_values(wl_col)
    df_num = df_num.rename(columns={wl_col: "wavelength_nm"})
    m = (df_num["wavelength_nm"] >= Wavelength_min) & (df_num["wavelength_nm"] <= Wavelength_max)
    df_num = df_num.loc[m].reset_index(drop=True)

    def pick(cols):
        if not cols: return None
        best, bestG = None, -1
        wl = df_num["wavelength_nm"].to_numpy()
        for c in cols:
            v = df_num[c].to_numpy()
            G = trapz_on_grid(wl, v)
            if np.isfinite(G) and G > bestG:
                best, bestG = c, G
        return best

    bn_sel = pick(bn_cols)
    gt_sel = pick(gt_cols)
    out = pd.DataFrame({"wavelength_nm": df_num["wavelength_nm"].to_numpy()})
    out["E_bn_Wm2nm"] = df_num[bn_sel].to_numpy() if bn_sel else np.nan
    out["E_gt_Wm2nm"] = df_num[gt_sel].to_numpy() if gt_sel else np.nan
    return out

def solar_altitude_deg(Lat: float, n: int, lt_hour: float, Lon_kst: float, Lon: float) -> float:
    # Duffie & Beckman / 태양 고도 및 천정각 계산
    delta = 23.45 * math.sin(math.radians(360/365 * (284 + n)))
    B = math.radians(360*(n-81)/364)
    EoT = 9.87*math.sin(2*B) - 7.53*math.cos(B) - 1.5*math.sin(B)
    TC  = EoT + 4*(Lon_kst - Lon)
    LST = lt_hour + TC/60
    omega = 15*(LST - 12)
    sin_alpha = (math.sin(math.radians(Lat))*math.sin(math.radians(delta)) +
                 math.cos(math.radians(Lat))*math.cos(math.radians(delta))*math.cos(math.radians(omega)))
    sin_alpha = max(-1, min(1, sin_alpha))
    alpha = math.degrees(math.asin(sin_alpha))
    return alpha

def iam_ashrae(theta_deg: float, b0: float = IAM_b0) -> float:
    # IAM 직달 보정 수식
    if theta_deg >= 90:
        return 0.0
    c = math.cos(math.radians(theta_deg))
    if c <= 0:
        return 0.0
    return max(0.0, 1.0 - b0*(1.0/c - 1.0))

def sam_effective_angles(beta_deg: float):
    theta_d = 59.7 - 0.1388*beta_deg + 0.001497*(beta_deg**2)     # sky diffuse
    # theta_g = 90.0 - 0.5788*beta_deg + 0.002693*(beta_deg**2)     # ground-reflected diffuse / 지붕면 경사가 없기 때문에 사용하지 않음
    return theta_d#, theta_g

def reindl2_decompose(GHI: float, cosZ: float, n: int) -> dict:
    """
    Reindl-2 diffuse fraction correlation
    kd = DHI/GHI
    """
    G_sc = 1367
    G_on = G_sc * (1 + 0.033 * math.cos(math.radians(360*n/365)))
    if cosZ <= 0 or G_on * cosZ <= 0 or GHI <= 0:
        return {"DHI": 0, "DNI": 0, "Ib_h": 0, "Kt": 0}

    Kt = max(0, min(1.2, GHI / (G_on * cosZ)))
    sin_alpha = max(0.0, min(1.0, cosZ))

    if Kt <= 0.3:
        Fd = 1.020 - 0.254*Kt + 0.0123*sin_alpha
        Fd = min(Fd, 1.0)
    elif Kt < 0.78:
        Fd = 1.400 - 1.749*Kt + 0.177*sin_alpha
        Fd = min(max(Fd, 0.1), 0.97)
    else:
        Fd = 0.486*Kt - 0.182*sin_alpha
        Fd = max(Fd, 0.1)

    Fd = max(0.0, min(1.0, Fd))

    DHI = max(0.0, min(GHI, Fd * GHI))
    Ib_h = max(0.0, GHI - DHI)
    DNI = Ib_h / cosZ if cosZ > 0 else 0.0

    return {"DHI": DHI, "DNI": DNI, "Ib_h": Ib_h, "Kt": Kt}

def build_diffuse_shape_from_gt_minus_dn(wl, E_gt, E_bn):
    E_tilt_diff = np.maximum(E_gt - E_bn, 0)
    G_tilt_diff = trapz_on_grid(wl, E_tilt_diff)
    if not np.isfinite(G_tilt_diff) or G_tilt_diff <= 0:
        S = np.maximum(E_bn, 0)
        G = trapz_on_grid(wl, S)
        return S / max(G, 1e-12)
    return E_tilt_diff / G_tilt_diff

def compute_case(chosen_dt: pd.Timestamp, orientation: str):
    src = pd.read_excel(file)
    dt_col = guess_datetime_col(src)
    src[dt_col] = pd.to_datetime(src[dt_col])
    src = src.set_index(dt_col).sort_index()
    if GHI_column not in src.columns:
        raise KeyError(f"'{GHI_column}' 파일 열 재확인 필요")
    src["GHI_Wm2"] = convert_irradiance_unit(src[GHI_column], IRRAD_UNIT)
    if chosen_dt not in src.index:
        idx_ns = src.index.view("i8")
        target_ns = np.int64(pd.Timestamp(chosen_dt).value)
        pos = int(np.argmin(np.abs(idx_ns - target_ns)))
        chosen_dt = src.index[pos]
    GHI = float(src.loc[chosen_dt, "GHI_Wm2"])

    # 태양 고도/천정각 (수평 지붕에서는 방위 불필요)
    n = int(chosen_dt.timetuple().tm_yday)
    lt_hour = chosen_dt.hour + chosen_dt.minute/60
    alpha = solar_altitude_deg(Lat, n, lt_hour, Lon_kst, Lon)
    theta_z = max(0, 90 - alpha)
    cosZ = max(0, math.cos(math.radians(theta_z)))

    # Reindl-2: GHI → DHI, DNI
    parts = reindl2_decompose(GHI, cosZ, n)
    DHI, DNI, Ib_h, Kt = parts["DHI"], parts["DNI"], parts["Ib_h"], parts["Kt"]

    # AM1.5 스펙트럼 형상
    spec = load_am15_bn_and_gt()
    spec = spec[(spec["wavelength_nm"] >= Wavelength_min) & (spec["wavelength_nm"] <= Wavelength_max)].copy()
    if spec.shape[0] < 10:
        raise RuntimeError("스펙트럼 구간(280–2500 nm)의 유효 데이터 부족")
    wl = spec["wavelength_nm"].to_numpy()
    E_bn = spec["E_bn_Wm2nm"].to_numpy()
    E_gt = spec["E_gt_Wm2nm"].to_numpy()

    G_bn_ref = trapz_on_grid(wl, E_bn)
    if not np.isfinite(G_bn_ref) or G_bn_ref <= 0:
        raise RuntimeError("AM1.5 직달 분광 적분이 비정상입니다.")
    S_bn = E_bn / G_bn_ref
    S_d  = build_diffuse_shape_from_gt_minus_dn(wl, E_gt, E_bn)

    """
      수평 지붕(β=0):
    - 직달(수평) = Ib_h
    - 산란(수평) = DHI
    - 지면반사(경사면식) = 0(고려하지 않음)
    """

    w = np.array([1.0])

    E_bn_POA_seg = np.array([Ib_h])[:, None] * S_bn[None, :]
    E_d_sky_seg  = np.array([DHI])[:, None] * S_d[None, :]
    E_d_grd_seg  = np.zeros_like(E_d_sky_seg)

    Gb = float(Ib_h)
    Gd = float(DHI)

    beta_deg = beta_roof
    theta_d_deg = sam_effective_angles(beta_deg)

    # ASHRAE IAM을 K로 사용 (직달=θz, 산란=θd)
    Kb = iam_ashrae(theta_z, IAM_b0)
    Kd = iam_ashrae(theta_d_deg, IAM_b0)

    tau_b = Tau * Kb
    tau_d = Tau * Kd

    rho_b = max(0.0, 1.0 - tau_b - Alpha)
    rho_d = max(0.0, 1.0 - tau_d - Alpha)

    # 외부 입사광의 합
    E_ext_seg = E_bn_POA_seg + E_d_sky_seg + E_d_grd_seg

    # 투과/반사/흡수(방사) 스펙트럼
    E_trn_seg = (tau_b * E_bn_POA_seg) + (tau_d * E_d_sky_seg)
    E_ref_seg = (rho_b * E_bn_POA_seg) + (rho_d * E_d_sky_seg)
    E_abs_seg = (Alpha * E_bn_POA_seg) + (Alpha * E_d_sky_seg)

    E_trn_beam_seg = (tau_b * E_bn_POA_seg)
    E_trn_diff_seg = (tau_d * E_d_sky_seg)

    E_ref_beam_seg = (rho_b * E_bn_POA_seg)
    E_ref_diff_seg = (rho_d * E_d_sky_seg)

    E_abs_beam_seg = (Alpha * E_bn_POA_seg)
    E_abs_diff_seg = (Alpha * E_d_sky_seg)

    def area_avg(E_seg_lambda):
        return (w[:, None] * E_seg_lambda).sum(axis=0)

    E_ext = area_avg(E_ext_seg)
    E_trn = area_avg(E_trn_seg)
    E_ref = area_avg(E_ref_seg)
    E_abs = area_avg(E_abs_seg)

    E_trn_beam = area_avg(E_trn_beam_seg)
    E_trn_diff = area_avg(E_trn_diff_seg)

    E_ref_beam = area_avg(E_ref_beam_seg)
    E_ref_diff = area_avg(E_ref_diff_seg)

    E_abs_beam = area_avg(E_abs_beam_seg)
    E_abs_diff = area_avg(E_abs_diff_seg)

    G_ext = trapz_on_grid(wl, E_ext)
    G_trn = trapz_on_grid(wl, E_trn)
    G_ref = trapz_on_grid(wl, E_ref)
    G_abs = trapz_on_grid(wl, E_abs)

    G_trn_beam = trapz_on_grid(wl, E_trn_beam)
    G_trn_diff = trapz_on_grid(wl, E_trn_diff)

    G_ref_beam = trapz_on_grid(wl, E_ref_beam)
    G_ref_diff = trapz_on_grid(wl, E_ref_diff)

    G_abs_beam = trapz_on_grid(wl, E_abs_beam)
    G_abs_diff = trapz_on_grid(wl, E_abs_diff)

    # 차광 스크린
    Screen_on_flag = bool(Screen_on) and (G_trn >= shading_threshold)

    if Screen_on_flag:
        E_scr_beam_seg = trns_Beam * E_trn_beam_seg
        E_scr_diff_seg = trns_Diffuse * E_trn_diff_seg
    else:
        E_scr_beam_seg = E_trn_beam_seg
        E_scr_diff_seg = E_trn_diff_seg

    E_crop_seg = E_scr_beam_seg + E_scr_diff_seg

    E_scr_beam = area_avg(E_scr_beam_seg)
    E_scr_diff = area_avg(E_scr_diff_seg)
    E_crop = area_avg(E_crop_seg)

    G_scr_beam = trapz_on_grid(wl, E_scr_beam)
    G_scr_diff = trapz_on_grid(wl, E_scr_diff)
    G_crop = trapz_on_grid(wl, E_crop)

    meta = {
        "chosen_dt": chosen_dt,
        "surface": "flat",
        "orientation": orientation,
        "GHI": GHI, "DHI": DHI, "DNI": DNI, "Ib_h": Ib_h, "Kt": Kt,
        "alpha_deg": alpha, "theta_z_deg": theta_z,
        "Tau": Tau, "Alpha": Alpha,

        "beta_deg": beta_deg,
        "theta_d_deg": float(theta_d_deg),
        "K_b": float(Kb),
        "K_d": float(Kd),
        "Gb": Gb, "Gd": Gd,
        "GHI_coeff": float(Gb*Kb + Gd*Kd),

        "tau_b": float(tau_b),
        "tau_d": float(tau_d),

        "rho_b": float(rho_b),
        "rho_d": float(rho_d),

        "Screen_on": bool(Screen_on_flag),
        "shading_rate": float(shading_rate),
        "trns_Beam": float(trns_Beam),
        "trns_Diffuse": float(trns_Diffuse),
        "screen_threshold": float(shading_threshold),
        "G_trn_internal": float(G_trn),
        "G_after_screen": float(G_crop),

        "G_trn_beam": float(G_trn_beam),
        "G_trn_diff": float(G_trn_diff),
        "G_ref_beam": float(G_ref_beam),
        "G_ref_diff": float(G_ref_diff),
        "G_abs_beam": float(G_abs_beam),
        "G_abs_diff": float(G_abs_diff),
        "G_scr_beam": float(G_scr_beam),
        "G_scr_diff": float(G_scr_diff),

        "AOI_model": AOI_model,
    }

    return (
        wl, E_ext, E_trn, E_ref, E_abs,
        E_trn_beam, E_trn_diff,
        E_ref_beam, E_ref_diff,
        E_abs_beam, E_abs_diff,
        E_scr_beam, E_scr_diff,
        E_crop,
        G_ext, G_trn, G_ref, G_abs,
        G_trn_beam, G_trn_diff,
        G_ref_beam, G_ref_diff,
        G_abs_beam, G_abs_diff,
        G_scr_beam, G_scr_diff,
        G_crop,
        meta
    )

def save_excel_combined(wl, spectra_by_case, sums_by_case, chosen_dt: pd.Timestamp) -> Path:
    Output.mkdir(parents=True, exist_ok=True)
    base = chosen_dt.strftime("%y%mdd%H") if False else chosen_dt.strftime("%y%m%d%H")
    xlsx_path = Output / f"{base}_Optical_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        cols = {"파장[nm]": np.round(wl, 6)}
        for key, spec in spectra_by_case.items():
            tag = f"{key[0]}-{key[1]}"

            cols[f"{tag} 외부[W/m²/nm]"] = np.round(spec["ext"], decimals)

            cols[f"{tag} 투과-전체[W/m²/nm]"] = np.round(spec["trn"], decimals)
            cols[f"{tag} 투과-직달[W/m²/nm]"] = np.round(spec["trn_b"], decimals)
            cols[f"{tag} 투과-산란[W/m²/nm]"] = np.round(spec["trn_d"], decimals)

            cols[f"{tag} 반사-전체[W/m²/nm]"] = np.round(spec["ref"], decimals)
            cols[f"{tag} 반사-직달[W/m²/nm]"] = np.round(spec["ref_b"], decimals)
            cols[f"{tag} 반사-산란[W/m²/nm]"] = np.round(spec["ref_d"], decimals)

            cols[f"{tag} 흡수(방사)-전체[W/m²/nm]"] = np.round(spec["abs"], decimals)
            cols[f"{tag} 흡수-직달[W/m²/nm]"] = np.round(spec["abs_b"], decimals)
            cols[f"{tag} 흡수-산란[W/m²/nm]"] = np.round(spec["abs_d"], decimals)

            cols[f"{tag} 투과(스크린)-전체[W/m²/nm]"] = np.round(spec["scr"], decimals)
            cols[f"{tag} 투과(스크린)-직달[W/m²/nm]"] = np.round(spec["scr_b"], decimals)
            cols[f"{tag} 투과(스크린)-산란[W/m²/nm]"] = np.round(spec["scr_d"], decimals)

        df_spec = pd.DataFrame(cols)
        df_spec.to_excel(writer, sheet_name="spectral", index=False)

        rows = []
        for key, S in sums_by_case.items():
            tag_surf, tag_or = key
            row = {
                "표면": tag_surf, "방향": tag_or,
                "파장 범위": f"{int(Wavelength_min)}–{int(Wavelength_max)} nm",
                "외부(합)[W/m²]": round(S["G_ext"], decimals_sum),

                "투과-전체[W/m²]": round(S["G_trn"], decimals_sum),
                "투과-직달[W/m²]": round(S["G_trn_beam"], decimals_sum),
                "투과-산란[W/m²]": round(S["G_trn_diff"], decimals_sum),

                "반사-전체)[W/m²]": round(S["G_ref"], decimals_sum),
                "반사-직달[W/m²]": round(S["G_ref_beam"], decimals_sum),
                "반사-산란[W/m²]": round(S["G_ref_diff"], decimals_sum),

                "흡수(방사)-전체[W/m²]": round(S["G_abs"], decimals_sum),
                "흡수-직달[W/m²]": round(S["G_abs_beam"], decimals_sum),
                "흡수-산란[W/m²]": round(S["G_abs_diff"], decimals_sum),

                "AOI loss model": S.get("AOI_model", ""),
                "β[°]": round(S.get("beta_deg", 0.0), 2),
                "θd[°](SAM)": round(S.get("theta_d_deg", 0.0), 2),
                "K_b": round(S.get("K_b", 0.0), 3),
                "K_d": round(S.get("K_d", 0.0), 3),
                "GHI_coeff": round(S.get("GHI_coeff", 0.0), decimals_sum),

                "맑음 계수 Kt": round(S["Kt"], 3),
                "태양고도 α[°]": round(S["alpha_deg"], 2),
                "천정각 θz[°]": round(S["theta_z_deg"], 2),
            }

            if "G_crop" in S:
                row["투과(스크린)-전체[W/m²]"] = round(S["G_crop"], decimals_sum)
                row["투과(스크린)-직달[W/m²]"] = round(S.get("G_scr_beam", float("nan")), decimals_sum)
                row["투과(스크린)-산란[W/m²]"] = round(S.get("G_scr_diff", float("nan")), decimals_sum)

            if "Screen_on" in S:
                row["스크린ON"] = "Y" if S["Screen_on"] else "N"
                row["차광률"] = round(S.get("shading_rate", shading_rate), 2)
                row["스크린 τ_b"] = round(S.get("trns_Beam", trns_Beam), 3)
                row["스크린 τ_d"] = round(S.get("trns_Diffuse", trns_Diffuse), 3)
            if "screen_threshold" in S:
                row["스크린 임계[W/m²]"] = round(S["screen_threshold"], decimals_sum)
            if "G_trn_internal" in S:
                row["내부 기준광량(필름 통과)[W/m²]"] = round(S["G_trn_internal"], decimals_sum)

            row["τ_b(=Tau*K_b)"] = round(S.get("tau_b", float("nan")), 3)
            row["τ_d(=Tau*K_d)"] = round(S.get("tau_d", float("nan")), 3)
            row["ρ_b"] = round(S.get("rho_b", float("nan")), 3)
            row["ρ_d"] = round(S.get("rho_d", float("nan")), 3)

            rows.append(row)

        df_sum = pd.DataFrame(rows)
        df_sum.to_excel(writer, sheet_name="sum", index=False)

        for sheet, df in [("spectral", df_spec), ("sum", df_sum)]:
            ws = writer.sheets[sheet]
            for i, col in enumerate(df.columns):
                max_len = max([len(str(col))] + [len(str(v)) for v in df[col].astype(str).tolist()])
                ws.set_column(i, i, min(60, max(12, max_len + 2)))
    return xlsx_path

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        yymmddhh = sys.argv[1].strip()
    else:
        yymmddhh = input("날짜 및 시간 입력(24010100~24123123): ").strip()
    qdt = parse_yymmddhh(yymmddhh)

    try:
        cases = [("Flat", "EW")]
        spectra_by_case = {}
        sums_by_case = {}
        wl_ref = None
        meta_ref = None

        for surf, ori in cases:
            (
                wl, E_ext, E_trn, E_ref, E_abs,
                E_trn_beam, E_trn_diff,
                E_ref_beam, E_ref_diff,
                E_abs_beam, E_abs_diff,
                E_scr_beam, E_scr_diff,
                E_crop,
                G_ext, G_trn, G_ref, G_abs,
                G_trn_beam, G_trn_diff,
                G_ref_beam, G_ref_diff,
                G_abs_beam, G_abs_diff,
                G_scr_beam, G_scr_diff,
                G_crop,
                meta
            ) = compute_case(qdt, ori)

            if wl_ref is None:
                wl_ref = wl
                meta_ref = meta

            spectra_by_case[(surf, ori)] = {
                "ext": E_ext,

                "trn": E_trn, "trn_b": E_trn_beam, "trn_d": E_trn_diff,
                "ref": E_ref, "ref_b": E_ref_beam, "ref_d": E_ref_diff,
                "abs": E_abs, "abs_b": E_abs_beam, "abs_d": E_abs_diff,

                "scr": E_crop, "scr_b": E_scr_beam, "scr_d": E_scr_diff,
            }

            sums_by_case[(surf, ori)] = {
                "G_ext": G_ext,

                "G_trn": G_trn, "G_trn_beam": G_trn_beam, "G_trn_diff": G_trn_diff,
                "G_ref": G_ref, "G_ref_beam": G_ref_beam, "G_ref_diff": G_ref_diff,
                "G_abs": G_abs, "G_abs_beam": G_abs_beam, "G_abs_diff": G_abs_diff,

                "Kt": meta["Kt"],
                "alpha_deg": meta["alpha_deg"],
                "theta_z_deg": meta["theta_z_deg"],

                "AOI_model": meta.get("AOI_model", AOI_model),
                "beta_deg": meta["beta_deg"],
                "theta_d_deg": meta["theta_d_deg"],
                "K_b": meta["K_b"],
                "K_d": meta["K_d"],
                "GHI_coeff": meta["GHI_coeff"],

                "tau_b": meta["tau_b"],
                "tau_d": meta["tau_d"],
                "rho_b": meta["rho_b"],
                "rho_d": meta["rho_d"],

                "G_crop": G_crop,
                "G_scr_beam": meta.get("G_scr_beam", float("nan")),
                "G_scr_diff": meta.get("G_scr_diff", float("nan")),

                "Screen_on": meta["Screen_on"],
                "shading_rate": meta["shading_rate"],
                "trns_Beam": meta["trns_Beam"],
                "trns_Diffuse": meta["trns_Diffuse"],
                "screen_threshold": meta["screen_threshold"],
                "G_trn_internal": meta["G_trn_internal"],
            }

        out_path = save_excel_combined(wl_ref, spectra_by_case, sums_by_case, meta_ref["chosen_dt"])

    except Exception as e:
        print(f"[오류] {e}")
        sys.exit(1)

    m = meta_ref
    print("\n================ 1) 위치 및 온실 정보 ================")
    print(f"입력 시각              : {m['chosen_dt']}")
    print(f"위도, 경도              : {Lat:.2f}°, {Lon:.2f}°  (KST 표준경선 {Lon_kst:.1f}°)")
    print(f"필름 광학 계수           : Tau={Tau:.3f}, Alpha={Alpha:.3f} (ρ=1-τ(θ)-α, α 고정)")
    print(f"AOI model             : {m.get('AOI_model', AOI_model)}")
    print(f"지붕 경사 β[°]          : {m['beta_deg']:.2f} (수평 = 0)")

    print("\n================ 2) 태양 고도 및 입사각 계산 ================")
    print(f"연일 n                  : {meta_ref['chosen_dt'].timetuple().tm_yday}")
    print(f"표준시(LT) [h]           : {meta_ref['chosen_dt'].hour}")
    print(f"태양고도 α [°]           : {m['alpha_deg']:.3f}")
    print(f"천정각 θz [°]            : {m['theta_z_deg']:.3f}")

    print("\n================ 3) 일사 분해 (Reindl-2) ================")
    print(f"GHI [W/m²]              : {m['GHI']:.3f}")
    print(f"DHI [W/m²]              : {m['DHI']:.3f}")
    print(f"DNI [W/m²]              : {m['DNI']:.3f}   (Ib,h = {m['Ib_h']:.3f})")
    print(f"clearness index Kt      : {m['Kt']:.3f}")

    print("\n================ 4) NREL(SAM) IAM 보정 ================")
    print(f"θd[°]        : {m['theta_d_deg']:.3f} (sky diffuse)")
    print(f"K_b          : {m['K_b']:.3f}")
    print(f"K_d          : {m['K_d']:.3f}")
    print(f"GHI_coeff      : {m['GHI_coeff']:.3f} (Gb*K_b + Gd*K_d)")

    print("\n================ 5) 필름 투과율(성분별) ================")
    print(f"τ_b = Tau*K_b           : {m['tau_b']:.3f}")
    print(f"τ_d = Tau*K_d           : {m['tau_d']:.3f}")

    print("\n================ 6) 필름의 광학 계수 적분 결과 ================")
    for (surf, ori), S in sums_by_case.items():
        thr = S.get('screen_threshold', shading_threshold)
        onoff = "ON" if S.get('Screen_on', False) else "OFF"
        shade = S.get('shading_rate', shading_rate)

        print(
            f"[{surf}-{ori}][W/m²]\n"
            f"외부={S['G_ext']:.3f}[W/m²]\n"
            f"투과(전체)={S['G_trn']:.3f} (직달={S['G_trn_beam']:.3f}, 산란={S['G_trn_diff']:.3f})[W/m²]\n"
            f"흡수(전체)={S['G_abs']:.3f} (직달={S['G_abs_beam']:.3f}, 산란={S['G_abs_diff']:.3f})[W/m²]\n"
            f"반사(전체)={S['G_ref']:.3f} (직달={S['G_ref_beam']:.3f}, 산란={S['G_ref_diff']:.3f})[W/m²]\n"
            f"스크린후(전체)={S['G_crop']:.3f} (직달={S.get('G_scr_beam', float('nan')):.3f}, 산란={S.get('G_scr_diff', float('nan')):.3f})[W/m²]\n"
            f"스크린 임계={thr:.1f}[W/m²]\n"
            f"스크린={onoff}\n"
            f"차광률={shade:.2f}\n"
        )

    print("\n엑셀 저장(통합) :", out_path, "\n")

# 전주 내의 딸기 단동 플라스틱 온실을 기준으로 함(10-단동-6형 -> 스마트 딸기 온실(농사로) 규격 기준
# 태양광은 온실의 지붕 면적에만 입사 된다고 가정함(측면과 전후면은 무시함)-태양 고도에 따른 입사광이 위치 별로 전부 달라 고려하기 어려움
# 본 온실의 지붕 표면은 수평면을 기준으로 함(곡면 및 지붕각은 생각하지 않음)
# 정규화(적분 = 1)를 통해 일사의 스펙트럼 비율을 계산함 -> Reindl-2 모델을 가지고 일사를 직달과 산란으로 분리함
# 직달은 NREL에서 측정한 GHI의 37° 측정 각도와 무관함(법선-수직)이므로 해당 직달광 값에 입사각을 곱하고 스펙트럼 비율을 곱해 해당 스펙트럼의 직달 총량을 구함
# 직달광은 IAM 보정 수식에 입사각을 넣어서 필름의 투과, 흡수(방사), 반사된 정도를 계산할 때 계산함
# NREL(SAM)에서 제시한 산란광 대표 입사각 경험 수식을 이용하여 산란광의 각도를 계산하고 산란광에 대한 IAM 보정을 진행함
# 산란광은 Hay-Davies 모델을 활용해 등방 또는 비등방 여부를 고려해야 하지만 코드와 내용의 단순화를 위해 활용하지 않음
# 55% 차광률을 가진 차광 스크린을 온실 내부에 설치했다고 가정함 - 필름 하부에 위치함으로 투과율만 생각함
# 지붕면이 수평면일 때는 동서동 방향 등의 온실 위치 및 방향은 큰 의미가 없음(일단, 동서동 방향으로 설정함)
