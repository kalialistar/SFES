"""
Greenhouse Microclimate Simulator - Streamlit App
Based on GH_microclimate_model_gimje_smart_farm_lettuce.py
"""

import sys, os, io, math, tempfile, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, time, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GH_microclimate_model_gimje_smart_farm_lettuce import run_integrated_TRH_CO2_model

# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="온실 미기후 시뮬레이터 (김제 스마트팜)",
    page_icon="🌿", layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title{font-size:2rem;font-weight:700;color:#2d6a4f;
  border-bottom:3px solid #52b788;padding-bottom:.4rem;margin-bottom:1rem;}
.section-header{font-size:1.05rem;font-weight:600;color:#1b4332;
  background:#d8f3dc;border-left:4px solid #52b788;
  padding:.35rem .8rem;border-radius:0 6px 6px 0;margin:.9rem 0 .45rem 0;}
.pred-card{background:#f0faf4;border:1.5px solid #74c69d;border-radius:12px;
  padding:1rem 1rem .8rem;text-align:center;margin:.2rem;}
.pred-label{font-size:.82rem;color:#2d6a4f;font-weight:600;margin-bottom:.25rem;}
.pred-value{font-size:1.7rem;font-weight:800;color:#1b4332;line-height:1.1;}
.pred-unit{font-size:.75rem;color:#52b788;margin-top:.15rem;}
.obs-main-card{background:#fff5ee;border:1.5px solid #f4a261;border-radius:12px;
  padding:1rem 1rem .8rem;text-align:center;margin:.2rem;}
.obs-main-label{font-size:.82rem;color:#a0522d;font-weight:600;margin-bottom:.25rem;}
.obs-main-value{font-size:1.7rem;font-weight:800;color:#7f3600;line-height:1.1;}
.obs-main-unit{font-size:.75rem;color:#e07a39;margin-top:.15rem;}
.obs-meta-card{background:#f8f4ff;border:1px solid #c3aed6;border-radius:9px;
  padding:.55rem .7rem;text-align:center;margin:.15rem;}
.obs-meta-label{font-size:.72rem;color:#5a3e85;margin-bottom:.15rem;}
.obs-meta-value{font-size:1.15rem;font-weight:700;color:#3b1f6a;}
.obs-meta-unit{font-size:.66rem;color:#9b72cf;}
.cmp-card{background:#f0f4ff;border:1.5px solid #90b0e8;
  border-radius:12px;padding:.9rem;text-align:center;margin:.2rem;}
.stat-card{background:#fafafa;border:1px solid #ddd;
  border-radius:10px;padding:.7rem .5rem;text-align:center;margin:.15rem;}
.stat-label{font-size:.72rem;color:#555;}
.stat-value{font-size:1.1rem;font-weight:700;color:#1b4332;}
.info-box{background:#e8f4fd;border-left:4px solid #4da6db;
  padding:.6rem 1rem;border-radius:0 6px 6px 0;
  font-size:.87rem;color:#1a4a6b;margin:.5rem 0;}
.warn-box{background:#fff3cd;border-left:4px solid #ffc107;
  padding:.6rem 1rem;border-radius:0 6px 6px 0;
  font-size:.87rem;color:#664d00;margin:.5rem 0;}
</style>
""", unsafe_allow_html=True)

TIME_COL = "날짜&시간"

# 파라미터 정보: key -> (표시명, 단위, 색)
PARAM_INFO = {
    "Troom_C":("Troom_C","°C","#e63946"),
    "RHin_%":("RHin_%","%","#0077b6"),
    "Cin_ppm":("Cin_ppm","ppm","#0077b6"),
    "Cs_ppm":("Cs_ppm","ppm","#48cae4"),
    "Cc_ppm":("Cc_ppm","ppm","#023e8a"),
    "ACH":("ACH","h⁻¹","#457b9d"),
    "u_int":("u_int","m/s","#74b3ce"),
    "Ur":("Ur","W/m²K","#e9c46a"),
    "Uw":("Uw","W/m²K","#f4a261"),
    "transGlass":("transGlass","-","#a8dadc"),
    "PPFD_above_umol_m2_s":("PPFD_above_umol_m2_s","μmol/m²/s","#ffd166"),
    "PPFD_canopy_umol_m2_s":("PPFD_canopy_umol_m2_s","μmol/m²/s","#f4e04d"),
    "PPFD_canopy_absorbed_umol_m2_ground_s":("PPFD_canopy_absorbed","μmol/m²/s","#e9b44c"),
    "A_leaf_umol_m2_s":("A_leaf_umol_m2_s","μmol/m²/s","#118ab2"),
    "A_canopy_umol_m2_s":("A_canopy_umol_m2_s","μmol/m²/s","#06d6a0"),
    "A_canopy_gross_umol_m2_s":("A_canopy_gross_umol_m2_s","μmol/m²/s","#52b788"),
    "R_canopy_umol_m2_s":("R_canopy_umol_m2_s","μmol/m²/s","#e07a5f"),
    "J_leaf_umol_m2_s":("J_leaf_umol_m2_s","μmol/m²/s","#3d405b"),
    "Vcmax_leaf_umol_m2_s":("Vcmax_leaf_umol_m2_s","μmol/m²/s","#6d6875"),
    "gsw_mol_m2_s":("gsw_mol_m2_s","mol/m²/s","#073b4c"),
    "VPD_Pa":("VPD_Pa","Pa","#ef476f"),
    "Cbio_kg_per_h":("Cbio_kg_per_h","kg/h","#8338ec"),
    "Cvent_kg_per_h":("Cvent_kg_per_h","kg/h","#3a86ff"),
    "Cr_kg_per_h":("Cr_kg_per_h","kg/h","#ff006e"),
    "Cbio_ppm_per_h":("Cbio_ppm_per_h","ppm/h","#7209b7"),
    "Cvent_ppm_per_h":("Cvent_ppm_per_h","ppm/h","#4361ee"),
    "Cr_ppm_per_h":("Cr_ppm_per_h","ppm/h","#f72585"),
    "LAI":("LAI","-","#2dc653"),
    "Ld_cm":("Ld_cm","cm","#52b788"),
    "Lw_cm":("Lw_cm","cm","#95d5b2"),
    "Rn_Wm2":("Rn_Wm2","W/m²","#e9c46a"),
    "Rnl_crop_Wm2":("Rnl_crop_Wm2","W/m²","#f4a261"),
    "lambdaE_Wm2":("lambdaE_Wm2","W/m²","#4cc9f0"),
    "ra_leaf_s_m":("ra_leaf_s_m","s/m","#adb5bd"),
    "rs_s_m":("rs_s_m","s/m","#6c757d"),
    "rc_s_m":("rc_s_m","s/m","#495057"),
    "rhoAir_kgm3":("rhoAir_kgm3","kg/m³","#dee2e6"),
    "cAir_kJkgK":("cAir_kJkgK","kJ/kgK","#ced4da"),
    "mHouse_kg":("mHouse_kg","kg","#adb5bd"),
    "mHouse_dry_kg":("mHouse_dry_kg","kg","#6c757d"),
    "Qlw_env_total_kW":("Qlw_env_total_kW","kW","#9b2226"),
    "Qlw_env_roof_kW":("Qlw_env_roof_kW","kW","#ae2012"),
    "Qlw_env_side_kW":("Qlw_env_side_kW","kW","#bb3e03"),
    "Qlw_env_total_Wm2":("Qlw_env_total_Wm2","W/m²","#ca6702"),
    "Wtr_kgph":("Wtr_kgph","kg/h","#06d6a0"),
    "Wev_kgph":("Wev_kgph","kg/h","#74c69d"),
    "Wev_raw_kgph":("Wev_raw_kgph","kg/h","#b7e4c7"),
    "hsolution_Wm2K":("hsolution_Wm2K","W/m²K","#95d5b2"),
    "ra_solution_s_m":("ra_solution_s_m","s/m","#74c69d"),
    "hs_Wm2K":("hs_Wm2K","W/m²K","#40916c"),
    "h_cover_in_Wm2K":("h_cover_in_Wm2K","W/m²K","#2d6a4f"),
    "h_gap_fs_Wm2K":("h_gap_fs_Wm2K","W/m²K","#1b4332"),
    "h_gap_sc_Wm2K":("h_gap_sc_Wm2K","W/m²K","#081c15"),
    "Wcond_cover_kgph":("Wcond_cover_kgph","kg/h","#b5838d"),
    "qCond_cover_kW":("qCond_cover_kW","kW","#e5989b"),
    "qLatent_net_kW":("qLatent_net_kW","kW","#ffb4a2"),
    "Tfilm_ext_C":("Tfilm_ext_C","°C","#e07a5f"),
    "Tscreen_int_C":("Tscreen_int_C","°C","#f2cc8f"),
    "Tcurtain_int_C":("Tcurtain_int_C","°C","#e9c46a"),
    "Tgap_fs_C":("Tgap_fs_C","°C","#a8dadc"),
    "Tgap_sc_C":("Tgap_sc_C","°C","#457b9d"),
    "Tcover_in_C":("Tcover_in_C","°C","#f2cc8f"),
    "Tleaf_C":("Tleaf_C","°C","#81b29a"),
    "Tsky_C":("Tsky_C","°C","#3d405b"),
    "qRad_kW":("qRad_kW","kW","#e9c46a"),
    "qRoof_kW":("qRoof_kW","kW","#f4a261"),
    "qFloor_kW":("qFloor_kW","kW","#8ecae6"),
    "qSideWall_kW":("qSideWall_kW","kW","#a8dadc"),
    "qVent_kW":("qVent_kW","kW","#457b9d"),
    "QcropS_kW":("QcropS_kW","kW","#52b788"),
    "qLatent_kW":("qLatent_kW","kW","#b5838d"),
    "qFIR_kW":("qFIR_kW","kW","#6d6875"),
    "qt_kW":("qt_kW","kW","#e63946"),
}

DISPLAY_GROUPS = {
    "🌡️ Temp / RH / CO₂":    ["Troom_C","RHin_%","Cin_ppm","Cs_ppm","Cc_ppm"],
    "☀️ Energy Fluxes":       ["qRad_kW","qRoof_kW","qFloor_kW","qSideWall_kW",
                                "qVent_kW","qt_kW","qFIR_kW","qLatent_kW",
                                "qLatent_net_kW","QcropS_kW","qCond_cover_kW"],
    "🌿 Photosynthesis / LAI":["PPFD_above_umol_m2_s","PPFD_canopy_umol_m2_s",
                                "PPFD_canopy_absorbed_umol_m2_ground_s",
                                "A_canopy_umol_m2_s","A_canopy_gross_umol_m2_s",
                                "A_leaf_umol_m2_s","R_canopy_umol_m2_s",
                                "J_leaf_umol_m2_s","Vcmax_leaf_umol_m2_s",
                                "gsw_mol_m2_s","VPD_Pa","LAI","Ld_cm","Lw_cm"],
    "💧 CO₂ & Water Fluxes": ["Cbio_ppm_per_h","Cvent_ppm_per_h","Cr_ppm_per_h",
                                "Cbio_kg_per_h","Cvent_kg_per_h","Cr_kg_per_h",
                                "Wtr_kgph","Wev_kgph","Wcond_cover_kgph","lambdaE_Wm2"],
    "🏗️ Cover / Ventilation": ["ACH","u_int","Ur","Uw","transGlass",
                                "Tfilm_ext_C","Tscreen_int_C","Tcurtain_int_C",
                                "Tgap_fs_C","Tgap_sc_C","Tcover_in_C","Tleaf_C","Tsky_C",
                                "h_cover_in_Wm2K","h_gap_fs_Wm2K","h_gap_sc_Wm2K",
                                "hs_Wm2K","hsolution_Wm2K"],
    "📐 Resistances / Air":   ["ra_leaf_s_m","rs_s_m","rc_s_m","ra_solution_s_m",
                                "rhoAir_kgm3","cAir_kJkgK","mHouse_kg","mHouse_dry_kg",
                                "Rn_Wm2","Rnl_crop_Wm2",
                                "Qlw_env_total_kW","Qlw_env_roof_kW",
                                "Qlw_env_side_kW","Qlw_env_total_Wm2"],
}

# 실측 주요 카드 (온습도CO2 — 큰 크기)
OBS_MAIN = [
    ("내부온도",  "Indoor Temp (Obs)",   "°C"),
    ("내부습도",  "Indoor RH (Obs)",     "%"),
    ("내부CO2",   "Indoor CO₂ (Obs)",   "ppm"),
    ("외부CO2",   "Outdoor CO₂ (Obs)",  "ppm"),
]
# 실측 기상 카드 (작은 크기) — 강수량 포함
OBS_META = [
    ("외부 온도", "Outdoor Temp",    "°C"),
    ("외부 습도", "Outdoor RH",     "%"),
    ("일사",      "Solar Rad.",     "W/m²"),
    ("외부 풍속", "Wind Speed",     "m/s"),
    ("강수량",    "Precipitation",  "mm"),
    ("감우",      "Rain Sensor",    "0/1"),
    ("전운량",    "Cloud Cover",    "oktas"),
]

# ─── 유틸 함수 ────────────────────────────────────────────────────────
def fmt(v, digits=2):
    if v is None:
        return "—"
    try:
        fv = float(v)
    except Exception:
        return str(v)
    if not math.isfinite(fv):
        return "—"
    if abs(fv) >= 10000:
        return f"{fv:,.0f}"
    if abs(fv) >= 1000:
        return f"{fv:,.1f}"
    if abs(fv) >= 10:
        return f"{fv:.{digits}f}"
    return f"{fv:.{max(digits,3)}f}"


def pred_card_html(label, value, unit):
    return (f'<div class="pred-card"><div class="pred-label">{label}</div>'
            f'<div class="pred-value">{fmt(value)}</div>'
            f'<div class="pred-unit">{unit}</div></div>')


def obs_main_card_html(label, value, unit):
    return (f'<div class="obs-main-card"><div class="obs-main-label">{label}</div>'
            f'<div class="obs-main-value">{fmt(value)}</div>'
            f'<div class="obs-main-unit">{unit}</div></div>')


def obs_meta_card_html(label, value, unit):
    return (f'<div class="obs-meta-card"><div class="obs-meta-label">{label}</div>'
            f'<div class="obs-meta-value">{fmt(value)}</div>'
            f'<div class="obs-meta-unit">{unit}</div></div>')




def _canon_col(s):
    s = str(s).strip()
    rep = {" ":"", "_":"", "-":"", "&":"", "·":"", "(":"", ")":"", "[":"", "]":"", "/":"", "℃":"c", "°":"", "%":"percent", "₂":"2", "₃":"3"}
    for a,b in rep.items():
        s = s.replace(a,b)
    return s.lower()


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    alias_groups = {
        TIME_COL: ["날짜&시간", "날짜 시간", "datetime", "date time", "time", "timestamp", "날짜&시간ㅂ"],
        "외부 온도": ["외부 온도", "외부온도", "외기온도", "outside temp", "outdoor temp", "tout", "taout"],
        "외부 습도": ["외부 습도", "외부습도", "외기습도", "outside rh", "outdoor rh", "rhout"],
        "일사": ["일사", "일사량", "solar", "solar rad", "radiation", "swrad", "rs"],
        "외부 풍속": ["외부 풍속", "외부풍속", "풍속", "wind", "wind speed", "uout"],
        "전운량": ["전운량", "운량", "cloud", "cloud cover", "tcc"],
        "강수량": ["강수량", "강우량", "precipitation", "rainfall"],
        "감우": ["감우", "rain sensor", "rainsensor", "rain flag"],
        "내부온도": ["내부온도", "내부 온도", "실내온도", "실내 온도", "온실내부온도", "indoor temp", "tin", "t_in"],
        "내부습도": ["내부습도", "내부 습도", "실내습도", "실내 습도", "indoor rh", "rhin", "rh_in"],
        "내부CO2": ["내부co2", "내부 co2", "내부co₂", "내부 co₂", "실내co2", "실내 co2", "indoor co2", "cin", "c_in"],
        "외부CO2": ["외부co2", "외부 co2", "외부co₂", "외부 co₂", "outdoor co2", "cout", "c_out"],
    }
    existing = list(df.columns)
    canon_existing = {_canon_col(c): c for c in existing}
    rename_map = {}
    for target, aliases in alias_groups.items():
        if target in df.columns:
            continue
        for alias in aliases:
            hit = canon_existing.get(_canon_col(alias))
            if hit is not None:
                rename_map[hit] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def calc_metrics(pred_series, obs_series):
    """R², RMSE, MBE 계산 (NaN 행 제거 후 계산)"""
    pred_arr = np.array(pred_series, dtype=float)
    obs_arr  = np.array(obs_series,  dtype=float)
    mask = np.isfinite(pred_arr) & np.isfinite(obs_arr)
    p = pred_arr[mask]
    o = obs_arr[mask]
    n = len(p)
    if n < 2:
        return None, None, None
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    mbe  = float(np.mean(p - o))
    return r2, rmse, mbe


def calc_threshold_error_stats(pred_series, obs_series, threshold):
    """임계오차 초과 비율(%) 계산 (NaN 제거 후)"""
    pred_arr = np.array(pred_series, dtype=float)
    obs_arr  = np.array(obs_series,  dtype=float)
    mask = np.isfinite(pred_arr) & np.isfinite(obs_arr)
    p = pred_arr[mask]
    o = obs_arr[mask]
    n = len(p)
    if n < 1:
        return None, 0, 0
    exceed = np.abs(p - o) > float(threshold)
    exceed_count = int(np.sum(exceed))
    exceed_pct = float(exceed_count / n * 100.0)
    return exceed_pct, exceed_count, int(n)


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Result")
    return buf.getvalue()


def render_folium_gimje():
    """김제시 위치를 folium으로 표시하는 HTML 반환"""
    try:
        import folium

        gimje_lat = 35.8039
        gimje_lon = 126.8893
        m = folium.Map(
            location=[gimje_lat, gimje_lon],
            zoom_start=12,
            tiles="OpenStreetMap",
            control_scale=True,
        )
        folium.Marker(
            location=[gimje_lat, gimje_lon],
            popup=folium.Popup("<b>김제 스마트팜</b><br>전북특별자치도 김제시", max_width=220),
            tooltip="📍 김제 스마트팜",
            icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
        ).add_to(m)
        folium.Circle(
            location=[gimje_lat, gimje_lon],
            radius=2500, color="#52b788", fill=True,
            fill_color="#74c69d", fill_opacity=0.22,
        ).add_to(m)
        return m._repr_html_()
    except ImportError:
        return None


# ─── 세션 상태 ──────────────────────────────────────────────────────
if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# ─── 사이드바 ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 온실 시뮬레이터")
    st.markdown("**김제 스마트팜 상추 미기후 모델**")
    st.divider()

    st.markdown('<div class="section-header">📂 기상 데이터 업로드</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "엑셀 파일 (.xlsx) 업로드", type=["xlsx"],
        help="필수: 날짜&시간, 외부 온도, 외부 습도, 일사\n"
             "선택: 외부 풍속, 전운량, 강수량, 감우, 내부온도, 내부습도, 내부CO2, 외부CO2",
    )

    st.markdown('<div class="section-header">📍 관측 지점</div>', unsafe_allow_html=True)
    folium_html_sidebar = render_folium_gimje()
    if folium_html_sidebar:
        st.components.v1.html(folium_html_sidebar, height=240, scrolling=False)
    else:
        st.markdown("""<div class="warn-box">
        📍 <b>전라북도 김제시</b><br>
        위도: 35.804° N<br>경도: 126.889° E
        </div>""", unsafe_allow_html=True)

    run_btn = st.button("▶ 시뮬레이션 실행", type="primary", use_container_width=True)
    st.divider()
    st.caption("FvCB + Medlyn + Penman-Monteith + 다층 커버 모델")

# ─── 메인 ───────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌿 온실 미기후 시뮬레이터 | 김제 스마트팜</div>', unsafe_allow_html=True)

if run_btn:
    if uploaded is None:
        st.error("❌ 엑셀 파일을 먼저 업로드해 주세요.")
    else:
        with st.spinner("⚙️ 시뮬레이션 계산 중..."):
            try:
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_f:
                    tmp_f.write(uploaded.getbuffer())
                    tmp_path = tmp_f.name
                try:
                    raw_input_df = pd.read_excel(tmp_path)
                    raw_input_df = normalize_input_columns(raw_input_df)
                    result = run_integrated_TRH_CO2_model(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

                # 입력 엑셀의 실측/기상 컬럼을 결과 데이터프레임에 다시 병합
                merge_cols = [
                    TIME_COL, "외부 온도", "외부 습도", "일사", "외부 풍속", "전운량",
                    "강수량", "감우", "내부온도", "내부습도", "내부CO2", "외부CO2"
                ]
                keep_cols = [c for c in merge_cols if c in raw_input_df.columns]
                if TIME_COL in keep_cols:
                    raw_input_df = raw_input_df[keep_cols].copy()
                    raw_input_df[TIME_COL] = pd.to_datetime(raw_input_df[TIME_COL], errors="coerce")
                    raw_input_df = raw_input_df.dropna(subset=[TIME_COL]).drop_duplicates(subset=[TIME_COL], keep="last")
                    result[TIME_COL] = pd.to_datetime(result[TIME_COL], errors="coerce")
                    result = result.dropna(subset=[TIME_COL]).drop_duplicates(subset=[TIME_COL], keep="last")
                    result = result.merge(raw_input_df, on=TIME_COL, how="left", suffixes=("", "_obsraw"))
                    for c in keep_cols:
                        if c != TIME_COL and f"{c}_obsraw" in result.columns:
                            if c not in result.columns or result[c].isna().all():
                                result[c] = result[f"{c}_obsraw"]
                            else:
                                result[c] = result[c].where(result[c].notna(), result[f"{c}_obsraw"])
                            result = result.drop(columns=[f"{c}_obsraw"])

                st.session_state.result_df = result
                st.session_state.uploaded_filename = uploaded.name
                st.success(f"✅ 완료! 총 {len(result)}개 시간 스텝 계산됨.")
            except Exception as e:
                st.error(f"❌ 오류: {e}")
                st.exception(e)

result_df = st.session_state.result_df

if result_df is None:
    st.markdown("""<div class="info-box">
    📌 <b>사용 방법</b><br>
    1. 사이드바에서 엑셀 파일을 업로드합니다.<br>
    2. <b>시뮬레이션 실행</b>을 클릭합니다.<br>
    3. <b>기간 분석</b>: 날짜 범위 → 온도·습도·CO₂ 예측/실측 비교 그래프 + 성능지표 + 엑셀 다운로드<br>
    4. <b>시점 분석</b>: 날짜·시각 → 예측값·실측값 카드 + ±12h 가로 3개 그래프
    </div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**필수 컬럼**")
        st.table(pd.DataFrame({
            "컬럼명":["날짜&시간","외부 온도","외부 습도","일사"],
            "단위":["datetime","°C","%","W/m²"],
            "설명":["1시간 간격 권장","외기 온도","외기 상대습도","태양복사"],
        }))
    with c2:
        st.markdown("**선택 컬럼**")
        st.table(pd.DataFrame({
            "컬럼명":["외부 풍속","전운량","강수량","감우","내부온도","내부습도","내부CO2","외부CO2"],
            "단위":["m/s","0~10","mm","0/1","°C","%","ppm","ppm"],
            "설명":["외기 풍속","운량","강수량","강우 여부","실측 내부온도","실측 내부습도","실측 내부CO₂","실측 외부CO₂"],
        }))
    st.stop()

result_df[TIME_COL] = pd.to_datetime(result_df[TIME_COL])
t_min = result_df[TIME_COL].min().to_pydatetime()
t_max = result_df[TIME_COL].max().to_pydatetime()
d_min, d_max = t_min.date(), t_max.date()

st.markdown(f"""<div class="info-box">
📁 파일: <b>{st.session_state.uploaded_filename}</b> &nbsp;|&nbsp;
기간: <b>{t_min.strftime('%Y-%m-%d %H:%M')}</b> ~ <b>{t_max.strftime('%Y-%m-%d %H:%M')}</b> &nbsp;|&nbsp;
총 <b>{len(result_df)}</b> 스텝
</div>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📈 기간 분석", "🔍 시점 분석"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — 기간 분석
# ═══════════════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="section-header">📅 분석 기간 선택</div>', unsafe_allow_html=True)
    cs, ce = st.columns(2)
    with cs:
        # 디폴트: 전체 기간 시작
        start_date = st.date_input("시작 날짜", value=d_min, min_value=d_min, max_value=d_max, key="t1_sd")
        start_hour = st.selectbox("시작 시각", list(range(24)), index=0,
                                  format_func=lambda h: f"{h:02d}:00", key="t1_sh")
    with ce:
        # 디폴트: 전체 기간 종료
        end_date = st.date_input("종료 날짜", value=d_max, min_value=d_min, max_value=d_max, key="t1_ed")
        end_hour = st.selectbox("종료 시각", list(range(24)), index=t_max.hour,
                                format_func=lambda h: f"{h:02d}:00", key="t1_eh")

    start_dt = datetime.combine(start_date, time(start_hour))
    end_dt   = datetime.combine(end_date,   time(end_hour))

    if start_dt >= end_dt:
        st.warning("⚠️ 시작 시각이 종료 시각보다 앞이어야 합니다.")
        st.stop()

    sub_df = result_df[(result_df[TIME_COL] >= start_dt) & (result_df[TIME_COL] <= end_dt)].copy()
    if sub_df.empty:
        st.warning("⚠️ 선택 기간에 데이터가 없습니다.")
        st.stop()

    st.markdown(f"**선택 기간:** {start_dt.strftime('%Y-%m-%d %H:%M')} ~ {end_dt.strftime('%Y-%m-%d %H:%M')} | {len(sub_df)} 스텝")

    obs_count_t = int(sub_df["내부온도"].notna().sum()) if "내부온도" in sub_df.columns else 0
    obs_count_rh = int(sub_df["내부습도"].notna().sum()) if "내부습도" in sub_df.columns else 0
    obs_count_c = int(sub_df["내부CO2"].notna().sum()) if "내부CO2" in sub_df.columns else 0
    st.caption(f"실측 데이터 인식: 온도 {obs_count_t}개 | 습도 {obs_count_rh}개 | CO₂ {obs_count_c}개")

    # ── 섹션 1: 예측 vs 실측 겹침 그래프 (온도/습도/CO2 각각 하나의 그래프) ──
    st.markdown('<div class="section-header">📊 예측 vs 실측 비교 (온도·습도·CO₂)</div>', unsafe_allow_html=True)

    cmp_set = [
        ("Troom_C", "내부온도",  "Indoor Temperature", "°C",   "#e63946"),
        ("RHin_%",  "내부습도",  "Indoor RH",          "%",    "#0077b6"),
        ("Cin_ppm", "내부CO2",   "Indoor CO₂",         "ppm",  "#2d6a4f"),
    ]
    for pk, ok, title, unit, pred_clr in cmp_set:
        has_pred = pk in sub_df.columns and sub_df[pk].notna().any()
        has_obs  = ok in sub_df.columns and sub_df[ok].notna().any()

        fig_c = go.Figure()
        if has_pred:
            fig_c.add_trace(go.Scatter(
                x=sub_df[TIME_COL], y=sub_df[pk],
                mode="lines", name="예측 (Predicted)",
                line=dict(color=pred_clr, width=2.4),
                hovertemplate=f"Pred: %{{y:.2f}} {unit}<extra></extra>",
            ))
        if has_obs:
            fig_c.add_trace(go.Scatter(
                x=sub_df[TIME_COL], y=sub_df[ok],
                mode="lines", name="실측 (Observed)",
                line=dict(color="#000000", width=2.6, dash="dash"),
                hovertemplate=f"Obs: %{{y:.2f}} {unit}<extra></extra>",
            ))
        if not has_pred and not has_obs:
            continue

        fig_c.update_layout(
            title=dict(text=f"{title}  [{unit}]", font=dict(size=13, color="#1b4332")),
            height=270, margin=dict(l=58, r=20, t=45, b=35),
            legend=dict(orientation="h", y=1.18, x=0, font=dict(size=11)),
            hovermode="x unified", plot_bgcolor="#f9fffe",
            yaxis_title=unit,
            xaxis=dict(tickformat="%m/%d\n%H:%M"),
        )
        st.plotly_chart(fig_c, use_container_width=True)


    # ── 섹션 2: 기간별 예측 세부 항목 그래프 ─────────────────────────
    st.markdown('<div class="section-header">🧩 기간별 예측 세부 항목 그래프</div>', unsafe_allow_html=True)
    detail_group_t1 = st.selectbox('기간 분석 항목 그룹', list(DISPLAY_GROUPS.keys()), key='t1_detail_group')
    detail_candidates_t1 = [k for k in DISPLAY_GROUPS[detail_group_t1] if k in sub_df.columns]
    detail_defaults_t1 = [k for k in DISPLAY_GROUPS[detail_group_t1][:3] if k in sub_df.columns]
    detail_params_t1 = st.multiselect(
        '기간 분석 그래프 항목',
        options=detail_candidates_t1,
        default=detail_defaults_t1,
        format_func=lambda k: f"{PARAM_INFO.get(k,(k,'',''))[0]} [{PARAM_INFO.get(k,(k,'',''))[1]}]",
        key='t1_detail_params'
    )
    if detail_params_t1:
        fig_t1_detail = go.Figure()
        units_t1 = []
        for k in detail_params_t1:
            info = PARAM_INFO.get(k, (k, '', '#2d6a4f'))
            fig_t1_detail.add_trace(go.Scatter(
                x=sub_df[TIME_COL], y=sub_df[k], mode='lines', name=info[0],
                line=dict(color=info[2], width=2),
            ))
            if info[1] and info[1] not in units_t1:
                units_t1.append(info[1])
        fig_t1_detail.update_layout(
            height=380, hovermode='x unified', plot_bgcolor='#f9fffe',
            legend=dict(orientation='h', y=1.12, x=0),
            margin=dict(l=50, r=20, t=50, b=40),
            xaxis=dict(title='Time', tickformat='%m/%d\n%H:%M'),
            yaxis=dict(title=' / '.join(units_t1) if units_t1 else ''),
        )
        st.plotly_chart(fig_t1_detail, use_container_width=True)

    # ── 섹션 3: 성능 지표 (RMSE, MBE, Error 초과 비율) ─────────────────
    st.markdown('<div class="section-header">📐 예측 성능 지표 (선택 기간)</div>', unsafe_allow_html=True)
    m_pairs = [
        ("Troom_C", "내부온도", "Indoor Temp", "°C"),
        ("RHin_%",  "내부습도", "Indoor RH",   "%"),
        ("Cin_ppm", "내부CO2",  "Indoor CO₂",  "ppm"),
    ]
    metric_rows = []
    thresholds = {"Indoor Temp": 2.0, "Indoor RH": 7.0, "Indoor CO₂": 30.0}
    for pk, ok, lbl, unit in m_pairs:
        pred_ok = pk in sub_df.columns and sub_df[pk].notna().any()
        obs_ok  = ok in sub_df.columns and sub_df[ok].notna().any()
        rmse = mbe = exceed_pct = None
        if pred_ok and obs_ok:
            _, rmse, mbe = calc_metrics(sub_df[pk].values, sub_df[ok].values)
            exceed_pct, _, _ = calc_threshold_error_stats(
                sub_df[pk].values, sub_df[ok].values, thresholds[lbl]
            )
        metric_rows.append({
            "항목": lbl,
            "단위": unit,
            "Error 초과 비율(%)": None if exceed_pct is None or not math.isfinite(exceed_pct) else round(exceed_pct, 2),
            "RMSE": None if rmse is None or not math.isfinite(rmse) else round(rmse, 3),
            "MBE": None if mbe is None or not math.isfinite(mbe) else round(mbe, 3),
        })

    metric_df = pd.DataFrame(metric_rows)[["항목", "단위", "Error 초과 비율(%)", "RMSE", "MBE"]]
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    m_cols = st.columns(3)
    for col_obj, row_m in zip(m_cols, metric_rows):
        with col_obj:
            def show_val(v, signed=False):
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    return "—"
                return f"{v:+.3f}" if signed else f"{v:.3f}"
            def show_pct(v):
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    return "—"
                return f"{v:.2f}%"
            st.markdown(f"**{row_m['항목']}**")
            st.markdown(
                f'<div class="stat-card"><div class="stat-label">Error 초과 비율</div>'
                f'<div class="stat-value">{show_pct(row_m["Error 초과 비율(%)"])}</div></div>'
                f'<div class="stat-card"><div class="stat-label">RMSE ({row_m["단위"]})</div>'
                f'<div class="stat-value">{show_val(row_m["RMSE"])}</div></div>'
                f'<div class="stat-card"><div class="stat-label">MBE ({row_m["단위"]})</div>'
                f'<div class="stat-value">{show_val(row_m["MBE"], signed=True)}</div></div>',
                unsafe_allow_html=True,
            )

    # ── 섹션 5: 엑셀 다운로드 ─────────────────────────────────────────
    st.markdown('<div class="section-header">💾 데이터 다운로드</div>', unsafe_allow_html=True)
    try:
        excel_bytes = to_excel_bytes(sub_df)
        st.download_button(
            "📥 선택 기간 Excel 다운로드 (.xlsx)",
            data=excel_bytes,
            file_name=f"gh_result_{start_dt.strftime('%Y%m%d%H')}_{end_dt.strftime('%Y%m%d%H')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Excel 저장 실패 (openpyxl 필요): {e}")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — 시점 분석
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📅 날짜·시간 선택</div>', unsafe_allow_html=True)
    cd, ch = st.columns([2, 1])
    with cd:
        # 디폴트: 첫날
        sel_date = st.date_input("날짜 선택", value=d_min, min_value=d_min, max_value=d_max, key="t2_date")
    with ch:
        # 디폴트: 첫 시간(index=0)
        sel_hour = st.selectbox("시각 선택", list(range(24)), index=0,
                                format_func=lambda h: f"{h:02d}:00", key="t2_hour")

    sel_dt      = datetime.combine(sel_date, time(sel_hour))
    nearest_idx = (result_df[TIME_COL] - sel_dt).abs().idxmin()
    row         = result_df.loc[nearest_idx]
    actual_dt   = row[TIME_COL]

    st.markdown(f"""<div class="info-box">
    선택 시각: <b>{sel_dt.strftime('%Y-%m-%d %H:%M')}</b> &nbsp;→&nbsp;
    실제 데이터 시각: <b>{actual_dt.strftime('%Y-%m-%d %H:%M')}</b>
    </div>""", unsafe_allow_html=True)

    # ── 예측값 카드 ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔮 예측값 (모델 계산)</div>', unsafe_allow_html=True)
    pred_main_items = [
        ("Troom_C", "Indoor Temp (Pred)", "°C"),
        ("RHin_%",  "Indoor RH (Pred)",   "%"),
        ("Cin_ppm", "Indoor CO₂ (Pred)",  "ppm"),
    ]
    cols_p = st.columns(3)
    for col_obj, (key, label, unit) in zip(cols_p, pred_main_items):
        val = row.get(key, None)
        with col_obj:
            st.markdown(pred_card_html(label, val if pd.notna(val) else None, unit),
                        unsafe_allow_html=True)

    with st.expander("📋 예측 세부 항목 더보기"):
        skip_keys_pt = {"Troom_C","RHin_%","Cin_ppm","active_covers","cond_surface_name",
                        "areaHouse_m2","cropArea_m2","RHs_solution","Ld_m","Lw_m"}
        extra_keys = [k for k in PARAM_INFO if k not in skip_keys_pt]
        n_ec = 4
        for i in range(0, len(extra_keys), n_ec):
            batch = extra_keys[i:i+n_ec]
            cols_e = st.columns(n_ec)
            for col_obj, k in zip(cols_e, batch):
                info = PARAM_INFO.get(k, (k,"",""))
                val  = row.get(k, None)
                if val is not None and pd.notna(val):
                    with col_obj:
                        try:
                            display_val = fmt(float(val))
                        except Exception:
                            display_val = str(val)
                        st.markdown(
                            f'<div class="stat-card">'
                            f'<div class="stat-label">{info[0]}</div>'
                            f'<div class="stat-value">{display_val}</div>'
                            f'<div class="stat-label">{info[1]}</div></div>',
                            unsafe_allow_html=True,
                        )

    # ── 실측값 — 상단 큰 카드 ─────────────────────────────────────────
    st.markdown('<div class="section-header">📡 실측값 (관측 데이터)</div>', unsafe_allow_html=True)

    obs_main_avail = [(k, lbl, unit) for k, lbl, unit in OBS_MAIN
                      if k in row.index and pd.notna(row.get(k))]
    if obs_main_avail:
        cols_om = st.columns(len(obs_main_avail))
        for col_obj, (k, lbl, unit) in zip(cols_om, obs_main_avail):
            with col_obj:
                st.markdown(obs_main_card_html(lbl, row[k], unit), unsafe_allow_html=True)

    # 하단 작은 기상 카드 (강수량 포함)
    obs_meta_avail = [(k, lbl, unit) for k, lbl, unit in OBS_META
                      if k in row.index and pd.notna(row.get(k))]
    if obs_meta_avail:
        st.markdown("<br>", unsafe_allow_html=True)
        cols_mm = st.columns(len(obs_meta_avail))
        for col_obj, (k, lbl, unit) in zip(cols_mm, obs_meta_avail):
            with col_obj:
                st.markdown(obs_meta_card_html(lbl, row[k], unit), unsafe_allow_html=True)

    # ── 예측 vs 실측 비교 카드 ────────────────────────────────────────
    cmp_pairs_pt = [
        ("Troom_C","내부온도","Indoor Temp","°C"),
        ("RHin_%", "내부습도","Indoor RH",  "%"),
        ("Cin_ppm","내부CO2", "Indoor CO₂","ppm"),
    ]
    valid_cmp = [(pk,ok,lbl,unit) for pk,ok,lbl,unit in cmp_pairs_pt
                 if pk in row.index and ok in row.index and pd.notna(row.get(ok))]
    if valid_cmp:
        st.markdown('<div class="section-header">⚖️ 예측 vs 실측 비교</div>', unsafe_allow_html=True)
        cols_cmp = st.columns(len(valid_cmp))
        for col_obj, (pk, ok, lbl, unit) in zip(cols_cmp, valid_cmp):
            pv, ov = row.get(pk), row.get(ok)
            if pd.notna(pv) and pd.notna(ov):
                diff = float(pv) - float(ov)
                pct  = diff / float(ov) * 100 if abs(float(ov)) > 1e-6 else 0.0
                arrow = "▲" if diff > 0 else "▼" if diff < 0 else "●"
                clr   = "#e63946" if diff > 0 else "#0077b6" if diff < 0 else "#555"
                with col_obj:
                    st.markdown(f"""<div class="cmp-card">
                        <div style="font-weight:700;font-size:.9rem;color:#1b4332;margin-bottom:.4rem;">{lbl}</div>
                        <div style="display:flex;justify-content:space-around;align-items:center;">
                          <div style="text-align:center;">
                            <div style="font-size:.68rem;color:#555;">예측</div>
                            <div style="font-size:1.3rem;font-weight:800;color:#1b4332;">{fmt(float(pv))}</div>
                          </div>
                          <div style="font-size:1.4rem;color:{clr};">{arrow}</div>
                          <div style="text-align:center;">
                            <div style="font-size:.68rem;color:#555;">실측</div>
                            <div style="font-size:1.3rem;font-weight:800;color:#7f3600;">{fmt(float(ov))}</div>
                          </div>
                        </div>
                        <div style="font-size:.75rem;color:{clr};margin-top:.35rem;">
                          오차: {fmt(diff)} {unit} ({pct:+.1f}%)</div>
                    </div>""", unsafe_allow_html=True)

    # ── ±12h 추이 — 가로 3개 나란히 ─────────────────────────────────
    st.markdown('<div class="section-header">📈 선택 시각 ±12시간 추이</div>', unsafe_allow_html=True)
    w_df = result_df[
        (result_df[TIME_COL] >= sel_dt - timedelta(hours=12)) &
        (result_df[TIME_COL] <= sel_dt + timedelta(hours=12))
    ]

    if not w_df.empty:
        mini_cfg = [
            ("Troom_C", "내부온도",  "Indoor Temp", "°C",  "#e63946", "#000000"),
            ("RHin_%",  "내부습도",  "Indoor RH",   "%",   "#0077b6", "#000000"),
            ("Cin_ppm", "내부CO2",   "Indoor CO₂",  "ppm", "#2d6a4f", "#000000"),
        ]
        g1, g2, g3 = st.columns(3)
        col_objs = [g1, g2, g3]

        for col_obj, (pk, obs_k, title, unit, pred_clr, obs_clr) in zip(col_objs, mini_cfg):
            fig_m = go.Figure()
            if pk in w_df.columns:
                fig_m.add_trace(go.Scatter(
                    x=w_df[TIME_COL], y=w_df[pk],
                    mode="lines", name="Predicted",
                    line=dict(color=pred_clr, width=2),
                ))
            if obs_k in w_df.columns and w_df[obs_k].notna().any():
                fig_m.add_trace(go.Scatter(
                    x=w_df[TIME_COL], y=w_df[obs_k],
                    mode="lines", name="Observed",
                    line=dict(color=obs_clr, width=1.8, ),
                ))
            fig_m.add_vline(x=sel_dt, line_dash="dash", line_color="red", opacity=0.75)
            fig_m.update_layout(
                title=dict(text=f"{title} [{unit}]", font=dict(size=12)),
                height=300, margin=dict(l=48, r=10, t=42, b=35),
                legend=dict(orientation="h", y=1.2, x=0, font=dict(size=10)),
                hovermode="x unified", plot_bgcolor="#f9fffe",
                yaxis_title=unit,
                xaxis=dict(tickformat="%m/%d\n%H:%M", tickangle=-30),
            )
            with col_obj:
                st.plotly_chart(fig_m, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("📌 모델: FvCB 광합성 (Medlyn 기공) + Penman-Monteith 증산 + 다층 커버 열전달 (PO필름/AL스크린/보온커튼) | 온실: 80×48×6 m | 작물: 상추")