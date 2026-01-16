import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

in_path = "climate_data_2.xlsx"  # 시간별 전북 정읍시 기상 데이터 엑셀 파일

df = pd.read_excel(in_path)

col_map = {}
if "일시" in df.columns:
    col_map["일시"] = "datetime"
elif "시간" in df.columns:
    col_map["시간"] = "datetime"
elif "date" in df.columns:
    col_map["date"] = "datetime"

if "기온" in df.columns:
    col_map["기온"] = "Tout_C"
elif "평균 기온" in df.columns:
    col_map["평균 기온"] = "Tout_C"
elif "외기온" in df.columns:
    col_map["외기온"] = "Tout_C"

if "일사" in df.columns:
    col_map["일사"] = "G_MJ_m2_hr"
elif "일사량" in df.columns:
    col_map["일사량"] = "G_MJ_m2_hr"
elif "합계 일사량(MJ/m2)" in df.columns:
    col_map["합계 일사량(MJ/m2)"] = "G_MJ_m2_hr"

df = df.rename(columns=col_map)

required = ["datetime", "Tout_C", "G_MJ_m2_hr"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"필수 컬럼이 없습니다: {missing}\n현재 컬럼: {list(df.columns)}")

df["datetime"] = pd.to_datetime(df["datetime"])

eta_sc = 0.6       # 태양열 집열기 효율
Us = 0.20          # 샌드위치패널 U [W/m2K]
Un = 4.77          # 부직포 U [W/m2K]

tau_film = 0.71    # 플라스틱 필름의 광투과율

Agh_air = 66.0      # m2
Agh_ground = 134.0  # m2

Twater_C = 30.0     # 축열조 물 온도 [°C]

Qdryer_kW = 7.0              # 건조기 히터용량(필요 열량) [kWth]
A_panel = 2.0                # 집열판 1장 유효 면적 [m2]
price_per_panel = 1_000_000  # 집열판 1장 가격 [원]

# 여름철(6~8월)만 매일 가동
summer_months = [6, 7, 8]
dryer_on = df["datetime"].dt.month.isin(summer_months)

# 지중온도
def tground_from_month(m: int) -> float:
    if m in [12, 1, 2]:   # winter
        return 10.0
    if m in [3, 4, 5]:    # spring
        return 15.0
    if m in [6, 7, 8]:    # summer
        return 20.0
    if m in [9, 10, 11]:  # fall
        return 15.0
    return np.nan

df["Tground_C"] = df["datetime"].dt.month.map(tground_from_month)

# 재료 물성
k_film = 0.28        # W/mK
L_film = 0.0001      # m  (0.1 mm)
k_fabric = 0.035     # W/mK
L_fabric = 0.005     # m  (0.5 cm)

# 대류계수 가정(자연대류)
h_air_nat = 15.0      # W/m2K
h_water = 50.0      # W/m2K

R_cover = (1.0 / h_air_nat) + (L_film / k_film) + (1.0 / h_air_nat)
R_fabric = (1.0 / h_air_nat) + (L_fabric / k_fabric) + (1.0 / h_water)

df["Tin_C"] = (R_fabric * df["Tout_C"] + R_cover * Twater_C) / (R_cover + R_fabric)

df["G_kWh_m2_hr"] = df["G_MJ_m2_hr"] / 3.6
df["G_W_m2_avg"] = df["G_MJ_m2_hr"] * 1e6 / 3600.0

df["G_in_kWh_m2_hr"] = df["G_kWh_m2_hr"] * tau_film
df["G_in_W_m2_avg"]  = df["G_W_m2_avg"]  * tau_film

df["dT_ground_K"] = (Twater_C - df["Tground_C"]).clip(lower=0)
df["dT_air_K"] = (Twater_C - df["Tin_C"]).clip(lower=0)

df["Qloss_ground_W"] = Us * Agh_ground * df["dT_ground_K"]
df["Qloss_air_W"] = Un * Agh_air * df["dT_air_K"]
df["Qloss_total_W"] = df["Qloss_ground_W"] + df["Qloss_air_W"]

df["Qloss_ground_kWh_hr"] = df["Qloss_ground_W"] / 1000.0
df["Qloss_air_kWh_hr"] = df["Qloss_air_W"] / 1000.0
df["Qloss_total_kWh_hr"] = df["Qloss_total_W"] / 1000.0

df["EWT_C"] = Twater_C
df["COP_h"] = 3.01 + 0.062 * df["EWT_C"]
df["COP_h"] = df["COP_h"].clip(lower=1.1)

# 건조기 필요 열량(if summer = 7,else 0)
df["Qdryer_kWh_hr"] = np.where(dryer_on, Qdryer_kW, 0.0)

# 히트펌프 소비전력
df["PHP_kW"] = df["Qdryer_kWh_hr"] / df["COP_h"]

# Qhp: 축열조에서 히트펌프가 사용하는 열
df["Qhp_from_tank_kWh_hr"] = df["Qdryer_kWh_hr"] - df["PHP_kW"]

# 축열조 총 에너지 부하(열 손실)
df["Qtank_total_kWh_hr"] = df["Qloss_total_kWh_hr"] + df["Qhp_from_tank_kWh_hr"]

den = eta_sc * df["G_in_kWh_m2_hr"]
df["Asc_m2"] = np.nan
mask_posG = df["G_in_kWh_m2_hr"] > 0
df.loc[mask_posG, "Asc_m2"] = df.loc[mask_posG, "Qtank_total_kWh_hr"] / den.loc[mask_posG]

df["date_only"] = df["datetime"].dt.date

mask_200 = df["G_in_W_m2_avg"] >= 200

daily_asc_eff = (
    df.loc[mask_200]
      .groupby("date_only")["Asc_m2"]
      .mean()
      .reset_index(name="Asc_eff_daily_mean")
)

daily_asc_eff["date_dt"] = pd.to_datetime(daily_asc_eff["date_only"])
daily_asc_eff = daily_asc_eff.sort_values("date_dt")

monthly_asc_eff = (
    daily_asc_eff
      .set_index("date_dt")["Asc_eff_daily_mean"]
      .resample("MS")
      .mean()
      .reset_index()
)
monthly_asc_eff = monthly_asc_eff.rename(columns={"date_dt": "month_dt", "Asc_eff_daily_mean": "Asc_eff_monthly_mean"})

plt.figure()

plt.bar(
    monthly_asc_eff["month_dt"],
    monthly_asc_eff["Asc_eff_monthly_mean"],
    width=25,
    color="red",
    alpha=0.35,
    label="Monthly mean"
)

plt.plot(
    daily_asc_eff["date_dt"],
    daily_asc_eff["Asc_eff_daily_mean"],
    linestyle="--",
    color="black",
    linewidth=1.2,
    label="Daily mean"
)

plt.xlabel("Date")
plt.ylabel("Asc [m2]")
plt.title("Asc Graph")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

daily_mean_asc_200 = (
    df.loc[mask_200]
      .groupby("date_only")["Asc_m2"]
      .mean()
)

df["Asc_mean_Gge200_daily"] = np.nan
midnight_mask = df["datetime"].dt.hour.eq(0) & df["datetime"].dt.minute.eq(0)
df.loc[midnight_mask, "Asc_mean_Gge200_daily"] = df.loc[midnight_mask, "date_only"].map(daily_mean_asc_200)

out = df[[
    "datetime",
    "Tout_C", "Tin_C", "Tground_C",
    "G_W_m2_avg",
    "G_in_W_m2_avg",
    "G_kWh_m2_hr",
    "G_in_kWh_m2_hr",
    "Qloss_ground_kWh_hr", "Qloss_air_kWh_hr", "Qloss_total_kWh_hr",
    "Qdryer_kWh_hr", "COP_h", "PHP_kW", "Qhp_from_tank_kWh_hr", "Qtank_total_kWh_hr",
    "Asc_m2",
    "Asc_mean_Gge200_daily",
]].copy()

out = out.rename(columns={
    "datetime": "Date&Time",
    "Tout_C": "Tout[℃]",
    "Tin_C": "Tin[℃]",
    "Tground_C": "Tground[℃]",

    "G_W_m2_avg": "G_out[W/m2]",
    "G_in_W_m2_avg": "G_in[W/m2]",
    "G_kWh_m2_hr": "G_out[kWh/m2h]",
    "G_in_kWh_m2_hr": "G_in[kWh/m2h]",

    "Qloss_ground_kWh_hr": "Qloss_ground[kWh/h]",
    "Qloss_air_kWh_hr": "Qloss_air[kWh/h]",
    "Qloss_total_kWh_hr": "Qloss_total[kWh/h]",

    "Qdryer_kWh_hr": "Qdryer_out[kWh/h](60C)",
    "COP_h": "COP_h[-]",
    "PHP_kW": "Php[kW]",
    "Qhp_from_tank_kWh_hr": "Qhp_from_tank[kWh/h](30C)",
    "Qtank_total_kWh_hr": "Qtank_total[kWh/h]",

    "Asc_m2": "Asc[m2]",
    "Asc_mean_Gge200_daily": "Asc_eff[m2]",
})

base_name = "solar_collector_area_jeongeup_2025.xlsx"
out_path = base_name

if os.path.exists(out_path):
    i = 1
    while True:
        candidate = base_name.replace(".xlsx", f"_v{i}.xlsx")
        if not os.path.exists(candidate):
            out_path = candidate
            break
        i += 1

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    out.to_excel(writer, index=False, sheet_name="Hourly_Area")

    const = pd.DataFrame({
        "parameter": [
            "eta_sc",
            "Us_W_m2K",
            "Un_W_m2K",
            "Agh_air_m2",
            "Agh_ground_m2",
            "Twater_C",
            "Qdryer_kWth",
            "dryer_on_months",
            "COP_formula(Method1)",
            "A_panel_m2_per_panel",
            "price_per_panel_KRW",
            "k_film_W_mK",
            "L_film_m",
            "k_fabric_W_mK",
            "L_fabric_m",
            "h_air_nat_W_m2K",
            "h_water_W_m2K",
            "G_unit_assumption",
            "R_cover_m2K_W",
            "R_fabric_m2K_W",
            "Tground_winter_C(Dec-Jan-Feb)",
            "Tground_spring_C(Mar-Apr-May)",
            "Tground_summer_C(Jun-Jul-Aug)",
            "Tground_fall_C(Sep-Oct-Nov)",
            "Night_handling",
            "Asc_daily_mean_condition",
        ],
        "value": [
            eta_sc, Us, Un, Agh_air, Agh_ground, Twater_C,
            Qdryer_kW,
            str(summer_months),
            "COP = 3.01 + 0.062 * EWT, EWT ~ Twater_C",
            A_panel, price_per_panel,
            k_film, L_film, k_fabric, L_fabric, h_air_nat, h_water,
            "G_MJ_m2_hr is MJ/m2 per hour (input)",
            R_cover, R_fabric,
            10, 15, 20, 15,
            "Method B: if G==0 then Asc=NaN (not calculated)",
            "Daily mean Asc computed using hours with G_in_W_m2 >= 200 (after film), stored at 00:00 row",
        ]
    })
    const.to_excel(writer, index=False, sheet_name="Constants")

print("Saved:", out_path)

# 겨울/여름 평균 집열판 면적 각각 산출
winter_months = [12, 1, 2]

mask_winter_200 = df["datetime"].dt.month.isin(winter_months) & (df["G_in_W_m2_avg"] >= 200)
mask_summer_200 = df["datetime"].dt.month.isin(summer_months) & (df["G_in_W_m2_avg"] >= 200)

Asc_design_winter = df.loc[mask_winter_200, "Asc_m2"].mean()
Asc_design_summer = df.loc[mask_summer_200, "Asc_m2"].mean()

N_winter = int(np.ceil(Asc_design_winter / A_panel)) if np.isfinite(Asc_design_winter) else np.nan
N_summer = int(np.ceil(Asc_design_summer / A_panel)) if np.isfinite(Asc_design_summer) else np.nan

Cost_winter = N_winter * price_per_panel if np.isfinite(N_winter) else np.nan
Cost_summer = N_summer * price_per_panel if np.isfinite(N_summer) else np.nan

print("\n===== Panel Purchase Result (Seasonal means, G>=200) =====")
print(f"Winter months {winter_months}: Design Asc = {Asc_design_winter:.2f} m^2 -> Panels = {N_winter} -> Cost = {Cost_winter:,} KRW")
print(f"Summer months {summer_months}: Design Asc = {Asc_design_summer:.2f} m^2 -> Panels = {N_summer} -> Cost = {Cost_summer:,} KRW")

# Parameter Graphs

#  6개 주요 파라미터 민감도 분석(Tornado Plot)

def compute_design_asc(
    df_in: pd.DataFrame,
    eta_sc_local: float,
    Us_local: float,
    Un_local: float,
    Twater_C_local: float,
    Qdryer_kW_local: float,
    dryer_months_local: list
):
    Tout = df_in["Tout_C"].to_numpy(dtype=float)
    Tground = df_in["Tground_C"].to_numpy(dtype=float)
    G_kWh = df_in["G_in_kWh_m2_hr"].to_numpy(dtype=float)

    Tin = (R_fabric * Tout + R_cover * Twater_C_local) / (R_cover + R_fabric)

    dT_ground = np.clip(Twater_C_local - Tground, 0.0, None)
    dT_air = np.clip(Twater_C_local - Tin, 0.0, None)

    Qloss_ground_kWh_hr_local = (Us_local * Agh_ground * dT_ground) / 1000.0
    Qloss_air_kWh_hr_local = (Un_local * Agh_air * dT_air) / 1000.0
    Qloss_total_kWh_hr_local = Qloss_ground_kWh_hr_local + Qloss_air_kWh_hr_local

    COP_h_local = 3.01 + 0.062 * Twater_C_local
    COP_h_local = np.clip(COP_h_local, 1.1, None)

    dryer_on_local = df_in["datetime"].dt.month.isin(dryer_months_local).to_numpy()
    Qdryer_kWh_hr_local = np.where(dryer_on_local, Qdryer_kW_local, 0.0)

    PHP_kW_local = Qdryer_kWh_hr_local / COP_h_local
    Qhp_from_tank_kWh_hr_local = Qdryer_kWh_hr_local - PHP_kW_local

    Qtank_total_kWh_hr_local = Qloss_total_kWh_hr_local + Qhp_from_tank_kWh_hr_local

    den_local = eta_sc_local * G_kWh
    Asc_local = np.full_like(G_kWh, np.nan, dtype=float)
    mask_pos = G_kWh > 0
    Asc_local[mask_pos] = Qtank_total_kWh_hr_local[mask_pos] / den_local[mask_pos]

    mask_g200 = df_in["G_in_W_m2_avg"].to_numpy(dtype=float) >= 200.0
    month = df_in["datetime"].dt.month.to_numpy()

    winter_months_local = [12, 1, 2]
    summer_months_local = dryer_months_local

    mask_winter = np.isin(month, winter_months_local) & mask_g200
    mask_summer = np.isin(month, summer_months_local) & mask_g200

    Asc_design_winter_local = np.nanmean(Asc_local[mask_winter]) if np.any(mask_winter) else np.nan
    Asc_design_summer_local = np.nanmean(Asc_local[mask_summer]) if np.any(mask_summer) else np.nan

    return Asc_design_winter_local, Asc_design_summer_local

base_w, base_s = compute_design_asc(
    df,
    eta_sc_local=eta_sc,
    Us_local=Us,
    Un_local=Un,
    Twater_C_local=Twater_C,
    Qdryer_kW_local=Qdryer_kW,
    dryer_months_local=summer_months
)

sens = []

# 1) 집열기 효율
sens.append(("eta_sc",
             compute_design_asc(df, eta_sc*(1-0.2), Us, Un, Twater_C, Qdryer_kW, summer_months),
             compute_design_asc(df, eta_sc*(1+0.2), Us, Un, Twater_C, Qdryer_kW, summer_months)))

# 2) 부직포 U-value
sens.append(("Un",
             compute_design_asc(df, eta_sc, Us, Un*(1-0.2), Twater_C, Qdryer_kW, summer_months),
             compute_design_asc(df, eta_sc, Us, Un*(1+0.2), Twater_C, Qdryer_kW, summer_months)))

# 3) 샌드위치 패널 U-value
sens.append(("Us",
             compute_design_asc(df, eta_sc, Us*(1-0.2), Un, Twater_C, Qdryer_kW, summer_months),
             compute_design_asc(df, eta_sc, Us*(1+0.2), Un, Twater_C, Qdryer_kW, summer_months)))

# 4) 건조기 열 공급량
sens.append(("Qdryer_kW",
             compute_design_asc(df, eta_sc, Us, Un, Twater_C, Qdryer_kW*(1-0.2), summer_months),
             compute_design_asc(df, eta_sc, Us, Un, Twater_C, Qdryer_kW*(1+0.2), summer_months)))

# 5) 축열조 물 온도
sens.append(("Twater_C(30↔60)",
             compute_design_asc(df, eta_sc, Us, Un, 30.0, Qdryer_kW, summer_months),
             compute_design_asc(df, eta_sc, Us, Un, 60.0, Qdryer_kW, summer_months)))

# 6) 여름철 건조기 가동 시간 조정
dryer_low = [7, 8]
dryer_high = [5, 6, 7, 8, 9]
sens.append(("dryer_on_months",
             compute_design_asc(df, eta_sc, Us, Un, Twater_C, Qdryer_kW, dryer_low),
             compute_design_asc(df, eta_sc, Us, Un, Twater_C, Qdryer_kW, dryer_high)))

labels = [s[0] for s in sens]

label_map = {
    "eta_sc": "집열기 효율",
    "Twater_C(30↔60)": "축열조(물) 온도",
    "Qdryer_kW": "건조기 공급 열량",
    "Un": "부직포 U-value",
    "Us": "샌드위치 패널 U-value",
    "dryer_on_months": "건조기 가동 시간",
}

low_w  = np.array([s[1][0] for s in sens], dtype=float)
high_w = np.array([s[2][0] for s in sens], dtype=float)

low_s  = np.array([s[1][1] for s in sens], dtype=float)
high_s = np.array([s[2][1] for s in sens], dtype=float)

def heatmap_sensitivity(labels, base_w, low_w, high_w, base_s, low_s, high_s):

    winter_low_pct  = (low_w  - base_w) / base_w * 100.0
    winter_high_pct = (high_w - base_w) / base_w * 100.0
    summer_low_pct  = (low_s  - base_s) / base_s * 100.0
    summer_high_pct = (high_s - base_s) / base_s * 100.0

    M = np.vstack([winter_low_pct, winter_high_pct, summer_low_pct, summer_high_pct]).T
    cols = ["겨울철 -20%", "겨울철 +20%", "여름철 -20%", "여름철 +20%"]

    impact = np.max(np.abs(M), axis=1)
    order = np.argsort(impact)[::-1]
    M = M[order, :]
    labels_sorted = [label_map.get(labels[i], labels[i]) for i in order]

    plt.figure(figsize=(9, 4.8))
    im = plt.imshow(M, aspect="auto", cmap="YlOrRd", vmin=np.nanmin(M), vmax=np.nanmax(M))
    plt.colorbar(im, label="태양열 집열판 면적 변화 (%)")
    plt.xticks(np.arange(len(cols)), cols, rotation=15, ha="right")
    plt.yticks(np.arange(len(labels_sorted)), labels_sorted)
    plt.title("주요 파라미터 민감도 히트맵(±20%)")
    plt.tight_layout()
    plt.show()

if not np.isfinite(base_w) or not np.isfinite(base_s) or base_w == 0 or base_s == 0:
    print("[Heatmap] Baseline 설계면적(base_w 또는 base_s)이 NaN/0 입니다. (G>=200 데이터 부족 가능)")
else:
    heatmap_sensitivity(labels, base_w, low_w, high_w, base_s, low_s, high_s)

def compute_asc_timeseries(df_in: pd.DataFrame, Twater_C_local: float, Qdryer_kW_local: float) -> pd.Series:
    Tout = df_in["Tout_C"].to_numpy(dtype=float)
    Tground = df_in["Tground_C"].to_numpy(dtype=float)
    G_kWh = df_in["G_in_kWh_m2_hr"].to_numpy(dtype=float)

    Tin = (R_fabric * Tout + R_cover * Twater_C_local) / (R_cover + R_fabric)

    dT_ground = np.clip(Twater_C_local - Tground, 0.0, None)
    dT_air = np.clip(Twater_C_local - Tin, 0.0, None)

    Qloss_ground_kWh_hr_local = (Us * Agh_ground * dT_ground) / 1000.0
    Qloss_air_kWh_hr_local = (Un * Agh_air * dT_air) / 1000.0
    Qloss_total_kWh_hr_local = Qloss_ground_kWh_hr_local + Qloss_air_kWh_hr_local

    COP_h_local = 3.01 + 0.062 * Twater_C_local
    COP_h_local = np.clip(COP_h_local, 1.1, None)

    dryer_on_local = df_in["datetime"].dt.month.isin(summer_months).to_numpy()
    Qdryer_kWh_hr_local = np.where(dryer_on_local, Qdryer_kW_local, 0.0)

    PHP_kW_local = Qdryer_kWh_hr_local / COP_h_local
    Qhp_from_tank_kWh_hr_local = Qdryer_kWh_hr_local - PHP_kW_local

    Qtank_total_kWh_hr_local = Qloss_total_kWh_hr_local + Qhp_from_tank_kWh_hr_local

    den_local = eta_sc * G_kWh
    Asc_local = np.full_like(G_kWh, np.nan, dtype=float)
    mask_pos = G_kWh > 0
    Asc_local[mask_pos] = Qtank_total_kWh_hr_local[mask_pos] / den_local[mask_pos]
    return pd.Series(Asc_local, index=df_in.index, name="Asc_param_m2")

# 축열조 물 온도 vs 집열판 면적 그래프
twater_list = [30.0, 40.0, 50.0, 60.0]
asc_mean_by_twater = []

for tw in twater_list:
    Asc_tw = compute_asc_timeseries(df, Twater_C_local=tw, Qdryer_kW_local=Qdryer_kW)
    Asc_tw_valid = Asc_tw.where(mask_200).replace([np.inf, -np.inf], np.nan).dropna()
    asc_mean_by_twater.append(float(Asc_tw_valid.mean()) if len(Asc_tw_valid) > 0 else np.nan)

plt.figure()
plt.plot(twater_list, asc_mean_by_twater, marker="o", linestyle="-", color='black')
plt.xlabel("축열조 물 온도 [℃]")
plt.ylabel("평균 집열판 면적 [m2]")
plt.title("축열조 물 온도 대비 평균 집열판 면적 그래프")
plt.grid(False)
plt.tight_layout()
plt.show()

# 건조기 공급량 vs 집열판 면적 그래프
qdryer_list = list(np.arange(0.0, 14.0 + 1.0, 1.0))

asc_mean_by_qdryer = []
asc_mean_by_qdryer_summer = []

for qd in qdryer_list:
    Asc_qd = compute_asc_timeseries(df, Twater_C_local=Twater_C, Qdryer_kW_local=float(qd))

    Asc_all_valid = Asc_qd.where(mask_200).replace([np.inf, -np.inf], np.nan).dropna()
    asc_mean_by_qdryer.append(float(Asc_all_valid.mean()) if len(Asc_all_valid) > 0 else np.nan)

    Asc_summer_valid = Asc_qd.where(mask_summer_200).replace([np.inf, -np.inf], np.nan).dropna()
    asc_mean_by_qdryer_summer.append(float(Asc_summer_valid.mean()) if len(Asc_summer_valid) > 0 else np.nan)

plt.figure()
plt.plot(qdryer_list, asc_mean_by_qdryer, marker="o", linestyle="-", label="전체 기간", color='red')
plt.plot(qdryer_list, asc_mean_by_qdryer_summer, marker="o", linestyle="--", label="여름철", color='black')
plt.xlabel("건조기 공급량 [kWth]")
plt.ylabel("평균 집열판 면적 [m2]")
plt.title(f"건조기 공급량 대비 평균 집열판 면적 그래프")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()
