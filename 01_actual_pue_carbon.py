"""
=============================================================================
SCRIPT 1: GOOGLE ACTUAL PUE — CARBON & ELECTRICITY COST ANALYSIS
=============================================================================
Article section: "To understand the actual scale of this cost..."

What this script does:
  1. Loads Google's actual 2025 facility-level PUE data
  2. Computes cooling overhead, electricity cost, CO2, and carbon cost
     for ALL 29 active Google facilities (2025 data)
     [Note: "New Albany, Ohio" is a legacy name superseded by the three
      "Central Ohio (Lancaster/Columbus/New Albany)" entries as of 2024 Q3]
  3. Computes the same for the 10 growing data center markets
     (using predicted PUE from regression models — see Script 2)
  4. Exports results to CSV and generates charts

Inputs required (same folder):
  google_datacenter_pue_final.csv   — Google's quarterly facility PUE data
  model_c.pkl                       — Regression Model C (run Script 2 first)
  model_d.pkl                       — Regression Model D (run Script 2 first)

Outputs:
  google_all_facilities_results.csv — All 29 Google facilities, all metrics
  growing_markets_results.csv       — 10 growing markets, all metrics
  pue_carbon_analysis.png           — Charts

Run order:
  python3 02_regression_analysis.py   (generates model_c.pkl, model_d.pkl)
  python3 01_actual_pue_carbon.py     (this script)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib, warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1 — CONSTANTS AND ASSUMPTIONS
# Mirrors the article's methodology disclosure
# =============================================================================

IT_LOAD_MW     = 100    # Standard hyperscale campus assumption
HOURS_PER_YEAR = 8760   # Hours in a year
EU_ETS_PRICE   = 70.37  # EU ETS price, World Bank Carbon Pricing Dashboard
TREE_KG_CO2_YR = 21     # kg CO2 absorbed per mature tree (20+ yrs) per year
                        # Source: Arbor Day Foundation / USDA Forest Service

print("=" * 70)
print("SCRIPT 1: GOOGLE ACTUAL PUE — CARBON & ELECTRICITY COST ANALYSIS")
print("=" * 70)
print(f"\n  Assumptions:")
print(f"    IT load:          {IT_LOAD_MW} MW (standard hyperscale campus)")
print(f"    EU ETS benchmark: ${EU_ETS_PRICE}/ton CO2 (World Bank Carbon Pricing Dashboard)")
print(f"    Tree absorption:  {TREE_KG_CO2_YR} kg CO2/yr at maturity (20+ years)")


# =============================================================================
# SECTION 2 — CARBON PRICES BY COUNTRY
# Source: World Bank Carbon Pricing Dashboard
# Note: US state-level grid intensities used where available (EIA eGRID)
# =============================================================================

CARBON_PRICES = {
    # EU — national carbon tax stacked on top of EU ETS
    "Denmark":              178.80,
    "Netherlands":          165.20,
    "Ireland":              138.87,
    "Finland":              137.26,
    "Belgium":               70.37,   # EU ETS only (no national top-up)
    # Non-EU with carbon pricing
    "Chile":                  5.00,   # Chile national carbon tax (World Bank 2024)
    "Singapore":             18.62,
    "South Korea":            6.45,
    "Japan":                  1.93,
    "Indonesia":              0.72,
    # No federal carbon price
    "United States":          0.00,
    "Taiwan":                 0.00,
    "Saudi Arabia":           0.00,
    "United Arab Emirates":   0.00,
    "India":                  0.00,
    "Thailand":               0.00,
    "Malaysia":               0.00,
}

# State-level grid carbon intensities for US facilities (gCO2/kWh)
# Source: EIA eGRID 2023/2024 — replaces US national average (383.8)
# These are passed directly to ALL_FACILITIES below, not used for lookup.
# Kept here as a documented reference for each state's source.
STATE_GRID_REFERENCE = {
    # Original article facilities
    "Midlothian, Texas":               370.0,   # ERCOT 2024
    "Council Bluffs, Iowa":            286.0,   # Iowa 63% wind, EIA 2024
    "Council Bluffs, Iowa (2nd)":      286.0,   # Same subgrid
    "Mayes County, Oklahoma":          297.0,   # Oklahoma EIA 2023
    # Extended facilities
    "Central Ohio (Lancaster), Ohio":  390.0,   # Ohio RFCW subregion, EIA 2024
    "Central Ohio (Columbus), Ohio":   390.0,   # Same subgrid
    "Central Ohio (New Albany), Ohio": 390.0,   # Same subgrid
    "Douglas County, Georgia":         350.0,   # Georgia Power, EIA 2024
    "Loudoun County, Virginia":        240.0,   # PJME — nuclear-heavy, EIA 2024
    "Loudoun County, Virginia (2nd)":  240.0,   # Same subgrid
    "Montgomery County, Tennessee":    210.0,   # TVA — nuclear + hydro, EIA 2024
    "Berkeley County, South Carolina": 245.0,   # SC — nuclear dominant, EIA 2024
    "Lenoir, North Carolina":          350.0,   # Duke Energy Carolinas, EIA 2024
    "Jackson County, Alabama":         360.0,   # Alabama Power / Southern, EIA 2024
    "Papillion, Nebraska":             340.0,   # OPPD — coal + nuclear + wind, EIA 2024
    "Henderson, Nevada":               280.0,   # NV Energy / WECC, EIA 2024
    "Storey County, Nevada":           290.0,   # NV Energy / WECC, EIA 2024
    "The Dalles, Oregon":               95.0,   # PNW / BPA — hydro dominant, EIA 2024
    "The Dalles, Oregon (2nd)":         95.0,   # Same subgrid
}


# =============================================================================
# SECTION 3 — HELPER FUNCTIONS
# =============================================================================

def wet_bulb(temp, humidity):
    """Stull (2011) approximation of wet bulb temperature."""
    return (temp * np.arctan(0.151977 * (humidity + 8.313659)**0.5)
            + np.arctan(temp + humidity)
            - np.arctan(humidity - 1.676331)
            + 0.00391838 * humidity**1.5 * np.arctan(0.023101 * humidity)
            - 4.686035)


def calc_metrics(facility, country, actual_pue, elec_price,
                 grid_co2_gkwh, carbon_price_per_ton):
    """
    Given a facility's PUE and data inputs, compute all cost and carbon metrics.

    Formula chain (mirrors article methodology):
      Overhead (MW)  = (PUE - 1) x IT_LOAD_MW
      Overhead (kWh) = Overhead_MW x 1000 x 8760
      Cooling cost   = Overhead_kWh x electricity_price
      CO2 (tons)     = Overhead_kWh x grid_CO2 / 1,000,000
      Carbon today   = CO2_tons x local_carbon_price
      Carbon at ETS  = CO2_tons x EU_ETS_PRICE
      Carbon gap     = Carbon_today - Carbon_at_ETS  (negative = underpaying)
      Trees needed   = CO2_tons x 1000 / 21 kg
    """
    overhead_mw  = (actual_pue - 1) * IT_LOAD_MW
    overhead_kwh = overhead_mw * 1000 * HOURS_PER_YEAR

    cooling_cost = overhead_kwh * elec_price if elec_price else None
    co2_tons     = overhead_kwh * grid_co2_gkwh / 1_000_000
    carbon_today = co2_tons * carbon_price_per_ton
    carbon_ets   = co2_tons * EU_ETS_PRICE
    carbon_gap   = carbon_today - carbon_ets   # negative = undercharging

    trees_needed = co2_tons * 1000 / TREE_KG_CO2_YR

    return {
        "facility":             facility,
        "country":              country,
        "pue":                  round(actual_pue, 3),
        "overhead_mw":          round(overhead_mw, 2),
        "overhead_kwh_M":       round(overhead_kwh / 1e6, 1),
        "elec_price_usd_kwh":   elec_price,
        "cooling_cost_usd":     round(cooling_cost, 0) if cooling_cost else None,
        "grid_co2_gco2_kwh":    grid_co2_gkwh,
        "co2_tons":             round(co2_tons, 0),
        "carbon_price_usd_ton": carbon_price_per_ton,
        "carbon_today_usd":     round(carbon_today, 0),
        "carbon_ets_usd":       round(carbon_ets, 0),
        "carbon_gap_usd":       round(carbon_gap, 0),  # negative = underpaying
        "trees_needed":         round(trees_needed, 0),
    }


# =============================================================================
# SECTION 4 — LOAD GOOGLE 2025 ACTUAL PUE DATA
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: LOADING GOOGLE 2025 ACTUAL PUE DATA")
print("=" * 70)

df = pd.read_csv("google_datacenter_pue_final.csv")
df["wet_bulb_c"] = wet_bulb(df["avg_temp_c"], df["avg_humidity_pct"])

# Average across all 2025 quarters (Q1-Q4)
df_2025 = (df[df["year"] == 2025]
           .groupby(["facility", "country"])
           .agg(avg_pue=("pue", "mean"), quarters=("quarter", "count"),
                electricity=("electricity_business_usd", "mean"))
           .reset_index()
           .sort_values("avg_pue"))

print(f"\n  Facilities with 2025 data: {len(df_2025)}")
print(f"  Quarters per facility:     {df_2025['quarters'].min()}–{df_2025['quarters'].max()} (Q1–Q4)")
print(f"\n  {'Facility':<45} {'Avg PUE':>8} {'Quarters':>9}")
print(f"  {'-'*65}")
for _, r in df_2025.iterrows():
    print(f"  {r['facility']:<45} {r['avg_pue']:>8.3f} {r['quarters']:>9}")


# =============================================================================
# SECTION 5 — COMPUTE METRICS FOR ALL GOOGLE FACILITIES
#
# Tiers:
#   🟢 FAVORABLE      — EU/ETS carbon pricing (at or above EU ETS benchmark)
#   ⬜ US / AMERICAS  — No federal carbon price; varying grid cleanliness
#   🟡 MATERIAL GAP   — Higher PUE or dirty grid; symbolic/zero carbon price
#   🔴 LARGEST GAP    — High PUE, dirty grid, near-zero or zero carbon price
#
# Grid CO2 intensities:
#   US — EIA eGRID 2023/2024 state-level subregions (see STATE_GRID_REFERENCE)
#   Non-US — Our World in Data / Ember / IEA 2024 national averages
# Electricity prices:
#   Non-US — GlobalPetrolPrices business rates
#   US — N/A (state prices not in GlobalPetrolPrices national dataset)
#   Chile — GlobalPetrolPrices business rate ($0.166/kWh)
# Energy mix:
#   Ember / IEA 2024 national/regional averages
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: ALL GOOGLE FACILITIES — FULL CARBON & COST ANALYSIS")
print("(29 facilities with 2025 data — sorted by 2025 avg PUE within tier)")
print("=" * 70)

ALL_FACILITIES = [
    # (facility_name, country, grid_co2_gkwh, elec_price_usd_kwh, primary_energy_source)
    # elec_price = None for US (state prices not in GlobalPetrolPrices dataset)

    # ──────────────────────────────────────────────────────────────────────────
    # 🟢 FAVORABLE — EU/ETS carbon pricing; clean or moderate grid
    # ──────────────────────────────────────────────────────────────────────────
    ("Fredericia, Denmark",             "Denmark",      155.0, 0.234,
     "Wind 58%, Solar 14%, Bioenergy 16%"),
    ("Eemshaven, Netherlands",          "Netherlands",  253.2, 0.220,
     "Natural Gas 40%, Wind 30%, Solar 14%"),
    ("Dublin, Ireland",                 "Ireland",      255.9, 0.270,
     "Natural Gas 50%, Wind 35%, Solar 5%"),
    ("St. Ghislain, Belgium",           "Belgium",      155.0, 0.261,
     "Nuclear 48%, Gas 25%, Wind 15%, Solar 8%"),
    ("Hamina, Finland",                 "Finland",       56.8, 0.124,
     "Nuclear 35%, Hydro 28%, Wind 24%"),

    # ──────────────────────────────────────────────────────────────────────────
    # ⬜ US / AMERICAS — No federal carbon price · Varying grid cleanliness
    # Sorted within this tier by carbon_gap (smallest gap first)
    # ──────────────────────────────────────────────────────────────────────────

    # Clean US grids — PNW hydro and TVA nuclear/hydro keep the gap small
    ("The Dalles, Oregon (2nd)",        "United States",  95.0, None,
     "Hydro 60%, Wind 20%, Natural Gas 13%, Nuclear 4%"),
    ("The Dalles, Oregon",              "United States",  95.0, None,
     "Hydro 60%, Wind 20%, Natural Gas 13%, Nuclear 4%"),
    ("Montgomery County, Tennessee",    "United States", 210.0, None,
     "Nuclear 42%, Gas 24%, Hydro 16%, Coal 10%"),
    ("Loudoun County, Virginia (2nd)",  "United States", 240.0, None,
     "Nuclear 39%, Natural Gas 37%, Wind 10%, Solar 8%"),
    ("Loudoun County, Virginia",        "United States", 240.0, None,
     "Nuclear 39%, Natural Gas 37%, Wind 10%, Solar 8%"),
    ("Berkeley County, South Carolina", "United States", 245.0, None,
     "Nuclear 56%, Natural Gas 22%, Coal 10%, Solar 8%"),
    # Ohio facilities — exceptional PUE offsets the mixed grid
    ("Central Ohio (Lancaster), Ohio",  "United States", 390.0, None,
     "Natural Gas 37%, Coal 34%, Nuclear 14%, Wind 10%"),
    ("Central Ohio (Columbus), Ohio",   "United States", 390.0, None,
     "Natural Gas 37%, Coal 34%, Nuclear 14%, Wind 10%"),
    ("Central Ohio (New Albany), Ohio", "United States", 390.0, None,
     "Natural Gas 37%, Coal 34%, Nuclear 14%, Wind 10%"),
    # Mixed US grids
    ("Henderson, Nevada",               "United States", 280.0, None,
     "Natural Gas 70%, Solar 15%, Hydro 10%"),
    ("Council Bluffs, Iowa (2nd)",      "United States", 286.0, None,
     "Wind 58%, Natural Gas 20%, Nuclear 10%"),
    ("Papillion, Nebraska",             "United States", 340.0, None,
     "Coal 35%, Wind 25%, Nuclear 20%, Gas 15%"),
    ("Douglas County, Georgia",         "United States", 350.0, None,
     "Natural Gas 52%, Nuclear 24%, Coal 11%, Solar 9%"),
    ("Lenoir, North Carolina",          "United States", 350.0, None,
     "Nuclear 36%, Natural Gas 32%, Coal 17%, Solar 11%"),
    ("Jackson County, Alabama",         "United States", 360.0, None,
     "Nuclear 35%, Gas 30%, Coal 23%, Hydro 8%"),
    ("Mayes County, Oklahoma",          "United States", 297.0, None,
     "Natural Gas 40%, Wind 35%, Coal 15%"),
    ("Council Bluffs, Iowa",            "United States", 286.0, None,
     "Wind 58%, Natural Gas 20%, Nuclear 10%"),
    ("Midlothian, Texas",               "United States", 370.0, None,
     "Natural Gas 43%, Wind 26%, Coal 15%"),
    # Americas — Chile has modest carbon tax ($5/ton) but no federal mandate
    ("Quilicura, Chile",                "Chile",         340.0, 0.166,
     "Natural Gas 35%, Coal 28%, Solar 20%, Hydro 12%"),

    # ──────────────────────────────────────────────────────────────────────────
    # 🟡 MATERIAL GAP — Higher PUE / dirty grid / symbolic or zero carbon price
    # ──────────────────────────────────────────────────────────────────────────
    ("Inzai, Japan",                    "Japan",         483.4, 0.202,
     "LNG 35%, Coal 30%, Nuclear 10%"),
    ("Singapore",                       "Singapore",     498.7, 0.265,
     "Natural Gas 95%, Solar 3%"),
    # Storey County NV: regression predicts ~1.09 for this climate — actual 1.145, operational outlier
    ("Storey County, Nevada",           "United States", 290.0, None,
     "Natural Gas 70%, Solar 15%, Hydro 10%"),

    # ──────────────────────────────────────────────────────────────────────────
    # 🔴 LARGEST GAP — High PUE · Dirty grid · Near-zero or zero carbon price
    # ──────────────────────────────────────────────────────────────────────────
    ("Changhua County, Taiwan",         "Taiwan",        635.2, 0.187,
     "Coal 45%, LNG 35%, Nuclear 10%"),
    ("Singapore (2nd)",                 "Singapore",     498.7, 0.265,
     "Natural Gas 95%, Solar 3%"),
]

# Match each facility to its actual 2025 PUE and compute metrics
results_all = []
for facility, country, grid_co2, elec_price, energy_mix in ALL_FACILITIES:
    match = df_2025[df_2025["facility"] == facility]
    if len(match) == 0:
        print(f"  ⚠️  No 2025 data found for: {facility}")
        continue
    actual_pue   = match["avg_pue"].values[0]
    carbon_price = CARBON_PRICES.get(country, 0.0)
    row = calc_metrics(facility, country, actual_pue,
                       elec_price, grid_co2, carbon_price)
    row["primary_energy_source"] = energy_mix
    results_all.append(row)

df_all = pd.DataFrame(results_all)

# Print results grouped by tier
tiers = [
    ("🟢 FAVORABLE — EU/ETS carbon pricing · At or above EU ETS benchmark",
     ["Fredericia, Denmark", "Eemshaven, Netherlands",
      "Dublin, Ireland", "St. Ghislain, Belgium", "Hamina, Finland"]),

    ("⬜ US / AMERICAS — No federal carbon price · Varying grid cleanliness",
     ["The Dalles, Oregon (2nd)", "The Dalles, Oregon",
      "Montgomery County, Tennessee", "Loudoun County, Virginia (2nd)",
      "Loudoun County, Virginia", "Berkeley County, South Carolina",
      "Central Ohio (Lancaster), Ohio", "Central Ohio (Columbus), Ohio",
      "Central Ohio (New Albany), Ohio",
      "Henderson, Nevada", "Council Bluffs, Iowa (2nd)",
      "Papillion, Nebraska", "Douglas County, Georgia",
      "Lenoir, North Carolina", "Jackson County, Alabama",
      "Mayes County, Oklahoma", "Council Bluffs, Iowa",
      "Midlothian, Texas", "Quilicura, Chile"]),

    ("🟡 MATERIAL GAP — Higher PUE · Dirty grid · Symbolic or zero carbon price",
     ["Inzai, Japan", "Singapore", "Storey County, Nevada"]),

    ("🔴 LARGEST GAP — High PUE · Dirty grid · Near-zero or zero carbon price",
     ["Changhua County, Taiwan", "Singapore (2nd)"]),
]

print(f"\n  {'Facility':<38} {'PUE':>5} {'OH(MW)':>7} {'CO2(t)':>8} "
      f"{'C$/ton':>7} {'Today($K)':>10} {'ETS($M)':>8} {'Gap($M)':>8}")
print(f"  {'-'*97}")

for tier_label, facilities in tiers:
    print(f"\n  {tier_label}")
    print(f"  {'-'*97}")
    for fac in facilities:
        r = df_all[df_all["facility"] == fac]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        cool_str  = f"${r['cooling_cost_usd']/1e6:.1f}M" if r["cooling_cost_usd"] else "N/A*"
        today_str = f"${r['carbon_today_usd']/1e3:.0f}K"
        ets_str   = f"${r['carbon_ets_usd']/1e6:.2f}M"
        gap_str   = f"${r['carbon_gap_usd']/1e6:.2f}M"
        print(f"  {r['facility']:<38} {r['pue']:>5.3f} {r['overhead_mw']:>7.1f} "
              f"{r['co2_tons']:>8,.0f} "
              f"${r['carbon_price_usd_ton']:>6.2f} {today_str:>10} "
              f"{ets_str:>8} {gap_str:>8}")

print(f"\n  * US electricity prices vary by state and are not in GlobalPetrolPrices.")
print(f"  Carbon gap = Today − EU ETS: negative = underpaying, positive = overpaying.")
print(f"  Belgium at exactly EU ETS benchmark ($70.37/ton — no national top-up).")
print(f"  Chile carbon tax ~$5/ton applies to emitters above 25,000 tCO2/yr.")
print(f"  Storey County, NV flagged: regression predicts ~1.09 for this climate — actual 1.145, operational outlier.")

# Export
df_all.to_csv("google_all_facilities_results.csv", index=False)
print(f"\n  ✅ Saved: google_all_facilities_results.csv ({len(df_all)} facilities)")


# =============================================================================
# SECTION 6 — PUE SENSITIVITY ANALYSIS
# Article: "A 0.01 improvement in PUE at Changhua County..."
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: PUE SENSITIVITY ANALYSIS")
print("(How much does a 0.01 PUE improvement save?)")
print("=" * 70)

# Changhua Taiwan — worst in article table
changhua = df_all[df_all["facility"] == "Changhua County, Taiwan"].iloc[0]
pue_base = changhua["pue"]
grid_co2 = changhua["grid_co2_gco2_kwh"]
elec     = changhua["elec_price_usd_kwh"]

for delta in [0.01, 0.05, 0.10]:
    pue_new  = pue_base - delta
    oh_new   = (pue_new - 1) * IT_LOAD_MW * 1000 * HOURS_PER_YEAR
    oh_base  = (pue_base - 1) * IT_LOAD_MW * 1000 * HOURS_PER_YEAR
    pct_chg  = (oh_new - oh_base) / oh_base * 100
    co2_save = (oh_base - oh_new) * grid_co2 / 1_000_000
    cost_save= (oh_base - oh_new) * elec
    trees    = co2_save * 1000 / TREE_KG_CO2_YR
    print(f"\n  Changhua PUE {pue_base:.3f} → {pue_new:.3f} (−{delta:.2f}):")
    print(f"    Overhead change:    {pct_chg:+.1f}%")
    print(f"    Cooling cost saved: ${cost_save/1e6:.2f}M/yr")
    print(f"    CO2 avoided:        {co2_save:,.0f} tons/yr")
    print(f"    Trees offset:       {trees:,.0f} mature trees needed annually")

# Hamina vs Changhua full comparison
print(f"\n  Hamina (Finland) vs Changhua (Taiwan) — same 100MW IT load:")
hamina = df_all[df_all["facility"] == "Hamina, Finland"].iloc[0]
print(f"    {'Metric':<30} {'Hamina':>15} {'Changhua':>15} {'Ratio':>8}")
print(f"    {'-'*70}")
for label, h_val, c_val in [
    ("Actual PUE", hamina["pue"], changhua["pue"]),
    ("CO2 from cooling (tons)", hamina["co2_tons"], changhua["co2_tons"]),
    ("Trees needed (M)", hamina["trees_needed"]/1e6, changhua["trees_needed"]/1e6),
]:
    ratio = c_val / h_val if h_val > 0 else 0
    print(f"    {label:<30} {h_val:>15,.1f} {c_val:>15,.1f} {ratio:>7.1f}x")

# Also compare best US grid (Oregon) vs worst US grid (Texas)
print(f"\n  The Dalles, OR (2nd) vs Midlothian, TX — same US grid, no carbon price:")
oregon = df_all[df_all["facility"] == "The Dalles, Oregon (2nd)"].iloc[0]
texas  = df_all[df_all["facility"] == "Midlothian, Texas"].iloc[0]
print(f"    {'Metric':<30} {'Oregon (2nd)':>15} {'Texas':>15} {'Ratio':>8}")
print(f"    {'-'*70}")
for label, o_val, t_val in [
    ("Actual PUE", oregon["pue"], texas["pue"]),
    ("Grid CO2 (gCO2/kWh)", oregon["grid_co2_gco2_kwh"], texas["grid_co2_gco2_kwh"]),
    ("CO2 from cooling (tons)", oregon["co2_tons"], texas["co2_tons"]),
    ("Carbon gap vs ETS ($M)", abs(oregon["carbon_gap_usd"])/1e6,
                               abs(texas["carbon_gap_usd"])/1e6),
]:
    ratio = t_val / o_val if o_val > 0 else 0
    print(f"    {label:<30} {o_val:>15,.1f} {t_val:>15,.1f} {ratio:>7.1f}x")


# =============================================================================
# SECTION 7 — GROWING DATA CENTER MARKETS (PREDICTED PUE)
# Article: "Here are the predicted PUEs for the top 10 growing markets"
# Uses average of Model C and Model D from Script 2
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: TOP 10 GROWING MARKETS — PREDICTED PUE")
print("(Using average of Model C and Model D)")
print("=" * 70)

# Load regression models from Script 2
try:
    model_c = joblib.load("model_c.pkl")
    model_d = joblib.load("model_d.pkl")
    print("\n  ✅ Models loaded: model_c.pkl, model_d.pkl")
except FileNotFoundError:
    raise FileNotFoundError(
        "model_c.pkl or model_d.pkl not found.\n"
        "Please run 02_regression_analysis.py first."
    )

GROWING_MARKETS = [
    # (city, country, temp_c, humidity_pct, latitude, elec_usd_kwh,
    #  grid_co2_gkwh, carbon_price_usd_ton, primary_energy_source)
    ("Tokyo, Japan",        "Japan",                15.5, 68, 35.7,  0.202, 483.4, 1.93,
     "LNG 35%, Coal 30%, Nuclear 10%"),
    ("Seoul, South Korea",  "South Korea",          12.5, 65, 37.6,  0.114, 415.5, 6.45,
     "Coal 36%, LNG 29%, Nuclear 25%"),
    ("Riyadh, Saudi Arabia","Saudi Arabia",         32.5, 28, 24.7,  0.038, 692.0, 0.00,
     "Gas 60%, Oil 40%"),
    ("Dubai, UAE",          "United Arab Emirates", 32.0, 58, 25.2,  0.058, 467.5, 0.00,
     "Gas 99%, Solar 1%"),
    ("Taipei, Taiwan",      "Taiwan",               23.0, 76, 25.0,  0.187, 635.2, 0.00,
     "Coal 45%, LNG 35%, Nuclear 10%"),
    ("Singapore",           "Singapore",            27.5, 84,  1.4,  0.265, 498.7, 18.62,
     "Gas 95%, Solar 3%"),
    ("Bangkok, Thailand",   "Thailand",             29.0, 78, 13.8,  0.083, 554.7, 0.00,
     "Gas 56%, Coal 20%, Renewables 14%"),
    ("Kuala Lumpur, Malaysia","Malaysia",           27.5, 78,  3.1,  0.083, 604.4, 0.00,
     "Gas 51%, Coal 34%, Hydro 10%"),
    ("Mumbai, India",       "India",                28.5, 75, 19.1,  0.078, 707.4, 0.00,
     "Coal 72%, Gas 8%, Renewables 17%"),
    ("Jakarta, Indonesia",  "Indonesia",            28.0, 80, -6.2,  0.070, 680.2, 0.72,
     "Coal 60%, Gas 22%, Oil 9%"),
]

results_growing = []
for (city, country, temp, hum, lat, elec, grid_co2,
     carbon_price, energy_mix) in GROWING_MARKETS:

    wb    = wet_bulb(temp, hum)
    pue_c = model_c.predict(pd.DataFrame([[wb, lat]],
                columns=["wet_bulb_c", "latitude"]))[0]
    pue_d = model_d.predict(pd.DataFrame([[temp, lat]],
                columns=["avg_temp_c", "latitude"]))[0]
    avg_pue = (pue_c + pue_d) / 2

    row = calc_metrics(city, country, avg_pue, elec, grid_co2, carbon_price)
    row["pue_model_c"]          = round(pue_c, 3)
    row["pue_model_d"]          = round(pue_d, 3)
    row["primary_energy_source"]= energy_mix
    results_growing.append(row)

df_growing = pd.DataFrame(results_growing).sort_values("pue")

print(f"\n  {'City':<25} {'C':>6} {'D':>6} {'Avg':>6} {'CO2(t)':>8} "
      f"{'C$/ton':>7} {'Today($K)':>10} {'ETS($M)':>8} {'Gap($M)':>8} {'Trees(M)':>9}")
print(f"  {'-'*100}")
for _, r in df_growing.iterrows():
    today_str = f"${r['carbon_today_usd']/1e3:.0f}K"
    gap_str   = f"${r['carbon_gap_usd']/1e6:.2f}M"
    print(f"  {r['facility']:<25} {r['pue_model_c']:>6.3f} {r['pue_model_d']:>6.3f} "
          f"{r['pue']:>6.3f} {r['co2_tons']:>8,.0f} "
          f"${r['carbon_price_usd_ton']:>6.2f} {today_str:>10} "
          f"${r['carbon_ets_usd']/1e6:>7.2f}M {gap_str:>8} "
          f"{r['trees_needed']/1e6:>8.1f}M")

print(f"\n  Note: Predicted PUE assumes Google-level operational excellence.")
print(f"        This is the efficiency floor — actual PUE will likely be higher.")
print(f"        Carbon gap (negative) = undercharging vs EU ETS benchmark.")

df_growing.to_csv("growing_markets_results.csv", index=False)
print(f"\n  ✅ Saved: growing_markets_results.csv")


# =============================================================================
# SECTION 8 — CHARTS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 8: GENERATING CHARTS")
print("=" * 70)

# Color by tier
TIER_COLORS = {
    "🟢": "#1D9E75",   # green — EU favorable
    "⬜": "#AAAAAA",   # gray — US / Americas
    "🟡": "#EF9F27",   # amber — material gap
    "🔴": "#D85A30",   # red — largest gap
}

FACILITY_TIERS = {}
for fac in ["Fredericia, Denmark", "Eemshaven, Netherlands",
            "Dublin, Ireland", "St. Ghislain, Belgium", "Hamina, Finland"]:
    FACILITY_TIERS[fac] = "🟢"
for fac in ["The Dalles, Oregon (2nd)", "The Dalles, Oregon",
            "Montgomery County, Tennessee", "Loudoun County, Virginia (2nd)",
            "Loudoun County, Virginia", "Berkeley County, South Carolina",
            "Central Ohio (Lancaster), Ohio", "Central Ohio (Columbus), Ohio",
            "Central Ohio (New Albany), Ohio", "Henderson, Nevada",
            "Council Bluffs, Iowa (2nd)", "Papillion, Nebraska",
            "Douglas County, Georgia", "Lenoir, North Carolina",
            "Jackson County, Alabama", "Mayes County, Oklahoma",
            "Council Bluffs, Iowa", "Midlothian, Texas", "Quilicura, Chile"]:
    FACILITY_TIERS[fac] = "⬜"
for fac in ["Inzai, Japan", "Singapore", "Storey County, Nevada"]:
    FACILITY_TIERS[fac] = "🟡"
for fac in ["Changhua County, Taiwan", "Singapore (2nd)"]:
    FACILITY_TIERS[fac] = "🔴"

def get_color(facility):
    tier = FACILITY_TIERS.get(facility, "⬜")
    return TIER_COLORS[tier]

# ── Chart layout: 2×2, taller to accommodate 29 facilities ──
fig = plt.figure(figsize=(18, 22))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

fig.suptitle(
    "Google Data Center PUE — Carbon & Electricity Cost Analysis\n"
    f"All 29 active facilities | Actual 2025 avg PUE | "
    f"100MW IT load | EU ETS benchmark ${EU_ETS_PRICE}/ton",
    fontsize=13, fontweight="bold"
)

# Chart 1: Actual 2025 PUE — all 29 facilities
ax1 = fig.add_subplot(gs[0, 0])
df_plot  = df_all.sort_values("pue")
colors   = [get_color(f) for f in df_plot["facility"]]
ax1.barh(df_plot["facility"], df_plot["pue"], color=colors,
         edgecolor="white", height=0.7)
ax1.axvline(x=1.09, color="black", linestyle="--", linewidth=1.2,
            label="Google fleet avg 1.09")
ax1.set_xlabel("Actual 2025 Average PUE (Q1–Q4)")
ax1.set_title(
    "Actual 2025 PUE — All 29 Google Facilities\n"
    "(green=EU/ETS | gray=US/Americas | amber=material gap | red=largest gap)"
)
ax1.tick_params(axis='y', labelsize=7)
ax1.legend(fontsize=8)

# Chart 2: Annual CO2 from cooling — all 29 facilities
ax2 = fig.add_subplot(gs[0, 1])
df_co2 = df_all.sort_values("co2_tons")
colors2 = [get_color(f) for f in df_co2["facility"]]
ax2.barh(df_co2["facility"], df_co2["co2_tons"] / 1000, color=colors2,
         edgecolor="white", height=0.7)
ax2.set_xlabel("Annual CO2 from cooling (thousand tons)")
ax2.set_title("Annual CO2 from Cooling Overhead\nAll 29 facilities | 100MW IT load")
ax2.tick_params(axis='y', labelsize=7)

# Chart 3: Carbon cost gap — all 29 facilities
ax3 = fig.add_subplot(gs[1, 0])
df_gap   = df_all.sort_values("carbon_gap_usd", ascending=False)
gap_cols = ["#1D9E75" if g > 0 else "#D85A30" for g in df_gap["carbon_gap_usd"]]
ax3.barh(df_gap["facility"], df_gap["carbon_gap_usd"] / 1e6,
         color=gap_cols, edgecolor="white", height=0.7)
ax3.axvline(x=0, color="black", linewidth=0.8)
ax3.set_xlabel("Carbon cost gap vs EU ETS ($M/yr)\nPositive = overpays | Negative = underpays")
ax3.set_title(f"Carbon Policy Gap — All 29 Google Facilities\n"
              f"(vs EU ETS ${EU_ETS_PRICE}/ton benchmark)")
ax3.tick_params(axis='y', labelsize=7)

# Chart 4: Growing markets — predicted PUE and CO2
ax4 = fig.add_subplot(gs[1, 1])
df_gm    = df_growing.sort_values("co2_tons")
ax4.barh(df_gm["facility"], df_gm["co2_tons"] / 1000,
         color="#D85A30", edgecolor="white", height=0.7, alpha=0.85)
ax4b = ax4.twiny()
ax4b.barh(df_gm["facility"], df_gm["carbon_gap_usd"].abs() / 1e6,
          color="#1F4E79", edgecolor="white", height=0.3, alpha=0.5,
          label="Carbon gap ($M, top axis)")
ax4.set_xlabel("Annual CO2 from cooling (thousand tons)", color="#D85A30")
ax4b.set_xlabel("Carbon gap vs EU ETS ($M/yr)", color="#1F4E79")
ax4.set_title("Growing Markets — CO2 Output & Carbon Gap\n"
              "(predicted PUE at Google-level efficiency)")
ax4.tick_params(axis='y', labelsize=8)

plt.savefig("pue_carbon_analysis.png", dpi=150, bbox_inches="tight")
print(f"\n  ✅ Saved: pue_carbon_analysis.png")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY — FILES CREATED")
print("=" * 70)
print(f"  google_all_facilities_results.csv — All {len(df_all)} Google facilities")
print(f"  growing_markets_results.csv       — 10 growing markets, predicted PUE")
print(f"  pue_carbon_analysis.png           — 4 charts")

total_gap  = df_all["carbon_gap_usd"].sum()
max_gap    = df_all.loc[df_all["carbon_gap_usd"].idxmin()]
max_co2    = df_all.loc[df_all["co2_tons"].idxmax()]
min_co2    = df_all.loc[df_all["co2_tons"].idxmin()]
best_pue   = df_all.loc[df_all["pue"].idxmin()]
worst_pue  = df_all.loc[df_all["pue"].idxmax()]

print(f"\n  Key findings:")
print(f"    Best PUE:             {best_pue['facility']} ({best_pue['pue']:.3f})")
print(f"    Worst PUE:            {worst_pue['facility']} ({worst_pue['pue']:.3f})")
print(f"    Lowest CO2/yr:        {min_co2['facility']} ({min_co2['co2_tons']:,.0f} tons)")
print(f"    Highest CO2/yr:       {max_co2['facility']} ({max_co2['co2_tons']:,.0f} tons)")
print(f"    Largest carbon gap:   {max_gap['facility']} "
      f"(${abs(max_gap['carbon_gap_usd'])/1e6:.2f}M/yr underpaying)")
print(f"    Total fleet gap:      ${abs(total_gap)/1e6:.1f}M/yr vs EU ETS benchmark")
print(f"    Growing market (highest gap): "
      f"{df_growing.loc[df_growing['carbon_gap_usd'].idxmin(), 'facility']} "
      f"(${abs(df_growing['carbon_gap_usd'].min())/1e6:.2f}M/yr)")
print(f"\n  Tier breakdown:")
for tier_label, facilities in tiers:
    count = sum(1 for f in facilities if f in df_all["facility"].values)
    print(f"    {tier_label[:2]}  {count} facilities")
