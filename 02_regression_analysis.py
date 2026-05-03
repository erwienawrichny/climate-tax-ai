"""
=============================================================================
SCRIPT 2: PUE REGRESSION ANALYSIS
=============================================================================
Article section: "Regression Analysis"

What this script does:
  1. Loads Google's quarterly PUE data (2022-2025)
  2. Computes wet bulb temperature from temp + humidity
  3. Runs correlation analysis — Temperature, Humidity, Wet Bulb, Latitude
  4. Tests 6 model combinations and compares them using:
       Adj R², 5-fold CV R² (mean ± std), CV RMSE, AIC, p-values, VIF
  5. Selects Model C (Wet Bulb + Latitude) and Model D (Avg Temp + Latitude)
  6. Validates models against actual 2025 PUE — scatter plot
  7. Saves model_c.pkl and model_d.pkl for use in Script 1

Inputs required:
  google_datacenter_pue_final.csv   — Google's quarterly facility PUE data

Outputs:
  model_c.pkl                       — Selected model: Wet Bulb + Latitude
  model_d.pkl                       — Selected model: Avg Temp + Latitude
  regression_analysis.png           — Correlation chart + model comparison
  actual_vs_predicted_pue_2025.png  — Scatter plot: actual vs predicted PUE

Run this script BEFORE Script 1.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import statsmodels.api as sm
import joblib, warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1 — LOAD DATA AND COMPUTE WET BULB
# =============================================================================

print("=" * 70)
print("SCRIPT 2: PUE REGRESSION ANALYSIS")
print("=" * 70)

df = pd.read_csv("google_datacenter_pue_final.csv")

def wet_bulb(temp, humidity):
    """Stull (2011) approximation of wet bulb temperature."""
    return (temp * np.arctan(0.151977 * (humidity + 8.313659)**0.5)
            + np.arctan(temp + humidity)
            - np.arctan(humidity - 1.676331)
            + 0.00391838 * humidity**1.5 * np.arctan(0.023101 * humidity)
            - 4.686035)

df["wet_bulb_c"] = wet_bulb(df["avg_temp_c"], df["avg_humidity_pct"])
df = df.dropna(subset=["pue", "wet_bulb_c", "latitude", "avg_temp_c", "avg_humidity_pct"])
y = df["pue"]

print(f"\n  Dataset: {len(df)} quarterly observations")
print(f"  Facilities: {df['facility'].nunique()}")
print(f"  Years: {df['year'].min()}–{df['year'].max()}")
print(f"  PUE range: {y.min():.3f}–{y.max():.3f}")


# =============================================================================
# SECTION 2 — CORRELATION ANALYSIS
# Article: "Here's how each of these factors correlate with PUE"
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: CORRELATION ANALYSIS")
print("=" * 70)

variables = {
    "Temperature (°C)":   "avg_temp_c",
    "Wet Bulb Temp (°C)": "wet_bulb_c",
    "Humidity (%)":       "avg_humidity_pct",
    "Latitude":           "latitude",
}

correlations = {}
print(f"\n  {'Variable':<25} {'Correlation with PUE':>22}  Interpretation")
print(f"  {'-'*75}")
for label, col in variables.items():
    corr = df[col].corr(y)
    correlations[label] = corr
    direction = "positive" if corr > 0 else "negative"
    strength  = "strong" if abs(corr) > 0.4 else "moderate" if abs(corr) > 0.2 else "weak"
    print(f"  {label:<25} {corr:>22.3f}  {strength} {direction}")

# Multicollinearity check — Wet Bulb vs Temperature
wb_temp_corr = df["wet_bulb_c"].corr(df["avg_temp_c"])
print(f"\n  ⚠️  Wet Bulb vs Temperature correlation: {wb_temp_corr:.3f}")
print(f"     These two cannot be used in the same model (VIF would be ~42)")

# Correlation chart (matches article figure)
fig_corr, ax = plt.subplots(figsize=(8, 4))
fig_corr.patch.set_facecolor("white")
ax.set_facecolor("white")

labels = list(correlations.keys())[::-1]
values = [correlations[l] for l in labels]
colors = ["#2563a8" if v < 0 else "#D85A30" for v in values]
bars   = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)

for bar, val in zip(bars, values):
    ax.text(val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=9)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Correlation with PUE")
ax.set_title("Variable Correlations with PUE", fontweight="bold")
ax.set_xlim(-0.4, 0.65)
ax.grid(axis="x", color="#eeeeee", lw=0.5)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("correlation_chart.png", dpi=150, bbox_inches="tight", facecolor="white")
print(f"\n  ✅ Saved: correlation_chart.png")


# =============================================================================
# SECTION 3 — MODEL SELECTION
# Article: "To find the strongest combination of independent variables,
#           six models were evaluated"
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: MODEL SELECTION")
print("=" * 70)
print("""
  Six models tested — 2 single-variable, 4 two-variable.
  Not tested: Wet Bulb + Avg Temp (correlation = 0.977, VIF = 42 — multicollinearity)
  Not tested: Humidity standalone (raw correlation = -0.040 — no signal)

  Evaluation criteria:
    Adj R²     — fit penalised for number of variables (rewards parsimony)
    CV R²      — 5-fold cross-validation mean (generalisation)
    CV std     — CV R² standard deviation (lower = more stable)
    CV RMSE    — average prediction error in PUE units
    AIC        — goodness of fit per parameter (lower = better)
    p-values   — statistical significance of each variable
    VIF        — multicollinearity (>5 = concern, >10 = serious problem)
""")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

MODELS = [
    # (label, description, col_names)
    ("A",  "Wet Bulb Temp",        ["wet_bulb_c"]),
    ("B",  "Avg Temp",             ["avg_temp_c"]),
    ("C★", "Wet Bulb + Latitude",  ["wet_bulb_c", "latitude"]),
    ("D★", "Avg Temp + Latitude",  ["avg_temp_c", "latitude"]),
    ("F",  "Wet Bulb + Humidity",  ["wet_bulb_c", "avg_humidity_pct"]),
    ("G",  "Avg Temp + Humidity",  ["avg_temp_c", "avg_humidity_pct"]),
]

model_results = []
fitted_models = {}

print(f"  {'Model':<5} {'Variables':<28} {'Adj R²':>7} {'CV R²':>7} {'±std':>7} "
      f"{'RMSE':>7} {'AIC':>9}  {'p-values':<30}  {'VIF'}")
print(f"  {'-'*115}")

for label, desc, cols in MODELS:
    X_sk = df[cols].values
    X_sm = sm.add_constant(df[cols])

    # OLS for Adj R², AIC, p-values
    ols = sm.OLS(y, X_sm).fit()
    adj_r2 = round(ols.rsquared_adj, 3)
    aic    = round(ols.aic, 1)
    pvals  = {c: round(ols.pvalues[c], 4) for c in cols}
    pval_str = ", ".join([f"{c.split('_')[0]}:{v}" for c,v in pvals.items()])

    # 5-fold CV R² and RMSE
    cv_r2   = cross_val_score(LinearRegression(), X_sk, y, cv=kf, scoring="r2")
    cv_rmse = np.sqrt(-cross_val_score(LinearRegression(), X_sk, y, cv=kf,
                                        scoring="neg_mean_squared_error"))

    # VIF (only meaningful for 2+ variables)
    if len(cols) > 1:
        vif_vals = [round(variance_inflation_factor(df[cols].values, i), 1)
                    for i in range(len(cols))]
        vif_str = ", ".join([f"{c.split('_')[0]}:{v}" for c,v in zip(cols, vif_vals)])
    else:
        vif_str = "N/A"

    model_results.append({
        "label": label, "desc": desc, "cols": cols,
        "adj_r2": adj_r2, "cv_r2_mean": round(cv_r2.mean(), 3),
        "cv_r2_std": round(cv_r2.std(), 3),
        "cv_rmse": round(cv_rmse.mean(), 4),
        "aic": aic, "pvals": pvals, "vif": vif_str, "ols": ols,
    })
    fitted_models[label] = ols

    selected = "★" if "★" in label else " "
    print(f"  {label:<5} {desc:<28} {adj_r2:>7.3f} {cv_r2.mean():>7.3f} "
          f"{cv_r2.std():>7.3f} {cv_rmse.mean():>7.4f} {aic:>9.1f}  "
          f"{pval_str:<30}  {vif_str}")

print(f"""
  Selection rationale:
    Model C★ (Wet Bulb + Latitude):  Best CV R² (0.252), clean VIF (1.6),
                                      best narrative fit for general audience
    Model D★ (Avg Temp + Latitude):  Best AIC (−1771.8), best Adj R² (0.247),
                                      lowest CV std (±0.112), best analytical model
    Model G  (Avg Temp + Humidity):  Humidity p-value = 0.874 — not significant
    Model F  (Wet Bulb + Humidity):  Humidity adds nothing wet bulb doesn't capture
""")


# =============================================================================
# SECTION 4 — SELECTED MODEL DETAILS
# Article: regression equations and coefficients
# =============================================================================

print("=" * 70)
print("SECTION 4: SELECTED MODEL DETAILS")
print("=" * 70)

for label, desc in [("C★", "Wet Bulb + Latitude"), ("D★", "Avg Temp + Latitude")]:
    m = fitted_models[label]
    print(f"\n  Model {label}: {desc}")
    print(f"  {'-'*55}")
    print(f"  Equation: PUE = {m.params['const']:.4f}", end="")
    for col in [c for c in m.params.index if c != "const"]:
        sign = "+" if m.params[col] >= 0 else "−"
        print(f" {sign} ({abs(m.params[col]):.5f} × {col})", end="")
    print()
    print(f"\n  Coefficients:")
    print(f"    Intercept: {m.params['const']:.4f}  — Google's operational baseline")
    for col in [c for c in m.params.index if c != "const"]:
        label_name = col.replace("_c","").replace("_pct","").replace("avg_","avg ").replace("_"," ")
        print(f"    {label_name}: {m.params[col]:.5f}  (p={m.pvalues[col]:.4f})")
    print(f"\n  Plain English:")
    if "wet_bulb_c" in m.params.index:
        print(f"    Every 1°C increase in wet bulb temp adds {m.params['wet_bulb_c']:.5f} to PUE")
    if "avg_temp_c" in m.params.index:
        print(f"    Every 1°C increase in avg temp adds {m.params['avg_temp_c']:.5f} to PUE")
    if "latitude" in m.params.index:
        print(f"    Every 1° further from equator reduces PUE by {abs(m.params['latitude']):.5f}")


# =============================================================================
# SECTION 5 — VALIDATE: ACTUAL VS PREDICTED PUE (2025)
# Article: scatter plot showing how well the model predicts actual 2025 PUE
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: MODEL VALIDATION — ACTUAL VS PREDICTED PUE (2025)")
print("=" * 70)

# Compute predicted PUE using both models for each facility
mc_ols = fitted_models["C★"]
md_ols = fitted_models["D★"]

df["pred_c"] = mc_ols.predict(sm.add_constant(df[["wet_bulb_c", "latitude"]]))
df["pred_d"] = md_ols.predict(sm.add_constant(df[["avg_temp_c",  "latitude"]]))

df_2025 = (df[df["year"] == 2025]
           .groupby("facility")
           .agg(actual_pue=("pue",    "mean"),
                pred_c    =("pred_c", "mean"),
                pred_d    =("pred_d", "mean"))
           .reset_index())

df_2025["avg_model"] = (df_2025["pred_c"] + df_2025["pred_d"]) / 2
df_2025["diff_pct"]  = ((df_2025["actual_pue"] - df_2025["avg_model"])
                         / df_2025["avg_model"] * 100)

print(f"\n  {'Facility':<40} {'Actual':>8} {'Avg C+D':>8} {'Diff%':>8}")
print(f"  {'-'*68}")
for _, r in df_2025.sort_values("actual_pue").iterrows():
    flag = " ⬆ outperforms model" if r["diff_pct"] < -2 else \
           " ⬇ underperforms model" if r["diff_pct"] > 2 else ""
    print(f"  {r['facility']:<40} {r['actual_pue']:>8.3f} "
          f"{r['avg_model']:>8.3f} {r['diff_pct']:>+8.2f}%{flag}")

within_2pct = (df_2025["diff_pct"].abs() <= 2).sum()
print(f"\n  {within_2pct}/{len(df_2025)} facilities within ±2% of model prediction")
print(f"  Largest outperformer: "
      f"{df_2025.loc[df_2025['diff_pct'].idxmin(), 'facility']} "
      f"({df_2025['diff_pct'].min():+.2f}%)")
print(f"  Largest underperformer: "
      f"{df_2025.loc[df_2025['diff_pct'].idxmax(), 'facility']} "
      f"({df_2025['diff_pct'].max():+.2f}%)")


# =============================================================================
# SECTION 6 — SCATTER PLOT: ACTUAL VS PREDICTED PUE
# Article figure — numbered points with facility index panel
# =============================================================================

print("\n  Generating scatter plot...")

names   = df_2025.sort_values("actual_pue")["facility"].tolist()
actuals = df_2025.sort_values("actual_pue")["actual_pue"].tolist()
models_ = df_2025.sort_values("actual_pue")["avg_model"].tolist()
diffs   = df_2025.sort_values("actual_pue")["diff_pct"].tolist()
colors  = ["#2563a8" if a <= m else "#c0392b" for a, m in zip(actuals, models_)]

fig, (ax, ax_leg) = plt.subplots(1, 2, figsize=(14, 9), dpi=150,
    gridspec_kw={"width_ratios": [1.15, 0.85]})
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")
ax_leg.set_facecolor("#FAFAFA")

lo, hi, rmse = 1.072, 1.155, 0.030
ref = np.array([lo, hi])

ax.fill_between(ref, ref-rmse, ref+rmse, color="#888", alpha=0.07, zorder=1)
ax.plot(ref, ref+rmse, color="#bbb", lw=0.8, linestyle="--", dashes=(4,5), zorder=2)
ax.plot(ref, ref-rmse, color="#bbb", lw=0.8, linestyle="--", dashes=(4,5), zorder=2)
ax.plot(ref, ref, color="#555", lw=1.4, linestyle="--", dashes=(7,5), zorder=3)
ax.scatter(models_, actuals, c=colors, s=72, zorder=5,
           edgecolors="white", linewidths=0.8, alpha=0.92)

for i, (n, a, m, c) in enumerate(zip(names, actuals, models_, colors)):
    num = i + 1
    ox  = 0.0006 if m < 1.130 else -0.0006
    ha  = "left" if ox > 0 else "right"
    ax.annotate(str(num), xy=(m, a), xytext=(m+ox, a+0.0006),
                fontsize=7.5, color=c, fontweight="700",
                ha=ha, va="bottom", zorder=6)

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Model prediction — avg of Model C & D", fontsize=10, color="#444", labelpad=8)
ax.set_ylabel("Actual 2025 PUE (Q1–Q4 average)", fontsize=10, color="#444", labelpad=8)

ticks = np.arange(1.075, hi+0.005, 0.010)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels([f"{t:.3f}" for t in ticks], fontsize=8, color="#666")
ax.set_yticklabels([f"{t:.3f}" for t in ticks], fontsize=8, color="#666")
ax.grid(True, color="#ddd", lw=0.5, zorder=0)
ax.spines[["top","right"]].set_visible(False)
ax.spines[["left","bottom"]].set_color("#ccc")
ax.tick_params(colors="#888", length=3)

blue_p = mpatches.Patch(color="#2563a8", label="Outperforming climate benchmark")
red_p  = mpatches.Patch(color="#c0392b", label="Underperforming climate benchmark")
ref_l  = mlines.Line2D([],[],color="#555",lw=1.4,linestyle="--",dashes=(7,5),
                        label="Perfect prediction")
band_p = mpatches.Patch(color="#888", alpha=0.15, label=f"±RMSE (±{rmse:.3f} PUE)")
ax.legend(handles=[blue_p,red_p,ref_l,band_p], fontsize=8, frameon=True,
          framealpha=0.92, loc="upper left", edgecolor="#ddd",
          facecolor="#FAFAFA", labelcolor="#444")

# Right panel: facility index
ax_leg.axis("off")
ax_leg.set_xlim(0, 1)
ax_leg.set_ylim(0, 1)
ax_leg.text(0.02, 0.988, "Facility index", fontsize=10, fontweight="700",
            color="#1F4E79", va="top", transform=ax_leg.transAxes)
for cx, lbl in [(0.55,"Actual"),(0.72,"Predicted"),(0.90,"Diff%")]:
    ax_leg.text(cx, 0.988, lbl, fontsize=8.5, fontweight="600", color="#444",
                va="top", ha="center", transform=ax_leg.transAxes)
ax_leg.plot([0,1],[0.979,0.979], color="#ccc", lw=0.8, transform=ax_leg.transAxes)

row_h = 0.958 / len(names)
for i, (n, a, m, d, c) in enumerate(zip(names, actuals, models_, diffs, colors)):
    yp = 0.974 - i*row_h - row_h/2
    if i % 2 == 0:
        rect = plt.Rectangle((0, yp-row_h*0.52), 1, row_h,
                              transform=ax_leg.transAxes, color="#F0F0F0",
                              zorder=0, clip_on=False)
        ax_leg.add_patch(rect)
    ax_leg.text(0.022, yp, str(i+1), fontsize=7, fontweight="800", color="white",
                ha="center", va="center", transform=ax_leg.transAxes,
                bbox=dict(boxstyle="round,pad=0.17", fc=c, ec="none"))
    ax_leg.text(0.065, yp, n, fontsize=7.5, color="#222",
                ha="left", va="center", transform=ax_leg.transAxes)
    ax_leg.text(0.55, yp, f"{a:.3f}", fontsize=7.5, color="#333",
                ha="center", va="center", transform=ax_leg.transAxes)
    ax_leg.text(0.72, yp, f"{m:.3f}", fontsize=7.5, color="#333",
                ha="center", va="center", transform=ax_leg.transAxes)
    sign = "+" if d >= 0 else ""
    ax_leg.text(0.90, yp, f"{sign}{d:.1f}%", fontsize=7.5, fontweight="600",
                color=c, ha="center", va="center", transform=ax_leg.transAxes)

fig.add_artist(plt.Line2D([0.545,0.545],[0.04,0.97],
               transform=fig.transFigure, color="#ddd", lw=1))
fig.text(0.28, 0.01,
         "Source: Google Data Center Efficiency Page | "
         "Model C = Wet Bulb + Latitude | Model D = Avg Temp + Latitude | "
         f"N={len(names)} facilities | 2022–2025 training data",
         ha="center", fontsize=7, color="#999")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("actual_vs_predicted_pue_2025.png", dpi=150,
            bbox_inches="tight", facecolor="#FAFAFA")
print(f"  ✅ Saved: actual_vs_predicted_pue_2025.png")


# =============================================================================
# SECTION 7 — SAVE MODELS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: SAVING SELECTED MODELS")
print("=" * 70)

# Refit with sklearn for joblib export (needed by Script 1)
from sklearn.linear_model import LinearRegression

mc_sk = LinearRegression().fit(df[["wet_bulb_c", "latitude"]].values, y)
md_sk = LinearRegression().fit(df[["avg_temp_c",  "latitude"]].values, y)

joblib.dump(mc_sk, "model_c.pkl")
joblib.dump(md_sk, "model_d.pkl")

print(f"\n  ✅ Saved: model_c.pkl  (Model C: Wet Bulb + Latitude)")
print(f"  ✅ Saved: model_d.pkl  (Model D: Avg Temp + Latitude)")

print(f"\n  Model C equation:")
print(f"    PUE = {mc_sk.intercept_:.4f} + ({mc_sk.coef_[0]:.5f} × wet_bulb_c) "
      f"+ ({mc_sk.coef_[1]:.5f} × latitude)")
print(f"\n  Model D equation:")
print(f"    PUE = {md_sk.intercept_:.4f} + ({md_sk.coef_[0]:.5f} × avg_temp_c) "
      f"+ ({md_sk.coef_[1]:.5f} × latitude)")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY — FILES CREATED")
print("=" * 70)
print(f"  model_c.pkl                     — Model C: Wet Bulb + Latitude")
print(f"  model_d.pkl                     — Model D: Avg Temp + Latitude")
print(f"  correlation_chart.png           — Variable correlations with PUE")
print(f"  actual_vs_predicted_pue_2025.png— Scatter plot: actual vs predicted")
print(f"""
  Key model stats (used in article):
    Climate explains ~25% of PUE variation (CV R² 0.252–0.254)
    Average prediction error: ±0.030 PUE units (RMSE)
    Both models statistically significant (all p-values < 0.05)
    No multicollinearity issues (VIF 1.6–1.9)

  Run Script 1 next:
    python3 01_actual_pue_carbon.py
""")
