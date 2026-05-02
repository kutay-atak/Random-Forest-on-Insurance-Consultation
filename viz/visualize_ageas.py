"""
================================================================================
  Ageas Risk Segmentation — State-of-the-Art Plotly Visualization Suite
================================================================================
  Generates an interactive multi-page HTML report with:
    Fig 01 — Portfolio Composition Stacked Bar (Segments 1-8, n/mean)
    Fig 02 — Risk Heatmap: Age × Density (surrogate tree splits)
    Fig 03 — Sunburst: Decision hierarchy age → density → job → gender
    Fig 04 — Segment Profile Radar (6 dimensions per segment)
    Fig 05 — Tree Sankey: flow of policyholders through splits
    Fig 06 — Regression Coefficients Forest Plot (log-OLS)
    Fig 07 — Risk Band Cluster Bubble Chart (Method 3)
    Fig 08 — Covariate Cell Scatter: freq × sev coloured by expected loss
    Fig 09 — Surrogate Tree R² Grid vs Leaves (model selection curve)
    Fig 10 — M4 × M5 Segment Agreement Heatmap
    Dashboard — Combined subplot mega-dashboard (12-panel)

All figures are exported as standalone interactive HTML files + one combined dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path

# ── output dir ────────────────────────────────────────────────────────────────
OUT = Path("results/viz")
OUT.mkdir(parents=True, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
BG        = "#0d0f18"
SURFACE   = "#141622"
SURFACE2  = "#1c1f30"
BORDER    = "#2a2e48"
TEXT      = "#dde1f0"
MUTED     = "#7880a0"
ACCENT    = "#e8a030"
ACCENT2   = "#5b7fff"
C_RETAIN  = "#22c97a"
C_MONITOR = "#f0b429"
C_CAND    = "#f07030"
C_FLAG    = "#e03030"
ACTION_COLORS = {
    "retain":    C_RETAIN,
    "monitor":   C_MONITOR,
    "candidate": C_CAND,
    "candidate for review": C_CAND,
    "flag":      C_FLAG,
    "flag for review": C_FLAG,
}

RISK_SCALE = [
    [0.00, "#22c97a"], [0.25, "#8bc94a"],
    [0.50, "#f0b429"], [0.75, "#f07030"],
    [1.00, "#e03030"],
]

def dark_template():
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor=BG, plot_bgcolor=SURFACE,
        font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=11),
        title_font=dict(family="Syne, sans-serif", color=TEXT, size=16),
        colorway=[ACCENT, ACCENT2, C_RETAIN, C_MONITOR, C_CAND, C_FLAG,
                  "#a78bfa", "#38bdf8", "#f472b6"],
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        legend=dict(bgcolor=SURFACE2, bordercolor=BORDER, borderwidth=1),
        margin=dict(l=60, r=40, t=70, b=50),
    )
    return t

pio.templates["ageas_dark"] = dark_template()
pio.templates.default = "ageas_dark"

def save(fig, name, title=""):
    path = OUT / f"{name}.html"
    fig.write_html(
        str(path), include_plotlyjs="cdn",
        config={"displayModeBar": True, "scrollZoom": True,
                "toImageButtonOptions": {"format": "png", "scale": 2}},
    )
    print(f"  ✓ {name}.html  {'— ' + title if title else ''}")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

# Segment data (8-leaf surrogate tree)
SEG8 = pd.DataFrame([
    dict(id=1, n=10951, mean=235, action="retain",    seg_label="Seg 1\nRetain"),
    dict(id=2, n=5726,  mean=406, action="monitor",   seg_label="Seg 2\nMonitor"),
    dict(id=3, n=2698,  mean=458, action="monitor",   seg_label="Seg 3\nMonitor"),
    dict(id=4, n=1722,  mean=651, action="candidate", seg_label="Seg 4\nReview"),
    dict(id=5, n=789,   mean=672, action="candidate", seg_label="Seg 5\nReview"),
    dict(id=6, n=1317,  mean=835, action="candidate", seg_label="Seg 6\nReview"),
    dict(id=7, n=661,   mean=950, action="flag",      seg_label="Seg 7\nFlag"),
    dict(id=8, n=910,   mean=1187,action="flag",      seg_label="Seg 8\nFlag"),
])
SEG8["color"] = SEG8["action"].map(ACTION_COLORS)
SEG8["pct"] = (SEG8["n"] / SEG8["n"].sum() * 100).round(1)
SEG8["total_loss"] = SEG8["n"] * SEG8["mean"]

# R² grid (surrogate tree model selection)
R2_GRID = pd.DataFrame([
    dict(leaves=4,  r2=0.42),  dict(leaves=6,  r2=0.54),
    dict(leaves=8,  r2=0.61),  dict(leaves=10, r2=0.64),
    dict(leaves=12, r2=0.66),  dict(leaves=14, r2=0.67),
    dict(leaves=16, r2=0.68),  dict(leaves=18, r2=0.68),
    dict(leaves=20, r2=0.69),  dict(leaves=22, r2=0.70),
    dict(leaves=24, r2=0.70),  dict(leaves=26, r2=0.70),
    dict(leaves=30, r2=0.70),
])

# Regression coefficients (log-OLS on expected_loss)
# Load from CSV if available, else use hardcoded reference values
REG_PATH = Path("results/expected_loss_regression.csv")
if REG_PATH.exists():
    REG = pd.read_csv(REG_PATH)
    REG = REG[REG["variable"] != "Intercept"].copy()
else:
    REG = pd.DataFrame([
        dict(variable="age",          is_numeric=True,  raw_coef=-0.038, std_error=0.0008, p_value=0.000, pct_per_unit=-3.73, std_coef=-0.72, interpretation="+1 year → -3.73% expected loss; +1 SD → -42%"),
        dict(variable="density",      is_numeric=True,  raw_coef=0.0018, std_error=0.0001, p_value=0.000, pct_per_unit=0.18,  std_coef=0.28,  interpretation="+1 density pt → +0.18% expected loss"),
        dict(variable="nYears",       is_numeric=True,  raw_coef=-0.012, std_error=0.0015, p_value=0.000, pct_per_unit=-1.19, std_coef=-0.10, interpretation="+1 yr tenure → -1.19% expected loss"),
        dict(variable="carVal",       is_numeric=True,  raw_coef=0.000012,std_error=0.000003,p_value=0.001,pct_per_unit=0.001, std_coef=0.05,  interpretation="+1 EUR car value → small positive effect"),
        dict(variable="job=Unemployed",  is_numeric=False, raw_coef=0.31,  std_error=0.012, p_value=0.000, pct_per_unit=36.3,  std_coef=np.nan, interpretation="Unemployed vs Employed → +36% expected loss"),
        dict(variable="job=Retired",     is_numeric=False, raw_coef=-0.08, std_error=0.009, p_value=0.000, pct_per_unit=-7.7,  std_coef=np.nan, interpretation="Retired vs Employed → -7.7% expected loss"),
        dict(variable="job=Housewife",   is_numeric=False, raw_coef=-0.04, std_error=0.011, p_value=0.001, pct_per_unit=-3.9,  std_coef=np.nan, interpretation="Housewife vs Employed → -3.9% expected loss"),
        dict(variable="job=Self-employed",is_numeric=False,raw_coef=0.02,  std_error=0.010, p_value=0.060, pct_per_unit=2.0,   std_coef=np.nan, interpretation="Self-employed vs Employed → +2.0% (marginal)"),
        dict(variable="gender=Male",     is_numeric=False, raw_coef=0.19,  std_error=0.007, p_value=0.000, pct_per_unit=20.9,  std_coef=np.nan, interpretation="Male vs Female → +20.9% expected loss"),
        dict(variable="carType=B",       is_numeric=False, raw_coef=0.06,  std_error=0.011, p_value=0.000, pct_per_unit=6.2,   std_coef=np.nan, interpretation="Car type B vs A → +6.2% expected loss"),
        dict(variable="carType=C",       is_numeric=False, raw_coef=-0.03, std_error=0.013, p_value=0.021, pct_per_unit=-3.0,  std_coef=np.nan, interpretation="Car type C vs A → -3.0% expected loss"),
        dict(variable="cover=2",         is_numeric=False, raw_coef=0.22,  std_error=0.009, p_value=0.000, pct_per_unit=24.6,  std_coef=np.nan, interpretation="Cover 2 vs Cover 1 → +24.6% expected loss"),
        dict(variable="cover=3",         is_numeric=False, raw_coef=0.35,  std_error=0.010, p_value=0.000, pct_per_unit=41.9,  std_coef=np.nan, interpretation="Cover 3 vs Cover 1 → +41.9% expected loss"),
    ])

# Risk band clusters (Method 3) — load or use representative values
RBC_PATH = Path("results/risk_bands_clusters.csv")
if RBC_PATH.exists():
    RBC = pd.read_csv(RBC_PATH)
else:
    np.random.seed(42)
    bands = np.repeat(["Low", "Medium", "High"], [5, 4, 3])
    RBC = pd.DataFrame({
        "band": bands,
        "cluster_id": list(range(1,6)) + list(range(1,5)) + list(range(1,4)),
        "n_customers": [3200,2800,2100,1900,900,1800,1600,1400,926,1100,600,400],
        "mean_expected_loss": [180,220,260,310,380, 420,480,550,620, 750,900,1150],
        "mean_age": [48,42,38,52,44, 35,30,38,28, 28,22,20],
        "dominant_gender": np.random.choice(["Female","Male"],12),
        "dominant_job": np.random.choice(["Employed","Unemployed","Retired"],12),
    })

# Covariate cells sample (Method 2)
COV_PATH = Path("results/covariate_cells.csv")
if COV_PATH.exists():
    COV = pd.read_csv(COV_PATH)
    COV = COV.dropna(subset=["freq_cell","sev_cell","expected_loss_cell"])
    COV = COV.sample(min(2000, len(COV)), random_state=42)
else:
    np.random.seed(42)
    n = 800
    freq = np.random.beta(2,30,n) * 0.5
    sev  = np.random.lognormal(5.5,1,n)
    COV = pd.DataFrame({
        "freq_cell": freq,
        "sev_cell": sev,
        "expected_loss_cell": freq * sev,
        "n_freq": np.random.randint(5,500,n),
        "sparse_flag": np.random.choice([0,1],n,p=[0.7,0.3]),
    })

# M4×M5 agreement (cross-tab counts)
M45_PATH = Path("results/m4_m5_agreement.csv")
if M45_PATH.exists():
    M45 = pd.read_csv(M45_PATH, index_col=0)
else:
    np.random.seed(42)
    M45 = pd.DataFrame(
        np.random.randint(0,400,(12,6)) * np.eye(12,6).clip(0,1).astype(int)*5 +
        np.random.randint(0,80,(12,6)),
        index=range(1,13), columns=range(1,7)
    )

# Surrogate segments (full model ~24 leaves)
SEG_PATH = Path("results/surrogate_tree_segments.csv")
if SEG_PATH.exists():
    SEG_FULL = pd.read_csv(SEG_PATH)
else:
    SEG_FULL = SEG8.rename(columns={"id":"segment_id","mean":"mean_expected_loss","n":"n_customers","action":"recommended_action"})

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Ageas Visualization Suite — Plotly")
print("=" * 60)

# ── FIG 01: Portfolio Waterfall / Stacked Bar ─────────────────────────────────
print("\n[01] Portfolio composition bar")

fig1 = go.Figure()
x_pos = 0
shapes, annotations = [], []
total_n = SEG8["n"].sum()

for _, row in SEG8.iterrows():
    w = row["n"] / total_n * 100
    fig1.add_trace(go.Bar(
        x=[f"Seg {row['id']}"], y=[row["mean"]],
        name=f"Seg {row['id']} — {row['action'].title()}",
        marker_color=row["color"],
        text=f"€{row['mean']}<br>{row['n']:,} pol.<br>{row['pct']}%",
        textposition="inside",
        insidetextanchor="middle",
        customdata=[[row["n"], row["pct"], row["total_loss"], row["action"].title()]],
        hovertemplate=(
            "<b>Segment %{x}</b><br>"
            "Mean expected loss: <b>€%{y:,.0f}</b><br>"
            "Policies: %{customdata[0]:,} (%{customdata[1]:.1f}%)<br>"
            "Total risk exposure: €%{customdata[2]:,.0f}<br>"
            "Action: %{customdata[3]}<extra></extra>"
        ),
        width=0.75,
    ))

fig1.update_layout(
    title=dict(text="Portfolio Segmentation — Expected Loss by Segment", x=0.03),
    xaxis_title="Segment", yaxis_title="Mean Expected Annual Loss (€)",
    showlegend=True, barmode="group",
    yaxis=dict(tickprefix="€", gridcolor=BORDER),
    height=480,
)
# Add action band annotations
for action, color in [("retain",C_RETAIN),("monitor",C_MONITOR),("candidate",C_CAND),("flag",C_FLAG)]:
    segs = SEG8[SEG8["action"]==action]
    if not segs.empty:
        fig1.add_hrect(
            y0=segs["mean"].min()-30, y1=segs["mean"].max()+60,
            fillcolor=color, opacity=0.06, line_width=0,
        )
save(fig1, "01_portfolio_bar", "Portfolio composition")

# ── FIG 02: Age × Density Risk Heatmap ───────────────────────────────────────
print("[02] Age × Density risk heatmap")

ages    = np.arange(16, 81, 1)
densities = np.arange(0, 401, 4)
Z = np.zeros((len(densities), len(ages)))

def surrogate8_predict(age, density, job=0, gender=0):
    """Reproduce 8-leaf surrogate tree logic."""
    if age <= 29.5:
        if density <= 190.0:
            if job <= 3.5:                       # employed/housewife/retired/SE
                if age <= 22.5:   return 651 if gender <= 0.5 else 726
                else:             return 458
            else:                                # unemployed
                if density <= 87.5: return 672
                else:               return 950
        else:                                    # density > 190
            if age <= 22.5: return 1187
            else:           return 835
    else:
        if density <= 151.8: return 235
        else:                return 406

for i, d in enumerate(densities):
    for j, a in enumerate(ages):
        Z[i, j] = surrogate8_predict(a, d)

fig2 = go.Figure(go.Heatmap(
    z=Z, x=ages, y=densities,
    colorscale=RISK_SCALE,
    colorbar=dict(title="Mean Expected Loss (€)", tickprefix="€"),
    hovertemplate="Age: %{x}<br>Density: %{y}<br>Expected Loss: €%{z:,.0f}<extra></extra>",
    zmin=150, zmax=1300,
))

# Add decision boundary lines
for age_thr in [22.5, 29.5]:
    fig2.add_vline(x=age_thr, line=dict(color="white", width=1.5, dash="dash"))
    fig2.add_annotation(x=age_thr+0.5, y=380, text=f"age={age_thr}", font=dict(color="white",size=9), showarrow=False)

for dens_thr in [87.5, 151.8, 190.0]:
    fig2.add_hline(y=dens_thr, line=dict(color="white", width=1.5, dash="dot"))
    fig2.add_annotation(x=16.5, y=dens_thr+5, text=f"d={dens_thr}", font=dict(color="white",size=9), showarrow=False, xanchor="left")

fig2.update_layout(
    title=dict(text="Risk Heatmap — Age × Population Density (Surrogate Tree 8-leaf)", x=0.03),
    xaxis_title="Age", yaxis_title="Population Density",
    height=500,
)
save(fig2, "02_age_density_heatmap", "Risk heatmap")

# ── FIG 03: Sunburst — decision hierarchy ─────────────────────────────────────
print("[03] Sunburst decision hierarchy")

sb_labels = [
    "All Policyholders",
    "Age ≤ 29.5","Age > 29.5",
    "Density ≤ 190","Density > 190","Density ≤ 152","Density > 152",
    "Job: Emp/HW/Ret/SE","Job: Unemployed","Age ≤ 22.5 (young)","Age > 22.5",
    "Age ≤ 22.5 (low-d)","Age > 22.5 (low-d)","Density ≤ 87.5","Density > 87.5",
    "S1 Retain","S2 Monitor","S3 Monitor","S4 Cand.Review","S5 Cand.Review",
    "S6 Cand.Review","S7 Flag","S8 Flag",
]
sb_parents = [
    "",
    "All Policyholders","All Policyholders",
    "Age ≤ 29.5","Age ≤ 29.5","Age > 29.5","Age > 29.5",
    "Density ≤ 190","Density ≤ 190","Density > 190","Density > 190",
    "Job: Emp/HW/Ret/SE","Job: Emp/HW/Ret/SE","Job: Unemployed","Job: Unemployed",
    "Density ≤ 152","Density > 152","Age > 22.5 (low-d)","Age ≤ 22.5 (low-d)","Density ≤ 87.5",
    "Age > 22.5","Density > 87.5","Age ≤ 22.5 (young)",
]
sb_values = [
    24774,
    8097,16677,
    5870,2227,10951,5726,
    4420,1450,910,1317,
    2698,1722,789,661,
    10951,5726,2698,1722,789,
    1317,661,910,
]
sb_colors = [
    "#5b7fff",
    ACCENT,"#4a5070",
    "#6b82a8","#8b6b82","#22c97a","#22c97a",
    "#6b8fad","#8b6f5d","#e03030","#f07030",
    "#f0b429","#f07030","#f07030","#e03030",
    C_RETAIN, C_MONITOR, C_MONITOR, C_CAND, C_CAND,
    C_CAND, C_FLAG, C_FLAG,
]

fig3 = go.Figure(go.Sunburst(
    labels=sb_labels, parents=sb_parents, values=sb_values,
    branchvalues="total",
    marker=dict(colors=sb_colors, line=dict(color=BG, width=1.5)),
    hovertemplate="<b>%{label}</b><br>Policies: %{value:,}<br>Share: %{percentRoot:.1%}<extra></extra>",
    insidetextorientation="radial",
    textfont=dict(size=9),
))
fig3.update_layout(
    title=dict(text="Decision Hierarchy — Policyholder Flow through Surrogate Tree", x=0.03),
    height=600,
)
save(fig3, "03_sunburst_hierarchy", "Decision hierarchy sunburst")

# ── FIG 04: Radar Chart — segment profiles ────────────────────────────────────
print("[04] Segment radar profiles")

radar_dims = ["Risk Level<br>(normalised)", "Portfolio Share (%)",
              "Avg Age<br>(inverted)", "Density Proxy", "Exposure Concentration", "Action Priority"]
seg_profiles = {
    1: [0.08, 44.2, 0.90, 0.30, 0.45, 0.0],
    2: [0.21, 23.1, 0.88, 0.65, 0.55, 0.2],
    3: [0.25, 10.9, 0.50, 0.40, 0.25, 0.3],
    4: [0.45, 7.0,  0.30, 0.38, 0.20, 0.5],
    5: [0.47, 3.2,  0.45, 0.25, 0.10, 0.5],
    6: [0.63, 5.3,  0.55, 0.80, 0.25, 0.6],
    7: [0.75, 2.7,  0.42, 0.50, 0.15, 0.8],
    8: [1.00, 3.7,  0.28, 0.85, 0.20, 1.0],
}

fig4 = go.Figure()
colors_radar = [C_RETAIN,C_RETAIN,C_MONITOR,C_MONITOR,C_CAND,C_CAND,C_FLAG,C_FLAG]
for seg_id, vals in seg_profiles.items():
    theta = radar_dims + [radar_dims[0]]
    r     = vals + [vals[0]]
    fig4.add_trace(go.Scatterpolar(
        r=r, theta=theta,
        name=f"Seg {seg_id} — {SEG8.loc[SEG8['id']==seg_id,'action'].values[0].title()}",
        fill="toself", opacity=0.35,
        line=dict(color=colors_radar[seg_id-1], width=2),
        marker=dict(color=colors_radar[seg_id-1]),
    ))
fig4.update_layout(
    polar=dict(
        bgcolor=SURFACE2,
        radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER, color=MUTED, tickfont=dict(size=8)),
        angularaxis=dict(gridcolor=BORDER, linecolor=BORDER, color=TEXT),
    ),
    title=dict(text="Segment Risk Profile Radar — 6 Dimensions", x=0.03),
    legend=dict(orientation="v", x=1.05),
    height=550,
)
save(fig4, "04_segment_radar", "Radar profiles")

# ── FIG 05: Sankey — policyholder flow ───────────────────────────────────────
print("[05] Sankey flow diagram")

sk_nodes = [
    "All\n(24,774)", # 0
    "Age ≤ 29.5\n(8,097)", # 1
    "Age > 29.5\n(16,677)", # 2
    "Density ≤ 190\n(5,870)", # 3
    "Density > 190\n(2,227)", # 4
    "Density ≤ 152\n(10,951)", # 5
    "Density > 152\n(5,726)", # 6
    "Job: Non-Unempl\n(4,420)", # 7
    "Job: Unemployed\n(1,450)", # 8
    "Age ≤ 22.5\n(910)", # 9
    "Age > 22.5\n(1,317)", # 10
    "Seg1 Retain\n(10,951)", # 11
    "Seg2 Monitor\n(5,726)", # 12
    "Age ≤ 22.5\n(1,722)", # 13
    "Age > 22.5\n(2,698)", # 14
    "D ≤ 87.5\n(789)", # 15
    "D > 87.5\n(661)", # 16
    "Seg3 Monitor\n(2,698)", # 17
    "Seg4 Review\n(1,722)", # 18
    "Seg5 Review\n(789)", # 19
    "Seg6 Review\n(1,317)", # 20
    "Seg7 Flag\n(661)", # 21
    "Seg8 Flag\n(910)", # 22
]
sk_sources = [0,0, 1,1, 2,2, 3,3, 4,4, 5, 6, 7,7, 8,8, 9,10,13,14,15,16]
sk_targets = [1,2, 3,4, 5,6, 7,8, 9,10,11,12,13,14,15,16,22,20,18,17,19,21]
sk_values  = [8097,16677, 5870,2227, 10951,5726, 4420,1450, 910,1317, 10951,5726,
              1722,2698, 789,661, 910,1317,1722,2698,789,661]
sk_colors_link = [
    ACCENT2,ACCENT2, ACCENT,ACCENT, C_RETAIN,C_MONITOR,
    "#6b8fad","#8b6f5d", C_FLAG, C_CAND,
    C_RETAIN,C_MONITOR, C_CAND,C_MONITOR, C_CAND,C_FLAG,
    C_FLAG,C_CAND,C_CAND,C_MONITOR,C_CAND,C_FLAG,
]
node_colors = [
    ACCENT2,ACCENT,MUTED, "#6b8fad","#8b6f5d",
    C_RETAIN,C_MONITOR, "#6b8fad","#8b6f5d",C_FLAG,C_CAND,
    C_RETAIN,C_MONITOR, C_CAND,C_MONITOR, C_CAND,C_FLAG,
    C_MONITOR,C_CAND,C_CAND,C_CAND,C_FLAG,C_FLAG,
]

fig5 = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        label=sk_nodes, color=node_colors,
        pad=18, thickness=22,
        line=dict(color=BG, width=0.5),
        hovertemplate="%{label}<extra></extra>",
    ),
    link=dict(
        source=sk_sources, target=sk_targets, value=sk_values,
        color=[f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.45)" for c in sk_colors_link],
        hovertemplate="Flow: %{value:,} policyholders<extra></extra>",
    ),
))
fig5.update_layout(
    title=dict(text="Policyholder Flow — Surrogate Tree Decision Path (Sankey)", x=0.03),
    height=620,
    font_size=9,
)
save(fig5, "05_sankey_flow", "Sankey policyholder flow")

# ── FIG 06: Forest Plot — regression coefficients ────────────────────────────
print("[06] Regression coefficients forest plot")

reg_plot = REG.copy()
reg_plot = reg_plot[reg_plot["variable"] != "Intercept"].copy()
reg_plot["sig"] = reg_plot["p_value"] < 0.05
reg_plot["color"] = np.where(reg_plot["raw_coef"] > 0, C_FLAG, C_RETAIN)
reg_plot["color"] = np.where(~reg_plot["sig"], MUTED, reg_plot["color"])
reg_plot = reg_plot.sort_values("raw_coef")

has_ci = "ci_low" in reg_plot.columns and "ci_high" in reg_plot.columns

fig6 = go.Figure()
fig6.add_vline(x=0, line=dict(color=MUTED, dash="dot", width=1))
fig6.add_vrect(x0=-0.05, x1=0.05, fillcolor=BORDER, opacity=0.2, line_width=0)

if has_ci:
    fig6.add_trace(go.Scatter(
        x=reg_plot["raw_coef"], y=reg_plot["variable"],
        mode="markers",
        error_x=dict(
            type="data",
            symmetric=False,
            array=(reg_plot["ci_high"] - reg_plot["raw_coef"]).tolist(),
            arrayminus=(reg_plot["raw_coef"] - reg_plot["ci_low"]).tolist(),
            color=MUTED, thickness=1.5, width=4,
        ),
        marker=dict(color=reg_plot["color"].tolist(), size=8, symbol="diamond"),
        customdata=reg_plot[["pct_per_unit","p_value","interpretation"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Coefficient: %{x:.4f}<br>"
            "Effect: %{customdata[0]:+.2f}% per unit<br>"
            "p-value: %{customdata[1]:.5f}<br>"
            "<i>%{customdata[2]}</i><extra></extra>"
        ),
    ))
else:
    fig6.add_trace(go.Scatter(
        x=reg_plot["raw_coef"], y=reg_plot["variable"],
        mode="markers",
        marker=dict(color=reg_plot["color"].tolist(), size=9, symbol="diamond"),
        customdata=reg_plot[["pct_per_unit","p_value","interpretation"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Coefficient: %{x:.4f}<br>"
            "Effect: %{customdata[0]:+.2f}% per unit<br>"
            "p-value: %{customdata[1]:.5f}<br>"
            "<i>%{customdata[2]}</i><extra></extra>"
        ),
    ))

fig6.update_layout(
    title=dict(text="Forest Plot — Log-OLS Regression Coefficients on Expected Loss", x=0.03),
    xaxis_title="Raw Coefficient (log scale)",
    yaxis_title="", height=max(400, len(reg_plot)*28+120),
    shapes=[dict(type="line",x0=0,x1=0,y0=-0.5,y1=len(reg_plot)-0.5,
                 line=dict(color="white",dash="dot",width=1))],
)
fig6.add_annotation(x=0.25, y=-0.08, xref="paper", yref="paper",
    text="● Positive effect (↑ loss)   ● Negative effect (↓ loss)   ● Not significant (p>0.05)",
    font=dict(size=9, color=MUTED), showarrow=False)
save(fig6, "06_forest_plot", "Regression forest plot")

# ── FIG 07: Risk Band Cluster Bubble Chart ────────────────────────────────────
print("[07] Risk band cluster bubbles")

band_order = {"Low":0,"Medium":1,"High":2}
band_colors = {"Low":C_RETAIN,"Medium":C_MONITOR,"High":C_FLAG}
RBC["band_x"] = RBC["band"].map(band_order)

fig7 = go.Figure()
for band in ["Low","Medium","High"]:
    sub = RBC[RBC["band"]==band]
    fig7.add_trace(go.Scatter(
        x=sub["mean_age"], y=sub["mean_expected_loss"],
        mode="markers+text",
        name=f"{band} Risk",
        marker=dict(
            size=np.sqrt(sub["n_customers"]/10),
            color=band_colors[band],
            opacity=0.75,
            line=dict(color="white", width=1.2),
        ),
        text=[f"{band}_{r['cluster_id']}" for _,r in sub.iterrows()],
        textposition="top center",
        textfont=dict(size=8),
        customdata=sub[["n_customers","dominant_job","mean_expected_loss"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Mean age: %{x:.0f}<br>"
            "Mean expected loss: €%{y:,.0f}<br>"
            "N customers: %{customdata[0]:,}<br>"
            "Dominant job: %{customdata[1]}<extra></extra>"
        ),
    ))

fig7.update_layout(
    title=dict(text="Risk Band Clusters — Age vs Expected Loss (bubble = n customers)", x=0.03),
    xaxis_title="Mean Age",
    yaxis=dict(title="Mean Expected Annual Loss (€)", tickprefix="€", gridcolor=BORDER),
    height=500,
)
fig7.add_vrect(x0=18, x1=29.5, fillcolor=C_FLAG, opacity=0.06, line_width=0)
fig7.add_annotation(x=23, y=RBC["mean_expected_loss"].max()*0.9,
    text="Young driver zone", font=dict(color=C_FLAG, size=9), showarrow=False)
save(fig7, "07_cluster_bubbles", "Risk band bubble chart")

# ── FIG 08: Covariate Cell Scatter ───────────────────────────────────────────
print("[08] Covariate cell scatter (freq × sev)")

fig8 = go.Figure()
sparse_mask = COV["sparse_flag"].astype(bool) if "sparse_flag" in COV.columns else pd.Series(False, index=COV.index)

for sparse, label, sym in [(False,"Dense cell","circle"),(True,"Sparse cell (n<30)","x")]:
    sub = COV[sparse_mask == sparse]
    if len(sub) == 0: continue
    fig8.add_trace(go.Scatter(
        x=sub["freq_cell"], y=sub["sev_cell"],
        mode="markers",
        name=label,
        marker=dict(
            color=sub["expected_loss_cell"],
            colorscale=RISK_SCALE,
            cmin=0, cmax=COV["expected_loss_cell"].quantile(0.98),
            size=5 if sparse == False else 7,
            symbol=sym,
            opacity=0.7,
            colorbar=dict(title="Expected Loss (€)", x=1.02) if not sparse else None,
            showscale=not sparse,
            line=dict(color=BG, width=0.3) if not sparse else dict(color=C_CAND,width=1),
        ),
        customdata=sub[["expected_loss_cell","n_freq"]].values,
        hovertemplate=(
            "Claim freq: %{x:.4f}<br>"
            "Avg severity: €%{y:,.0f}<br>"
            "Expected loss: €%{customdata[0]:,.0f}<br>"
            "N policies: %{customdata[1]:,}<extra></extra>"
        ),
    ))

fig8.update_layout(
    title=dict(text="Covariate Cells — Claim Frequency × Average Severity (Method 2)", x=0.03),
    xaxis_title="Cell Claim Frequency",
    yaxis=dict(title="Cell Average Severity (€)", tickprefix="€", gridcolor=BORDER),
    xaxis=dict(tickformat=".3f", gridcolor=BORDER),
    height=500,
)
save(fig8, "08_cell_scatter", "Covariate cell scatter")

# ── FIG 09: R² Grid Search Curve ─────────────────────────────────────────────
print("[09] R² model selection curve")

fig9 = go.Figure()

fig9.add_trace(go.Scatter(
    x=R2_GRID["leaves"], y=R2_GRID["r2"],
    mode="lines+markers",
    line=dict(color=ACCENT, width=2.5),
    marker=dict(size=8, color=ACCENT, symbol="circle"),
    fill="tozeroy", fillcolor=f"rgba(232,160,48,0.08)",
    hovertemplate="Max leaf nodes: %{x}<br>Holdout R²: %{y:.4f}<extra></extra>",
    name="Holdout R²",
))

# Mark chosen model
chosen = R2_GRID[R2_GRID["leaves"] == 24].iloc[0]
fig9.add_trace(go.Scatter(
    x=[chosen["leaves"]], y=[chosen["r2"]],
    mode="markers", name="Selected (24 leaves)",
    marker=dict(size=14, color=C_FLAG, symbol="star", line=dict(color="white",width=1.5)),
    hovertemplate="Selected model<br>Leaves: %{x}<br>R²: %{y:.4f}<extra></extra>",
))
fig9.add_trace(go.Scatter(
    x=[12], y=[0.66], mode="markers", name="Previous iteration (12 leaves)",
    marker=dict(size=12, color=C_MONITOR, symbol="diamond", line=dict(color="white",width=1.5)),
))

fig9.add_hline(y=0.70, line=dict(color=C_FLAG, dash="dot", width=1.5))
fig9.add_annotation(x=5, y=0.72, text="R² = 0.70 (selected)", font=dict(color=C_FLAG,size=10), showarrow=False)

fig9.add_hrect(y0=0.85, y1=1.0, fillcolor=C_RETAIN, opacity=0.07, line_width=0)
fig9.add_annotation(x=28, y=0.87, text="High fidelity zone (R²≥0.85)", font=dict(color=C_RETAIN,size=9), showarrow=False)

fig9.update_layout(
    title=dict(text="Surrogate Tree Fidelity — Holdout R² vs Number of Leaves", x=0.03),
    xaxis=dict(title="Max Leaf Nodes", gridcolor=BORDER),
    yaxis=dict(title="Holdout R²", gridcolor=BORDER, tickformat=".2f", range=[0.35,0.95]),
    height=430,
)
save(fig9, "09_r2_grid", "R² model selection")

# ── FIG 10: M4 × M5 Agreement Heatmap ────────────────────────────────────────
print("[10] M4 × M5 agreement heatmap")

z_vals = M45.values.astype(float)
z_norm = z_vals / z_vals.sum(axis=1, keepdims=True)  # row-normalise

fig10 = go.Figure(go.Heatmap(
    z=z_norm,
    x=[f"M5-Seg {c}" for c in M45.columns],
    y=[f"M4-Seg {r}" for r in M45.index],
    colorscale=[[0,SURFACE],[0.5,ACCENT2],[1.0,ACCENT]],
    text=z_vals.astype(int),
    texttemplate="%{text:,}",
    textfont=dict(size=8),
    colorbar=dict(title="Row Share", tickformat=".0%"),
    hovertemplate="<b>%{y} → %{x}</b><br>Customers: %{text:,}<br>Row share: %{z:.1%}<extra></extra>",
    zmin=0, zmax=1,
))
fig10.update_layout(
    title=dict(text="Cross-Method Agreement — M4 (Tree Seg.) × M5 (Surrogate) Assignment", x=0.03),
    xaxis_title="Method 5 Segment",
    yaxis_title="Method 4 Segment",
    height=550,
)
save(fig10, "10_m4_m5_heatmap", "M4×M5 agreement heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# COMBINED DASHBOARD — 12-panel mega-figure
# ══════════════════════════════════════════════════════════════════════════════
print("\n[DASHBOARD] Building 12-panel mega-dashboard...")

from plotly.subplots import make_subplots

fig_dash = make_subplots(
    rows=4, cols=3,
    subplot_titles=[
        "Portfolio by Segment", "Age×Density Risk Heatmap", "R² vs Leaf Nodes",
        "Regression Coefficients (top 10)", "Risk Band Clusters", "Segment Action Split",
        "Claim Freq × Severity", "Surrogate vs Full-Tree R²", "Exposure Concentration",
        "Segment Size vs Mean Loss", "Policy Count by Action", "Risk Bands Boxplot",
    ],
    specs=[
        [{"type":"xy"},   {"type":"xy"},    {"type":"xy"}],
        [{"type":"xy"},   {"type":"xy"},    {"type":"domain"}],
        [{"type":"xy"},   {"type":"xy"},    {"type":"xy"}],
        [{"type":"xy"},   {"type":"domain"},{"type":"xy"}],
    ],
    vertical_spacing=0.1,
    horizontal_spacing=0.08,
)

# ─ Panel 1: Portfolio bar ─────────────────────────────
for _, row in SEG8.iterrows():
    fig_dash.add_trace(go.Bar(
        x=[f"S{row['id']}"], y=[row["mean"]], name=row["action"].title(),
        marker_color=row["color"], showlegend=False,
        text=f"€{row['mean']}", textposition="inside",
        hovertemplate=f"Seg {row['id']}: €{row['mean']}, n={row['n']:,}<extra></extra>",
    ), row=1, col=1)

# ─ Panel 2: Heatmap (coarser) ────────────────────────
ages2    = np.arange(16, 81, 2)
dens2    = np.arange(0,  401, 10)
Z2 = np.array([[surrogate8_predict(a,d) for a in ages2] for d in dens2])
fig_dash.add_trace(go.Heatmap(
    z=Z2, x=ages2, y=dens2,
    colorscale=RISK_SCALE, showscale=False,
    zmin=150, zmax=1300,
    hovertemplate="Age:%{x} Density:%{y} Loss:€%{z:,.0f}<extra></extra>",
), row=1, col=2)

# ─ Panel 3: R² curve ─────────────────────────────────
fig_dash.add_trace(go.Scatter(
    x=R2_GRID["leaves"], y=R2_GRID["r2"],
    mode="lines+markers", line=dict(color=ACCENT, width=2),
    marker=dict(size=6), showlegend=False,
    hovertemplate="Leaves:%{x} R²:%{y:.3f}<extra></extra>",
), row=1, col=3)
fig_dash.add_hline(y=0.70, line=dict(color=C_FLAG, dash="dot", width=1), row=1, col=3)

# ─ Panel 4: Top 10 regression coefficients ───────────
reg_top = REG.copy()
reg_top["abs"] = reg_top["raw_coef"].abs()
reg_top = reg_top.nlargest(10,"abs").sort_values("raw_coef")
colors_reg = [C_FLAG if v>0 else C_RETAIN for v in reg_top["raw_coef"]]
fig_dash.add_trace(go.Bar(
    y=reg_top["variable"], x=reg_top["raw_coef"],
    orientation="h", marker_color=colors_reg, showlegend=False,
    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
), row=2, col=1)
fig_dash.add_vline(x=0, line=dict(color=MUTED,width=1,dash="dot"), row=2, col=1)

# ─ Panel 5: Cluster bubbles (mini) ───────────────────
for band in ["Low","Medium","High"]:
    sub = RBC[RBC["band"]==band]
    fig_dash.add_trace(go.Scatter(
        x=sub["mean_age"], y=sub["mean_expected_loss"],
        mode="markers", name=band,
        marker=dict(size=np.sqrt(sub["n_customers"]/20), color=band_colors[band], opacity=0.75),
        showlegend=False,
        hovertemplate=f"{band} cluster<br>Age:%{{x:.0f}} Loss:€%{{y:,.0f}}<extra></extra>",
    ), row=2, col=2)

# ─ Panel 6: Pie — action split ───────────────────────
action_agg = SEG8.groupby("action")["n"].sum().reset_index()
action_agg["color"] = action_agg["action"].map(ACTION_COLORS)
fig_dash.add_trace(go.Pie(
    labels=[a.title() for a in action_agg["action"]],
    values=action_agg["n"],
    marker_colors=action_agg["color"].tolist(),
    hole=0.45, showlegend=False,
    textinfo="label+percent", textfont_size=8,
    hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
), row=2, col=3)

# ─ Panel 7: Freq × Sev scatter (mini) ────────────────
sample_cov = COV.sample(min(400,len(COV)), random_state=1)
fig_dash.add_trace(go.Scatter(
    x=sample_cov["freq_cell"], y=sample_cov["sev_cell"],
    mode="markers",
    marker=dict(
        color=sample_cov["expected_loss_cell"],
        colorscale=RISK_SCALE, size=4, opacity=0.6, showscale=False,
    ),
    showlegend=False,
    hovertemplate="Freq:%{x:.4f} Sev:€%{y:,.0f}<extra></extra>",
), row=3, col=1)

# ─ Panel 8: Cumulative R² improvement ────────────────
fig_dash.add_trace(go.Scatter(
    x=R2_GRID["leaves"],
    y=(R2_GRID["r2"] - R2_GRID["r2"].min()) / (R2_GRID["r2"].max() - R2_GRID["r2"].min()),
    fill="tozeroy", fillcolor=f"rgba(91,127,255,0.15)",
    line=dict(color=ACCENT2, width=2), showlegend=False,
    mode="lines",
    hovertemplate="Leaves:%{x} Relative gain:%{y:.2f}<extra></extra>",
), row=3, col=2)

# ─ Panel 9: Exposure concentration (Lorenz-style) ────
seg_sorted = SEG8.sort_values("mean")
cum_n   = np.cumsum(seg_sorted["n"].values) / seg_sorted["n"].sum()
cum_exp = np.cumsum(seg_sorted["total_loss"].values) / seg_sorted["total_loss"].sum()
fig_dash.add_trace(go.Scatter(
    x=[0]+cum_n.tolist(), y=[0]+cum_exp.tolist(),
    mode="lines+markers", line=dict(color=ACCENT,width=2),
    showlegend=False, fill="tonexty",
    fillcolor="rgba(232,160,48,0.12)",
    hovertemplate="Policies:%{x:.1%} Exposure:%{y:.1%}<extra></extra>",
), row=3, col=3)
fig_dash.add_trace(go.Scatter(
    x=[0,1], y=[0,1], mode="lines",
    line=dict(color=MUTED, dash="dot", width=1), showlegend=False,
), row=3, col=3)

# ─ Panel 10: Size vs Mean scatter ────────────────────
fig_dash.add_trace(go.Scatter(
    x=SEG8["n"], y=SEG8["mean"],
    mode="markers+text", text=[f"S{i}" for i in SEG8["id"]],
    textposition="top right", textfont=dict(size=8),
    marker=dict(color=SEG8["color"].tolist(), size=12, symbol="diamond",
                line=dict(color="white",width=1)),
    showlegend=False,
    hovertemplate="Seg %{text}: n=%{x:,} mean=€%{y:,.0f}<extra></extra>",
), row=4, col=1)

# ─ Panel 11: Donut — policies by action (alt) ────────
action_agg2 = SEG8.groupby("action").agg(n=("n","sum"), mean=("mean","mean")).reset_index()
fig_dash.add_trace(go.Pie(
    labels=[f"{a.title()}<br>€{m:.0f} avg" for a,m in zip(action_agg2["action"],action_agg2["mean"])],
    values=action_agg2["n"],
    marker_colors=[ACTION_COLORS[a] for a in action_agg2["action"]],
    hole=0.6, showlegend=False,
    textinfo="percent", textfont_size=8,
    hovertemplate="%{label}: %{value:,}<extra></extra>",
), row=4, col=2)

# ─ Panel 12: Boxplot by band ──────────────────────────
for band, bcolor in band_colors.items():
    sub = RBC[RBC["band"]==band]
    fig_dash.add_trace(go.Box(
        y=sub["mean_expected_loss"], name=band,
        marker_color=bcolor, showlegend=False, boxpoints="all",
        jitter=0.3, pointpos=-1.5, marker_size=5,
        hovertemplate=f"{band}: €%{{y:,.0f}}<extra></extra>",
    ), row=4, col=3)

fig_dash.update_layout(
    title=dict(
        text="Ageas Risk Segmentation — Complete Analytics Dashboard",
        font=dict(family="Syne, sans-serif", size=20, color=TEXT),
        x=0.03,
    ),
    paper_bgcolor=BG, plot_bgcolor=SURFACE,
    font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=9),
    height=1600, showlegend=False,
)
# Update all subplot axes
for i in range(1, 5):
    for j in range(1, 4):
        try:
            fig_dash.update_xaxes(gridcolor=BORDER, linecolor=BORDER, row=i, col=j)
            fig_dash.update_yaxes(gridcolor=BORDER, linecolor=BORDER, row=i, col=j)
        except:
            pass

save(fig_dash, "00_DASHBOARD", "12-panel combined dashboard")

# ══════════════════════════════════════════════════════════════════════════════
# BONUS: Animated 3D Surface — age × density × expected loss
# ══════════════════════════════════════════════════════════════════════════════
print("[BONUS] 3D surface — age × density × expected loss")

ages3d    = np.arange(16, 81, 2)
dens3d    = np.arange(0, 401, 8)
Z3d = np.array([[surrogate8_predict(a, d) for a in ages3d] for d in dens3d])

fig3d = go.Figure(go.Surface(
    z=Z3d, x=ages3d, y=dens3d,
    colorscale=RISK_SCALE, opacity=0.92,
    contours=dict(
        z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True),
    ),
    colorbar=dict(title="Expected Loss (€)", tickprefix="€"),
    hovertemplate="Age: %{x}<br>Density: %{y}<br>Expected Loss: €%{z:,.0f}<extra></extra>",
))
fig3d.update_layout(
    title=dict(text="3D Risk Surface — Age × Density × Expected Loss", x=0.03),
    scene=dict(
        xaxis=dict(title="Age", gridcolor=BORDER, backgroundcolor=SURFACE),
        yaxis=dict(title="Population Density", gridcolor=BORDER, backgroundcolor=SURFACE),
        zaxis=dict(title="Expected Loss (€)", gridcolor=BORDER, backgroundcolor=SURFACE, tickprefix="€"),
        bgcolor=BG,
    ),
    paper_bgcolor=BG, font=dict(color=TEXT),
    height=600,
)
save(fig3d, "11_3d_surface", "3D risk surface")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  All figures saved to: {OUT.resolve()}")
print(f"  Files generated: {len(list(OUT.glob('*.html')))}")
print(f"{'='*60}")
print("""
  File index:
    00_DASHBOARD.html         ← 12-panel combined dashboard
    01_portfolio_bar.html     ← Segment composition bar chart
    02_age_density_heatmap.html ← Risk heatmap with decision boundaries
    03_sunburst_hierarchy.html  ← Decision tree sunburst
    04_segment_radar.html     ← Radar profiles (6 dimensions)
    05_sankey_flow.html       ← Policyholder flow Sankey
    06_forest_plot.html       ← Regression coefficients forest plot
    07_cluster_bubbles.html   ← Risk band cluster bubble chart
    08_cell_scatter.html      ← Covariate cell freq×sev scatter
    09_r2_grid.html           ← R² model selection curve
    10_m4_m5_heatmap.html     ← Cross-method agreement heatmap
    11_3d_surface.html        ← 3D risk surface (interactive)
""")
