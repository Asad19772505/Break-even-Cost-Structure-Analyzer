# app.py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="Break-even Analyzer", layout="wide")
st.title("📉 Break-even & Cost-Structure Analyzer")

with st.sidebar:
    st.header("🔧 Base Inputs")
    price = st.number_input("Price per unit (P)", min_value=0.0, value=10.0, step=0.5, format="%.2f")
    vc_a  = st.number_input("Variable cost per unit – Structure A (VCₐ)", min_value=0.0, value=4.0, step=0.5, format="%.2f")
    f_a   = st.number_input("Fixed costs – Structure A (Fₐ)", min_value=0.0, value=40000.0, step=1000.0, format="%.0f")

    st.markdown("---")
    compare = st.checkbox("Compare with Structure B", value=True)
    if compare:
        vc_b = st.number_input("Variable cost per unit – Structure B (VCᵦ)", min_value=0.0, value=6.0, step=0.5, format="%.2f")
        f_b  = st.number_input("Fixed costs – Structure B (Fᵦ)", min_value=0.0, value=20000.0, step=1000.0, format="%.0f")

    st.markdown("---")
    st.header("📈 Scenario Range")
    q_max = st.number_input("Max units to chart", min_value=1000, value=20000, step=1000)
    q_step = st.number_input("Step size", min_value=100, value=250, step=50)
    q_now = st.number_input("Current/Target units (for MOS & DOL)", min_value=0, value=10000, step=500)

# ---------- Core functions ----------
def be_units(price, vc, fixed):
    cm = price - vc
    return math.inf if cm <= 0 else fixed / cm

def metrics(price, vc, fixed, q_now):
    cm_u = price - vc
    be_u = be_units(price, vc, fixed)
    be_sales = be_u * price if math.isfinite(be_u) else np.nan
    revenue_now = price * q_now
    vc_now = vc * q_now
    contrib_now = cm_u * q_now
    profit_now = contrib_now - fixed
    mos_units = max(q_now - (be_u if math.isfinite(be_u) else 0), 0)
    mos_pct = (mos_units / q_now * 100) if q_now > 0 else 0
    mos_sales = max(revenue_now - (be_sales if math.isfinite(be_sales) else 0), 0)
    dol = (contrib_now / profit_now) if profit_now != 0 else np.nan
    return {
        "CM/unit": cm_u,
        "BE units": be_u,
        "BE sales": be_sales,
        "Revenue @Q": revenue_now,
        "VC @Q": vc_now,
        "Contribution @Q": contrib_now,
        "Profit @Q": profit_now,
        "MOS units": mos_units,
        "MOS %": mos_pct,
        "MOS $": mos_sales,
        "DOL @Q": dol
    }

def profit_curve(price, vc, fixed, q_array):
    cm_u = price - vc
    return cm_u * q_array - fixed

# ---------- Compute ----------
q = np.arange(0, q_max + 1, q_step, dtype=int)
met_a = metrics(price, vc_a, f_a, q_now)
df = pd.DataFrame({
    "Units": q,
    "Profit_A": profit_curve(price, vc_a, f_a, q)
})

if compare:
    met_b = metrics(price, vc_b, f_b, q_now)
    df["Profit_B"] = profit_curve(price, vc_b, f_b, q)

# Indifference quantity (optional)
indiff_q = None
if compare:
    # (P-VCa)Q - Fa = (P-VCb)Q - Fb  ->  Q = (Fa - Fb) / ((P-VCa) - (P-VCb))
    num = f_a - f_b
    den = (price - vc_a) - (price - vc_b)
    if den != 0:
        indiff_q = num / den

# ---------- KPI cards ----------
def kpi_block(label, value, fmt="{:,.0f}"):
    col = st.container()
    col.metric(label, "—" if value is None or (isinstance(value, float) and np.isnan(value)) else fmt.format(value))

col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi_block("A • BE Units", met_a["BE units"], "{:,.0f}")
with col2:
    kpi_block("A • CM/Unit", met_a["CM/unit"], "${:,.2f}")
with col3:
    kpi_block("A • MOS % @Q", met_a["MOS %"], "{:,.1f}%")
with col4:
    kpi_block("A • DOL @Q", met_a["DOL @Q"], "{:,.2f}")

if compare:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_block("B • BE Units", met_b["BE units"], "{:,.0f}")
    with col2:
        kpi_block("B • CM/Unit", met_b["CM/unit"], "${:,.2f}")
    with col3:
        kpi_block("B • MOS % @Q", met_b["MOS %"], "{:,.1f}%")
    with col4:
        kpi_block("B • DOL @Q", met_b["DOL @Q"], "{:,.2f}")

    if indiff_q is not None and indiff_q >= 0 and np.isfinite(indiff_q):
        st.info(f"📍 **Indifference Volume** (profits equal A vs B): **{indiff_q:,.0f} units**")

# ---------- Summary table ----------
def tidy(m, label):
    return pd.Series({
        "Structure": label,
        "CM/unit": m["CM/unit"],
        "BE units": m["BE units"],
        "BE sales": m["BE sales"],
        "Profit @Q": m["Profit @Q"],
        "MOS units": m["MOS units"],
        "MOS %": m["MOS %"] / 100.0,
        "MOS $": m["MOS $"],
        "DOL @Q": m["DOL @Q"],
    })

summary_rows = [tidy(met_a, "A")]
if compare:
    summary_rows.append(tidy(met_b, "B"))
summary = pd.DataFrame(summary_rows)
st.subheader("📊 Summary")
st.dataframe(summary.style.format({
    "CM/unit": "${:,.2f}",
    "BE units": "{:,.0f}",
    "BE sales": "${:,.0f}",
    "Profit @Q": "${:,.0f}",
    "MOS units": "{:,.0f}",
    "MOS %": "{:.1%}",
    "MOS $": "${:,.0f}",
    "DOL @Q": "{:,.2f}",
}), use_container_width=True)

# ---------- Download ----------
csv = summary.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Summary (CSV)", data=csv, file_name="breakeven_summary.csv", mime="text/csv")

# ---------- Chart ----------
st.subheader("🧭 Profit vs Units")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(0, linewidth=1)
ax.plot(df["Units"], df["Profit_A"], label="Profit – Structure A")
if compare:
    ax.plot(df["Units"], df["Profit_B"], label="Profit – Structure B")

# Mark BE points
if np.isfinite(met_a["BE units"]):
    ax.scatter([met_a["BE units"]], [0], marker="o")
    ax.annotate(f"BE A: {met_a['BE units']:,.0f}", (met_a["BE units"], 0), xytext=(10, 10),
                textcoords="offset points")
if compare and np.isfinite(met_b["BE units"]):
    ax.scatter([met_b["BE units"]], [0], marker="o")
    ax.annotate(f"BE B: {met_b['BE units']:,.0f}", (met_b["BE units"], 0), xytext=(10, -20),
                textcoords="offset points")

# Current Q marker
ax.axvline(q_now, linewidth=1)
ax.annotate(f"Q = {q_now:,.0f}", (q_now, 0), xytext=(5, -25), textcoords="offset points", rotation=90)

ax.set_xlabel("Units")
ax.set_ylabel("Profit ($)")
ax.set_title("Profit vs Units with Break-even Points")
ax.legend()
st.pyplot(fig)

# ---------- Guidance ----------
with st.expander("📌 How to interpret"):
    st.markdown("""
- **Break-even units** = Fixed Costs ÷ (Price − Variable Cost per unit).  
- **Margin of Safety (MOS)** measures how far current/target volume is above break-even.  
- **Degree of Operating Leverage (DOL)** ≈ sensitivity of profit to % change in sales at the selected volume.  
- Use the **indifference volume** (if comparing A vs B) to choose the structure above/below which each structure wins.
""")
