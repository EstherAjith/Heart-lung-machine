"""
HeartLungLab - Virtual laboratory
Experiment 03: Study of Heart Lung Machine (cardiopulmonary bypass) model

Educational simulation only - not connected to medical equipment and not for real patient care.
Run:  streamlit run app.py
"""
import hashlib
import math
import random
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

st.set_page_config(page_title="Heart Lung Machine", page_icon="🫀", layout="wide")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
STEPS = [
    "1. Operating Room",
    "2. Equipment Room",
    "3. Circuit Assembly",
    "4. Priming & Leak Check",
    "5. Run the Machine",
    "6. Viva & Report",
]
EQUIPMENT = [
    (1, "Heart lung machine model", "1"),
    (2, "Connecting cables and power cords", "As required"),
    (3, "Effluent pump", "1"),
    (4, "Blood pump", "1"),
]
SCENARIOS = [
    "Coronary artery bypass grafting (CABG)",
    "Heart valve replacement",
    "Congenital heart defect repair",
]
SLOT_OPTIONS = [
    "Select component",
    "Venous reservoir",
    "Roller (blood) pump",
    "Oxygenator + heat exchanger",
    "Arterial line filter",
]
CORRECT_ORDER = [
    "Venous reservoir",
    "Roller (blood) pump",
    "Oxygenator + heat exchanger",
    "Arterial line filter",
]
ML_PER_REV = 100.0  # roller pump output, mL per revolution (model tubing)
RESERVOIR_ML = 1500.0
SVR = 14.0  # mmHg per (L/min), normothermic
HB = 12.0  # g/dL

BG = "#0e1117"
FG = "#e6e6e6"
BLUE = "#4f7dff"
RED = "#ff4b4b"

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.hero{background:linear-gradient(120deg,#2b1f55 0%,#3d2a7a 60%,#4a2f8f 100%);
      padding:1.4rem 1.8rem;border-radius:18px;margin-bottom:1.2rem;}
.hero h1{margin:0 0 .6rem 0;font-size:2.6rem;}
.hero p{margin:.2rem 0;color:#e8e3ff;}
.hero small{color:#cfc6f5;}
.block-container{padding-top:2rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# State helpers
# ----------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("step", 0)
    ss.setdefault("unlocked", 0)
    ss.setdefault("patient", {})
    ss.setdefault("circuit_ok", False)
    ss.setdefault("oxy_type", "Membrane oxygenator")
    ss.setdefault("primed", False)
    ss.setdefault("prime_log", [])
    ss.setdefault("fault", "none")
    ss.setdefault("fault_found", False)
    ss.setdefault("runs", [])
    ss.setdefault("viva_score", None)
    ss.setdefault("low_start_done", False)


def set_step(i):
    st.session_state.step = i


def advance(i):
    """Mark step i complete and move to the next one."""
    ss = st.session_state
    ss.unlocked = max(ss.unlocked, i + 1)
    ss.step = min(i + 1, len(STEPS) - 1)
    st.rerun()


def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def bsa_mosteller(height_cm, weight_kg):
    return math.sqrt(height_cm * weight_kg / 3600.0)


def header(title, blurb):
    st.markdown(f"## {title}")
    st.write(blurb)


def need(step_index, text):
    """Stop rendering the page if a previous step is not completed."""
    if st.session_state.unlocked < step_index:
        st.warning(text)
        st.stop()


# ----------------------------------------------------------------------------
# Circuit drawing
# ----------------------------------------------------------------------------
PATH = [(1.2, 2.9), (1.2, 2.0), (7.4, 2.0), (8.4, 2.0), (8.4, 5.2), (1.2, 5.2), (1.2, 4.3)]
_seg = np.array([math.dist(PATH[i], PATH[i + 1]) for i in range(len(PATH) - 1)])
_cum = np.concatenate([[0], np.cumsum(_seg)])
PATH_LEN = _cum[-1]
OXY_EXIT_LEN = _cum[2]


def _point_at(s):
    s = s % PATH_LEN
    i = int(np.searchsorted(_cum, s, side="right") - 1)
    i = min(i, len(_seg) - 1)
    f = (s - _cum[i]) / _seg[i]
    x = PATH[i][0] + f * (PATH[i + 1][0] - PATH[i][0])
    y = PATH[i][1] + f * (PATH[i + 1][1] - PATH[i][1])
    return x, y, s


def _box(ax, x, y, w, h, fc, ec="#9aa0b4", z=3, lw=1.4):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=fc, ec=ec, lw=lw, zorder=z,
        )
    )


def _temp_color(t):
    f = min(max((t - 15) / 25.0, 0), 1)
    return (0.25 + 0.75 * f, 0.45 + 0.05 * f, 1.0 - 0.8 * f)


def draw_circuit(flow=0.0, phase=0.0, arm=0.0, temp=37.0, level=70.0,
                 effluent_fill=0.0, blood_fill=1.0, sat=None, oxy_type="Membrane",
                 bath_on=False, assembled=True, title=None):
    fig, ax = plt.subplots(figsize=(10, 4.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # tubing
    xs, ys = zip(*PATH)
    ax.plot(xs, ys, color="#5b6075", lw=7, solid_capstyle="round", zorder=1)
    ax.plot(xs, ys, color="#1b1e2b", lw=4, solid_capstyle="round", zorder=1.1)
    if blood_fill > 0:
        n = int(len(PATH) * blood_fill)
        # coloured fluid in line (venous blue until oxygenator, arterial red after)
        ax.plot([1.2, 1.2, 7.4], [2.9, 2.0, 2.0], color=BLUE, lw=3, alpha=0.55 * blood_fill, zorder=1.2)
        ax.plot([7.4, 8.4, 8.4, 1.2, 1.2], [2.0, 2.0, 5.2, 5.2, 4.3], color=RED, lw=3,
                alpha=0.55 * blood_fill, zorder=1.2)

    # patient
    _box(ax, 0.25, 2.9, 1.9, 1.4, "#2a2f45")
    ax.text(1.2, 3.6, "PATIENT\n(heart stopped)", ha="center", va="center", color=FG, fontsize=8.5, zorder=5)
    ax.text(1.35, 2.45, "SVC / IVC\n(venous drain)", color=BLUE, fontsize=7, ha="left", va="center", zorder=5)
    ax.text(1.35, 4.75, "Aorta / femoral artery\n(arterial return)", color=RED, fontsize=7, ha="left", va="center", zorder=5)

    # reservoir
    _box(ax, 2.4, 1.4, 1.2, 1.2, "#161a26")
    h = 1.16 * (level / 100.0)
    ax.add_patch(Rectangle((2.42, 1.42), 1.16, h, fc=BLUE, alpha=0.55, zorder=3.2, lw=0))
    ax.text(3.0, 2.85, "Venous\nreservoir", color=FG, fontsize=7.5, ha="center", va="bottom", zorder=5)

    # blood pump
    ax.add_patch(Circle((4.6, 2.0), 0.55, fc="#161a26", ec="#9aa0b4", lw=1.4, zorder=3))
    for k in range(2):
        a = arm + k * math.pi
        ax.plot([4.6, 4.6 + 0.42 * math.cos(a)], [2.0, 2.0 + 0.42 * math.sin(a)], color="#cfd3e6", lw=2.5, zorder=4)
        ax.add_patch(Circle((4.6 + 0.42 * math.cos(a), 2.0 + 0.42 * math.sin(a)), 0.09, fc="#ffb347", zorder=4.2))
    ax.text(4.6, 2.85, "Blood pump\n(roller)", color=FG, fontsize=7.5, ha="center", va="bottom", zorder=5)

    # oxygenator + heat exchanger
    _box(ax, 5.8, 1.4, 1.6, 1.2, "#1b2236")
    for k in range(5):
        ax.plot([5.95, 7.25], [1.55 + k * 0.22] * 2, color="#3b4468", lw=1, zorder=3.3)
    ax.text(6.6, 2.85, "Oxygenator +\nheat exchanger", color=FG, fontsize=7.5, ha="center", va="bottom", zorder=5)
    ax.text(6.6, 1.2, "O$_2$ in / CO$_2$ out" if oxy_type.startswith("Membrane") else "95% O$_2$ + 5% CO$_2$ bubbles",
            color="#8fd19e", fontsize=6.5, ha="center", va="center", zorder=5)

    # water bath / heater
    _box(ax, 5.8, 0.15, 1.6, 0.7, _temp_color(temp), ec="#9aa0b4", z=3)
    ax.text(6.6, 0.5, f"Water bath {temp:.1f} °C" + ("" if bath_on else " (heater off)"),
            color="#10131c", fontsize=7.5, ha="center", va="center", zorder=5, fontweight="bold")
    ax.plot([6.6, 6.6], [0.85, 1.4], color="#9aa0b4", lw=2, ls=":", zorder=2)

    # arterial filter
    _box(ax, 8.1, 3.2, 0.6, 0.9, "#1b2236")
    ax.text(8.8, 3.65, "Arterial\nfilter", color=FG, fontsize=7.5, ha="left", va="center", zorder=5)

    # effluent pump + container
    ax.plot([3.0, 3.0], [1.4, 0.95], color="#5b6075", lw=3, zorder=1)
    ax.add_patch(Circle((3.0, 0.65), 0.3, fc="#161a26", ec="#9aa0b4", lw=1.3, zorder=3))
    ax.plot([3.0 + 0.2 * math.cos(-arm), 3.0 - 0.2 * math.cos(-arm)],
            [0.65 + 0.2 * math.sin(-arm), 0.65 - 0.2 * math.sin(-arm)], color="#cfd3e6", lw=2, zorder=4)
    ax.plot([3.3, 3.8], [0.65, 0.65], color="#5b6075", lw=3, zorder=1)
    _box(ax, 3.8, 0.2, 1.0, 0.9, "#161a26")
    ax.add_patch(Rectangle((3.82, 0.22), 0.96, 0.86 * effluent_fill, fc="#e0c35a", alpha=0.6, zorder=3.2, lw=0))
    ax.text(3.0, 0.15, "Effluent pump", color=FG, fontsize=7, ha="center", va="top", zorder=5)
    ax.text(4.3, 1.2, "Effluent\ncontainer", color=FG, fontsize=7, ha="center", va="bottom", zorder=5)

    # moving fluid dots
    if flow > 0 and assembled:
        n = 22
        for k in range(n):
            x, y, s = _point_at(phase * PATH_LEN + k * PATH_LEN / n)
            ax.add_patch(Circle((x, y), 0.075, fc=BLUE if s < OXY_EXIT_LEN else RED, ec="none", zorder=4.5))

    if sat is not None:
        ax.text(8.4, 5.45, f"SaO$_2$ {sat:.0f}%", color=RED, fontsize=8, ha="center", zorder=5)
        ax.text(2.0, 1.15, "SvO$_2$ ~65%", color=BLUE, fontsize=8, ha="center", zorder=5)
    if title:
        ax.text(5.0, 5.85, title, color=FG, fontsize=10, ha="center", va="center")
    fig.tight_layout(pad=0.3)
    return fig


def show_fig(fig, slot=None):
    (slot or st).pyplot(fig, clear_figure=True)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Physiology model
# ----------------------------------------------------------------------------
def physio(p, t_s, bsa):
    """State of the model after t_s seconds of running."""
    on = p["power"]
    rpm = p["rpm"] if on else 0.0
    q = ML_PER_REV * rpm / 1000.0  # L/min
    tau = 180.0
    if p["heater"] and on:
        temp = p["setpoint"] + (37.0 - p["setpoint"]) * math.exp(-t_s / tau)
    else:
        temp = 37.0
    ratio = (p["gas"] / q) if q > 0 else 0.0
    eff = 1 - math.exp(-4 * ratio)
    sat = 65 + (99 - 65) * eff - (1.0 if p["oxy"].startswith("Bubble") else 0.0) if p["power"] and q > 0 else 65.0
    sat = min(sat, 99.5)
    vo2 = 250.0 * (bsa / 1.8) * 2 ** ((temp - 37.0) / 10.0)
    do2 = q * 10 * 1.34 * HB * sat / 100.0
    map_ = q * SVR * (1 + 0.015 * (37 - temp))
    eff_ml = p["effluent"] if on else 0.0
    level = max(0.0, p["level0"] - eff_ml * (t_s / 60.0) / RESERVOIR_ML * 100.0)
    collected = min(1.0, eff_ml * (t_s / 60.0) / 1500.0)
    return dict(q=q, temp=temp, sat=sat, do2=do2, vo2=vo2, map=map_, level=level, collected=collected)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
init_state()
ss = st.session_state

with st.sidebar:
    st.markdown("### Experiment navigation")
    for i, label in enumerate(STEPS):
        prefix = "➤ " if i == ss.step else ("✓ " if i < ss.unlocked else "")
        st.button(prefix + label, key=f"nav{i}", disabled=i > ss.unlocked, on_click=set_step, args=(i,))
    st.divider()
    st.caption("Ex. No. 3 · Study of Heart Lung Machine model")
    st.button("↺ Restart experiment", on_click=reset_all)

st.markdown(
    """
<div class="hero">
  <h1>🫀 Heart Lung Machine</h1>
  <p>Interactive virtual laboratory · Experiment 03: Study the Working of a Heart Lung Machine</p>
  <small>Educational simulation only — not connected to medical equipment and not for real patient care.</small>
</div>
""",
    unsafe_allow_html=True,
)

step = ss.step

# ----------------------------------------------------------------------------
# 1. Operating room
# ----------------------------------------------------------------------------
if step == 0:
    header("01 · Operating room",
           "Register the simulated patient, gather a brief history and choose the teaching scenario. "
           "Required fields are marked with *.")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Patient name / case ID *", value=ss.patient.get("name", ""))
            age = st.number_input("Age (years) *", 1, 100, ss.patient.get("age", 55))
            sex = st.selectbox("Sex", ["Male", "Female"], index=0 if ss.patient.get("sex", "Male") == "Male" else 1)
            weight = st.number_input("Weight (kg) *", 20.0, 200.0, float(ss.patient.get("weight", 70.0)), step=1.0)
            height = st.number_input("Height (cm) *", 100.0, 220.0, float(ss.patient.get("height", 170.0)), step=1.0)
        with c2:
            hist = st.text_area("Relevant medical history / allergies",
                                value=ss.patient.get("hist", ""),
                                placeholder="Enter known history or 'unknown'", height=130)
            scen = st.selectbox("Surgical scenario", SCENARIOS,
                                index=SCENARIOS.index(ss.patient.get("scenario", SCENARIOS[0])))
        bsa = bsa_mosteller(height, weight)
        target = 2.4 * bsa
        m1, m2, m3 = st.columns(3)
        m1.metric("Body surface area", f"{bsa:.2f} m²")
        m2.metric("Target pump flow (2.4 L/min/m²)", f"{target:.2f} L/min")
        m3.metric("Target roller speed", f"{target * 1000 / ML_PER_REV:.0f} rpm")
    st.info("The heart-lung machine takes over circulation (cardio) and oxygenation (pulmonary) "
            "while the surgeon works on a stopped heart.")
    if st.button("Register patient & go to equipment room", type="primary"):
        if not name.strip():
            st.error("Please enter a patient name / case ID.")
        else:
            ss.patient = dict(name=name.strip(), age=age, sex=sex, weight=weight, height=height,
                              hist=hist, scenario=scen, bsa=bsa, target=target)
            seed = int(hashlib.md5(name.strip().encode()).hexdigest(), 16)
            ss.fault = random.Random(seed).choice(["none", "bubble", "leak"])
            advance(0)

# ----------------------------------------------------------------------------
# 2. Equipment room
# ----------------------------------------------------------------------------
elif step == 1:
    need(1, "Complete step 1 first.")
    p = ss.patient
    header("02 · Equipment room", "Collect and verify the equipment listed for the experiment.")
    st.success(f"Patient: **{p['name']}** · Scenario: **{p['scenario']}** · "
               f"BSA {p['bsa']:.2f} m² · Target flow {p['target']:.2f} L/min")
    st.markdown("#### Equipment checklist")
    st.table(pd.DataFrame(EQUIPMENT, columns=["S.No", "Name of the Equipment", "Quantity"]).set_index("S.No"))
    st.write("Tick each item once you have verified it on the bench. Real blood is replaced by "
             "**water** in the teaching model.")
    ok = []
    for sno, nm, _ in EQUIPMENT:
        ok.append(st.toggle(f"Verified: {nm}", key=f"eq{sno}"))
    with st.expander("Theory in brief"):
        st.markdown(
            "- **Pump** – drives blood in the arterial system (roller pump, ~2.4 L/min/m², ≈5 L/min adult).\n"
            "- **Oxygenator** – exchanges O₂ and CO₂ (bubble or membrane type).\n"
            "- **Heat exchanger** – water-jacketed tubes to cool/rewarm blood; hypothermia lowers O₂ demand.\n"
            "- **Tubing & filter** – connect the circuit and trap clots/debris."
        )
    if all(ok):
        st.success("All equipment verified.")
        if st.button("Continue to circuit assembly", type="primary"):
            advance(1)
    else:
        st.warning("Verify all four items to continue.")

# ----------------------------------------------------------------------------
# 3. Circuit assembly
# ----------------------------------------------------------------------------
elif step == 2:
    need(2, "Complete the equipment check first.")
    header("03 · Circuit assembly",
           "Connect the tubing as per the circuit diagram. Blood leaves the patient through the venous "
           "cannula and returns through the arterial cannula.")
    st.caption("Training illustration only. Follow the actual device manufacturer's instructions in clinical practice.")
    left, right = st.columns([3, 2])
    with right:
        st.markdown("**Start:** Venous cannula (SVC/IVC → right atrium)")
        slots = [st.selectbox(f"Component {i + 1}", SLOT_OPTIONS, key=f"slot{i}") for i in range(4)]
        st.markdown("**End:** Arterial cannula (ascending aorta / femoral artery)")
        oxy = st.radio("Oxygenator type", ["Membrane oxygenator", "Bubble oxygenator"], horizontal=True)
        st.caption("Membrane: fine tubes, O₂/CO₂ diffuse through walls. "
                   "Bubble: gas bubbles (95% O₂ + 5% CO₂) with a defoaming reservoir.")
        eff_conn = st.toggle("Connect effluent pump to the reservoir")
        pump_conn = st.toggle("Connect blood pump and power cords")
        check = st.button("Check circuit", type="primary")
    with left:
        show_fig(draw_circuit(flow=0, assembled=False, level=0, blood_fill=0.0, oxy_type=oxy,
                              title="Circuit diagram (reference)"))
    if check:
        if "Select component" in slots or len(set(slots)) < 4:
            st.error("Fill all four positions with different components.")
        elif slots != CORRECT_ORDER:
            bad = [i + 1 for i in range(4) if slots[i] != CORRECT_ORDER[i]]
            st.error(f"Position(s) {bad} are wrong. Hint: blood is collected, pushed by the pump, "
                     "oxygenated/temperature-controlled, then filtered before returning to the artery.")
        elif not (eff_conn and pump_conn):
            st.error("Connect both the effluent pump and the blood pump / power cords.")
        else:
            ss.circuit_ok = True
            ss.oxy_type = oxy
            st.success("Circuit assembled correctly: venous cannula → reservoir → roller pump → "
                       "oxygenator + heat exchanger → arterial filter → arterial cannula.")
    if ss.circuit_ok:
        if st.button("Continue to priming"):
            advance(2)

# ----------------------------------------------------------------------------
# 4. Priming and leak check
# ----------------------------------------------------------------------------
elif step == 3:
    need(3, "Assemble the circuit first.")
    header("04 · Priming & leak check",
           "Fill the effluent and blood compartments with water, prime the tubing at low speed and look "
           "for air bubbles or leaks.")
    c1, c2 = st.columns([2, 3])
    with c1:
        bfill = st.slider("Blood compartment fill (%)", 0, 100, 0, step=5)
        efill = st.slider("Effluent compartment fill (%)", 0, 100, 0, step=5)
        rpm_prime = st.slider("Priming speed - blood pump (rpm)", 5, 100, 60)
        st.caption("Priming at high speed can entrain air. Use low speed (≤ 20 rpm).")
        go = st.button("Prime tubing", type="primary")
    with c2:
        slot = st.empty()
        show_fig(draw_circuit(level=bfill, effluent_fill=0.0, blood_fill=bfill / 100.0,
                              oxy_type=ss.oxy_type, title="Priming"), slot)
    if go:
        if bfill < 70 or efill < 50:
            st.error("Fill the blood compartment to at least 70% and effluent compartment to at least 50%.")
        elif rpm_prime > 20:
            st.error("Priming speed too high - air entrainment risk. Set the pump to low speed (≤ 20 rpm).")
        else:
            bar = st.progress(0, text="Priming tubing...")
            arm = 0.0
            for f in range(31):
                arm += 0.5
                show_fig(draw_circuit(flow=rpm_prime * 0.1, phase=f / 30 * 0.5, arm=arm, level=bfill,
                                      blood_fill=bfill / 100.0, oxy_type=ss.oxy_type, title="Priming"), slot)
                bar.progress(f / 30, text="Priming tubing...")
            bar.empty()
            ss.prime_log = ["Compartments filled", f"Primed at {rpm_prime} rpm"]
            ss.primed = False
            ss.fault_found = False
            ss.prime_done = True
    if ss.get("prime_done"):
        st.markdown("#### Inspection")
        finding = st.radio("What do you observe in the primed tubing?",
                           ["No issue", "Air bubbles in the line", "Water leak at a connector"], index=None)
        action_map = {
            "bubble": ("Air bubbles in the line", "Purge air by recirculating at low speed"),
            "leak": ("Water leak at a connector", "Tighten the connector and re-prime"),
            "none": ("No issue", None),
        }
        truth, fix = action_map[ss.fault]
        if finding is not None:
            if finding != truth:
                st.error("Look again - your observation does not match the circuit state.")
            elif fix is None:
                st.success("Tubing is clear: no bubbles and no leaks.")
                ss.primed = True
            else:
                st.warning(f"Correct: {truth.lower()}. Choose the corrective action.")
                act = st.selectbox("Corrective action", ["Select action", "Increase pump speed to flush",
                                                         "Purge air by recirculating at low speed",
                                                         "Tighten the connector and re-prime", "Ignore and proceed"])
                if act == fix:
                    st.success("Fault corrected and circuit re-checked.")
                    ss.primed = True
                elif act != "Select action":
                    st.error("That action is not appropriate for this fault.")
    if ss.primed:
        if st.button("Continue to run the machine", type="primary"):
            ss.prime_done = False
            advance(3)

# ----------------------------------------------------------------------------
# 5. Run the machine
# ----------------------------------------------------------------------------
elif step == 4:
    need(4, "Prime the circuit first.")
    p = ss.patient
    header("05 · Run the machine",
           "Set blood pump at low speed, switch on the heater and power the modules, then observe how blood "
           "is circulated and oxygenated.")
    c1, c2, c3 = st.columns(3)
    with c1:
        power = st.toggle("Power ON modules (pumps, oxygenator gas)")
        heater = st.toggle("Switch ON heater / water bath")
        setpoint = st.slider("Heater setpoint (°C)", 20, 37, 32)
    with c2:
        rpm = st.slider("Blood pump speed (rpm)", 0, 100, 20)
        eff = st.slider("Effluent pump (mL/min)", 0, 200, 50, step=10)
    with c3:
        gas = st.slider("Sweep gas flow (L/min)", 0.0, 10.0, 5.0, step=0.5)
        st.metric("Target flow", f"{p['target']:.2f} L/min", help="2.4 L/min/m² × BSA")
        st.caption(f"Model flow = {ML_PER_REV:.0f} mL × rpm → {rpm * ML_PER_REV / 1000:.1f} L/min")

    params = dict(power=power, heater=heater, setpoint=setpoint, rpm=rpm, effluent=eff, gas=gas,
                  oxy=ss.oxy_type, level0=70.0)
    left, right = st.columns([3, 2])
    fig_slot = left.empty()
    met_slot = right.empty()
    show_fig(draw_circuit(level=70, oxy_type=ss.oxy_type, bath_on=heater, title="Heart-lung model (stopped)"), fig_slot)

    run = st.button("▶ Run simulation (10 min of bypass)", type="primary")
    if run:
        if not power:
            st.error("Power on the modules first.")
        elif heater is False and setpoint < 37:
            st.error("Switch on the heater to control the water-bath temperature.")
        elif rpm > 30 and not ss.low_start_done:
            st.error("Start the first run at low speed (≤ 30 rpm), then increase gradually.")
        elif rpm == 0:
            st.error("Blood pump speed is 0 - nothing will circulate.")
        else:
            if rpm <= 30:
                ss.low_start_done = True
            frames, dur = 40, 600.0
            arm = phase = 0.0
            series = []
            for f in range(frames + 1):
                t = dur * f / frames
                s = physio(params, t, p["bsa"])
                arm += s["q"] * 0.12
                phase = (phase + s["q"] * 0.012) % 1.0
                series.append(dict(minute=t / 60, Temperature=s["temp"], Level=s["level"]))
                show_fig(draw_circuit(flow=s["q"], phase=phase, arm=arm, temp=s["temp"], level=s["level"],
                                      effluent_fill=s["collected"], sat=s["sat"], oxy_type=ss.oxy_type,
                                      bath_on=heater, title=f"Running · t = {t / 60:.1f} min"), fig_slot)
                with met_slot.container():
                    a, b = st.columns(2)
                    a.metric("Pump flow", f"{s['q']:.2f} L/min")
                    b.metric("Blood temp", f"{s['temp']:.1f} °C")
                    a.metric("Arterial SaO₂", f"{s['sat']:.1f} %")
                    b.metric("Mean art. pressure", f"{s['map']:.0f} mmHg")
                    a.metric("O₂ delivery / demand", f"{s['do2'] / s['vo2']:.1f}")
                    b.metric("Reservoir level", f"{s['level']:.0f} %")
                time.sleep(0.04)
            s = physio(params, dur, p["bsa"])
            ss.runs.append({
                "Run": len(ss.runs) + 1, "Oxygenator": ss.oxy_type.split()[0],
                "Blood pump (rpm)": rpm, "Flow (L/min)": round(s["q"], 2),
                "Target (L/min)": round(p["target"], 2), "Heater setpoint (°C)": setpoint if heater else "off",
                "Final blood temp (°C)": round(s["temp"], 1), "Sweep gas (L/min)": gas,
                "Arterial SaO2 (%)": round(s["sat"], 1), "MAP (mmHg)": round(s["map"]),
                "DO2:VO2": round(s["do2"] / s["vo2"], 1), "Effluent (mL/min)": eff,
                "Reservoir end (%)": round(s["level"]),
            })
            ss.last_series = pd.DataFrame(series).set_index("minute")
            ss.last_state = s
            ss.last_params = params
            st.rerun()

    if ss.get("last_state"):
        s, lp = ss.last_state, ss.last_params
        st.markdown("#### Result of last run")
        msgs = []
        q_err = abs(s["q"] - p["target"]) / p["target"]
        msgs.append(("success" if q_err <= 0.15 else "warning",
                     f"Flow {s['q']:.2f} L/min is {'within' if q_err <= 0.15 else 'outside'} ±15% of target "
                     f"({p['target']:.2f} L/min)."))
        msgs.append(("success" if s["sat"] >= 95 else "warning",
                     f"Arterial saturation {s['sat']:.1f}% — "
                     f"{'well oxygenated' if s['sat'] >= 95 else 'low: increase sweep gas relative to blood flow'}."))
        msgs.append(("success" if 50 <= s["map"] <= 80 else "warning",
                     f"Mean arterial pressure {s['map']:.0f} mmHg "
                     f"({'acceptable (50–80)' if 50 <= s['map'] <= 80 else 'outside 50–80 mmHg'})."))
        msgs.append(("success" if s["do2"] / s["vo2"] >= 2 else "error",
                     f"O₂ delivery is {s['do2']:.0f} mL/min vs demand {s['vo2']:.0f} mL/min "
                     f"(cooling to {s['temp']:.1f} °C lowers demand)."))
        msgs.append(("success" if s["level"] >= 25 else "error",
                     f"Reservoir level {s['level']:.0f}% — "
                     f"{'adequate' if s['level'] >= 25 else 'LOW: air could enter the circuit; reduce effluent rate'}."))
        for kind, text in msgs:
            getattr(st, kind)(text)
        ch1, ch2 = st.columns(2)
        ch1.caption("Blood temperature (°C) vs time (min)")
        ch1.line_chart(ss.last_series[["Temperature"]])
        ch2.caption("Reservoir level (%) vs time (min)")
        ch2.line_chart(ss.last_series[["Level"]])
        st.info("Try: bubble vs membrane oxygenator, different heater setpoints, low sweep gas, high effluent rate. "
                "Each run is added to the observation table.")
    if ss.runs:
        if st.button("Continue to viva & report", type="primary"):
            advance(4)

# ----------------------------------------------------------------------------
# 6. Viva & report
# ----------------------------------------------------------------------------
else:
    need(5, "Run the machine at least once first.")
    p = ss.patient
    header("06 · Viva & report", "Review your observations, answer the viva questions and download the report.")
    st.markdown("### Observation table")
    df = pd.DataFrame(ss.runs)
    st.dataframe(df, hide_index=True)

    QUIZ = [
        ("What are the main components of a heart–lung machine?",
         "Pump, oxygenator and heat exchanger (with tubing and filter)",
         ["Pump, oxygenator and heat exchanger (with tubing and filter)",
          "Ventilator, defibrillator and ECG monitor", "Dialyser, syringe pump and heater",
          "Pacemaker, oxygenator and ultrasound probe"]),
        ("Which pump type is most commonly used in a heart–lung machine?",
         "Roller pump", ["Roller pump", "Centrifugal turbine only", "Piston syringe pump", "Peristaltic fan pump"]),
        ("Why is the blood cooled during bypass?",
         "To reduce the oxygen requirement of the tissues",
         ["To reduce the oxygen requirement of the tissues", "To make blood clot faster",
          "To increase carbon dioxide level", "To raise blood pressure"]),
        ("Typical adult pump flow rate is about:",
         "2.4 L/min/m² (≈ 5 L/min)", ["2.4 L/min/m² (≈ 5 L/min)", "0.5 L/min", "15 L/min", "50 mL/min"]),
        ("A bubble oxygenator exposes blood to a gas mixture of:",
         "95% O₂ and 5% CO₂", ["95% O₂ and 5% CO₂", "50% N₂ and 50% O₂", "100% CO₂", "21% O₂ and 79% N₂"]),
        ("Where is blood drained from and returned to the patient?",
         "Drained from vena cavae / right atrium; returned to aorta or femoral artery",
         ["Drained from vena cavae / right atrium; returned to aorta or femoral artery",
          "Drained from aorta; returned to vena cava", "Drained from lungs; returned to left ventricle",
          "Drained from femoral vein; returned to the right atrium"]),
    ]
    st.markdown("### Viva")
    with st.form("viva"):
        answers = []
        for i, (q, correct, opts) in enumerate(QUIZ):
            o = opts[:]
            random.Random(i + 7).shuffle(o)
            answers.append(st.radio(f"Q{i + 1}. {q}", o, index=None, key=f"viva{i}"))
        sub = st.form_submit_button("Submit viva", type="primary")
    if sub:
        if None in answers:
            st.error("Answer all questions.")
        else:
            score = sum(a == q[1] for a, q in zip(answers, QUIZ))
            ss.viva_score = score
            for i, (a, q) in enumerate(zip(answers, QUIZ)):
                (st.success if a == q[1] else st.error)(
                    f"Q{i + 1}: {'Correct' if a == q[1] else 'Incorrect'} — {q[1]}")
    if ss.viva_score is not None:
        st.metric("Viva score", f"{ss.viva_score} / {len(QUIZ)}")

    # report
    lines = [
        "# Ex. No. 3 — Study of Heart Lung Machine Model",
        f"Date: {datetime.now():%d-%m-%Y}", "",
        "## Aim", "To study the working of heart lung machine.", "",
        "## Equipment", *[f"- {n} (qty: {q})" for _, n, q in EQUIPMENT], "",
        "## Case", f"- Patient/case ID: {p['name']}", f"- Age/Sex: {p['age']} / {p['sex']}",
        f"- Weight/Height: {p['weight']} kg / {p['height']} cm", f"- BSA: {p['bsa']:.2f} m²",
        f"- Target flow: {p['target']:.2f} L/min", f"- Scenario: {p['scenario']}", "",
        "## Procedure followed",
        "1. Tubing connected: venous cannula → reservoir → roller pump → oxygenator + heat exchanger → "
        "arterial filter → arterial cannula",
        "2. Effluent pump and blood pump connected",
        "3. Effluent and blood compartments filled with water",
        "4. Tubing primed; checked for bubbles/leaks", "5. Blood pump set at low speed",
        "6. Heater switched on and modules powered",
        "7. Circulation and oxygenation observed", "",
        "## Observation table", "```\n" + df.to_string(index=False) + "\n```", "",
        f"## Viva score\n{ss.viva_score if ss.viva_score is not None else 'not attempted'} / {len(QUIZ)}", "",
        "## Result", "Thus the working of heart lung machine is studied with a working model.",
    ]
    report = "\n".join(lines)
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Download report (.md)", report, file_name="HLM_lab_report.md")
    d2.download_button("⬇ Download observations (.csv)", df.to_csv(index=False), file_name="HLM_observations.csv")
    st.success("Result: Thus the working of heart lung machine is studied with a working model.")
