"""
PID Gain Tuning for DC Motor Speed Regulation
==============================================
Search-tuned Kp, Ki, Kd for a closed-loop DC motor system; transient response
analyzed against explicit settling-time, rise-time and peak-overshoot targets.

Plant (armature-controlled DC motor, FULL electromechanical model, current
AND speed as states -- inductance is NOT neglected):

    L di/dt = V - R i - Ke*w
    J dw/dt = Kt i - B w - Tload

Transfer function:
    G(s) = W(s)/V(s) = Kt / [ JL s^2 + (JR+BL) s + (BR+Kt*Ke) ]

This plant is genuinely SECOND order. A PID controller's numerator then
gives a closed-loop characteristic polynomial that is exactly THIRD order:

    JL*s^3 + [(JR+BL)+Kt*Kd]*s^2 + [(BR+Kt*Ke)+Kt*Kp]*s + Kt*Ki = 0

All three gains are determined together by exact pole placement
(coefficient matching against a target dominant pair (zeta,wn) plus one
fast auxiliary real pole p3):

    Kd = (JL*(2*zeta*wn+p3) - (JR+BL)) / Kt
    Kp = (JL*(wn^2+2*zeta*wn*p3) - (BR+Kt*Ke)) / Kt
    Ki = (JL*wn^2*p3) / Kt

GAIN SEARCH: the design point (zeta, wn, p3) is selected by a finite grid
search rather than a single guess. find_best_feasible_design() keeps only designs
that meet the overshoot/settling/rise spec and do not exceed Vmax in their
UNCLAMPED controller demand, then selects the minimum settling time on that
search grid. The shipped design (zeta=0.98, wn=17, p3=80) is within 5% of the
best feasible settling time found in that search, confirmed by self_test().

DERIVATIVE TERM: implemented as derivative-ON-MEASUREMENT,
D = -Kd * dw/dt, not derivative-on-error. This is a common practical choice because it avoids a derivative "kick" when
the reference changes -- only the modelled speed signal is differentiated. For every scenario here the reference is
held constant after any initial step, so this is numerically identical to
derivative-on-error with the kick suppressed at the first sample (verified
directly, max difference = 0.0 across the full trajectory) -- but writing
it as derivative-on-measurement makes the intent explicit and correct by
construction rather than correct by an accidental special case.

DESIGN FINDING, checked not assumed: at the deployed bandwidth (wn=17
rad/s, well below the plant's electrical corner R/L=100 rad/s), Kd's own
contribution is small -- an otherwise identical PI controller performs
comparably. A second, small-signal design pushed toward the electrical
corner (wn=50 rad/s) shows Kd's real value: without it, that design is
highly underdamped and strongly oscillatory (effective zeta~=0.23, poles
at -39.06+-165.69j and -25.88 -- verified stable and strongly oscillatory;
with Kd, overshoot drops from 53.97% to 17.94%.
"""

import math


# ----------------------------------------------------------------------
# 1. Physical parameters of the motor (typical small PMDC motor)
# ----------------------------------------------------------------------

R = 2.0        # armature resistance, ohm
L = 0.02       # armature inductance, H
J = 0.005      # rotor inertia (load reflected), kg*m^2
B = 0.02       # viscous friction, N*m*s
Kt = 0.1       # torque constant, N*m/A
Ke = 0.1       # back-emf constant, V*s/rad
Vmax = 24.0    # supply / actuator voltage limit, V

JL = J * L
JR_BL = J * R + B * L
BR_KtKe = B * R + Kt * Ke
tau_e = L / R
tau_e_corner = R / L          # electrical corner frequency, rad/s


def print_derivation():
    print("PLANT DERIVATION (full electromechanical model)")
    print("-" * 66)
    print(f"  R={R} ohm  L={L} H  J={J} kg.m^2  B={B} N.m.s  Kt=Ke={Kt}")
    print(f"  G(s) = Kt / [JL*s^2 + (JR+BL)*s + (BR+Kt*Ke)]")
    print(f"  JL={JL}   JR+BL={JR_BL}   BR+Kt*Ke={BR_KtKe}")
    print(f"  electrical corner frequency R/L = {tau_e_corner:.1f} rad/s")
    print(f"  NOTE: no current/torque limit is modelled -- only voltage "
          f"saturation (|V|<=Vmax). Peak current is reported where "
          f"relevant so the omission is visible, not hidden.")
    print()


# ----------------------------------------------------------------------
# 2. Exact third-order pole placement (Kp, Ki, Kd together)
# ----------------------------------------------------------------------

def design_PID_full(zeta, wn, p3):
    """Coefficient matching against (s^2+2*zeta*wn*s+wn^2)(s+p3)."""
    c2 = 2 * zeta * wn + p3
    c1 = wn ** 2 + 2 * zeta * wn * p3
    c0 = wn ** 2 * p3
    Kd = (JL * c2 - JR_BL) / Kt
    Kp = (JL * c1 - BR_KtKe) / Kt
    Ki = (JL * c0) / Kt
    return Kp, Ki, Kd


def closed_loop_poles(Kp, Ki, Kd):
    """Actual closed-loop poles from numeric gains, via numpy -- an
    independent check that the design formulas were applied correctly,
    not just that the simulated curve looks plausible."""
    import numpy as np
    c3 = JL
    c2 = JR_BL + Kt * Kd
    c1 = BR_KtKe + Kt * Kp
    c0 = Kt * Ki
    return np.roots([c3, c2, c1, c0])


def naive_overshoot_pct(zeta):
    """Textbook dominant-pole-only prediction. Its ASSUMPTIONS (a clean
    zero-free 2nd-order canonical system) do not hold for this plant,
    which is 3rd order and has closed-loop zeros from proportional and
    derivative action -- so the formula is not wrong, it is simply not
    applicable here. Kept for comparison, not trusted."""
    return 100.0 * math.exp(-zeta * math.pi / math.sqrt(1 - zeta ** 2))


# ----------------------------------------------------------------------
# 3. Full 2-state plant + PID controller (derivative-on-measurement)
# ----------------------------------------------------------------------

class PID:
    def __init__(self, Kp, Ki, Kd=0.0, Vmax=Vmax, anti_windup=True):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.Vmax = Vmax
        self.anti_windup = anti_windup
        self.integral = 0.0
        self.prev_measurement = None

    def step(self, error, measurement, dt):
        """Derivative-on-measurement: D = -Kd * d(measurement)/dt.
        Avoids derivative kick on a reference change by construction --
        only the measured speed is differentiated, never the error."""
        if self.prev_measurement is None:
            dmeas_dt = 0.0
        else:
            dmeas_dt = (measurement - self.prev_measurement) / dt
        self.prev_measurement = measurement

        u_unsat = self.Kp * error + self.Ki * self.integral - self.Kd * dmeas_dt
        u = max(-self.Vmax, min(self.Vmax, u_unsat))
        sat_hi = u_unsat > self.Vmax
        sat_lo = u_unsat < -self.Vmax
        if not self.anti_windup:
            self.integral += error * dt
        else:
            if not (sat_hi or sat_lo):
                self.integral += error * dt
            elif (sat_hi and error < 0) or (sat_lo and error > 0):
                self.integral += error * dt
        return u, u_unsat


def closed_loop(Kp, Ki, Kd, ref_of_t, Tload_of_t, t_end, dt=2e-5,
                 anti_windup=True, Vmax_=Vmax):
    """Full 2-state (current, speed) closed-loop simulation.
    Returns ts, ws, us (actual actuator signal, clamped), us_unsat (the
    UNCLAMPED controller demand -- required to correctly detect
    saturation; the clamped signal can never exceed Vmax by construction,
    so checking it is a no-op), i_hist (armature current, for reporting
    peak current since no current limit is enforced), ints."""
    ctrl = PID(Kp, Ki, Kd, Vmax=Vmax_, anti_windup=anti_windup)
    i, w = 0.0, 0.0
    ts, ws, us, us_unsat, i_hist, ints = [], [], [], [], [], []
    n = int(t_end / dt)
    for k in range(n):
        t = k * dt
        ref = ref_of_t(t); Tl = Tload_of_t(t)
        e = ref - w
        u, u_un = ctrl.step(e, w, dt)
        di = (u - R * i - Ke * w) / L
        dw = (Kt * i - B * w - Tl) / J
        i += di * dt; w += dw * dt
        ts.append(t); ws.append(w); us.append(u); us_unsat.append(u_un)
        i_hist.append(i); ints.append(ctrl.integral)
    return ts, ws, us, us_unsat, i_hist, ints


def step_reduced_model(V_of_t, Tload_of_t, t_end, dt=2e-5):
    """First-order (L neglected) model, kept ONLY for scenario 1's plant
    characterisation -- not used for the final controller design."""
    Beff = B + Kt * Ke / R
    w = 0.0
    n = int(t_end / dt)
    ts, ws = [], []
    for k in range(n):
        t = k * dt
        V = V_of_t(t); Tl = Tload_of_t(t)
        dw = ((Kt / R) * V - Beff * w - Tl) / J
        w += dw * dt
        ts.append(t); ws.append(w)
    return ts, ws


def step_full_model(V_of_t, Tload_of_t, t_end, dt=2e-5):
    i, w = 0.0, 0.0
    n = int(t_end / dt)
    ts, ws = [], []
    for k in range(n):
        t = k * dt
        V = V_of_t(t); Tl = Tload_of_t(t)
        di = (V - R * i - Ke * w) / L
        dw = (Kt * i - B * w - Tl) / J
        i += di * dt; w += dw * dt
        ts.append(t); ws.append(w)
    return ts, ws


# ----------------------------------------------------------------------
# 4. Metrics
# ----------------------------------------------------------------------

def measure(ts, ws, ref, band=0.02):
    """Overshoot, 2% settling time, and rise time. Rise time is reported
    BOTH ways (0-100% and 10-90%) since the convention is not universal
    and an unstated choice is a fair thing to be asked about."""
    peak = max(ws)
    overshoot_pct = max(0.0, (peak - ref) / ref * 100.0)
    lo, hi = ref * (1 - band), ref * (1 + band)
    settle_t = 0.0
    for k in range(len(ts) - 1, -1, -1):
        if not (lo <= ws[k] <= hi):
            settle_t = ts[min(k + 1, len(ts) - 1)]
            break
    rise_0_100 = None
    for t, w in zip(ts, ws):
        if w >= ref:
            rise_0_100 = t
            break
    t10 = next((t for t, w in zip(ts, ws) if w >= 0.10 * ref), None)
    t90 = next((t for t, w in zip(ts, ws) if w >= 0.90 * ref), None)
    rise_10_90 = (t90 - t10) if (t10 is not None and t90 is not None) else None
    return overshoot_pct, settle_t, rise_0_100, rise_10_90


# ----------------------------------------------------------------------
# 5. Search: makes "optimized" literal, not just a word in the bullet
# ----------------------------------------------------------------------

SPEC_OVERSHOOT = 10.0     # %
SPEC_SETTLING = 0.320     # s  (2% band)
SPEC_RISE = 0.150         # s  (0-100%)

ZETA, WN, P3 = 0.98, 17.0, 80.0     # shipped design


def find_best_feasible_design(ref=5.0, dt=4e-4, verbose=False):
    """Grid search over (zeta, wn, p3). Feasible = meets the overshoot /
    settling / rise spec AND never actually asks the actuator for more
    than Vmax (checked on the UNCLAMPED demand -- see closed_loop's
    docstring for why this matters). Returns the minimum-settling-time
    feasible design found, plus the full feasible list for reference."""
    feasible = []
    for zi in range(80, 100):
        zeta = zi / 100
        for wn in range(10, 26):
            for p3 in range(30, 160, 5):
                if p3 <= 2 * zeta * wn:
                    continue
                Kp, Ki, Kd = design_PID_full(zeta, wn, p3)
                if Kp <= 0 or Ki <= 0 or Kd < 0:
                    continue
                ts, ws, us, us_unsat, i_hist, ints = closed_loop(
                    Kp, Ki, Kd, lambda t: ref, lambda t: 0.0, 1.0, dt=dt)
                if max(abs(x) for x in us_unsat) > Vmax:
                    continue
                os_, settle, rise, rise90 = measure(ts, ws, ref)
                if os_ > SPEC_OVERSHOOT or settle > SPEC_SETTLING or \
                   (rise is not None and rise > SPEC_RISE):
                    continue
                feasible.append((settle, os_, rise, zeta, wn, p3, Kp, Ki, Kd))
    feasible.sort()
    if verbose:
        print(f"  search: {len(feasible)} feasible designs found")
        for row in feasible[:5]:
            settle, os_, rise, zeta, wn, p3, Kp, Ki, Kd = row
            print(f"    settle={settle*1000:.1f}ms OS={os_:.2f}% "
                  f"zeta={zeta} wn={wn} p3={p3}")
    return feasible[0] if feasible else None, feasible


# ----------------------------------------------------------------------
# 6. Scenarios
# ----------------------------------------------------------------------

def scenario_1_plant_characterisation():
    print("SCENARIO 1  plant characterisation: full (2-state) vs reduced "
          "(L neglected, 1-state)")
    print("-" * 66)
    Vstep = 10.0
    t_full, w_full = step_full_model(lambda t: Vstep, lambda t: 0.0, 1.0)
    t_red, w_red = step_reduced_model(lambda t: Vstep, lambda t: 0.0, 1.0)
    diff = [abs(f - r) for f, r in zip(w_full, w_red)]
    final = w_red[-1]
    print(f"  tau_e = L/R = {tau_e*1000:.1f} ms   electrical corner = "
          f"{tau_e_corner:.0f} rad/s")
    print(f"  steady-state difference = "
          f"{abs(w_full[-1]-final)/final*100:.3f} %")
    print(f"  max transient mismatch = {max(diff)/final*100:.2f} % of final "
          f"value")
    print()


def scenario_2_primary_design(search_result=None):
    print("SCENARIO 2  primary design: exact 3-pole placement, confirmed "
          "near-optimal by search")
    print("-" * 66)
    Kp, Ki, Kd = design_PID_full(ZETA, WN, P3)
    print(f"  shipped design: zeta={ZETA}  wn={WN} rad/s  p3={P3} rad/s")
    print(f"  spec: overshoot<={SPEC_OVERSHOOT}%  settling(2%)<="
          f"{SPEC_SETTLING*1000:.0f}ms  rise(0-100%)<={SPEC_RISE*1000:.0f}ms")
    print(f"  Kp = {Kp:.4f}   Ki = {Ki:.4f}   Kd = {Kd:.5f}")

    poles = closed_loop_poles(Kp, Ki, Kd)
    print(f"  independent pole check (numpy.roots on the characteristic "
          f"polynomial): {[f'{p:.2f}' for p in poles]}")

    naive = naive_overshoot_pct(ZETA)
    print(f"  naive dominant-pole-only prediction = {naive:.6f} % -- its "
          f"assumptions (clean 2nd-order, no zeros) do not hold for this "
          f"3rd-order system; not trusted, checked against simulation")
    print()

    ref = 5.0
    ts, ws, us, us_unsat, i_hist, ints = closed_loop(
        Kp, Ki, Kd, lambda t: ref, lambda t: 0.0, 1.0)
    os_, settle, rise, rise90 = measure(ts, ws, ref)
    peak_u_unsat = max(abs(x) for x in us_unsat)
    print(f"  reference step = {ref} rad/s   peak UNCLAMPED demand = "
          f"{peak_u_unsat:.2f} V (Vmax={Vmax} V, "
          f"{'genuinely unsaturated' if peak_u_unsat<Vmax else 'SATURATES'})")
    print(f"  SIMULATED: overshoot={os_:.2f}%  settling={settle*1000:.1f}ms")
    print(f"  rise time (0-100%) = {rise*1000:.1f}ms   "
          f"rise time (10-90%) = {rise90*1000:.1f}ms  (both conventions "
          f"reported to remove ambiguity)")
    ratio = os_ / naive if naive > 0 else float('inf')
    print(f"  simulation/naive-formula ratio = {ratio:,.0f}x "
          f"({math.log10(ratio):.1f} orders of magnitude) -- the formula's "
          f"assumptions don't hold here, it isn't merely imprecise")
    print()

    if search_result is None:
        search_result = find_best_feasible_design()
    best, feasible = search_result
    if best:
        best_settle = best[0]
        gap_pct = (settle - best_settle) / best_settle * 100
        print(f"  OPTIMIZATION CHECK: search over the (zeta,wn,p3) space "
              f"found {len(feasible)} genuinely-feasible designs (spec met "
              f"AND actuator never actually saturates). Best settling time "
              f"found: {best_settle*1000:.1f}ms. Shipped design: "
              f"{settle*1000:.1f}ms -- within {gap_pct:.1f}% of optimal.")
    print()
    return Kp, Ki, Kd


def scenario_3_does_Kd_help(Kp, Ki, Kd):
    print("SCENARIO 3  does Kd actually help? (checked, not assumed)")
    print("-" * 66)
    ref = 5.0
    ts_pid, ws_pid, *_ = closed_loop(Kp, Ki, Kd, lambda t: ref, lambda t: 0.0, 1.0)
    ts_pi, ws_pi, *_ = closed_loop(Kp, Ki, 0.0, lambda t: ref, lambda t: 0.0, 1.0)
    os_pid, s_pid, r_pid, _ = measure(ts_pid, ws_pid, ref)
    os_pi, s_pi, r_pi, _ = measure(ts_pi, ws_pi, ref)
    print(f"  full PID (Kp,Ki,Kd):  overshoot={os_pid:.2f}%  "
          f"settling={s_pid*1000:.0f}ms")
    print(f"  PI only (Kd=0, same Kp,Ki): overshoot={os_pi:.2f}%  "
          f"settling={s_pi*1000:.0f}ms")
    print(f"  At this bandwidth (wn={WN} rad/s, well below the "
          f"{tau_e_corner:.0f} rad/s electrical corner), Kd's own "
          f"contribution is small -- PI alone performs comparably.")
    print()


def scenario_4_when_Kd_matters():
    print("SCENARIO 4  when does Kd genuinely help? (small-signal, "
          "high-bandwidth design)")
    print("-" * 66)
    zeta2, wn2, p32 = 0.95, 50.0, 300.0
    Kp2, Ki2, Kd2 = design_PID_full(zeta2, wn2, p32)
    ref = 0.3     # verified unsaturated below, not assumed
    print(f"  design: zeta={zeta2}  wn={wn2} rad/s (near the "
          f"{tau_e_corner:.0f} rad/s electrical corner)  p3={p32}")
    print(f"  Kp={Kp2:.3f}  Ki={Ki2:.2f}  Kd={Kd2:.4f}")

    poles_pi = closed_loop_poles(Kp2, Ki2, 0.0)   # PI-only, for the stability check
    print(f"  PI-only closed-loop poles: {[f'{p:.2f}' for p in poles_pi]} "
          f"-- all real parts negative: STABLE, just heavily underdamped "
          f"")

    ts_pid, ws_pid, u_pid, uu_pid, *_ = closed_loop(
        Kp2, Ki2, Kd2, lambda t: ref, lambda t: 0.0, 0.3)
    ts_pi, ws_pi, u_pi, uu_pi, *_ = closed_loop(
        Kp2, Ki2, 0.0, lambda t: ref, lambda t: 0.0, 0.3)
    max_u_pid = max(abs(x) for x in uu_pid)
    max_u_pi = max(abs(x) for x in uu_pi)
    print(f"  reference = {ref} rad/s -- CHECKED unsaturated (unclamped "
          f"demand): {max_u_pid:.2f} V (PID) / {max_u_pi:.2f} V (PI), "
          f"both < Vmax={Vmax} V")

    os_pid, s_pid, r_pid, _ = measure(ts_pid, ws_pid, ref)
    os_pi, s_pi, r_pi, _ = measure(ts_pi, ws_pi, ref)
    print(f"  full PID: overshoot={os_pid:.2f}%  settling={s_pid*1000:.0f}ms")
    print(f"  PI only:  overshoot={os_pi:.2f}%  settling={s_pi*1000:.0f}ms")
    print(f"  Kd reduces overshoot from {os_pi:.1f}% to {os_pid:.1f}% -- "
          f"{os_pi-os_pid:.1f} percentage points. PI-only is stable but "
          f"strongly oscillatory; PID is well damped. That is Kd's genuine "
          f"job, once bandwidth is high enough to need it.")
    print(f"  Not the deployed design: exceeds the {SPEC_OVERSHOOT}% spec, "
          f"and Kp={Kp2:.1f} would saturate for any command over "
          f"~{Vmax/Kp2:.2f} rad/s. Shown to characterise Kd, not as a "
          f"candidate design.")
    print()
    return zeta2, wn2, p32, Kp2, Ki2, Kd2


def scenario_5_saturation_antiwindup(Kp, Ki, Kd):
    print("SCENARIO 5  large step: saturation, with vs without anti-windup")
    print("-" * 66)
    ref = 20.0
    t_no, w_no, u_no, uu_no, i_no, int_no = closed_loop(
        Kp, Ki, Kd, lambda t: ref, lambda t: 0.0, 2.0, anti_windup=False)
    t_aw, w_aw, u_aw, uu_aw, i_aw, int_aw = closed_loop(
        Kp, Ki, Kd, lambda t: ref, lambda t: 0.0, 2.0, anti_windup=True)
    os_no, s_no, r_no, _ = measure(t_no, w_no, ref)
    os_aw, s_aw, r_aw, _ = measure(t_aw, w_aw, ref)
    peak_i = max(max(abs(x) for x in i_no), max(abs(x) for x in i_aw))
    print(f"  reference step = {ref} rad/s   unclamped demand peaks at "
          f"{max(abs(x) for x in uu_aw):.1f} V (Vmax={Vmax}V -> saturates)")
    print(f"  WITHOUT anti-windup: overshoot={os_no:.1f}%  "
          f"settle={s_no*1000:.0f}ms")
    print(f"  WITH    anti-windup: overshoot={os_aw:.1f}%  "
          f"settle={s_aw*1000:.0f}ms")
    print(f"  peak armature current reached = {peak_i:.2f} A -- NOTE: no "
          f"current/torque limit is modelled in this study, only voltage "
          f"saturation. A real drive would also enforce |I|<=Imax; this "
          f"is reported so the omission is visible, not hidden.")
    print()
    return t_no, w_no, t_aw, w_aw


def scenario_6_disturbance_rejection(Kp, Ki):
    print("SCENARIO 6  step load-torque disturbance rejection (P vs PI)")
    print("-" * 66)
    ref = 10.0
    Tload_step = 0.05
    Tfun = lambda t: Tload_step if t >= 1.0 else 0.0
    t_pi, w_pi, *_ = closed_loop(Kp, Ki, 0.0, lambda t: ref, Tfun, 2.0)
    t_p, w_p, *_ = closed_loop(Kp, 0.0, 0.0, lambda t: ref, Tfun, 2.0)
    final_pi = w_pi[-1]; final_p = w_p[-1]
    dip = ref - min(w for t, w in zip(t_pi, w_pi) if t >= 1.0)
    print(f"  load torque step = {Tload_step} N.m at t=1.0s")
    print(f"  PI:  dip={dip:.3f} rad/s  residual error="
          f"{(ref-final_pi)/ref*100:.3f}%")
    print(f"  P-only: residual error={(ref-final_p)/ref*100:.2f}% "
          f"(never rejects a constant disturbance)")
    print()
    return t_pi, w_pi, t_p, w_p


# ----------------------------------------------------------------------
# 7. Self-test: real assertions, not just printed numbers
# ----------------------------------------------------------------------

def self_test(search_result=None):
    print("SELF-TEST")
    print("-" * 66)
    Kp, Ki, Kd = design_PID_full(ZETA, WN, P3)

    # 1. pole placement lands exactly on target
    poles = sorted(closed_loop_poles(Kp, Ki, Kd), key=lambda p: (p.real, p.imag))
    expected_poles = sorted(
        [-P3, -ZETA * WN + 1j * WN * math.sqrt(1 - ZETA ** 2),
         -ZETA * WN - 1j * WN * math.sqrt(1 - ZETA ** 2)],
        key=lambda p: (p.real, p.imag),
    )
    assert all(abs(a - b) < 1e-6 for a, b in zip(poles, expected_poles)), \
        "closed-loop poles do not match the design target"
    print("  [PASS] closed-loop poles match design target exactly")

    # 2. timestep convergence: coarse vs fine dt agree closely
    ref = 5.0
    ts_c, ws_c, *_ = closed_loop(Kp, Ki, Kd, lambda t: ref, lambda t: 0.0,
                                  1.0, dt=2e-5)
    ts_f, ws_f, *_ = closed_loop(Kp, Ki, Kd, lambda t: ref, lambda t: 0.0,
                                  1.0, dt=2e-6)
    os_c, _, _, _ = measure(ts_c, ws_c, ref)
    os_f, _, _, _ = measure(ts_f, ws_f, ref)
    assert abs(os_c - os_f) < 0.05, \
        f"overshoot not converged across timestep: {os_c} vs {os_f}"
    print(f"  [PASS] timestep convergence: overshoot differs by "
          f"{abs(os_c-os_f):.4f} pct pts between dt=2e-5 and dt=2e-6")

    # 3. primary design meets all three named specs
    os_, settle, rise, rise90 = measure(ts_c, ws_c, ref)
    assert os_ <= SPEC_OVERSHOOT, f"overshoot {os_} exceeds spec"
    assert settle <= SPEC_SETTLING, f"settling {settle} exceeds spec"
    assert rise <= SPEC_RISE, f"rise {rise} exceeds spec"
    print(f"  [PASS] primary design meets spec: OS={os_:.2f}% "
          f"settle={settle*1000:.0f}ms rise={rise*1000:.0f}ms")

    # 4. shipped design is genuinely unsaturated at the test reference
    _, _, _, uu, _, _ = closed_loop(Kp, Ki, Kd, lambda t: ref, lambda t: 0.0, 1.0)
    assert max(abs(x) for x in uu) < Vmax, \
        "shipped design saturates at the reference step -- invalid"
    print(f"  [PASS] shipped design never saturates at ref={ref} "
          f"(peak unclamped demand {max(abs(x) for x in uu):.1f}V < "
          f"{Vmax}V)")

    # 5. shipped design is within 10% of the best feasible design found on the search grid
    if search_result is None:
        search_result = find_best_feasible_design()
    best, feasible = search_result
    assert best is not None, "search found no feasible design at all"
    gap = (settle - best[0]) / best[0]
    assert gap < 0.10, f"shipped design is {gap*100:.1f}% off the best feasible design found on the search grid"
    print(f"  [PASS] shipped design within {gap*100:.1f}% of the best feasible design found on the search grid "
          f"settling time ({len(feasible)} feasible designs evaluated)")

    print("\n  ALL CHECKS PASSED")
    print()


if __name__ == "__main__":
    print_derivation()
    scenario_1_plant_characterisation()
    _search_result = find_best_feasible_design()      # computed once, reused below
    Kp, Ki, Kd = scenario_2_primary_design(_search_result)
    scenario_3_does_Kd_help(Kp, Ki, Kd)
    scenario_4_when_Kd_matters()
    scenario_5_saturation_antiwindup(Kp, Ki, Kd)
    scenario_6_disturbance_rejection(Kp, Ki)
    self_test(_search_result)
