#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import plot_torus

import h5py
import numpy as np


def find_snapshot(run_dir, tag, cycle):
    hits = sorted(glob.glob(os.path.join(run_dir, f"*.{tag}.{cycle:05d}.athdf")))
    if not hits:
        raise FileNotFoundError(f"no {tag} snapshot at cycle {cycle} in {run_dir}")
    return hits[0]


def load_state(prim_path):
    with h5py.File(prim_path, "r") as f:
        prim = np.asarray(f["prim"])
        B    = np.asarray(f["B"])
        x1v  = f["x1v"][...]
        x2v  = f["x2v"][...]
    # prim var order (5 vars): rho, press, vel1, vel2, vel3
    p    = prim[1]
    utilde1 = prim[2]
    utilde2 = prim[3]
    utilde3 = prim[4]
    r  = x1v[:, None, None, :]
    th = x2v[:, None, :, None]
    return p, utilde1, utilde2, utilde3, B, r, th


def bsq_faraday_ks(B, utilde1, utilde2, utilde3, r, th, a):
    """Compute b² = b^μ b_μ (Faraday form) in Kerr-Schild coordinates.

    (M = 1 units):
        1. Recover 4-velocity u^μ from ũ^i (Athena prim vel) using
           γ_E = √(1 + g_ij ũ^i ũ^j), α = √(-1/g^tt),
           u^0 = γ_E / α, u^i = ũ^i − α γ_E g^{0i}.
        2. Lower u^μ → u_μ via KS covariant metric.
        3. b^0 = u_i · B^i (spatial i sum).
        4. b^i = (B^i + b^0 u^i) / u^0.
        5. Lower b^μ → b_μ, then b² = b^μ b_μ.
    """
    # KS metric components (M = 1 units)
    Sigma  = r * r + a * a * np.cos(th) ** 2
    s2     = np.sin(th) ** 2
    two_M_r_over_S = 2.0 * r / Sigma

    # Covariant g_μν
    g_tt = -(1.0 - two_M_r_over_S)
    g_tr =  two_M_r_over_S
    g_tp = -a * s2 * two_M_r_over_S
    g_rr =  1.0 + two_M_r_over_S
    g_rp = -a * s2 * (1.0 + two_M_r_over_S)
    g_thth = Sigma
    g_pp = (r * r + a * a + 2.0 * r * a * a * s2 / Sigma) * s2

    # Contravariant g^μν (matching Athena kerr-schild.cpp:905-910 EXACTLY).
    # Athena's KS convention: g^{tφ} = 0  (the a/Σ term lives in g^{rφ},
    # NOT g^{tφ}).  This is the CANONICAL KS inverse metric — the radial
    # shift g^{tr}=2Mr/Σ absorbs the entire frame-dragging effect.
    gi_tt = -(1.0 + two_M_r_over_S)
    gi_tr =  two_M_r_over_S
    # gi_tp is ZERO in Athena's KS slicing (NOT a/Σ)

    # Step 1: 4-velocity from ũ (Athena vel1/vel2/vel3 = normal-frame utilde)
    # γ_E² = 1 + g_ij ũ^i ũ^j  (KS spatial metric off-diagonals: g_rp only;
    #                            g_rθ = g_θφ = 0 in KS)
    tmp = (g_rr   * utilde1 * utilde1
           + 2.0 * g_rp * utilde1 * utilde3
           + g_thth * utilde2 * utilde2
           + g_pp   * utilde3 * utilde3)
    gamma_E = np.sqrt(1.0 + tmp)
    alpha   = np.sqrt(-1.0 / gi_tt)

    u0 = gamma_E / alpha
    u1 = utilde1 - alpha * gamma_E * gi_tr
    u2 = utilde2                                # gi_t2 = 0 in KS
    u3 = utilde3                                # gi_t3 = 0 in Athena's KS convention

    # Step 2: Lower u^μ → u_μ
    u_t = g_tt * u0 + g_tr * u1 + g_tp * u3
    u_r = g_tr * u0 + g_rr * u1 + g_rp * u3
    u_th = g_thth * u2
    u_p = g_tp * u0 + g_rp * u1 + g_pp * u3

    # Step 3-4: 4-magnetic field
    Br, Bt, Bp = B[0], B[1], B[2]
    # b^0 = u_i · B^i  (spatial i = r, θ, φ)
    b0 = u_r * Br + u_th * Bt + u_p * Bp
    b1 = (Br + b0 * u1) / u0
    b2 = (Bt + b0 * u2) / u0
    b3 = (Bp + b0 * u3) / u0

    # Step 5: Lower b^μ → b_μ
    b_t = g_tt * b0 + g_tr * b1 + g_tp * b3
    b_r = g_tr * b0 + g_rr * b1 + g_rp * b3
    b_th = g_thth * b2
    b_p = g_tp * b0 + g_rp * b1 + g_pp * b3

    return b0 * b_t + b1 * b_r + b2 * b_th + b3 * b_p


def bsq_lab_ks(B, r, th, a):
    """Lab-frame B² = g_ij B^i B^j (LEGACY, kept for cross-check).

    This is the LAB-frame Eulerian 3-vector squared — does NOT include
    fluid velocity coupling. Agrees with Faraday b² when u^i = 0 only.
    """
    Sigma  = r * r + a * a * np.cos(th) ** 2
    s2     = np.sin(th) ** 2
    factor = 1.0 + 2.0 * r / Sigma
    g_rr   = factor
    g_th   = Sigma
    g_ph   = (r * r + a * a + 2.0 * r * a * a * s2 / Sigma) * s2
    g_rph  = -a * s2 * factor
    Br, Bt, Bp = B[0], B[1], B[2]
    return g_rr * Br * Br + g_th * Bt * Bt + g_ph * Bp * Bp + 2.0 * g_rph * Br * Bp


def calibrate(run_dir, pot_amp_used, spin=0.9375, target=100.0, cycle=0,
              form="faraday"):
    prim_path = find_snapshot(run_dir, "prim", cycle)
    p, u1, u2, u3, B, r, th = load_state(prim_path)

    if form == "faraday":
        bsq = bsq_faraday_ks(B, u1, u2, u3, r, th, spin)
    elif form == "lab":
        bsq = bsq_lab_ks(B, r, th, spin)
    else:
        raise ValueError(f"unknown form '{form}'; use 'faraday' or 'lab'")

    p_max    = float(p.max())
    pmag_max = float(bsq.max()) / 2.0
    if pmag_max <= 0.0:
        raise RuntimeError("magnetic pressure max <= 0; check pot_amp / field_config")

    beta = p_max / pmag_max
    scale = float(np.sqrt(beta / target))
    pot_amp_target = pot_amp_used * scale

    # Cross-check: also compute lab-form β for comparison when using Faraday
    beta_lab = None
    if form == "faraday":
        bsq_lab_max = float(bsq_lab_ks(B, r, th, spin).max()) / 2.0
        if bsq_lab_max > 0.0:
            beta_lab = p_max / bsq_lab_max

    return {
        "prim_path": prim_path,
        "form": form,
        "p_max": p_max,
        "pmag_max": pmag_max,
        "beta": beta,
        "beta_lab_crosscheck": beta_lab,
        "scale": scale,
        "pot_amp_used": pot_amp_used,
        "pot_amp_target": pot_amp_target,
        "target": target,
        "spin": spin,
    }


def report(r):
    print(f"prim          : {os.path.basename(r['prim_path'])}")
    print(f"spin a        : {r['spin']}")
    print(f"pot_amp used  : {r['pot_amp_used']:.6f}")
    print(f"form          : {r['form']}   "
          f"({'Faraday b² = b^μ b_μ (HARM convention)' if r['form']=='faraday' else 'lab B² = g_ij B^i B^j'})")
    print(f"p_max         : {r['p_max']:.6e}")
    if r['form'] == 'faraday':
        print(f"p_mag_max     : {r['pmag_max']:.6e}   (= max(b²)/2)")
    else:
        print(f"p_mag_max     : {r['pmag_max']:.6e}   (= max(B²)/2)")
    print(f"beta_min      : {r['beta']:.6f}   (= p_max / p_mag_max)")
    if r['beta_lab_crosscheck'] is not None:
        rel = (r['beta_lab_crosscheck'] - r['beta']) / r['beta'] * 100.0
        print(f"beta_min (lab): {r['beta_lab_crosscheck']:.6f}   "
              f"(cross-check; {rel:+.2f}% vs Faraday)")
    print(f"target        : {r['target']:.6f}")
    print(f"scale         : x{r['scale']:.6f}")
    print(f"pot_amp new   : {r['pot_amp_target']:.6f}")


def main():
    ap = argparse.ArgumentParser(
        description="Calibrate pot_amp to hit target β_min in FM torus IC. "
                    "Uses fluid-frame Faraday b² by default (matches HARM 2003 "
                    "and Athena's own history-flux convention).")
    ap.add_argument("run_dir")
    ap.add_argument("pot_amp_used", type=float)
    ap.add_argument("--spin", type=float, default=0.9375)
    ap.add_argument("--target", type=float, default=100.0)
    ap.add_argument("--cycle", type=int, default=0)
    ap.add_argument("--form", choices=["faraday", "lab"], default="faraday",
                    help="magnetic pressure form: faraday (b² = b^μ b_μ, HARM "
                         "convention) or lab (B² = g_ij B^i B^j, legacy)")
    args = ap.parse_args()

    result = calibrate(args.run_dir, args.pot_amp_used,
                       spin=args.spin, target=args.target, cycle=args.cycle,
                       form=args.form)
    report(result)


if __name__ == "__main__":
    main()
