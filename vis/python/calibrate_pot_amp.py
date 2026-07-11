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
    p  = prim[1]
    r  = x1v[:, None, None, :]
    th = x2v[:, None, :, None]
    return p, B, r, th


def bsq_ks(B, r, th, a):
    Sigma  = r * r + a * a * np.cos(th) ** 2
    s2     = np.sin(th) ** 2
    factor = 1.0 + 2.0 * r / Sigma
    g_rr   = factor
    g_th   = Sigma
    g_ph   = (r * r + a * a + 2.0 * r * a * a * s2 / Sigma) * s2
    g_rph  = -a * s2 * factor
    Br, Bt, Bp = B[0], B[1], B[2]
    return g_rr * Br * Br + g_th * Bt * Bt + g_ph * Bp * Bp + 2.0 * g_rph * Br * Bp


def calibrate(run_dir, pot_amp_used, spin=0.9375, target=100.0, cycle=0):
    prim_path = find_snapshot(run_dir, "prim", cycle)
    p, B, r, th = load_state(prim_path)
    bsq = bsq_ks(B, r, th, spin)

    p_max    = float(p.max())
    pmag_max = float(bsq.max()) / 2.0
    if pmag_max <= 0.0:
        raise RuntimeError("magnetic pressure max <= 0; check pot_amp / field_config")

    beta = p_max / pmag_max
    scale = float(np.sqrt(beta / target))
    pot_amp_target = pot_amp_used * scale

    return {
        "prim_path": prim_path,
        "p_max": p_max,
        "pmag_max": pmag_max,
        "beta": beta,
        "scale": scale,
        "pot_amp_used": pot_amp_used,
        "pot_amp_target": pot_amp_target,
        "target": target,
        "spin": spin,
    }


def report(r):
    print(f"prim         : {os.path.basename(r['prim_path'])}")
    print(f"spin a       : {r['spin']}")
    print(f"pot_amp used : {r['pot_amp_used']:.6f}")
    print(f"p_max        : {r['p_max']:.6e}")
    print(f"p_mag_max    : {r['pmag_max']:.6e}   (= max(B^2)/2)")
    print(f"beta         : {r['beta']:.6f}   (= p_max / p_mag_max)")
    print(f"target       : {r['target']:.6f}")
    print(f"scale        : x{r['scale']:.6f}")
    print(f"pot_amp new  : {r['pot_amp_target']:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("pot_amp_used", type=float)
    ap.add_argument("--spin", type=float, default=0.9375)
    ap.add_argument("--target", type=float, default=100.0)
    ap.add_argument("--cycle", type=int, default=0)
    args = ap.parse_args()

    result = calibrate(args.run_dir, args.pot_amp_used,
                       spin=args.spin, target=args.target, cycle=args.cycle)
    report(result)


if __name__ == "__main__":
    main()
