#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cost model for Gemma4 chunked prefill steps (BH Galaxy 8x4, CP8 / TP4).

    step(batch) = dense(M_total) + sum_r attn(C_r, P_r) + fixed

per layer type, times 10 global + 50 sliding layers. M = rows per device = sum(C_r) / CP. Data lives in a JSON file
(model_data_revc.json next to this script). `cost_model.py` prints the fitted coefficients and the reconstruction of
the measured whole-model runs.
"""

import argparse
import bisect
import json
import math
import os

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data_revc.json")
CP = 8
N_LAYERS = {"global": 10, "sliding": 50}
# Per device, per layer: 8 local Q heads x head_dim 512, for the global attention's FLOP rate in the report.
GLOBAL_FLOP_PER_QK = 4 * 8 * 512  # x (C/CP) x (P + C/2)


class Model:
    @classmethod
    def load(cls, path=DEFAULT_DATA):
        with open(path) as f:
            return cls(json.load(f))

    def __init__(self, data):
        self.d = data
        # dense(M) per layer type: piecewise linear through measured M points (no extrapolation past the ends).
        self.dense_pts = {L: sorted((int(m), v) for m, v in data["dense_us"][L].items()) for L in N_LAYERS}
        # attn_global(C, P) = a_C + b_C * P per layer, from measured (P, us) points by least squares.
        self.g = {}
        for C, pts in data["gattn_us"].items():
            xs = [p for p, _ in pts]
            ys = [t for _, t in pts]
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
            self.g[int(C)] = (my - b * mx, b, pts)
        self.l = {int(C): v for C, v in data["lattn_us"].items()}
        self.fixed_us = data.get("fixed_step_us", 0.0)
        self.cal = data["calibration"]

    def calibration(self, C, budget):
        """(device scale, host ms per step) measured on silicon for a scheduler with this unit and budget."""
        c = self.cal[str(C) if C == budget else "batched"]
        return c["scale"], c["host_ms"]

    # ---- components (per layer, us) ----
    def dense(self, L, M):
        pts = self.dense_pts[L]
        ms = [m for m, _ in pts]
        if M <= ms[0]:
            # Below the smallest measured M: hold flat (rows are free when weights dominate).
            return pts[0][1]
        if M > ms[-1]:
            raise ValueError(f"M={M} past the last measured point {ms[-1]}; not extrapolating")
        i = bisect.bisect_left(ms, M)
        if ms[i] == M:
            return pts[i][1]
        (m0, v0), (m1, v1) = pts[i - 1], pts[i]
        return v0 + (v1 - v0) * (M - m0) / (m1 - m0)

    def gattn(self, C, P):
        if C not in self.g:  # a multi-slot piece with no measured chunk size: run it as unit-size chunks
            u = max(c for c in self.g if C % c == 0)
            return sum(self.gattn(u, P + j * u) for j in range(C // u))
        a, b, _ = self.g[C]
        return a + b * P

    def lattn(self, C):
        if C not in self.l:
            u = max(c for c in self.l if C % c == 0)
            return (C // u) * self.l[u]
        return self.l[C]

    # ---- one step ----
    def step(self, batch):
        """batch: list of (C, P). Returns dict of per-step us by component (all layers summed)."""
        M = sum(C for C, _ in batch) // CP
        out = {
            "dense": sum(N_LAYERS[L] * self.dense(L, M) for L in N_LAYERS),
            "fixed": self.fixed_us,
            "gattn": N_LAYERS["global"] * sum(self.gattn(C, P) for C, P in batch),
            "lattn": N_LAYERS["sliding"] * sum(self.lattn(C) for C, _ in batch),
        }
        out["total"] = sum(out.values())
        return out

    def unbatched_prompt_s(self, C, tokens):
        n = math.ceil(tokens / C)
        return sum(self.step([(C, i * C)])["total"] for i in range(n)) / 1e6


# ---- reports ----
def report_fits(m):
    print("dense(M) per layer, us (M = rows per device):")
    for L in N_LAYERS:
        print(f"  {L:8s} " + "  ".join(f"M={mm}: {v:.0f}" for mm, v in m.dense_pts[L]))
    ms = [mm for mm, _ in m.dense_pts["sliding"]]
    print("dense per step, ms:  " + "  ".join(f"M={mm}: {m.step([(mm * CP, 0)])['dense'] / 1e3:.1f}" for mm in ms))
    for i in range(1, len(ms)):
        m0, m1 = ms[i - 1], ms[i]
        d0, d1 = (m.step([(x * CP, 0)])["dense"] for x in (m0, m1))
        print(
            f"   marginal {m0}->{m1}: {(d1 - d0) / (m1 - m0):.1f} us per row ({(d1 - d0) / (m1 - m0) / CP * 1e3:.0f} ns per token)"
        )
    print("attn_global(C, P) = a + b*P per layer:")
    for C, (a, b, pts) in sorted(m.g.items()):
        rate = GLOBAL_FLOP_PER_QK * (C / CP) / (b * 1e-6) / 1e12 if b else float("nan")
        resid = max(abs(a + b * p - t) for p, t in pts)
        print(
            f"  C={C:5d}: a={a:7.0f} us  b={b * 1e3:6.2f} us per 1k prefix tokens  slope rate {rate:5.0f} TFLOP/s"
            f"  ({len(pts)} pts, max resid {resid:.0f} us)"
        )
    print("attn_sliding(C) per layer, us: " + "  ".join(f"C={C}: {v:.0f}" for C, v in sorted(m.l.items())))
    print("unbatched reconstruction (device time, s) vs measured:")
    for C, meas in m.d.get("measured_unbatched", {}).items():
        C = int(C)
        if C not in m.g or C not in m.l:
            continue
        first = m.step([(C, 0)])["total"] / 1e3
        r100 = m.unbatched_prompt_s(C, 13 * 8192)
        r256 = m.unbatched_prompt_s(C, 262144)
        print(
            f"  C={C}: first {first:.1f} ms (meas {meas.get('first_ms')})  106k {r100:.2f} s (meas {meas.get('mid_s')})"
            f"  256k {r256:.2f} s (meas {meas.get('full_s')})"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    report_fits(Model.load(ap.parse_args().data))


if __name__ == "__main__":
    main()
