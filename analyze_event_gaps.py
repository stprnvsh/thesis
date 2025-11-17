#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_events(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "events" in data:
        ev = data["events"]
    else:
        raise RuntimeError("Input pickle must contain key 'events'.")
    t = np.asarray(ev["t"], dtype=float)
    u = np.asarray(ev["u"], dtype=int)
    e = np.asarray(ev["e"], dtype=int)
    return t, u, e


def compute_gaps(t: np.ndarray):
    if t.size <= 1:
        return np.array([])
    t_sorted = np.sort(t)
    return np.diff(t_sorted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data pickle (with 'events')")
    ap.add_argument("--save", help="Optional path to save the figure")
    ap.add_argument("--per-node", action="store_true", help="Report per-node average gaps")
    ap.add_argument("--per-mark", action="store_true", help="Report per-mark average gaps")
    args = ap.parse_args()

    t, u, e = load_events(args.data)
    gaps = compute_gaps(t)

    print(f"Events: {len(t)}")
    if gaps.size:
        print(f"Average gap (all events): {np.mean(gaps):.6f} time units")
        print(f"Median gap: {np.median(gaps):.6f}")
        q10, q90 = np.quantile(gaps, [0.1, 0.9])
        print(f"P10/P90 gaps: {q10:.6f} / {q90:.6f}")
    else:
        print("Not enough events to compute gaps.")

    if args.per_node:
        nodes = np.unique(u)
        vals = []
        for node in nodes:
            g = compute_gaps(t[u == node])
            if g.size:
                vals.append(np.mean(g))
                print(f"Node {int(node)} avg gap: {np.mean(g):.6f}")
        if vals:
            print(f"Per-node avg gap mean: {np.mean(vals):.6f}")

    if args.per_mark:
        marks = np.unique(e)
        for mk in marks:
            g = compute_gaps(t[e == mk])
            if g.size:
                print(f"Mark {int(mk)} avg gap: {np.mean(g):.6f}")

    # Plot all timestamps and gap histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # left: timestamps colored by mark
    marks = np.unique(e)
    for mk in marks:
        sel = (e == mk)
        axes[0].scatter(t[sel], np.full(np.sum(sel), mk), s=6, alpha=0.6, label=f"mark {int(mk)}")
    axes[0].set_xlabel("time (same units as data)")
    axes[0].set_ylabel("mark")
    axes[0].set_title("Event timestamps")
    if len(marks) <= 10:
        axes[0].legend(loc="best", fontsize=8)

    # right: gap histogram
    if gaps.size:
        axes[1].hist(gaps, bins=30, color="salmon", alpha=0.8, density=False)
        axes[1].set_title("Inter-event gaps (all events)")
        axes[1].set_xlabel("gap")
        axes[1].set_ylabel("count")
    else:
        axes[1].text(0.5, 0.5, "n/a", ha="center", va="center")
        axes[1].axis("off")

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main() 