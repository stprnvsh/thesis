#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import numpy as np

def _rmse(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(diff**2)))

def _rel_fro_err(a, b, eps=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a) + eps
    return float(100.0 * num / den)

def _print_array(name, arr, max_rows=None, max_cols=None, precision=6):
    arr = np.asarray(arr)
    print(f"\n{name} (shape={arr.shape}):")
    with np.printoptions(precision=precision, suppress=False, linewidth=140):
        if max_rows is not None or max_cols is not None:
            # Manual truncation preview
            r, c = arr.shape if arr.ndim == 2 else (arr.size, 1)
            r_show = min(r, max_rows if max_rows is not None else r)
            c_show = min(c, max_cols if max_cols is not None else c)
            if arr.ndim == 1:
                print(arr[:r_show])
                if r_show < r:
                    print(f"... ({r - r_show} more)")
            elif arr.ndim == 2:
                preview = arr[:r_show, :c_show]
                print(preview)
                if r_show < r or c_show < c:
                    print(f"... (truncated; total shape {arr.shape})")
            else:
                print(arr)
        else:
            print(arr)

def _save_csv(save_dir, basename, arr):
    path = os.path.join(save_dir, f"{basename}.csv")
    np.savetxt(path, np.asarray(arr), delimiter=",")
    print(f"  - saved: {path}")

def summarize_param(name, est, truth=None, save_dir=None, save_prefix=None,
                    print_full=True, precision=6):
    """Print full matrices/vectors and error stats; optionally save CSVs."""
    if print_full:
        _print_array(f"{name} (inferred)", est, precision=precision)

    if truth is not None:
        if print_full:
            _print_array(f"{name} (true)", truth, precision=precision)

        err = np.asarray(est) - np.asarray(truth)
        abs_err = np.abs(err)
        print(f"\n{name} errors:")
        print(f"  RMSE          : {_rmse(est, truth):.6f}")
        print(f"  rel Fro error : {_rel_fro_err(truth, est):.3f}%")
        print(f"  max |error|   : {abs_err.max():.6f} at index {np.unravel_index(abs_err.argmax(), abs_err.shape)}")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            prefix = (save_prefix or name).replace(" ", "_")
            _save_csv(save_dir, f"{prefix}__true", truth)
            _save_csv(save_dir, f"{prefix}__est", est)
            _save_csv(save_dir, f"{prefix}__error", err)

def main():
    ap = argparse.ArgumentParser(description="Compare true vs inferred parameters for Hawkes inference.")
    ap.add_argument("--result", default="inference_result.pickle",
                    help="Path to inference_result.pickle (posterior means saved by your inference script).")
    ap.add_argument("--mcmc", default="mcmc_state.npz",
                    help="Path to mcmc_state.npz (full MCMC draws); optional but enables quantile summaries.")
    ap.add_argument("--save_dir", default=None,
                    help="If set, save CSVs for each parameter: truth, estimate, error.")
    ap.add_argument("--precision", type=int, default=6, help="Print precision.")
    ap.add_argument("--no_full_print", action="store_true", help="Do not print full arrays (only stats).")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load posterior means and truths (if present)
    # ------------------------------------------------------------------
    with open(args.result, "rb") as f:
        res = pickle.load(f)

    # Inferred (posterior mean or MAP)
    mu_hat    = np.asarray(res["mu_hat"])
    K_hat     = np.asarray(res["K_hat"])        # masked & spatially weighted in your model
    M_K_hat   = np.asarray(res["M_K_hat"])
    sigma_hat = float(res["sigma_hat"])
    omega_hat = float(res["omega_hat"])

    # Optional truths (may be missing)
    mu_true    = res.get("mu_true", None)
    K_true     = res.get("K_true", None)
    M_K_true   = res.get("M_K_true", None)
    sigma_true = res.get("sigma_true", None)
    omega_true = res.get("omega_true", None)

    # Optionally load MCMC draws (for intervals / quantiles)
    mcmc = None
    if os.path.exists(args.mcmc):
        try:
            mcmc = np.load(args.mcmc)
            print(f"Loaded MCMC draws from {args.mcmc}. Available arrays: {list(mcmc.keys())}")
        except Exception as e:
            print(f"Warning: failed to load {args.mcmc}: {e}")

    print("\n=== Scalars ===")
    if sigma_true is not None:
        print(f"sigma: true={sigma_true:.6f}  inferred={sigma_hat:.6f}  rel.err={100.0*abs(sigma_hat-sigma_true)/(abs(sigma_true)+1e-12):.3f}%")
    else:
        print(f"sigma (inferred): {sigma_hat:.6f}")
    if omega_true is not None:
        print(f"omega: true={omega_true:.6f}  inferred={omega_hat:.6f}  rel.err={100.0*abs(omega_hat-omega_true)/(abs(omega_true)+1e-12):.3f}%")
    else:
        print(f"omega (inferred): {omega_hat:.6f}")
    def _fmt_val(x):
        x = np.asarray(x)
        # scalar (0-d) -> float
        if x.shape == ():
            return f"{float(x):.6f}"
        # 1-d or higher -> array string
        return np.array2string(
            x, precision=6, floatmode="fixed",
            suppress_small=False, max_line_width=140
        )

    # If you saved draws for sigma/omega, print intervals
    if mcmc is not None:
        for sname in ("sigma", "omega"):
            if sname in mcmc:
                s = mcmc[sname]
                q05, q50, q95 = np.quantile(s, [0.05, 0.50, 0.95], axis=0)
                print(f"{sname} 5–50–95%: {_fmt_val(q05)}, {_fmt_val(q50)}, {_fmt_val(q95)}")

    # Matrices: print full + error metrics and save CSVs if requested
    print_full = not args.no_full_print

    summarize_param("M_K", M_K_hat, truth=M_K_true, save_dir=args.save_dir,
                    save_prefix="M_K", print_full=print_full, precision=args.precision)

    # NOTE on K_hat:
    # Your K_hat is already the *effective* kernel you used in the intensity after masking & spatial kernel:
    #   G = (softplus(K_uncon) * reach_mask) * kappa
    # and you stored posterior mean of "K_masked * kappa" as K_hat.
    # The generator's K_true (if present) is likely the *pre-spatial, pre-mask base* K (NxN).
    # If you want to compare "like with like", either:
    #   (A) compare K_hat to "K_true_eff = (K_true * reach_mask) * kappa_true"
    #       (needs reach_mask and kappa_true); or
    #   (B) compare softplus(K_uncon) posterior mean to K_true (both unmasked, pre-spatial).
    #
    # Below we do a direct numeric compare if K_true is present, but also warn about the interpretation.

    if K_true is not None:
        print("\n[Warning] Comparing K_hat (effective, masked*spatial) to K_true (base).")
        print("          Consider adjusting to the same space if you want strictly apples-to-apples.")
    summarize_param("K", K_hat, truth=K_true if K_true is not None else None,
                    save_dir=args.save_dir, save_prefix="K", print_full=print_full, precision=args.precision)

    summarize_param("mu", mu_hat, truth=mu_true if mu_true is not None else None,
                    save_dir=args.save_dir, save_prefix="mu", print_full=print_full, precision=args.precision)

    # If you want credible intervals for matrices from MCMC:
    if mcmc is not None:
        def mat_quantiles(key, label):
            if key in mcmc:
                arr = mcmc[key]         # shape: (num_draws, *param_shape)
                q05 = np.quantile(arr, 0.05, axis=0)
                q50 = np.quantile(arr, 0.50, axis=0)
                q95 = np.quantile(arr, 0.95, axis=0)
                print(f"\n{label} quantiles (5%, 50%, 95%)")
                _print_array(f"{label} q05", q05, precision=args.precision)
                _print_array(f"{label} q50", q50, precision=args.precision)
                _print_array(f"{label} q95", q95, precision=args.precision)
                if args.save_dir:
                    _save_csv(args.save_dir, f"{label}_q05", q05)
                    _save_csv(args.save_dir, f"{label}_q50", q50)
                    _save_csv(args.save_dir, f"{label}_q95", q95)

        # These keys depend on how you saved the state; adapt as needed:
        mat_quantiles("M_K", "M_K")
        mat_quantiles("K_masked", "K")   # effective K used in intensity
        mat_quantiles("mu", "mu")

    print("\nDone.")

if __name__ == "__main__":
    main()
