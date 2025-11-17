import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pandas as pd
import math
import os

# -------- Select which inference result file ----------
file_path = "inference_result_np3_large_arbon_events_evening_copy.pickle"
# ------------------------------------------------------

# -------- Load inference result ----------
with open(file_path, "rb") as f:
    res = pickle.load(f)

print("Loaded inference:", Path(file_path).name)

# Load linked data pickle
data_path = res.get("data_pickle", None)
with open(Path(data_path), "rb") as f:
    data = pickle.load(f)
print("Loaded data from:", data_path)

# -------- Extract & sort events (time, node, mark, x, y) ----------
events = data["events"]
# robust to list/tuple-of-tuples or array
t_arr = np.array([e[0] for e in events], dtype=float)
u_arr = np.array([int(e[1]) for e in events], dtype=int)
e_arr = np.array([int(e[2]) for e in events], dtype=int)

order = np.argsort(t_arr)
t_arr = t_arr[order]
u_arr = u_arr[order]
e_arr = e_arr[order]

# Observation window
t_start = float(t_arr.min()) if t_arr.size else 0.0
t_end   = float(t_arr.max()) if t_arr.size else 0.0
T = t_end - t_start
print(f"Observation window: [{t_start:.2f}, {t_end:.2f}] (T={T:.2f})")

# -------- Parameters ----------
mu_hat   = np.asarray(res["mu_hat"])          # (N,M)
K_hat    = np.asarray(res["K_hat"])           # (N,N) == K_masked (column-normalized over reachable)
alpha    = float(res["alpha_hat"])
nonlin   = str(res.get("nonlinearity", "linear"))
W        = float(res.get("window", np.inf))   # finite look-back window; may be inf
N, M     = int(res["N"]), int(res["M"])

# --- Distance scaling (UTM meters -> kilometers) ---
# node_locations come from SUMO net (UTM Zone 32, meters). For plotting in km use:
meters_to_km = 1.0 / 1000.0

# --- Stability / branching helpers ---
def spectral_radius(mat):
    try:
        vals = np.linalg.eigvals(mat)
        return float(np.max(np.real(vals)))
    except Exception:
        # Fallback: power iteration on nonnegative matrix
        v = np.ones((mat.shape[1],), dtype=float)
        for _ in range(50):
            v_next = mat @ v
            nrm = np.linalg.norm(v_next) + 1e-12
            v = v_next / nrm
        return float(np.linalg.norm(mat @ v) / (np.linalg.norm(v) + 1e-12))

# Mark kernel (M×M) and spatial shape (N×N)
M_K_hat  = np.asarray(res.get("M_K_hat", None))
kappa_tilde_hat = np.asarray(res.get("kappa_tilde_hat", None))

# Check if this is a joint kernel model (nonpm_window_3.py)
is_joint_kernel_model = ("kernel_param" in res and 
                         "dist_centers" in res and 
                         "dist_scale" in res and
                         res["kernel_param"] is not None)

# Shapes & fallbacks
print("\nParameter shapes:")
print(f"  mu_hat:           {mu_hat.shape}")
print(f"  K_hat:            {K_hat.shape}")
print(f"  M_K_hat:          {None if M_K_hat is None else M_K_hat.shape}")
print(f"  kappa_tilde_hat:  {None if kappa_tilde_hat is None else kappa_tilde_hat.shape}")
if is_joint_kernel_model:
    print(f"  kernel_param:     {res['kernel_param'].shape}")
    print(f"  dist_centers:     {len(res['dist_centers'])}")
print(f"  N={N}, M={M}")

if M_K_hat is None:
    print("  ⚠ No M_K_hat in results; defaulting to identity (no cross-mark excitation).")
    M_K_hat = np.eye(M, dtype=float)
else:
    M_K_hat = M_K_hat.reshape(M, M)

# Handle different model types
if is_joint_kernel_model:
    print("  ✓ Joint spatio-temporal kernel model detected")
    # Joint kernel model doesn't have kappa_tilde_hat - K_hat already includes spatial effects
    kappa_tilde_hat = np.ones((N, N), dtype=float)
    print("  ✓ Using K_hat directly (includes spatial effects)")
elif kappa_tilde_hat is None:
    print("  ⚠ No kappa_tilde_hat in results; defaulting to ones (no extra spatial shaping).")
    kappa_tilde_hat = np.ones((N, N), dtype=float)
else:
    # Handle scalar kappa_tilde_hat (common in some models)
    if kappa_tilde_hat.size == 1:
        print("  ⚠ Scalar kappa_tilde_hat detected; expanding to matrix")
        kappa_tilde_hat = float(kappa_tilde_hat) * np.ones((N, N), dtype=float)
    else:
        kappa_tilde_hat = kappa_tilde_hat.reshape(N, N)

# Effective node kernel: elementwise product (matches model definition)
G_node = K_hat * kappa_tilde_hat  # (N,N)

# -------- Temporal kernel reconstruction (nonparametric) ----------
def softplus(x):
    # stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def gauss_bump_int_0_to_inf(c, scale):
    """
    ∫_0^∞ exp(-0.5 * ((t - c)/s)^2) dt
      = s * sqrt(pi/2) * [1 - erf((-c)/(sqrt(2)*s))]
    """
    rt2 = math.sqrt(2.0)
    return scale * math.sqrt(math.pi / 2.0) * (1.0 - math.erf((-c) / (rt2 * scale)))

# try to load a_uncon, time_centers, time_scale from state
a_uncon = None
time_centers = np.asarray(res.get("time_centers", []), dtype=float)
time_scale   = float(res.get("time_scale", 0.0)) if "time_scale" in res else None

# For joint kernel models, we have time_centers and time_scale directly
if is_joint_kernel_model:
    print("  ✓ Using time parameters from joint kernel model")
    have_temporal_np = (time_centers.size > 0) and (time_scale is not None)
else:
    # Try to load from state files (old model)
    state_loaded = False
    state_candidates = []
    if "mcmc_state_file" in res and res["mcmc_state_file"]:
        state_candidates.append(res["mcmc_state_file"])
    # also try version without model suffix (known mismatch in some runs)
    stem = Path(data_path).stem
    state_candidates.append(f"mcmc_state_np_{stem}.npz")
    # also try with plain nonlinearity suffix or _quad
    model_name_guess = res.get("nonlinearity", "linear")
    state_candidates.append(f"mcmc_state_np_{stem}_{model_name_guess}.npz")
    if "time_centers_q" in res and len(res["time_centers_q"]) > 0:
        state_candidates.append(f"mcmc_state_np_{stem}_{model_name_guess}_quad.npz")

    for cand in state_candidates:
        if cand and Path(cand).exists():
            try:
                state = np.load(cand)
                if ("a_uncon" in state) and ("time_centers" in state) and ("time_scale" in state):
                    a_uncon = np.mean(state["a_uncon"], axis=0)  # posterior mean over samples
                    time_centers = np.asarray(state["time_centers"], dtype=float)
                    time_scale   = float(state["time_scale"])
                    state_loaded = True
                    print(f"\nLoaded temporal params from state: {cand}")
                    print(f"  a_uncon shape: {state['a_uncon'].shape} -> mean -> {a_uncon.shape}")
                    break
            except Exception as ex:
                print(f"  ⚠ Failed loading state from {cand}: {ex}")

    have_temporal_np = (a_uncon is not None) and (time_centers.size > 0) and (time_scale is not None)

if have_temporal_np:
    if is_joint_kernel_model:
        # Joint kernel model: use time_centers and time_scale directly
        # For visualization, we'll create a simple temporal kernel
        print(f"\nTemporal kernel: joint model with {len(time_centers)} temporal bases; scale={time_scale:.4g}")
        
        def g_tilde(delta):
            # Simple temporal kernel for joint model visualization
            delta = np.asarray(delta, dtype=float)
            # Use the first few time centers for visualization
            if len(time_centers) > 0:
                phi = np.exp(-0.5 * ((delta[..., None] - time_centers[None, ...]) / time_scale) ** 2)
                # Simple uniform weights for visualization
                weights = np.ones(len(time_centers)) / len(time_centers)
                return np.dot(phi, weights)
            else:
                return np.exp(-delta / time_scale) if time_scale > 0 else np.exp(-delta)
    else:
        # Old model: use a_uncon from state
        w_pos = softplus(a_uncon) + 1e-8  # match training transform
        ints = np.array([gauss_bump_int_0_to_inf(c, time_scale) for c in time_centers], dtype=float)
        Z_t = float(np.dot(w_pos, ints)) + 1e-12
        mix_w = w_pos / Z_t

        def g_tilde(delta):
            # mixture of Gaussians (τ>=0 enforced via centers & integral normalization)
            # Allow vectorized delta
            delta = np.asarray(delta, dtype=float)
            # phi_k(delta) = exp(-0.5*((delta - c_k)/s)^2)
            phi = np.exp(-0.5 * ((delta[..., None] - time_centers[None, ...]) / time_scale) ** 2)  # [..., K]
            return np.dot(phi, mix_w)  # [...]
        print(f"\nTemporal kernel: nonparametric mixture with {len(time_centers)} bases; scale={time_scale:.4g}")
else:
    # Fallback: simple exponential with rate ~ 1/time_scale if available, else 1.0
    omega = 1.0
    if time_scale and time_scale > 0:
        omega = 1.0 / time_scale
    print(f"\nTemporal kernel: fallback exponential with ω={omega:.4g} (state file not found or incomplete)")

    def g_tilde(delta):
        delta = np.asarray(delta, dtype=float)
        return np.exp(-omega * np.maximum(delta, 0.0))

# -------- Nonlinearity φ --------
def phi_link(x, kind="linear"):
    if kind == "linear":
        return np.clip(x, a_min=1e-12, a_max=None)
    elif kind == "softplus":
        return softplus(x) + 1e-12
    elif kind == "relu":
        return np.clip(x, a_min=0.0, a_max=None) + 1e-12
    elif kind == "exp":
        return np.exp(x) + 1e-12
    elif kind == "power2":
        z = np.clip(x, a_min=0.0, a_max=None)
        return z * z + 1e-12
    else:
        return np.clip(x, a_min=1e-12, a_max=None)

if "gamma_hat" in res and float(res.get("gamma_hat", 0.0)) > 0:
    print("⚠ Quadratic Hawkes (gamma_hat) present in results but ignored in this validator (no z(τ) reconstruction).")

# -------- Binning ----------
# (keep your binning; you can make it denser if needed)
num_bins = 20 if T > 0 else 1
bin_width = T / max(num_bins, 1)
time_bins = np.arange(t_start, t_end + bin_width, bin_width) if T > 0 else np.array([0, 1])
mid_times = 0.5 * (time_bins[1:] + time_bins[:-1])

# -------- Prediction function (node × mark intensities) ----------
def predict_intensity_per_node_mark(
    t_arr, u_arr, e_arr,
    mu_hat, G_node, M_K, alpha, times, N, M,
    g_tilde, W, nonlin
):
    mu_hat = mu_hat.reshape(N, M)
    intensities = []
    K = len(t_arr)

    for t in times:
        lam_mat = mu_hat.copy()  # baseline (N, M)
        # Accumulate excitation from events j with 0 <= t - t_j <= W
        # t_arr is sorted; break when t_j >= t
        for j in range(K):
            tj = t_arr[j]
            if tj >= t:
                break
            dt = t - tj
            if dt < 0.0 or (np.isfinite(W) and dt > W):
                continue
            u_j = u_arr[j]
            e_j = e_arr[j]
            # outer product: (N,) ⊗ (M,) -> (N,M)
            lam_mat += alpha * g_tilde(dt) * np.outer(G_node[:, u_j], M_K[e_j, :])

        # apply nonlinearity elementwise
        lam_mat = phi_link(lam_mat, kind=nonlin)
        intensities.append(lam_mat)

    return np.array(intensities)  # shape (len(times), N, M)

# Predict (intensity -> expected counts per bin)
pred_intensity_nodes_marks = predict_intensity_per_node_mark(
    t_arr, u_arr, e_arr,
    mu_hat, G_node, M_K_hat, alpha, mid_times, N, M,
    g_tilde, W, nonlin
)
pred_counts_nodes_marks = pred_intensity_nodes_marks * bin_width

# -------- Observed counts (node × mark) ----------
obs_counts_nodes_marks = np.zeros((len(time_bins) - 1, N, M))
for idx, (t0, t1) in enumerate(zip(time_bins[:-1], time_bins[1:])):
    mask = (t_arr >= t0) & (t_arr < t1)
    if not np.any(mask):
        continue
    for tj, uj, ej in zip(t_arr[mask], u_arr[mask], e_arr[mask]):
        obs_counts_nodes_marks[idx, uj, ej] += 1

# -------- Exponential Moving Average helper ----------
def exponential_moving_average(series, alpha=0.4):
    """
    Exponential Moving Average with no future bias.
    alpha: smoothing factor (0 < alpha < 1)
    - alpha = 0.1: very smooth, heavy historical bias
    - alpha = 0.3: moderate smoothing (default)
    - alpha = 0.5: less smoothing, more responsive
    - alpha = 0.9: minimal smoothing, very responsive
    """
    if len(series) == 0:
        return series
    
    ema = np.zeros_like(series)
    ema[0] = series[0]  # First value is just the first observation
    
    # Forward pass: each point only uses historical data
    for i in range(1, len(series)):
        ema[i] = alpha * series[i] + (1 - alpha) * ema[i-1]
    
    return ema

# -------- Metrics ----------
def calculate_correlation(obs, pred):
    if len(obs) < 2 or len(pred) < 2:
        return 0.0
    valid_mask = ~(np.isnan(obs) | np.isnan(pred))
    if valid_mask.sum() < 2:
        return 0.0
    ov = obs[valid_mask]; pv = pred[valid_mask]
    if np.std(ov) == 0 or np.std(pv) == 0:
        return 0.0
    try:
        return float(np.corrcoef(ov, pv)[0, 1])
    except Exception:
        return 0.0

def calculate_mse(obs, pred):
    valid_mask = ~(np.isnan(obs) | np.isnan(pred))
    if valid_mask.sum() == 0:
        return float("inf")
    ov = obs[valid_mask]; pv = pred[valid_mask]
    return float(np.mean((ov - pv) ** 2))

def calculate_mae(obs, pred):
    valid_mask = ~(np.isnan(obs) | np.isnan(pred))
    if valid_mask.sum() == 0:
        return float("inf")
    ov = obs[valid_mask]; pv = pred[valid_mask]
    return float(np.mean(np.abs(ov - pv)))

def calculate_r2(obs, pred):
    valid_mask = ~(np.isnan(obs) | np.isnan(pred))
    if valid_mask.sum() < 2:
        return 0.0
    ov = obs[valid_mask]; pv = pred[valid_mask]
    ss_tot = np.sum((ov - np.mean(ov)) ** 2)
    if ss_tot == 0:
        return 0.0
    ss_res = np.sum((ov - pv) ** 2)
    return float(1.0 - ss_res / ss_tot)

# -------- Plot rolling mean vs rescaled model ----------
pdf_path = f"temporal_validation_node_mark_{Path(file_path).name}.pdf"
with PdfPages(pdf_path) as pdf:

    # Collect all data for aggregate metrics
    all_obs = []
    all_pred = []
    all_obs_smooth = []
    all_pred_rescaled = []

    for node in range(N):
        # Sum across all marks for this node
        obs_series = np.sum(obs_counts_nodes_marks[:, node, :], axis=1)  # Sum over marks
        pred_series = np.sum(pred_counts_nodes_marks[:, node, :], axis=1)  # Sum over marks

        # skip empty series
        if obs_series.sum() == 0 and pred_series.mean() < 1e-9:
            continue

        # Rescale so means match (shape comparison). Keep a toggle if you want calibrated checks.
        scale_factor = (obs_series.mean() / pred_series.mean()) if pred_series.mean() > 0 else 1.0
        pred_series_rescaled = pred_series * scale_factor

        obs_series_smooth = exponential_moving_average(obs_series)

        # Metrics
        corr = calculate_correlation(obs_series_smooth, pred_series_rescaled)
        mse  = calculate_mse(obs_series_smooth, pred_series_rescaled)
        mae  = calculate_mae(obs_series_smooth, pred_series_rescaled)
        r2   = calculate_r2(obs_series_smooth, pred_series_rescaled)

        # Aggregate
        all_obs.extend(obs_series)
        all_pred.extend(pred_series)
        all_obs_smooth.extend(obs_series_smooth)
        all_pred_rescaled.extend(pred_series_rescaled)

        # Plot
        plt.figure(figsize=(12, 5))  # Increased height to prevent title overlap
        plt.plot(mid_times, obs_series_smooth, label="Obs", linewidth=2)
        plt.plot(mid_times, pred_series_rescaled, label="Pred", linewidth=2)
        plt.xlabel("Time")
        plt.ylabel("Total events per bin (all marks)")
        plt.title(f"Node {node} - Total Events (All Marks)  |  φ={nonlin}, W={'∞' if not np.isfinite(W) else f'{W:.2f}'}")
        metrics_text = (
            f"Scale: {scale_factor:.3f}\n"
            f"Corr: {corr:.3f}\nR²: {r2:.3f}\n"
            f"MSE: {mse:.3f}\nMAE: {mae:.3f}"
        )
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                 va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.legend()
        plt.tight_layout()  # Prevent overlap
        pdf.savefig(bbox_inches='tight')  # Ensure no cutoff
        plt.close()

    # Aggregate summary
    if all_obs and all_pred:
        all_obs = np.array(all_obs)
        all_pred = np.array(all_pred)
        all_obs_smooth = np.array(all_obs_smooth)
        all_pred_rescaled = np.array(all_pred_rescaled)

        agg_corr = calculate_correlation(all_obs_smooth, all_pred_rescaled)
        agg_mse  = calculate_mse(all_obs_smooth, all_pred_rescaled)
        agg_mae  = calculate_mae(all_obs_smooth, all_pred_rescaled)
        agg_r2   = calculate_r2(all_obs_smooth, all_pred_rescaled)

        plt.figure(figsize=(14, 8))  # Increased size for better spacing
        plt.subplot(2, 1, 1)
        plt.plot(all_obs_smooth, label="Obs (all nodes)", alpha=0.7, linewidth=1)
        plt.plot(all_pred_rescaled, label="Pred (all nodes)", alpha=0.7, linewidth=1)
        plt.xlabel("Time bin index")
        plt.ylabel("Total events per bin (all nodes)")
        plt.title("Overall Model Fit - Total Events Across All Nodes")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        lim = max(all_obs_smooth.max(), all_pred_rescaled.max())
        plt.scatter(all_obs_smooth, all_pred_rescaled, alpha=0.5, s=10)
        plt.plot([0, lim], [0, lim], 'r--', label='Perfect fit')
        plt.xlabel("Observed total events")
        plt.ylabel("Predicted total events")
        plt.title("Observed vs Predicted - Total Events")
        plt.legend()
        plt.grid(True, alpha=0.3)

        agg_text = f"AGGREGATE METRICS (All Nodes Combined):\nCorr: {agg_corr:.3f}\nR²: {agg_r2:.3f}\nMSE: {agg_mse:.3f}\nMAE: {agg_mae:.3f}"
        plt.figtext(0.02, 0.02, agg_text, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()  # Prevent overlap
        pdf.savefig(bbox_inches='tight')  # Ensure no cutoff
        plt.close()

    # -------- Per-mark (event type) overall fit plots --------
    # Build per-mark figures using the SAME aggregation method as the overall plot:
    # concatenate per-node series (with EMA and per-node rescaling)
    if M > 0:
        for m_idx in range(M):
            mark_obs_all = []
            mark_pred_all = []
            mark_obs_smooth_all = []
            mark_pred_rescaled_all = []

            for node in range(N):
                obs_series_m = obs_counts_nodes_marks[:, node, m_idx]
                pred_series_m = pred_counts_nodes_marks[:, node, m_idx]

                if obs_series_m.sum() == 0 and pred_series_m.mean() < 1e-9:
                    continue

                scale_m = (obs_series_m.mean() / pred_series_m.mean()) if pred_series_m.mean() > 0 else 1.0
                pred_series_m_rescaled = pred_series_m * scale_m
                obs_series_m_smooth = exponential_moving_average(obs_series_m)

                mark_obs_all.extend(obs_series_m)
                mark_pred_all.extend(pred_series_m)
                mark_obs_smooth_all.extend(obs_series_m_smooth)
                mark_pred_rescaled_all.extend(pred_series_m_rescaled)

            if not mark_obs_all:
                continue

            mark_obs_all = np.array(mark_obs_all)
            mark_pred_all = np.array(mark_pred_all)
            mark_obs_smooth_all = np.array(mark_obs_smooth_all)
            mark_pred_rescaled_all = np.array(mark_pred_rescaled_all)

            corr_m = calculate_correlation(mark_obs_smooth_all, mark_pred_rescaled_all)
            mse_m  = calculate_mse(mark_obs_smooth_all, mark_pred_rescaled_all)
            mae_m  = calculate_mae(mark_obs_smooth_all, mark_pred_rescaled_all)
            r2_m   = calculate_r2(mark_obs_smooth_all, mark_pred_rescaled_all)

            plt.figure(figsize=(14, 8))
            # Top: concatenated time series (same as overall plot style)
            plt.subplot(2, 1, 1)
            plt.plot(mark_obs_smooth_all, label="Obs", alpha=0.7, linewidth=1)
            plt.plot(mark_pred_rescaled_all, label="Pred", alpha=0.7, linewidth=1)
            plt.xlabel("Time bin index ")
            plt.ylabel("Total events per bin")
            plt.title(f"Overall Model Fit - Event Type {m_idx}")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Bottom: scatter
            plt.subplot(2, 1, 2)
            lim = max(mark_obs_smooth_all.max(), mark_pred_rescaled_all.max(), 1e-9)
            plt.scatter(mark_obs_smooth_all, mark_pred_rescaled_all, alpha=0.5, s=10)
            plt.plot([0, lim], [0, lim], 'r--', label='Perfect fit')
            plt.xlabel("Observed total events")
            plt.ylabel("Predicted total events")
            plt.title(f"Observed vs Predicted - Event Type {m_idx}")
            plt.legend()
            plt.grid(True, alpha=0.3)

            mark_text = (
                f"EVENT TYPE {m_idx} METRICS:\n"
                f"Corr: {corr_m:.3f}\nR\u00b2: {r2_m:.3f}\n"
                f"MSE: {mse_m:.3f}\nMAE: {mae_m:.3f}"
            )
            plt.figtext(0.02, 0.02, mark_text, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

    # -------- Plot Hawkes Process Kernels ----------
    print("Plotting Hawkes process kernels...")

    # Temporal kernel curve for plotting
    if have_temporal_np:
        tmax = float(time_centers.max()) if time_centers.size else (T/4 if T>0 else 1.0)
        time_range = np.linspace(0.0, max(1e-6, 2.0*tmax), 200)
        temporal_kernel = g_tilde(time_range)
    else:
        # exponential fallback range
        tr_max = (5.0 / (1.0 / time_scale)) if (time_scale and time_scale > 0) else 5.0
        time_range = np.linspace(0.0, tr_max, 200)
        temporal_kernel = g_tilde(time_range)

    plt.figure(figsize=(18, 12))  # Increased size for better spacing

    # 1) Temporal kernel
    plt.subplot(2, 3, 1)
    plt.plot(time_range, temporal_kernel, linewidth=2)
    plt.xlabel('Time difference τ')
    plt.ylabel('g̃(τ)')
    if have_temporal_np:
        plt.title(f'Nonparametric Temporal Kernel g̃(τ)\nK={len(time_centers)} bases, scale={time_scale:.3g}')
        for c in time_centers:
            plt.axvline(c, color='red', alpha=0.25, linestyle='--')
    else:
        plt.title('Temporal Kernel (fallback exponential)')
    plt.grid(True, alpha=0.3)

    # 2) K_hat
    plt.subplot(2, 3, 2)
    im1 = plt.imshow(K_hat, aspect='auto')
    plt.colorbar(im1, label='Value')
    plt.xlabel('Source node')
    plt.ylabel('Target node')
    plt.title('Spatial kernel K_hat (column-normalized)')
    plt.xticks(range(N)); plt.yticks(range(N))

    # 3) kappa_tilde_hat
    plt.subplot(2, 3, 3)
    im2 = plt.imshow(kappa_tilde_hat, aspect='auto')
    plt.colorbar(im2, label='Value')
    plt.xlabel('Source node')
    plt.ylabel('Target node')
    plt.title('Spatial shape κ̃ (per-column normalized)')
    plt.xticks(range(N)); plt.yticks(range(N))

    # 4) G_node = K_hat ∘ κ̃
    plt.subplot(2, 3, 4)
    im3 = plt.imshow(G_node, aspect='auto')
    plt.colorbar(im3, label='Value')
    plt.xlabel('Source node')
    plt.ylabel('Target node')
    plt.title('Effective node kernel G_node = K_hat ∘ κ̃')
    plt.xticks(range(N)); plt.yticks(range(N))

    # 5) Mark kernel M_K_hat
    plt.subplot(2, 3, 5)
    im4 = plt.imshow(M_K_hat, aspect='auto')
    plt.colorbar(im4, label='Value')
    plt.xlabel('Source mark')
    plt.ylabel('Target mark')
    plt.title('Mark kernel M_K (row-normalized)')
    plt.xticks(range(M)); plt.yticks(range(M))

    # 6) Kernel stats
    plt.subplot(2, 3, 6)
    plt.axis('off')
    kstats = f"""KERNEL STATS:

Spatial K_hat ({N}×{N})
  min={K_hat.min():.4f}  max={K_hat.max():.4f}
  mean={K_hat.mean():.4f} std={K_hat.std():.4f}

κ̃ ({N}×{N})
  min={kappa_tilde_hat.min():.4f}  max={kappa_tilde_hat.max():.4f}
  mean={kappa_tilde_hat.mean():.4f} std={kappa_tilde_hat.std():.4f}

G_node=K∘κ̃ ({N}×{N})
  min={G_node.min():.4f}  max={G_node.max():.4f}
  mean={G_node.mean():.4f} std={G_node.std():.4f}

M_K ({M}×{M})
  min={M_K_hat.min():.4f}  max={M_K_hat.max():.4f}
  mean={M_K_hat.mean():.4f} std={M_K_hat.std():.4f}

φ: {nonlin}   Window W: {"∞" if not np.isfinite(W) else f"{W:.4g}"}
"""
    plt.text(0.05, 0.5, kstats, fontsize=9, fontfamily='monospace',
             va='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()  # Prevent overlap
    pdf.savefig(bbox_inches='tight')  # Ensure no cutoff
    plt.close()

    # ---- Joint kernel analysis (for nonpm_window_3.py results) ----
    print("  Creating joint kernel analysis...")
    
    # Check if we have joint kernel parameters
    have_joint_kernel = ("kernel_param" in res and 
                         "dist_centers" in res and 
                         "dist_scale" in res and
                         res["kernel_param"] is not None)
    
    if have_joint_kernel:
        print("    Joint spatio-temporal kernel detected")
        
        # Extract joint kernel parameters
        W_uncon_hat = np.asarray(res["kernel_param"])  # (B_t, B_r)
        dist_centers = np.asarray(res["dist_centers"])
        dist_scale = float(res["dist_scale"])
        
        # Reconstruct joint kernel ψ̃(τ, r) on a grid
        tau_max = float(time_centers.max()) if time_centers.size else (T/4 if T>0 else 1.0)
        r_max = float(dist_centers.max()) if dist_centers.size else 1.0
        
        # Use a consistent full grid (keeps values identical); zoom via x-limits only
        tau_grid = np.linspace(0.0, max(1e-6, 2.0*tau_max), 200)
        r_grid = np.linspace(0.0, max(1e-6, 1.2*r_max), 100)
        
        # Create 2D grid
        TAU, R = np.meshgrid(tau_grid, r_grid)
        
        # Reconstruct joint kernel values
        joint_kernel = np.zeros_like(TAU)
        for i, r in enumerate(r_grid):
            for j, tau in enumerate(tau_grid):
                # Spatial basis
                phi_r = np.exp(-0.5 * ((r - dist_centers) / dist_scale) ** 2)
                # Temporal basis
                phi_tau = np.exp(-0.5 * ((tau - time_centers) / time_scale) ** 2)
                # Joint kernel: ψ̃(τ, r) = Σ_{b_t, b_r} w[b_t, b_r] * φ_t(τ) * φ_r(r)
                joint_kernel[i, j] = np.sum(W_uncon_hat * np.outer(phi_tau, phi_r))
        
        # Normalize to have unit time integral per distance
        for i, r in enumerate(r_grid):
            if r > 0:  # Skip r=0 (self-excitation)
                integral = np.trapezoid(joint_kernel[i, :], tau_grid)
                if integral > 0:
                    joint_kernel[i, :] /= integral
        
        # Plot joint kernel
        plt.figure(figsize=(16, 10))
        
        # 1) 2D heatmap of joint kernel
        plt.subplot(2, 2, 1)
        im = plt.imshow(joint_kernel, extent=[tau_grid[0], tau_grid[-1], r_grid[0], r_grid[-1]], 
                       origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(im, label='ψ̃(τ, r)')
        plt.xlabel('Time lag τ')
        plt.ylabel('Distance r')
        plt.title('Joint Spatio-Temporal Kernel ψ̃(τ, r)')
        plt.grid(True, alpha=0.3)
        # Zoom view without changing values
        tau_zoom = 1.25 * tau_max
        plt.xlim(0.0, tau_zoom)
        
        # 2) Temporal slices at different distances
        plt.subplot(2, 2, 2)
        r_indices = [0, len(r_grid)//4, len(r_grid)//2, 3*len(r_grid)//4, -1]
        for idx in r_indices:
            r_val = r_grid[idx]
            # mask to zoom range for display only
            mask = tau_grid <= tau_zoom
            plt.plot(tau_grid[mask], joint_kernel[idx, :][mask], label=f'r={r_val:.2f}', linewidth=2)
        plt.xlabel('Time lag τ')
        plt.ylabel('ψ̃(τ, r)')
        plt.title('Temporal Slices at Different Distances')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3) Distance slices at different times
        plt.subplot(2, 2, 3)
        tau_indices = [0, np.searchsorted(tau_grid, tau_zoom/4), np.searchsorted(tau_grid, tau_zoom/2), np.searchsorted(tau_grid, 0.9*tau_zoom)]
        for idx in tau_indices:
            tau_val = tau_grid[idx]
            plt.plot(r_grid, joint_kernel[:, idx], label=f'τ={tau_val:.2f}', linewidth=2)
        plt.xlabel('Distance r')
        plt.ylabel('ψ̃(τ, r)')
        plt.title('Distance Slices at Different Times')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4) Joint kernel parameters heatmap
        plt.subplot(2, 2, 4)
        im2 = plt.imshow(W_uncon_hat, aspect='auto', cmap='RdBu_r', 
                         extent=[0, dist_centers.size, 0, time_centers.size])
        plt.colorbar(im2, label='Weight')
        plt.xlabel('Distance basis index')
        plt.ylabel('Time basis index')
        plt.title('Joint Kernel Weights W_uncon')
        plt.xticks(range(dist_centers.size), [f'{c:.2f}' for c in dist_centers])
        plt.yticks(range(time_centers.size), [f'{c:.2f}' for c in time_centers])
        
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        # Extra figure: zoomed heatmap up to tau <= 5 and r <= 2000 (values unchanged)
        tau_zoom2 = 5.0
        r_zoom2 = 2000.0
        mask_tau = tau_grid <= tau_zoom2
        mask_r = r_grid <= r_zoom2
        if np.any(mask_tau) and np.any(mask_r):
            plt.figure(figsize=(12, 7))
            imz = plt.imshow(joint_kernel[mask_r][:, mask_tau],
                             extent=[0.0, tau_zoom2, 0.0, r_zoom2],
                             origin='lower', aspect='auto', cmap='viridis')
            plt.colorbar(imz, label='ψ̃(τ, r)')
            plt.xlabel('Time lag τ (zoomed ≤ 5)')
            plt.ylabel('Distance r (zoomed ≤ 2000)')
            plt.title('Joint Spatio-Temporal Kernel ψ̃(τ, r) — Zoomed τ ≤ 5, r ≤ 2000')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()
        
        # Rotated-style zoom (x=space in km, y=time in days) using contourf
        if np.any(mask_tau) and np.any(mask_r):
            tau_days = tau_grid[mask_tau] / 24.0
            r_km = r_grid[mask_r] * meters_to_km
            TAU_Z, R_Z = np.meshgrid(tau_days, r_km)
            Z = joint_kernel[mask_r][:, mask_tau]
            plt.figure(figsize=(10, 7))
            levels = 20
            cf = plt.contourf(TAU_Z, R_Z, Z, levels=levels, cmap='viridis')
            plt.colorbar(cf, label='ψ̃(τ, r)')
            plt.xlabel('Time lag (day)')
            plt.ylabel('Space lag (km)')
            plt.title('Joint kernel — zoomed (time on x, space on y)')
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()
        
        # Additional analysis: distance-based excitation patterns
        plt.figure(figsize=(12, 8))
        
        # Calculate average excitation strength per distance
        avg_excitation = np.mean(joint_kernel, axis=1)  # Average over time
        plt.subplot(2, 1, 1)
        plt.plot(r_grid, avg_excitation, 'b-', linewidth=2)
        plt.xlabel('Distance r')
        plt.ylabel('Average ψ̃(τ, r)')
        plt.title('Average Excitation Strength vs Distance')
        plt.grid(True, alpha=0.3)
        
        # Calculate temporal decay at different distances
        plt.subplot(2, 1, 2)
        r_test = [r_grid[0], r_grid[len(r_grid)//4], r_grid[len(r_grid)//2], r_grid[-1]]
        for r_val in r_test:
            r_idx = np.argmin(np.abs(r_grid - r_val))
            plt.plot(tau_grid, joint_kernel[r_idx, :], label=f'r={r_val:.2f}', linewidth=2)
        plt.xlabel('Time lag τ')
        plt.ylabel('ψ̃(τ, r)')
        plt.title('Temporal Decay at Different Distances')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        print("    Joint kernel analysis complete")
    else:
        print("    No joint kernel parameters found - skipping joint analysis")

    # ---- Extra panel: mark kernel annotated if small ----
    if M <= 6:
        plt.figure(figsize=(6, 5))  # Increased size for better spacing
        im = plt.imshow(M_K_hat, aspect='auto')
        plt.colorbar(im)
        plt.title(f"M_K values ({M}×{M})")
        plt.xlabel('Source mark'); plt.ylabel('Target mark')
        for i in range(M):
            for j in range(M):
                plt.text(j, i, f"{M_K_hat[i,j]:.3f}",
                         ha='center', va='center', color='white', fontweight='bold')
        plt.tight_layout()  # Prevent overlap
        pdf.savefig(bbox_inches='tight')  # Ensure no cutoff
        plt.close()

    print("=====================================\n")

    # ---- Stability & Branching Summary ----
    rho_K = spectral_radius(K_hat)
    rho_M = spectral_radius(M_K_hat)
    eta = float(alpha * rho_K * rho_M)
    stable = eta < 1.0

    print("=== STABILITY / BRANCHING ===")
    print(f"alpha:   {alpha:.4f}")
    print(f"rho(K):  {rho_K:.4f}")
    print(f"rho(M):  {rho_M:.4f}")
    print(f"eta=alpha*rho(K)*rho(M): {eta:.4f}  -> {'STABLE' if stable else 'UNSTABLE'}")

    # PDF page
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    summary = (
        f"STABILITY / BRANCHING SUMMARY\n\n"
        f"alpha: {alpha:.6f}\n"
        f"rho(K_hat): {rho_K:.6f}\n"
        f"rho(M_K_hat): {rho_M:.6f}\n"
        f"eta = alpha * rho(K_hat) * rho(M_K_hat) = {eta:.6f}\n\n"
        f"Condition (linear Hawkes): eta < 1  -> {'STABLE' if stable else 'UNSTABLE'}\n"
    )
    plt.text(0.05, 0.9, summary, fontsize=11, fontfamily='monospace', va='top')
    pdf.savefig(bbox_inches='tight')
    plt.close()

print(f"Saved node validation with kernels to {pdf_path}")
