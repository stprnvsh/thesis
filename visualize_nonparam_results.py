#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NON-PARAMETRIC HAWKES MODEL RESULTS VISUALIZATION (supports v2 and v3)
- v2: separate temporal g̃ and spatial κ̃
- v3: joint spatio-temporal kernel ψ̃(τ,r) via kernel_param (B_t×B_r)
"""

import pickle
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from math import erf, sqrt, pi
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

# Global figure save config (set in main)
SAVE_FIGS = False
OUTDIR = "figs"

def _savefig(name: str):
    if SAVE_FIGS:
        os.makedirs(OUTDIR, exist_ok=True)
        path = os.path.join(OUTDIR, name)
        plt.savefig(path, dpi=150, bbox_inches='tight')


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _gauss_bump(x, c, s):
    z = (x - c) / s
    return np.exp(-0.5 * z * z)


def _gauss_int_0_to(x, c, s):
    rt2 = sqrt(2.0)
    pref = s * sqrt(pi / 2.0)
    return pref * (erf((x - c) / (rt2 * s)) - erf((-c) / (rt2 * s)))


def _gauss_int_0_to_inf(c, s):
    rt2 = sqrt(2.0)
    return s * sqrt(pi / 2.0) * (1.0 - erf((-c) / (rt2 * s)))


def _ks_uniform(pvals):
    if pvals.size == 0:
        return np.nan
    u = np.sort(pvals)
    n = len(u)
    grid = (np.arange(n) + 1) / n
    return float(np.max(np.abs(u - grid)))


def _spec_radius(mat):
    try:
        vals = np.linalg.eigvals(np.asarray(mat))
        return float(np.max(np.abs(vals)))
    except Exception:
        return np.nan


def load_results(path: str):
    path = path or 'inference_result_np_large_arbon_events_evening_copy_linear.pickle'
    print("Loading results:", path)
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results


# ---------------- Common plots ----------------

def plot_base_rates(results):
    mu_hat = results['mu_hat']
    N, M = mu_hat.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    im1 = ax1.imshow(mu_hat.T, cmap='viridis', aspect='auto')
    ax1.set_title('Base Rates μ (Node × Event)')
    ax1.set_xlabel('Node'); ax1.set_ylabel('Event')
    plt.colorbar(im1, ax=ax1, label='Rate')
    mean_rates = np.mean(mu_hat, axis=0)
    ax2.bar(range(M), mean_rates, color=['skyblue','salmon'][:M], alpha=0.7)
    ax2.set_title('Average Base Rates'); ax2.set_ylabel('Rate'); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); _savefig('base_rates.png'); plt.show()


def plot_spatial_coupling(results):
    K_hat = results['K_hat']; alpha_hat = results['alpha_hat']
    K_eff = alpha_hat * K_hat
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmax = max(abs(K_hat.min()), abs(K_hat.max())) or 1.0
    im1 = axes[0].imshow(K_hat, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Raw K'); plt.colorbar(im1, ax=axes[0])
    vmax2 = max(abs(K_eff.min()), abs(K_eff.max())) or 1.0
    im2 = axes[1].imshow(K_eff, cmap='RdBu_r', vmin=-vmax2, vmax=vmax2)
    axes[1].set_title(f'α×K (α={alpha_hat:.3f})'); plt.colorbar(im2, ax=axes[1])
    plt.tight_layout(); _savefig('spatial_coupling.png'); plt.show()


def plot_mark_kernel(results):
    M_K = results['M_K_hat']
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    vmax = max(abs(M_K.min()), abs(M_K.max())) or 1.0
    im = ax.imshow(M_K, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title('Mark Kernel M_K'); plt.colorbar(im, ax=ax)
    plt.tight_layout(); _savefig('mark_kernel.png'); plt.show()


def plot_network(results):
    node_xy = np.asarray(results.get('node_locations', []))
    reach = np.asarray(results.get('reach_mask', []))
    if node_xy.size == 0 or reach.size == 0:
        return
    N = node_xy.shape[0]
    plt.figure(figsize=(6,6))
    # edges
    for i in range(N):
        for j in range(i+1, N):
            if reach[i, j] > 0:
                x = [node_xy[i,0], node_xy[j,0]]
                y = [node_xy[i,1], node_xy[j,1]]
                plt.plot(x, y, color='lightgray', lw=0.8, alpha=0.7)
    # nodes
    plt.scatter(node_xy[:,0], node_xy[:,1], c='k', s=25)
    plt.title('Network (reachability edges)')
    plt.axis('equal'); plt.tight_layout(); _savefig('network_graph.png'); plt.show()


def plot_adjacency_heatmap(results):
    reach = np.asarray(results.get('reach_mask', []))
    if reach.size == 0:
        return
    plt.figure(figsize=(5,4))
    plt.imshow(reach, cmap='Greys', vmin=0, vmax=1)
    plt.title('Reachability (mask)'); plt.colorbar(); plt.tight_layout(); _savefig('reachability_heatmap.png'); plt.show()


# ---------------- v2-only (separate κ̃,g̃) ----------------

def _load_temporal_weights_v2(results):
    time_centers = np.asarray(results.get('time_centers', None))
    time_scale = float(results.get('time_scale', 1.0))
    
    if time_centers is None:
        print("Warning: Missing time_centers in results")
        return None, None, None
    
    state_file = results.get('mcmc_state_file', 'mcmc_state_np.npz')
    if not os.path.exists(state_file):
        state_file = 'mcmc_state_np.npz' if os.path.exists('mcmc_state_np.npz') else None
    if state_file is None: 
        print("Warning: No MCMC state file found")
        return None, None, None
    
    try:
        st = np.load(state_file, allow_pickle=True)
        a_uncon = st.get('a_uncon', None)
        if a_uncon is None: 
            print("Warning: No a_uncon found in MCMC state file")
            return None, None, None
        
        a_mean = np.asarray(a_uncon).mean(axis=0)
        
        # Use the actual basis size from the loaded weights
        actual_basis_size = len(a_mean)
        expected_basis_size = len(time_centers)
        
        if actual_basis_size != expected_basis_size:
            print(f"Info: Using actual basis size {actual_basis_size} (expected {expected_basis_size})")
            # Reconstruct time_centers with the correct size
            T = float(results.get('T', 1.0))
            time_centers = np.linspace(0.0, T, actual_basis_size)
        
        w_pos = _softplus(a_mean) + 1e-8
        ints = np.array([_gauss_int_0_to_inf(c, time_scale) for c in time_centers])
        Z = float(np.dot(w_pos, ints)) + 1e-12
        mix_w = w_pos / Z
        return mix_w, time_centers, time_scale
        
    except Exception as e:
        print(f"Warning: Error loading temporal weights: {e}")
        return None, None, None


def plot_temporal_kernel_v2(results):
    mix_w, time_centers, time_scale = _load_temporal_weights_v2(results)
    if mix_w is None: return
    T = float(results.get('T', float(time_centers.max())))
    W = results.get('window', T)
    tmax = max(min(T, W) if np.isfinite(W) else T, float(time_centers.max()))
    # Use number of grid points from results if available, otherwise default to 400
    num_grid_points = results.get('temporal_grid_points', 400)
    grid = np.linspace(0.0, tmax, num_grid_points)
    phi = np.stack([_gauss_bump(grid, c, time_scale) for c in time_centers], axis=-1)
    g_vals = phi @ mix_w
    plt.figure(figsize=(7,3))
    plt.plot(grid, g_vals, 'k-')
    plt.title('Temporal g̃(τ)')
    plt.xlabel('τ')
    plt.tight_layout()
    _savefig('temporal_kernel_g.png')
    plt.show()


def plot_spatial_kernel_v2(results):
    kappa = results.get('kappa_tilde_hat', None)
    if kappa is None: return
    plt.figure(figsize=(5,4))
    vmax = np.max(np.abs(kappa)) or 1.0
    plt.imshow(kappa, cmap='viridis', vmin=0, vmax=vmax)
    plt.title('Spatial κ̃ (normalized)'); plt.colorbar(); plt.tight_layout(); _savefig('spatial_kappa_v2.png'); plt.show()
    # radial profile over distances
    node_xy = np.asarray(results['node_locations'])
    D = np.linalg.norm(node_xy[:,None,:]-node_xy[None,:,:], axis=-1)
    reach = np.asarray(results['reach_mask'])
    mask = (np.arange(D.shape[0])[:,None] != np.arange(D.shape[0])[None,:]) & (reach>0)
    d_vals = D[mask].ravel(); k_vals = kappa[mask].ravel()
    if d_vals.size:
        # Use number of bins from results if available, otherwise default to 25
        num_bins = results.get('spatial_bins', 25)
        bins = np.linspace(d_vals.min(), d_vals.max(), num_bins)
        idx = np.digitize(d_vals, bins)
        med_x = []; med_y = []
        for b in range(1, len(bins)):
            sel = k_vals[idx==b]
            if sel.size:
                med_x.append((bins[b-1]+bins[b])/2.0)
                med_y.append(np.median(sel))
        if med_x:
            plt.figure(figsize=(6,3))
            plt.plot(med_x, med_y, 'b-')
            plt.xlabel('distance r')
            plt.ylabel('κ̃ median')
            plt.title('κ̃ vs distance (median)')
            plt.tight_layout()
            _savefig('kappa_distance_profile.png')
            plt.show()


# ---------------- v3 joint kernel ----------------

def _reconstruct_joint_params(results):
    W_uncon = results.get('kernel_param', None)  # (B_t,B_r)
    if W_uncon is None:
        return None
    try:
        W_uncon = np.asarray(W_uncon, dtype=float)
    except Exception:
        return None
    w_pos = _softplus(W_uncon) + 1e-8
    t_cent = np.asarray(results['time_centers']); t_scale = float(results['time_scale'])
    r_cent = np.asarray(results['dist_centers']); r_scale = float(results['dist_scale'])
    return w_pos, t_cent, t_scale, r_cent, r_scale


def plot_joint_kernel(results):
    params = _reconstruct_joint_params(results)
    if params is None: return
    w_pos, t_cent, t_scale, r_cent, r_scale = params
    # grids
    T = float(results['T']); W = results.get('window', T)
    tmax = min(T, W) if np.isfinite(W) else T
    # Use grid points from results if available, otherwise use defaults
    t_grid_points = results.get('joint_t_grid_points', 250)
    r_grid_points = results.get('joint_r_grid_points', 80)
    t_grid = np.linspace(0, max(tmax, float(t_cent.max())), t_grid_points)
    r_grid = np.linspace(0, float(r_cent.max()), r_grid_points)
    # time basis
    Phi_t = np.stack([_gauss_bump(t_grid, c, t_scale) for c in t_cent], axis=-1)  # (Nt,B_t)
    I_inf = np.array([_gauss_int_0_to_inf(c, t_scale) for c in t_cent])           # (B_t,)
    # build ψ(τ,r) on grid
    psi = np.zeros((len(r_grid), len(t_grid)))
    for i, r in enumerate(r_grid):
        psi_r = np.array([_gauss_bump(r, c, r_scale) for c in r_cent])            # (B_r,)
        S_t = w_pos @ psi_r                                                       # (B_t,)
        Z = float(np.dot(S_t, I_inf)) + 1e-12
        mix_t = S_t / Z
        psi[i] = Phi_t @ mix_t
    # heatmap
    plt.figure(figsize=(8,4));
    plt.imshow(psi, aspect='auto', origin='lower', extent=[t_grid[0], t_grid[-1], r_grid[0], r_grid[-1]], cmap='viridis')
    plt.colorbar(label='ψ̃(τ,r)'); plt.xlabel('τ'); plt.ylabel('distance r'); plt.title('Joint Kernel ψ̃(τ,r)');
    plt.tight_layout(); _savefig('joint_kernel_heatmap.png'); plt.show()
    # slices
    for frac in [0.25, 0.50, 0.75]:
        idx = int(frac * (len(r_grid)-1))
        plt.plot(t_grid, psi[idx], label=f"r≈{r_grid[idx]:.2f}")
    plt.title('Temporal slices of ψ̃(τ,r)'); plt.xlabel('τ'); plt.legend(); plt.tight_layout(); _savefig('joint_kernel_slices.png'); plt.show()
    return t_grid, r_grid, psi


def plot_psi_integral_vs_r(results, t_grid=None, r_grid=None, psi=None):
    if 'kernel_param' not in results: return
    if psi is None:
        triplet = plot_joint_kernel(results)
        if triplet is None: return
        t_grid, r_grid, psi = triplet
    W = results.get('window', np.inf)
    if np.isfinite(W):
        mask_t = t_grid <= float(W)
    else:
        mask_t = slice(None)
    area = np.trapz(psi[:, mask_t], t_grid[mask_t], axis=1)
    plt.figure(figsize=(7,3))
    plt.plot(r_grid, area, 'k-')
    plt.axhline(1.0, color='r', ls='--', lw=1)
    plt.xlabel('distance r'); plt.ylabel('∫₀^W ψ̃(τ,r) dτ'); plt.title('Mass vs distance (should ≈1 if W captures support)')
    plt.tight_layout(); _savefig('psi_integral_vs_r.png'); plt.show()


def plot_mean_tau_vs_r(results, t_grid=None, r_grid=None, psi=None):
    if 'kernel_param' not in results: return
    if psi is None:
        triplet = plot_joint_kernel(results)
        if triplet is None: return
        t_grid, r_grid, psi = triplet
    W = results.get('window', np.inf)
    if np.isfinite(W):
        mask_t = t_grid <= float(W)
    else:
        mask_t = slice(None)
    mass = np.trapz(psi[:, mask_t], t_grid[mask_t], axis=1) + 1e-12
    mean_tau = np.trapz((t_grid[mask_t] * psi[:, mask_t]), t_grid[mask_t], axis=1) / mass
    plt.figure(figsize=(7,3))
    plt.plot(r_grid, mean_tau, 'b-')
    plt.xlabel('distance r'); plt.ylabel('E[τ | r]'); plt.title('Mean lag given distance')
    plt.tight_layout(); _savefig('mean_tau_vs_r.png'); plt.show()


def plot_alphaK_vs_distance(results):
    K = np.asarray(results['K_hat'])
    alpha = float(results['alpha_hat'])
    node_xy = np.asarray(results['node_locations'])
    reach = np.asarray(results.get('reach_mask', np.ones_like(K)))
    N = K.shape[0]
    D = np.linalg.norm(node_xy[:,None,:]-node_xy[None,:,:], axis=-1)
    mask = (np.arange(N)[:,None] != np.arange(N)[None,:]) & (reach > 0)
    d_vals = D[mask].ravel(); g_vals = (alpha * K)[mask].ravel()
    # scatter + binned median
    plt.figure(figsize=(7,3))
    plt.scatter(d_vals, g_vals, s=6, alpha=0.2, color='gray')
    if d_vals.size > 0:
        # Use number of bins from results if available, otherwise default to 20
        num_bins = results.get('distance_bins', 20)
        bins = np.linspace(d_vals.min(), d_vals.max(), num_bins)
        idx = np.digitize(d_vals, bins)
        med_x = []; med_y = []
        for b in range(1, len(bins)):
            sel = g_vals[idx == b]
            if sel.size:
                med_x.append((bins[b-1]+bins[b])/2.0)
                med_y.append(np.median(sel))
        if med_x:
            plt.plot(med_x, med_y, 'r--', lw=2, label='binned median')
            plt.legend()
    plt.xlabel('distance r')
    plt.ylabel('α·K')
    plt.title('Effective pair scale vs distance')
    plt.tight_layout()
    _savefig('alphaK_vs_distance.png')
    plt.show()


def plot_time_rescaling_validation(results):
    # v3 joint kernel path, else v2 path
    params = _reconstruct_joint_params(results)
    data_path = results.get('data_pickle', None)
    if (params is None) or (data_path is None):
        # try v2
        return _validation_v2(results)
    try:
        w_pos, t_cent, t_scale, r_cent, r_scale = params
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        ev = data['events']
        t = np.asarray(ev['t']); u = np.asarray(ev['u']).astype(int); e = np.asarray(ev['e']).astype(int)
        order = np.argsort(t); t=t[order]; u=u[order]; e=e[order]
        N = int(results['N']); M = int(results['M'])
        mu = np.asarray(results['mu_hat']); K = np.asarray(results['K_hat']); MK = np.asarray(results['M_K_hat'])
        alpha = float(results['alpha_hat'])
        node_xy = np.asarray(results['node_locations'])
        # pairwise precompute S_t and denom
        D = np.linalg.norm(node_xy[:,None,:]-node_xy[None,:,:], axis=-1)
        Psi_r = np.stack([_gauss_bump(D, c, r_scale) for c in r_cent], axis=-1)      # (N,N,B_r)
        S_t = Psi_r @ w_pos.T                                                        # (N,N,B_t)
        I_inf = np.array([_gauss_int_0_to_inf(c, t_scale) for c in t_cent])
        denom = np.maximum(np.tensordot(S_t, I_inf, axes=([2],[0])), 1e-12)          # (N,N)
        def psi_int_cap(dt, ii, jj):
            dt = max(dt, 0.0)
            I_cap = np.array([_gauss_int_0_to(dt, c, t_scale) for c in t_cent])
            num = float(S_t[ii,jj] @ I_cap)
            return num / float(denom[ii,jj])
        W = results.get('window', np.inf); useW = np.isfinite(W); W = float(W) if useW else np.inf
        last_time = {(i,j):0.0 for i in range(N) for j in range(M)}
        s_vals = []
        for i_ev in range(len(t)):
            ui, ei, ti = int(u[i_ev]), int(e[i_ev]), float(t[i_ev])
            a = float(last_time[(ui, ei)]); b = ti
            base = mu[ui, ei]*(b-a)
            contrib = 0.0
            for j_ev in range(i_ev):
                tj = float(t[j_ev]); uj = int(u[j_ev]); ej = int(e[j_ev])
                up = b - tj; lo = a - tj
                if useW:
                    up = min(max(up,0.0),W); lo = min(max(lo,0.0),W)
                else:
                    up = max(up,0.0); lo = max(lo,0.0)
                if up > lo:
                    contrib += K[ui, uj]*MK[ej, ei]*(psi_int_cap(up, ui, uj) - psi_int_cap(lo, ui, uj))
            Delta = alpha*contrib + base
            s = 1.0 - np.exp(-Delta)
            s = min(max(s, 1e-9), 1-1e-9)
            s_vals.append(s)
            last_time[(ui, ei)] = b
        s_vals = np.asarray(s_vals)
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        u_theory = np.linspace(0,1,len(s_vals), endpoint=False)+0.5/len(s_vals)
        axes[0].plot(np.sort(u_theory), np.sort(s_vals), 'k.', ms=3)
        axes[0].plot([0,1],[0,1],'r--')
        axes[0].set_title('PIT QQ vs Uniform')
        axes[0].set_xlabel('Theoretical Quantiles')
        axes[0].set_ylabel('Empirical Quantiles')
        
        # Use number of bins from results if available, otherwise default to 20
        num_bins = results.get('num_bins', 20)
        axes[1].hist(s_vals, bins=num_bins, density=True, alpha=0.7)
        axes[1].axhline(1.0,color='r',ls='--')
        axes[1].set_title('PIT Histogram')
        axes[1].set_xlabel('PIT Values')
        axes[1].set_ylabel('Density')
        plt.tight_layout()
        _savefig('pit_validation_v3.png')
        plt.show()
    except Exception as ex:
        print('[Validation v3] skipped:', ex)


def _validation_v2(results):
    mix_w, t_cent, t_scale = _load_temporal_weights_v2(results)
    if mix_w is None: return
    try:
        with open(results['data_pickle'],'rb') as f: data = pickle.load(f)
        ev = data['events']
        t = np.asarray(ev['t']); u = np.asarray(ev['u']).astype(int); e = np.asarray(ev['e']).astype(int)
        order=np.argsort(t); t=t[order]; u=u[order]; e=e[order]
        N=int(results['N']); M=int(results['M'])
        mu=np.asarray(results['mu_hat']); K=np.asarray(results['K_hat']); MK=np.asarray(results['M_K_hat']); kappa=np.asarray(results['kappa_tilde_hat']); alpha=float(results['alpha_hat'])
        G_node = K*kappa
        def Gint(x):
            x=max(x,0.0); Phi=np.array([_gauss_int_0_to(x,c,t_scale) for c in t_cent]); return float(Phi@mix_w)
        W=results.get('window',np.inf); useW=np.isfinite(W); W=float(W) if useW else np.inf
        # One-step calibration: rescale alpha by average kernel mass captured by window
        if useW:
            T_val = float(results.get('T', float(t.max()) if t.size else 0.0))
            caps = np.maximum(np.minimum(W, T_val - t), 0.0)
            print(f"Caps: {caps}")
            if caps.size > 0:
                mW = float(np.mean([Gint(float(c)) for c in caps]))
            else:
                mW = 1.0
        else:
            mW = 1.0
        alpha_eval = alpha / mW
        print(f"Calibration: mW={mW:.3f} -> alpha_eval={alpha_eval:.6f} (was {alpha:.6f})")
        last={(i,j):0.0 for i in range(N) for j in range(M)}; s_vals=[]
        for i_ev in range(len(t)):
            ui,ei,ti=int(u[i_ev]),int(e[i_ev]),float(t[i_ev]); a=float(last[(ui,ei)]); b=ti
            base=mu[ui,ei]*(b-a); contrib=0.0
            for j_ev in range(i_ev):
                tj=float(t[j_ev]); uj=int(u[j_ev]); ej=int(e[j_ev])
                up=b-tj; lo=a-tj
                if useW: up=min(max(up,0.0),W); lo=min(max(lo,0.0),W)
                else: up=max(up,0.0); lo=max(lo,0.0)
                if up>lo: contrib += G_node[ui,uj]*MK[ej,ei]*(Gint(up)-Gint(lo))
            s = 1.0 - np.exp(- (alpha_eval*contrib + base))
            s=min(max(s,1e-9),1-1e-9); s_vals.append(s); last[(ui,ei)]=b
        s_vals=np.asarray(s_vals)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        u_th=np.linspace(0,1,len(s_vals),endpoint=False)+0.5/len(s_vals)
        axes[0].plot(np.sort(u_th),np.sort(s_vals),'k.',ms=3)
        axes[0].plot([0,1],[0,1],'r--')
        axes[0].set_title('PIT QQ vs Uniform')
        axes[0].set_xlabel('Theoretical Quantiles')
        axes[0].set_ylabel('Empirical Quantiles')
        
        # Use number of bins from results if available, otherwise default to 20
        num_bins = results.get('num_bins', 20)
        axes[1].hist(s_vals,bins=num_bins,density=True,alpha=0.7)
        axes[1].axhline(1.0,color='r',ls='--')
        axes[1].set_title('PIT Histogram')
        axes[1].set_xlabel('PIT Values')
        axes[1].set_ylabel('Density')
        plt.tight_layout()
        _savefig('pit_validation_v2.png')
        plt.show()
    except Exception as ex:
        print('[Validation v2] skipped:', ex)


def print_stability_metrics(results):
    alpha = float(results.get('alpha_hat', np.nan))
    K = results.get('K_hat', None)
    MK = results.get('M_K_hat', None)
    if K is None or MK is None or not np.isfinite(alpha):
        print('Stability: n/a')
        return
    rho_K = _spec_radius(K)
    rho_M = _spec_radius(MK)
    rho_G = alpha * rho_K * rho_M
    print(f"Spectral radii: ρ(K)={rho_K:.4f}, ρ(M_K)={rho_M:.4f}, ρ(G)=α·ρ(K)·ρ(M_K)={rho_G:.4f}")
    print(f"Stability margin 1-ρ(G): {1.0 - rho_G:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', type=str, default=None, help='Path to inference_result_np*.pickle')
    ap.add_argument('--holdout-frac', type=float, default=None, help='Fraction (0,1) for hold-out PIT')
    args = ap.parse_args()
    r = load_results(args.result)
    print('Keys:', list(r.keys()))

    # v3 joint kernel if present
    is_v3 = 'kernel_param' in r

    # Added: network + stability overview
    plot_network(r)
    plot_adjacency_heatmap(r)
    print_stability_metrics(r)

    if is_v3:
        plot_base_rates(r)
        plot_spatial_coupling(r)
        plot_mark_kernel(r)
        tg_rg_psi = plot_joint_kernel(r)
        plot_psi_integral_vs_r(r, *(tg_rg_psi or (None, None, None)))
        plot_mean_tau_vs_r(r, *(tg_rg_psi or (None, None, None)))
        plot_alphaK_vs_distance(r)
        plot_time_rescaling_validation(r)
        # Optional hold-out PIT (v3)
        if args.holdout_frac is not None and 0.0 < args.holdout_frac < 1.0:
            try:
                plot_time_rescaling_validation_holdout(r, args.holdout_frac)
            except NameError:
                pass
        # Optional speed report
        try:
            report_propagation_speed_v3(r)
        except NameError:
            pass
    else:
        plot_base_rates(r)
        plot_spatial_coupling(r)
        plot_mark_kernel(r)
        plot_temporal_kernel_v2(r)
        plot_spatial_kernel_v2(r)
        plot_time_rescaling_validation(r)
        if args.holdout_frac is not None and 0.0 < args.holdout_frac < 1.0:
            try:
                _validation_v2(r, holdout_frac=args.holdout_frac)
            except TypeError:
                # fallback to full if old signature
                _validation_v2(r)

    print('Done.')


if __name__ == '__main__':
    main() 