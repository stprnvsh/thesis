#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMO → Hawkes events pipeline with node reduction (K-Means clustering)

- Runs a SUMO simulation (TraCI) for a given .sumocfg
- Optionally clusters SUMO junctions into K super-nodes (--num-nodes)
- Aggregates per-(clustered)-node flow & speed in fixed time bins
- Detects Low-Flow and Congestion ONSET events per node
- Saves a pickle compatible with your Hawkes inference scripts:
    events: [('t', float), ('u', int), ('e', int), ('x', float), ('y', float)]
    num_nodes, num_event_types, node_locations (N,2), adjacency_matrix (N,N)
    neighbors_list, num_hops, mark_kernel_matrix, plus metadata

Event types:
    0 = LowFlow onset
    1 = Congestion onset
"""

import os
import sys
import argparse
import pickle
from collections import defaultdict
import xml.etree.ElementTree as ET

import numpy as np
import networkx as nx

# ---- SUMO imports (works if SUMO_HOME is set or sumo is on PATH) ----
def _ensure_sumo():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    try:
        import traci  # noqa: F401
        import sumolib  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Could not import traci/sumolib. Install SUMO and set SUMO_HOME or add SUMO tools to PYTHONPATH."
        ) from e

_ensure_sumo()
import traci
import sumolib

# Optional clustering
try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# ---------------- Utilities ----------------
def normalize_xy(xy):
    """Normalize 2D coords to roughly [0, 10] per axis (like your generator)."""
    xy = np.asarray(xy, dtype=float)
    if xy.size == 0:
        return xy
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    rng = np.where(maxs > mins, maxs - mins, 1.0)
    return 10.0 * (xy - mins) / rng


def compute_reachability(adjacency, num_hops=1):
    """Binary reachability within <= num_hops steps (includes 1-hop adjacency)."""
    A = (adjacency > 0).astype(np.int32)
    if num_hops <= 1:
        R = A.copy()
    else:
        R = A.copy()
        cur = A.copy()
        for _ in range(1, num_hops):
            cur = (cur @ A > 0).astype(np.int32)
            R = (R | cur).astype(np.int32)
    np.fill_diagonal(R, 0)
    return R.astype(np.float32)


def collapse_onsets(series_bool):
    """Return indices where series turns True from False (onset)."""
    s = np.asarray(series_bool, dtype=bool)
    if s.size == 0:
        return np.array([], dtype=int)
    prev = np.concatenate([[False], s[:-1]])
    return np.where(s & (~prev))[0]


# --------- Helpers to resolve net.xml from .sumocfg (FIX) ----------
def _resolve_netfile_from_sumocfg(sumocfg_path: str) -> str:
    """
    Parse the .sumocfg to find the <input><net-file value="..."/> entry
    and return an absolute path to the net.xml.
    """
    if not os.path.exists(sumocfg_path):
        raise FileNotFoundError(sumocfg_path)

    tree = ET.parse(sumocfg_path)
    root = tree.getroot()

    net_file = None
    # Typical structure: <configuration><input><net-file value="net.net.xml"/>
    for inp in root.findall(".//input"):
        elem = inp.find("net-file")
        if elem is not None:
            # SUMO allows either <net-file value="..."/> or text content
            net_file = elem.get("value") or elem.text
            if net_file:
                break

    if not net_file:
        raise RuntimeError(
            f"Could not find <net-file> in {sumocfg_path}. "
            "Please ensure your .sumocfg has <input><net-file value='...'/></input>."
        )

    # Resolve path relative to the .sumocfg directory
    cfg_dir = os.path.dirname(os.path.abspath(sumocfg_path))
    net_path = net_file if os.path.isabs(net_file) else os.path.join(cfg_dir, net_file)
    if not os.path.exists(net_path):
        raise FileNotFoundError(f"Resolved net-file not found: {net_path}")
    return net_path


# ------------- Parse net and build base topology -------------
def build_base_topology_from_cfg(sumocfg):
    """
    Returns (original graph level):
        node_ids: list of junction IDs
        node_xy:  (N,2) in SUMO XY
        adjacency: (N,N) binary undirected connectivity
        edge_to_nodes: dict edgeID -> (from_idx, to_idx) (original node indices)
        edge_speed: dict edgeID -> speedLimit (m/s)
        node_freeflow_speed: (N,) avg edge speed limit around node
        net: sumolib.net.Net
    """
    # --- FIX: older SUMO lacks sumolib.net.readNetFromConfig -> resolve and read net.xml manually
    net_xml_path = _resolve_netfile_from_sumocfg(sumocfg)
    net = sumolib.net.readNet(net_xml_path)

    nodes = list(net.getNodes())
    node_id_to_idx = {nd.getID(): i for i, nd in enumerate(nodes)}
    node_ids = [nd.getID() for nd in nodes]

    xy = np.zeros((len(nodes), 2), dtype=float)
    for i, nd in enumerate(nodes):
        # getCoord is present for projected nets; fall back to lon/lat if needed
        try:
            xy[i, 0], xy[i, 1] = nd.getCoord()
        except Exception:
            lon, lat = nd.getLonLat()
            xy[i, 0], xy[i, 1] = net.convertLonLat2XY(lon, lat)

    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    edge_to_nodes = {}
    edge_speed = {}
    node_speed_accum = defaultdict(list)

    for edge in net.getEdges():
        # Skip internal/special connectors if desired
        try:
            is_special = edge.isSpecial()
        except Exception:
            # some older APIs may not have isSpecial; treat connectors by ID pattern
            is_special = edge.getID().startswith(":")
        if is_special:
            continue

        fr = edge.getFromNode().getID()
        to = edge.getToNode().getID()
        if fr not in node_id_to_idx or to not in node_id_to_idx:
            continue
        i = node_id_to_idx[fr]
        j = node_id_to_idx[to]
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0  # treat as undirected connectivity for Hawkes reachability

        eid = edge.getID()
        edge_to_nodes[eid] = (i, j)
        try:
            sp = float(edge.getSpeed())  # m/s (speed limit)
        except Exception:
            sp = 0.0
        edge_speed[eid] = sp
        node_speed_accum[i].append(sp)
        node_speed_accum[j].append(sp)

    node_freeflow_speed = np.zeros(len(nodes), dtype=float)
    for idx in range(len(nodes)):
        vals = node_speed_accum.get(idx, [])
        node_freeflow_speed[idx] = float(np.mean(vals)) if vals else 0.0

    return (
        node_ids,
        xy,  # not normalized yet
        adjacency,
        edge_to_nodes,
        edge_speed,
        node_freeflow_speed,
        net,
    )


# ------------- Cluster junctions to K super-nodes -------------
def cluster_topology(node_xy, adjacency, edge_to_nodes, edge_speed, node_freeflow_speed,
                     num_nodes=None, random_state=42, n_init=10):
    """
    If num_nodes is None or >= original N, returns original topology.
    Else returns clustered topology and mapping.

    Returns:
        N_eff, labels (orig_node_idx -> cluster_id),
        centroids_xy (N_eff,2), adjacency_eff (N_eff,N_eff),
        incident_edges_by_cluster: list of lists of edgeIDs,
        freeflow_speed_by_cluster: (N_eff,)
    """
    N = node_xy.shape[0]
    if (num_nodes is None) or (num_nodes >= N):
        # Build per-node incident edge lists and per-node freeflow (already given)
        incident_edges_by_node = [[] for _ in range(N)]
        for eid, (i, j) in edge_to_nodes.items():
            incident_edges_by_node[i].append(eid)
            incident_edges_by_node[j].append(eid)
        return (
            N,
            np.arange(N, dtype=int),  # labels = identity
            node_xy.copy(),
            adjacency.copy(),
            incident_edges_by_node,
            node_freeflow_speed.copy(),
        )

    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for clustering. Install it or omit --num-nodes.")

    kmeans = KMeans(n_clusters=num_nodes, random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(node_xy)  # shape (N,)
    centroids = kmeans.cluster_centers_  # (K,2)
    K = centroids.shape[0]

    # Build cluster adjacency: any original edge between clusters adds a link
    adjacency_eff = np.zeros((K, K), dtype=np.float32)
    for eid, (i, j) in edge_to_nodes.items():
        ci, cj = labels[i], labels[j]
        if ci == cj:
            continue
        adjacency_eff[ci, cj] = 1.0
        adjacency_eff[cj, ci] = 1.0

    # Build incident edge lists for each cluster (edges touching any member node)
    incident_edges_by_cluster = [[] for _ in range(K)]
    for eid, (i, j) in edge_to_nodes.items():
        ci, cj = labels[i], labels[j]
        incident_edges_by_cluster[ci].append(eid)
        incident_edges_by_cluster[cj].append(eid)

    # Cluster free-flow speed = average of speed limits of its incident edges
    freeflow_speed_by_cluster = np.zeros(K, dtype=float)
    for c in range(K):
        eids = incident_edges_by_cluster[c]
        if eids:
            freeflow_speed_by_cluster[c] = float(np.mean([edge_speed[e] for e in eids]))
        else:
            freeflow_speed_by_cluster[c] = 0.0

    return (
        K,
        labels,
        centroids,
        adjacency_eff,
        incident_edges_by_cluster,
        freeflow_speed_by_cluster,
    )


# ---------------- Main pipeline ----------------
def run_pipeline(
    sumocfg: str,
    num_nodes: int = None,              # <= NEW: target K (like your generator)
    bin_secs: int = 60,
    low_flow_quantile: float = 0.2,
    congestion_frac: float = 0.4,
    max_steps: int = None,
    output_pickle: str = "traffic_hawkes_from_sumo.pickle",
    num_hops: int = 1,
    event_type_names=("LowFlow", "Congestion"),
    mark_kernel_matrix=((0.8, 0.9), (0.9, 0.9)),
    sumo_binary: str = None,
    cluster_random_state: int = 42,
    kmeans_n_init: int = 10,
):
    """
    Run SUMO, (optionally) cluster nodes, bin metrics, detect event onsets, save Hawkes dataset.
    """
    if sumo_binary is None:
        sumo_binary = os.environ.get("SUMO_BINARY", "sumo")
    if not os.path.exists(sumocfg):
        raise FileNotFoundError(sumocfg)

    # --- Build base topology from SUMO net ---
    (
        node_ids,
        node_xy_orig,             # SUMO XY (not normalized)
        adjacency_orig,
        edge_to_nodes,
        edge_speed,
        node_freeflow_speed_orig,
        _net,
    ) = build_base_topology_from_cfg(sumocfg)

    N0 = len(node_ids)
    if N0 == 0:
        raise RuntimeError("No nodes parsed from the network.")

    # --- Cluster to K super-nodes (or keep original) ---
    (
        N_eff,
        labels,                    # orig node idx -> cluster id
        centroids_xy,              # (N_eff,2) in SUMO XY
        adjacency_eff,             # (N_eff,N_eff)
        incident_edges_lists,      # list per cluster of edgeIDs
        freeflow_speed_eff,        # (N_eff,)
    ) = cluster_topology(
        node_xy=node_xy_orig,
        adjacency=adjacency_orig,
        edge_to_nodes=edge_to_nodes,
        edge_speed=edge_speed,
        node_freeflow_speed=node_freeflow_speed_orig,
        num_nodes=num_nodes,
        random_state=cluster_random_state,
        n_init=kmeans_n_init,
    )

    # Normalize XY for Hawkes
    node_xy_norm = normalize_xy(centroids_xy)

    # --- Launch SUMO (headless) and aggregate per cluster ---
    traci.start([sumo_binary, "-c", sumocfg, "--no-step-log", "true", "--time-to-teleport", "-1"])
    try:
        sim_time = 0.0
        next_bin_end = float(bin_secs)

        # accumulators within current bin
        flow_acc = np.zeros(N_eff, dtype=float)
        speed_sum = np.zeros(N_eff, dtype=float)
        speed_wgt = np.zeros(N_eff, dtype=float)

        flows_per_bin = []   # (T,N_eff)
        speeds_per_bin = []  # (T,N_eff)
        bin_times = []       # (T,)

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            # Aggregate per cluster using current step edge stats
            for c in range(N_eff):
                eids = incident_edges_lists[c]
                if not eids:
                    continue
                step_flow = 0.0
                edge_speeds = []
                edge_weights = []
                for eid in eids:
                    # vehicles and mean speed on the edge this step
                    vnum = traci.edge.getLastStepVehicleNumber(eid)
                    vspd = traci.edge.getLastStepMeanSpeed(eid)  # 0 if no vehicles
                    step_flow += float(vnum)
                    edge_speeds.append(float(vspd))
                    edge_weights.append(float(vnum))
                flow_acc[c] += step_flow
                if sum(edge_weights) > 0:
                    speed_sum[c] += np.dot(edge_speeds, edge_weights)
                    speed_wgt[c] += sum(edge_weights)

            # End of bin?
            if sim_time >= next_bin_end:
                flows_per_bin.append(flow_acc.copy())
                with np.errstate(divide="ignore", invalid="ignore"):
                    mean_speed = np.where(speed_wgt > 0, speed_sum / np.maximum(speed_wgt, 1e-12), 0.0)
                speeds_per_bin.append(mean_speed)
                bin_times.append(next_bin_end)

                # reset for next bin
                flow_acc.fill(0.0)
                speed_sum.fill(0.0)
                speed_wgt.fill(0.0)
                next_bin_end += float(bin_secs)

            if max_steps is not None and sim_time >= max_steps:
                break

        # flush partial bin if the last step crossed a boundary
        if sim_time > (next_bin_end - bin_secs):
            flows_per_bin.append(flow_acc.copy())
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_speed = np.where(speed_wgt > 0, speed_sum / np.maximum(speed_wgt, 1e-12), 0.0)
            speeds_per_bin.append(mean_speed)
            bin_times.append(sim_time)

    finally:
        traci.close(False)

    if len(flows_per_bin) == 0:
        raise RuntimeError("No bins recorded. Did the simulation produce any steps/vehicles?")

    flows = np.stack(flows_per_bin, axis=0)   # (Tbins, N_eff)
    speeds = np.stack(speeds_per_bin, axis=0) # (Tbins, N_eff)
    bin_times = np.asarray(bin_times, dtype=float)

    # ---- Thresholds (per cluster) ----
    # Low-flow: per-cluster quantile
    lf_thr = np.quantile(flows, low_flow_quantile, axis=0)  # (N_eff,)

    # Congestion: fraction of cluster free-flow speed (speed limits avg)
    ff_fallback = np.quantile(speeds, 0.8, axis=0)
    ff = np.where(freeflow_speed_eff > 0, freeflow_speed_eff, ff_fallback)
    cg_thr = congestion_frac * ff  # (N_eff,)

    # ---- Boolean series per cluster ----
    is_lowflow = flows <= lf_thr[None, :]
    is_congest = speeds <= cg_thr[None, :]

    # ---- ONSET events ----
    events = []
    for c in range(N_eff):
        # Low-flow onsets
        for idx in collapse_onsets(is_lowflow[:, c]):
            t = float(bin_times[idx])
            x, y = node_xy_norm[c]
            events.append((t, int(c), 0, float(x), float(y)))
        # Congestion onsets
        for idx in collapse_onsets(is_congest[:, c]):
            t = float(bin_times[idx])
            x, y = node_xy_norm[c]
            events.append((t, int(c), 1, float(x), float(y)))

    events.sort(key=lambda r: r[0])
    events_np = np.array(
        events,
        dtype=[('t', float), ('u', int), ('e', int), ('x', float), ('y', float)]
    )

    # Neighbors (1-hop) and optional reachability
    neighbors_list = [np.where(adjacency_eff[i] > 0)[0].astype(int) for i in range(N_eff)]

    out = {
        "events": events_np,
        "num_nodes": int(N_eff),
        "num_event_types": 2,
        "node_locations": node_xy_norm.astype(float),      # (N_eff,2)
        "adjacency_matrix": adjacency_eff.astype(float),   # (N_eff,N_eff)
        "neighbors_list": neighbors_list,
        "num_hops": int(num_hops),
        "mark_kernel_matrix": np.asarray(mark_kernel_matrix, dtype=float),
        "event_type_names": tuple(event_type_names),
        "params": None,
        # Metadata for reproducibility:
        "source": "SUMO",
        "bin_secs": int(bin_secs),
        "low_flow_quantile": float(low_flow_quantile),
        "congestion_frac": float(congestion_frac),
        "clustered_from_nodes": int(N0),
        "effective_nodes": int(N_eff),
        "cluster_labels": labels.astype(int),  # mapping (orig node idx -> cluster id)
        "freeflow_speed_mps": freeflow_speed_eff.astype(float),
        "bin_times": bin_times.astype(float),
        "flows_per_bin": flows.astype(float),
        "speeds_per_bin": speeds.astype(float),
    }

    with open(output_pickle, "wb") as f:
        pickle.dump(out, f)

    print(f"\nSaved dataset to: {output_pickle}")
    print(f"  events: {events_np.shape[0]}")
    print(f"  effective nodes (clusters): {N_eff} (from {N0})")
    print(f"  time bins: {flows.shape[0]}")
    print(f"Event types: 0={event_type_names[0]}, 1={event_type_names[1]}")
    return output_pickle


def main():
    ap = argparse.ArgumentParser(description="Run SUMO, detect flow/speed events, and export Hawkes-ready dataset (with optional node clustering).")
    ap.add_argument("--sumocfg", required=True, help="Path to .sumocfg")
    ap.add_argument("--num-nodes", type=int, default=None, help="Target number of super-nodes (K). If omitted, use all junctions.")
    ap.add_argument("--bin-secs", type=int, default=60, help="Time bin (seconds)")
    ap.add_argument("--low-flow-quantile", type=float, default=0.2, help="Quantile for low-flow threshold (per node)")
    ap.add_argument("--congestion-frac", type=float, default=0.4, help="Speed fraction of free-flow for congestion")
    ap.add_argument("--max-steps", type=int, default=None, help="Optional hard stop (seconds)")
    ap.add_argument("--num-hops", type=int, default=1, help="Reachability hops metadata (not used for detection)")
    ap.add_argument("--output", default="traffic_hawkes_from_sumo.pickle", help="Output pickle filename")
    ap.add_argument("--sumo-binary", default=None, help="Path to sumo or sumo-gui (default: env SUMO_BINARY or 'sumo')")
    ap.add_argument("--cluster-random-state", type=int, default=42, help="KMeans random_state")
    ap.add_argument("--kmeans-n-init", type=int, default=10, help="KMeans n_init")
    args = ap.parse_args()

    run_pipeline(
        sumocfg=args.sumocfg,
        num_nodes=args.num_nodes,
        bin_secs=args.bin_secs,
        low_flow_quantile=args.low_flow_quantile,
        congestion_frac=args.congestion_frac,
        max_steps=args.max_steps,
        output_pickle=args.output,
        num_hops=args.num_hops,
        sumo_binary=args.sumo_binary,
        cluster_random_state=args.cluster_random_state,
        kmeans_n_init=args.kmeans_n_init,
    )


if __name__ == "__main__":
    main()
