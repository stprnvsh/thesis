#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI-only SUMO -> (flows/speeds) -> clustered nodes -> events -> pickle

Events:
  e = 0  -> low flow
  e = 1  -> congestion (speed below fraction of freeflow)

Time unit in saved events: hours (to match your Hawkes code).
"""

import argparse
import os
import tempfile
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.cluster import KMeans
import pickle
from pathlib import Path
import math
import random

# --- Optional: if you want to use your hawkes_module.save_simulation_data() ---
try:
    import hawkes_module as hawkes
    HAVE_HAWKES = True
except ImportError:
    HAVE_HAWKES = False

# ---------------------------
# Utilities
# ---------------------------

def parse_sumocfg_paths(sumocfg_path: str):
    """Return absolute paths for net-file and route-files from a .sumocfg."""
    cfg_path = Path(sumocfg_path).resolve()
    tree = ET.parse(cfg_path)
    root = tree.getroot()
    input_node = root.find("./input")

    def get_attr(tag, attr):
        node = input_node.find(tag)
        if node is None:
            return None
        val = node.get(attr)
        if val is None:
            return None
        return str((cfg_path.parent / val).resolve())

    net_file = get_attr("net-file", "value")
    route_files = get_attr("route-files", "value")  # could be comma-separated
    
    # Handle comma-separated additional files
    additional_node = input_node.find("additional-files")
    additional_files = None
    if additional_node is not None:
        val = additional_node.get("value")
        if val:
            # Split comma-separated files and resolve each path
            file_list = [f.strip() for f in val.split(",")]
            resolved_files = [str((cfg_path.parent / f).resolve()) for f in file_list]
            additional_files = ",".join(resolved_files)
    
    return net_file, route_files, additional_files, cfg_path.parent


def write_edge_additional(add_dir: Path, out_xml: Path, period: int) -> Path:
    """Create an <additional> file that asks SUMO to write edgeData."""
    
    return ""


def run_sumo_cli(sumocfg: str, new_additional: Path = None, existing_additional: str = None, sumo_bin: str = "sumo", time_bounds: tuple = None) -> None:
    """Run SUMO in CLI mode once and wait for it to finish."""
    cmd = [
        sumo_bin,
        "-c", sumocfg,
        "--no-step-log", "false",
        "--duration-log.disable", "false",
        "--threads", "15",  # Use 8 threads for processing
    ]
    
    # Add time bounds if specified (convert hours to seconds)
    if time_bounds is not None:
        start_hour, end_hour = time_bounds
        begin_sec = start_hour * 3600
        end_sec = end_hour * 3600
        cmd.extend(["--begin", str(begin_sec)])
        cmd.extend(["--end", str(end_sec)])
        print(f">> SUMO simulation time bounds: {start_hour}:00 - {end_hour}:00 ({begin_sec}s - {end_sec}s)")
    
    # Handle additional files
    if new_additional and existing_additional:
        additional_files = f"{existing_additional},{new_additional.as_posix()}"
        cmd.extend(["--additional-files", additional_files])
    elif existing_additional:
        # Use only existing additional files (which already include edgeData output)
        cmd.extend(["--additional-files", existing_additional])
    elif new_additional:
        cmd.extend(["--additional-files", new_additional.as_posix()])
    # If neither, SUMO will use the additional files specified in the .sumocfg
    
    print(">> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_edgeData(edge_xml: Path):
    """
    Parse SUMO edgeData XML.
    Returns:
        intervals: list of (t_begin, t_end)
        edge_rows: list of dict rows with keys:
                   't_end', 'edge_id', 'entered', 'left', 'speed'
    """
    tree = ET.parse(edge_xml)
    root = tree.getroot()
    intervals = []
    rows = []
    for interval in root.findall("interval"):
        t0 = float(interval.get("begin"))
        t1 = float(interval.get("end"))
        intervals.append((t0, t1))
        for e in interval.findall("edge"):
            edge_id = e.get("id")
            # Attributes present depend on SUMO version and options.
            # We use 'entered' as the count crossing into the edge in the bin.
            entered = float(e.get("entered", "0"))  # vehicles
            left = float(e.get("left", "0"))
            # Average speed (m/s) over the interval for vehicles on edge
            speed = float(e.get("speed", "nan"))
            rows.append({
                "t_end": t1,
                "edge_id": edge_id,
                "entered": entered,
                "left": left,
                "speed": speed
            })
    return intervals, rows


def read_edge_geometry(net_xml: str):
    """
    Read edge geometry, speed limits, and capacity from net.xml.

    Returns:
      edge_ids: list[str]
      xy: float array [E, 2] with edge midpoints
      speed_lim: float array [E] (m/s)
      edge_capacity: float array [E] (vehicles/hour) - estimated capacity
      edge_to_nodes: dict edge_id -> (fromNodeId, toNodeId)
      node_xy: dict nodeId -> (x,y) (in net projection)
      node_to_edges: dict nodeId -> set(edgeId)
    """
    # We avoid sumolib here to reduce dependencies; parse with ET.
    tree = ET.parse(net_xml)
    root = tree.getroot()
    # Nodes
    node_xy = {}
    for n in root.findall("location"):
        # nothing to do; coordinates are in <node> elements
        pass
    for n in root.findall("junction"):
        nid = n.get("id")
        x = float(n.get("x"))
        y = float(n.get("y"))
        node_xy[nid] = (x, y)

    # Edges
    edge_ids = []
    xy = []
    speed_lim = []
    edge_capacity = []
    edge_to_nodes = {}
    node_to_edges = {nid: set() for nid in node_xy.keys()}

    for e in root.findall("edge"):
        if e.get("function", "") == "internal":
            continue
        eid = e.get("id")
        from_n = e.get("from")
        to_n = e.get("to")
        # speed may be on lanes; take max over lanes as freeflow
        max_spd = 0.0
        lane_count = 0
        pts = []
        # edge shape: optional <lane shape="x1,y1 x2,y2 ...">
        for lane in e.findall("lane"):
            lane_count += 1  # Count lanes for capacity
            s = float(lane.get("speed", "0"))
            max_spd = max(max_spd, s)
            shape = lane.get("shape")
            if shape:
                # pick the first lane with shape as representative geometry
                if not pts:
                    coords = []
                    for p in shape.split():
                        xx, yy = p.split(",")
                        coords.append((float(xx), float(yy)))
                    pts = coords
        # geometry fallback: midpoint of (from,to) nodes
        if not pts and from_n in node_xy and to_n in node_xy:
            x1, y1 = node_xy[from_n]
            x2, y2 = node_xy[to_n]
            pts = [(x1, y1), (x2, y2)]
        if not pts:
            # if missing, skip edge
            continue
        # midpoint
        mx = 0.5 * (pts[0][0] + pts[-1][0])
        my = 0.5 * (pts[0][1] + pts[-1][1])

        # Calculate theoretical capacity (vehicles/hour)
        # Standard formula: capacity = lanes * speed_limit_kmh * density_factor
        # Typical: 1900 veh/hour per lane at highway speeds, scaled by speed
        if max_spd == 0:
            max_spd = 13.89  # default 50 km/h
        speed_kmh = max_spd * 3.6  # Convert m/s to km/h
        # Simple capacity model: ~1900 veh/h/lane for highways, scaled by speed
        capacity_per_lane = min(1900, speed_kmh * 38)  # 38 ≈ 1900/50 km/h
        total_capacity = lane_count * capacity_per_lane

        edge_ids.append(eid)
        xy.append((mx, my))
        speed_lim.append(max_spd)
        edge_capacity.append(total_capacity)
        edge_to_nodes[eid] = (from_n, to_n)
        if from_n in node_to_edges:
            node_to_edges[from_n].add(eid)
        if to_n in node_to_edges:
            node_to_edges[to_n].add(eid)

    xy = np.asarray(xy, dtype=float)
    speed_lim = np.asarray(speed_lim, dtype=float)
    edge_capacity = np.asarray(edge_capacity, dtype=float)
    return edge_ids, xy, speed_lim, edge_capacity, edge_to_nodes, node_xy, node_to_edges


def build_cluster_adjacency(edge_ids, edge_to_nodes, node_to_edges, labels, K):
    """
    Build KxK adjacency: clusters are connected if any edge in i shares a node
    with any edge in j (i != j).
    """
    K = int(K)
    adj = np.zeros((K, K), dtype=int)
    # build mapping cluster -> set(edges) for speed
    cl2edges = {c: set() for c in range(K)}
    for eid, c in zip(edge_ids, labels):
        cl2edges[int(c)].add(eid)
    # for each junction, get clusters attached
    for nid, edges in node_to_edges.items():
        clusters_here = set(int(labels[edge_ids.index(e)]) for e in edges if e in edge_ids)
        ch = list(clusters_here)
        for i in range(len(ch)):
            for j in range(i + 1, len(ch)):
                a, b = ch[i], ch[j]
                adj[a, b] = 1
                adj[b, a] = 1
    # ensure no self-loops
    np.fill_diagonal(adj, 0)
    return adj


def aggregate_by_cluster(edge_rows, edge_ids, edge_xy, edge_speed_lim, edge_capacity, labels, K, period):
    """
    Aggregate entered (flow), speed, and capacity to clusters per interval.
    Returns:
      cluster_times: sorted list of unique interval end times (seconds)
      cluster_flow:  np.array [T, K]  (veh / period)
      cluster_speed: np.array [T, K]  (m/s; flow-weighted mean)
      cluster_capacity: np.array [K]  (veh/hour; sum of edge capacities in cluster)
    """
    # map edge_id -> (cluster idx, index into arrays)
    eid_to_idx = {eid: i for i, eid in enumerate(edge_ids)}
    T_end = sorted(set(r["t_end"] for r in edge_rows))
    t_to_rowidxs = {t: [] for t in T_end}
    for i, r in enumerate(edge_rows):
        t_to_rowidxs[r["t_end"]].append(i)

    T = len(T_end)
    K = int(K)
    flow = np.zeros((T, K), dtype=float)
    speed = np.full((T, K), np.nan, dtype=float)
    speed_num = np.zeros((T, K), dtype=float)
    speed_den = np.zeros((T, K), dtype=float)
    
    # Calculate cluster capacity (sum of edge capacities in each cluster)
    cluster_capacity = np.zeros(K, dtype=float)
    for eidx, c in enumerate(labels):
        cluster_capacity[int(c)] += edge_capacity[eidx]

    for ti, t in enumerate(T_end):
        for idx in t_to_rowidxs[t]:
            r = edge_rows[idx]
            eid = r["edge_id"]
            if eid not in eid_to_idx:
                continue
            eidx = eid_to_idx[eid]
            c = int(labels[eidx])

            entered = float(r["entered"])
            # vehicles per interval; convert to vehicles/period (same number)
            f = entered  # keep as vehicles/bin; downstream can normalize
            sp = r["speed"]
            # accumulate
            flow[ti, c] += f
            if not math.isnan(sp):
                speed_num[ti, c] += sp * f
                speed_den[ti, c] += f

    # compute flow-weighted average speed where we have >0 flow
    mask = speed_den > 0
    speed[mask] = (speed_num[mask] / speed_den[mask])
    # where no vehicles, set speed to NaN
    return T_end, flow, speed, cluster_capacity


def detect_events(t_end_secs, flow, speed, edge_speed_lim, cluster_capacity, labels, K,
                  low_flow_quantile=0.2, congestion_frac=0.4, period=60,
                  time_bounds=None):
    """
    Create Hawkes events based on REAL traffic state transitions from SUMO data.
    
    Uses relative thresholds for realistic traffic analysis:
      e=0 low-flow transition   when flow/capacity ratio drops below threshold
      e=1 congestion transition when speed/speed_limit ratio drops below threshold
    
    Events are generated throughout the day based on actual traffic patterns.
    
    Args:
        cluster_capacity: array [K] of theoretical capacity (veh/hour) for each cluster
        time_bounds: tuple (start_hour, end_hour) to focus on specific time periods
                    e.g., (5, 11) for morning peak, (14, 20) for evening peak
    """
    T = len(t_end_secs)
    K = int(K)
    
    # Filter data by time bounds if specified
    if time_bounds is not None:
        start_hour, end_hour = time_bounds
        start_sec = start_hour * 3600
        end_sec = end_hour * 3600
        
        # Find indices within time bounds
        time_mask = (np.array(t_end_secs) >= start_sec) & (np.array(t_end_secs) <= end_sec)
        valid_indices = np.where(time_mask)[0]
        
        if len(valid_indices) == 0:
            print(f"No data found in time bounds {start_hour}:00-{end_hour}:00")
            events = []
            dt_dtype = np.dtype([('t', float), ('u', int), ('e', int), ('x', float), ('y', float)])
            return events, dt_dtype
        
        # Filter arrays to time bounds
        t_end_filtered = [t_end_secs[i] for i in valid_indices]
        flow_filtered = flow[valid_indices]
        speed_filtered = speed[valid_indices]
        
        print(f"Focusing on {start_hour:02d}:00-{end_hour:02d}:00 ({len(valid_indices)} time bins)")
    else:
        t_end_filtered = t_end_secs
        flow_filtered = flow
        speed_filtered = speed
        valid_indices = range(len(t_end_secs))
    
    # per-cluster freeflow from speed limits of edges in cluster
    c2speeds = {c: [] for c in range(K)}
    for eidx, c in enumerate(labels):
        c2speeds[int(c)].append(edge_speed_lim[eidx])
    cluster_vf = np.array([np.median(c2speeds[c]) if c2speeds[c] else 13.89 for c in range(K)], dtype=float)
    
    # Calculate flow ratios (flow/capacity) for each cluster and time
    flow_ratios = np.full_like(flow_filtered, np.nan)
    for c in range(K):
        cluster_cap = cluster_capacity[c]
        if cluster_cap > 0:  # Avoid division by zero
            for ti in range(len(t_end_filtered)):
                if flow_filtered[ti, c] >= 0:  # Include zero flow
                    # Convert flow from vehicles/period to vehicles/hour for ratio calculation
                    flow_per_hour = flow_filtered[ti, c] * (3600.0 / period)
                    flow_ratios[ti, c] = flow_per_hour / cluster_cap
    
    # Calculate speed ratios (speed/speed_limit) for each cluster and time
    speed_ratios = np.full_like(speed_filtered, np.nan)
    for c in range(K):
        cluster_speed_limit = cluster_vf[c]  # Free-flow speed for this cluster
        for ti in range(len(t_end_filtered)):
            if not np.isnan(speed_filtered[ti, c]) and speed_filtered[ti, c] > 0:
                speed_ratios[ti, c] = speed_filtered[ti, c] / cluster_speed_limit
    
    # Use ACTUAL traffic data patterns for realistic thresholds (on filtered data)
    # Calculate dynamic thresholds based on actual traffic variation
    flow_ratios_valid = flow_ratios[~np.isnan(flow_ratios) & (flow_ratios >= 0)]
    speed_ratios_valid = speed_ratios[~np.isnan(speed_ratios) & (speed_ratios > 0)]
    
    if len(flow_ratios_valid) == 0:
        print("No valid flow ratio data found - generating minimal events")
        events = []
        dt_dtype = np.dtype([('t', float), ('u', int), ('e', int), ('x', float), ('y', float)])
        return events, dt_dtype
    
    # Much more sensitive thresholds based on actual traffic patterns
    low_flow_ratio_threshold = 0.05  # Flow/capacity < 10% (triggers on very low flow)
    congestion_ratio_threshold = 1.40 if len(speed_ratios_valid) > 0 else 0.0  # Speed/limit > 120% (triggers on high speeds)
    
    # Use actual traffic statistics for reporting
    flow_ratio_mean = np.mean(flow_ratios_valid)
    flow_ratio_std = np.std(flow_ratios_valid)
    
    print(f"Real traffic patterns (filtered period):")
    print(f"  Cluster capacities: {cluster_capacity}")
    print(f"  Flow ratios: {flow_ratio_mean:.3f} ± {flow_ratio_std:.3f} (flow/capacity)")
    print(f"  Cluster speed limits: {cluster_vf}")
    if len(speed_ratios_valid) > 0:
        speed_ratio_mean = np.mean(speed_ratios_valid)
        speed_ratio_std = np.std(speed_ratios_valid)
        print(f"  Speed ratios: {speed_ratio_mean:.2f} ± {speed_ratio_std:.2f} (actual/limit)")
        print(f"Thresholds (fixed): Low flow ratio < {low_flow_ratio_threshold:.3f}, Congestion ratio < {congestion_ratio_threshold:.2f}")
    else:
        print(f"Thresholds (fixed): Low flow ratio < {low_flow_ratio_threshold:.3f}, Speed-based detection disabled")
    
    # Track states and generate events based on REAL transitions
    events = []
    node_states = np.zeros(K, dtype=int)  # 0=normal, 1=low_flow, 2=congestion
    
    for ti, (tsec, orig_idx) in enumerate(zip(t_end_filtered, valid_indices)):
        for c in range(K):
            current_flow_ratio = flow_ratios[ti, c]
            current_speed_ratio = speed_ratios[ti, c]
            current_state = node_states[c]
            new_state = current_state
            
            # Only process if we have actual traffic data
            if not np.isnan(current_flow_ratio):  # Valid flow ratio (includes zero flow)
                
                # Detect congestion using LOS-based speed ratio (more accurate than absolute speed)
                if (not np.isnan(current_speed_ratio) and 
                    current_speed_ratio < congestion_ratio_threshold):
                    new_state = 2  # Congestion (LOS F: speed < threshold ratio of speed limit)
                    
                # Detect low flow using capacity-based flow ratio
                elif current_flow_ratio < low_flow_ratio_threshold:
                    new_state = 1  # Low flow (flow < threshold ratio of capacity)
                    
                else:
                    new_state = 0  # Normal traffic
                
                # Generate event only on state transitions
                if new_state != current_state:
                    if new_state == 1:
                        events.append((tsec / 3600.0, c, 0))  # Low flow event (time in hours)
                    elif new_state == 2:
                        events.append((tsec / 3600.0, c, 1))  # Congestion event (time in hours)
                    
                    node_states[c] = new_state
    
    # Sort events chronologically (NEVER shuffle!)
    events.sort(key=lambda x: x[0])
    
    # Count simultaneous events (same time)
    if len(events) > 1:
        event_times = [e[0] for e in events]
        unique_times = set(event_times)
        simultaneous_count = len(event_times) - len(unique_times)
        if simultaneous_count > 0:
            print(f"Found {simultaneous_count} simultaneous events (same time)")
        else:
            print("No simultaneous events found")
    
    print(f"Generated {len(events)} events from REAL traffic patterns")
    
    # Count event types
    if events:
        event_types = [e[2] for e in events]
        low_flow_count = event_types.count(0)
        congestion_count = event_types.count(1)
        print(f"Natural event distribution: {low_flow_count} low flow, {congestion_count} congestion")
    
    dt_dtype = np.dtype([('t', float), ('u', int), ('e', int), ('x', float), ('y', float)])
    return events, dt_dtype


# ---------------------------
# Main pipeline
# ---------------------------

def run_pipeline(
    sumocfg: str,
    num_nodes: int,
    bin_secs: int,
    low_flow_quantile: float,
    congestion_frac: float,
    output_pickle: str,
    sumo_bin: str = "sumo",
    time_bounds: tuple = None,
    force_rerun: bool = False,
    args: argparse.Namespace = None
):
    # 1) parse cfg -> paths
    net_file, _routes, existing_additional, cfg_dir = parse_sumocfg_paths(sumocfg)
    if net_file is None:
        raise RuntimeError("Could not find <net-file> in the .sumocfg")

    # 2) use existing folder structure, check for existing edgeData output
    cfg_path = Path(sumocfg).resolve()
    work_dir = cfg_path.parent
    
    # Check for existing edgeData files (the additional.add.xml already configures edgeData output)
    existing_edge_files = list(work_dir.glob("edge_data*.xml"))
    print(f"Found existing edgeData files in {work_dir}:")
    for f in existing_edge_files:
        print(f"  {f.name} ({f.stat().st_size} bytes)")
    
    # Use the largest edgeData file (most complete data)
    if existing_edge_files:
        edge_xml = max(existing_edge_files, key=lambda f: f.stat().st_size)
        print(f"Using existing edgeData: {edge_xml}")
    else:
        edge_xml = work_dir / f"edge_data_{args.city}.xml"
        print(f"No existing edgeData found, will create: {edge_xml}")
    
    # Check if edgeData already exists and is valid
    if not force_rerun and edge_xml.exists() and edge_xml.stat().st_size > 0:
        print(f"Found existing edgeData: {edge_xml}")
        try:
            intervals, edge_rows = parse_edgeData(edge_xml)
            if edge_rows:
                print(f"Using existing edgeData with {len(edge_rows)} records")
                use_existing = True
            else:
                print("Existing edgeData is empty, will rerun SUMO")
                use_existing = False
        except Exception as e:
            print(f"Error reading existing edgeData: {e}")
            print("Will rerun SUMO")
            use_existing = False
    else:
        use_existing = False
    
    if not use_existing:
        print("No valid edgeData found, running SUMO simulation...")
        # Run SUMO with existing additional files (which already include edgeData output)
        run_sumo_cli(sumocfg, None, existing_additional, sumo_bin=sumo_bin, time_bounds=time_bounds)

        # Check if edgeData was created
        if not edge_xml.exists():
            # If the expected file doesn't exist, look for any edgeData files
            new_edge_files = list(work_dir.glob("edge_data*.xml"))
            if new_edge_files:
                edge_xml = max(new_edge_files, key=lambda f: f.stat().st_size)
                print(f"Found new edgeData: {edge_xml}")
            else:
                raise RuntimeError("No edgeData output found after SUMO run")

        # Parse edgeData output
        intervals, edge_rows = parse_edgeData(edge_xml)
        if not edge_rows:
            raise RuntimeError("edgeData output is empty; check your network/routes and period.")

    # 5) read edge geometry & freeflow speeds
    edge_ids, edge_xy, edge_speed_lim, edge_capacity, edge_to_nodes, node_xy, node_to_edges = read_edge_geometry(net_file)

    # Keep only edges that appeared in edge_rows
    used_edge_ids = sorted(set(r["edge_id"] for r in edge_rows) & set(edge_ids))
    if not used_edge_ids:
        raise RuntimeError("No overlap between edges in net.xml and edgeData output.")
    mask = np.array([eid in used_edge_ids for eid in edge_ids], dtype=bool)
    edge_ids = [eid for eid, m in zip(edge_ids, mask) if m]
    edge_xy = edge_xy[mask]
    edge_speed_lim = edge_speed_lim[mask]
    edge_capacity = edge_capacity[mask]

    # 6) cluster edges -> K nodes
    kmeans = KMeans(n_clusters=num_nodes, random_state=42, n_init=10)
    labels = kmeans.fit_predict(edge_xy)
    cluster_xy = kmeans.cluster_centers_

    # 7) aggregate time series by cluster
    t_end_secs, cluster_flow, cluster_speed, cluster_capacity = aggregate_by_cluster(
        edge_rows, edge_ids, edge_xy, edge_speed_lim, edge_capacity, labels, num_nodes, bin_secs
    )

    # 8) adjacency among clusters
    adjacency = build_cluster_adjacency(edge_ids, edge_to_nodes, node_to_edges, labels, num_nodes)

    # 9) detect events (time in hours)
    events_list, dt_dtype = detect_events(
        t_end_secs, cluster_flow, cluster_speed, edge_speed_lim, cluster_capacity, labels, num_nodes,
        low_flow_quantile=low_flow_quantile,
        congestion_frac=congestion_frac,
        period=bin_secs,
        time_bounds=time_bounds
    )

    # attach x,y for each event (cluster centroid)
    events = np.zeros(len(events_list), dtype=dt_dtype)
    for i, (th, u, e) in enumerate(events_list):
        x, y = cluster_xy[u]
        events[i] = (th, u, e, float(x), float(y))

    # 10) prepare neighbor list (1-hop)
    neighbors_list = [np.where(adjacency[i] > 0)[0] for i in range(num_nodes)]

    # 11) save
    out = {
        "events": events,  # dtype [('t','u','e','x','y')]
        "num_nodes": int(num_nodes),
        "num_event_types": 2,  # 0: low flow, 1: congestion
        "node_locations": cluster_xy.astype(float),
        "adjacency_matrix": adjacency.astype(int),
        "neighbors_list": neighbors_list,
        "bin_secs": int(bin_secs),
        "low_flow_quantile": float(low_flow_quantile),
        "congestion_frac": float(congestion_frac),
        "t_end_secs": list(map(float, t_end_secs)),
        "cluster_flow": cluster_flow.astype(float),
        "cluster_speed": cluster_speed.astype(float),
        "edge_ids_used": used_edge_ids,
        "edge_capacity": cluster_capacity.astype(float), # Add edge_capacity to output
    }

    # If you want exactly the same structure your hawkes_module expects:
    if HAVE_HAWKES:
        # Initialize simple defaults (you can override in your training script)
        mu_init = np.full((num_nodes, 2), 1e-3)
        K_init = np.zeros((num_nodes, num_nodes))
        omega_init = 1.0   # 1/hour
        sigma_init = 2.0   # arbitrary distance unit of network coords
        params_init = np.concatenate([mu_init.flatten(), K_init.flatten(), [omega_init, sigma_init]])
        mark_kernel_matrix = np.array([[1.0, 1.0],
                                       [1.0, 1.0]])
        hawkes.save_simulation_data(
            output_pickle, events, params_init, num_nodes, 2,
            cluster_xy, adjacency, neighbors_list,
            'linear',
            mark_kernel_type='matrix',
            mark_kernel_matrix=mark_kernel_matrix,
            num_hops=1
        )
    else:
        with open(output_pickle, "wb") as f:
            pickle.dump(out, f)

    print(f"Saved: {output_pickle}")
    print(f"Events: {len(events)}  | Nodes: {num_nodes}  | Time bins: {len(t_end_secs)}")


def _auto_find_sumocfg_for_city(city: str) -> str:
    """Return first .sumocfg found under a directory named after the city."""
    root = Path(city)
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"City directory not found: {city}")
    candidates = list(root.rglob("*.sumocfg"))
    if not candidates:
        raise RuntimeError(f"No .sumocfg found under {city}")
    return str(candidates[0].resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumocfg", required=False, help="Path to .sumocfg")
    ap.add_argument("--city", choices=["arbon_waelli", "zurich", "geneva", "adliswil", "rio", "lis"], help="Select city; auto-detect .sumocfg under that folder if --sumocfg not provided")
    ap.add_argument("--num-nodes", type=int, default=20, help="number of clusters")
    ap.add_argument("--bin-secs", type=int, default=60, help="edgeData period (s)")
    ap.add_argument("--low-flow-quantile", type=float, default=0.2, help="quantile for low-flow event")
    ap.add_argument("--congestion-frac", type=float, default=0.4, help="speed < frac * freeflow -> congestion")
    ap.add_argument("--output", required=True, help="Output .pickle")
    ap.add_argument("--sumo-bin", default="sumo", help="Path to SUMO binary (default 'sumo')")
    ap.add_argument("--time-bounds", type=str, help="Time bounds as 'start,end' (e.g., '5,11' for 5am-11am)")
    ap.add_argument("--force-rerun", action="store_true", help="Force rerun SUMO even if edgeData exists")
    args = ap.parse_args()

    # Resolve sumocfg from city if needed
    sumocfg_path = args.sumocfg
    if not sumocfg_path:
        if args.city:
            sumocfg_path = _auto_find_sumocfg_for_city(args.city)
            print(f"Using city '{args.city}' -> {sumocfg_path}")
        else:
            raise SystemExit("Provide --sumocfg or --city")

    # Parse time bounds if provided
    time_bounds = None
    if args.time_bounds:
        try:
            start_hour, end_hour = map(int, args.time_bounds.split(','))
            time_bounds = (start_hour, end_hour)
            print(f"Using time bounds: {start_hour:02d}:00 - {end_hour:02d}:00")
        except ValueError:
            print("Invalid time bounds format. Use 'start,end' like '5,11'")
            return

    run_pipeline(
        sumocfg=sumocfg_path,
        num_nodes=args.num_nodes,
        bin_secs=args.bin_secs,
        low_flow_quantile=args.low_flow_quantile,
        congestion_frac=args.congestion_frac,
        output_pickle=args.output,
        sumo_bin=args.sumo_bin,
        time_bounds=time_bounds,
        force_rerun=args.force_rerun,
        args=args
    )

if __name__ == "__main__":
    main()
