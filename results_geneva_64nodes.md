# Model Evaluation Results: geneva_64nodes.pickle

## Dataset Overview

| Property | Value |
|----------|-------|
| **Total Events** | 1,717 |
| **Time Range** | 0.05 - 23.97 hours (≈24h) |
| **Duration** | 23.92 hours |
| **Num Nodes** | 12 |
| **Num Marks (Event Types)** | 2 |
| **Mark 0 (Low Flow)** | 815 events |
| **Mark 1 (Congestion)** | 902 events |
| **Event Rate** | 71.8 events/hour |
| **Events per Node** | min=59, max=242, mean=143.1, std=56.6 |
| **Network Edges** | 42 (avg degree: 3.5) |
| **Reachability Hops** | 3 |

---

## Model Comparison Summary

### Goodness-of-Fit (GOF) Results

| Model | Window | Ljung-Box | Scale Error | Overall |
|-------|--------|-----------|-------------|---------|
| **Linear** | 0.5h | ✓ PASS | 13.8% | ✓ PASS |
| **ReLU** | 0.5h | ✓ PASS | 11.7% | ✓ PASS |
| **Softplus** | 0.5h | ✓ PASS | 10.6% | ✓ PASS |
| **Linear** | 1.0h | ✓ PASS | 13.6% | ✓ PASS |
| **Linear** | 1.5h | ✓ PASS | 13.9% | ✓ PASS |
| **Linear** | 2.0h | ✓ PASS | 14.2% | ✓ PASS |

### Prediction Metrics (Test Period: 16h onwards, 15-min bins)

| Model | Network R² (raw) | Network R² (+EMA) | Per-Node R² | Top 5 R² |
|-------|------------------|-------------------|-------------|----------|
| **Linear** | 0.836 | 0.956 | 0.271 | 0.257 |
| **ReLU** | 0.879 | 0.965 | 0.293 | 0.273 |
| **Softplus** | 0.871 | 0.959 | 0.286 | 0.270 |
| **Joint (np3)** | 0.841 | 0.962 | **0.606** | **0.631** |

### Key Findings

1. **All linear/nonlinear models pass GOF** on this dataset
2. **ReLU performs slightly better** than Linear and Softplus for temporal prediction
3. **Joint kernel (np3) excels at spatial prediction** (R²=0.61 vs 0.27-0.29)
4. **EMA smoothing significantly improves R²** (0.84→0.96) by filtering noise

---

## Detailed GOF Results

### Linear Model (W=0.5h)

```
============================================================
GOF RESULTS (Spatio-Temporal Hawkes)
============================================================
Events: 1717
α = 0.9409
Window = 0.5

--- Calibration ---
Mean τ: 0.862 (expected: 1.0)
Scale error: 13.8%
Median τ: 0.472 (expected: 0.693)

--- Ljung-Box Test (temporal independence) ---
Lag 10: p=0.9984
Lag 20: p=1.0000
Lag 30: p=1.0000
Result: PASS

--- First-Order Residuals ---
Total observed: 1717
Baseline expected (μ×T): 177.7
Excitation ratio: 9.66×
Implied spectral radius: ~0.90

Per-node obs/exp ratio: min=4.27, max=19.46
-> Balanced across nodes
============================================================
```

### ReLU Model (W=0.5h)

```
Scale error: 11.7%
Excitation ratio: 9.88×
Implied spectral radius: ~0.90
```

### Softplus Model (W=0.5h)

```
Scale error: 10.6%
Excitation ratio: 7.91×
Implied spectral radius: ~0.87
```

---

## Detailed Prediction Results

### Linear Model (W=0.5h)

```
--- Network-Wide by Type (sum across nodes) ---
Raw:      R²=0.8359, Pearson=0.9184
+EMA:     R²=0.9559, Pearson=0.9821

--- Network-Wide Total (all events) ---
Raw:      R²=0.7268
+EMA:     R²=0.9202

--- Per-Node (spatial prediction) ---
Raw:      R²=0.2708, Pearson=0.5355
+EMA:     R²=0.3996, Pearson=0.6565

--- Top Nodes by Activity ---
Node   Events   Pred     R²       Pearson 
----------------------------------------
5      41       31.1     0.3622   0.6989  
9      37       28.7     0.5089   0.8165  
3      35       28.2     0.4753   0.8238  
6      33       31.2     -0.2896  -0.2467 
1      23       36.5     0.2126   0.6206  
```

### Joint Kernel Model (np3)

```
--- Network-Wide by Type ---
  Mark 0: Raw R²=0.7395, +EMA R²=0.8226
  Mark 1: Raw R²=0.8819, +EMA R²=0.9722

--- Network-Wide Total ---
Raw:   R²=0.8408, Pearson=0.9262
+EMA:  R²=0.9615, Pearson=0.9876

--- Per-Node ---
R²=0.6061

--- Top Nodes by Activity ---
Node   Events   Pred     R²       Pearson 
----------------------------------------
5      40       27.9     0.6117   0.9439  
9      36       30.9     0.6862   0.9175  
3      35       33.6     0.6130   0.8118  
6      33       22.8     0.5515   0.9511  
1      23       31.8     0.7349   0.8826  
```

---

## Methodology

### 1. Goodness-of-Fit (GOF) Testing

The GOF tests validate whether the model correctly captures the event dynamics using the **time rescaling theorem**.

#### Time Rescaling Theorem

If the model is correctly specified, the rescaled inter-arrival times should be i.i.d. Exp(1).

For each event at `(t_i, node_i, mark_i)`, compute:
```
τ_i = Λ_{node_i, mark_i}(t_i) - Λ_{node_i, mark_i}(t_prev)
```

Where:
- `Λ_{v,k}(t)` is the compensator (integrated intensity) for location `(v, k)`
- `t_prev` is the previous event time **at the same (node, mark) pair**

#### Compensator Calculation

For a specific (node, mark) pair `(v, k)`:

```
Λ_{v,k}(t_start → t_end) = ∫_{t_start}^{t_end} λ_{v,k}(s) ds
```

Where the intensity is:
```
λ_{v,k}(t) = μ_{v,k} + α × Σ_{j: t_j < t} K[v, u_j] × M_K[e_j, k] × g(t - t_j)
```

**Components:**
- `μ_{v,k}`: Baseline intensity for (node v, mark k)
- `α`: Global excitation strength
- `K[v, u_j]`: Spatial coupling from source node u_j to target node v
- `M_K[e_j, k]`: Mark coupling from source mark e_j to target mark k
- `g(τ)`: Temporal kernel (Gaussian mixture)

#### Tests Performed

1. **Ljung-Box Test** (Primary): Tests for autocorrelation in rescaled times
   - H₀: Rescaled times are independent (no autocorrelation)
   - PASS if p-values > 0.05 at lags 10, 20, 30
   - **This is the key test** - if temporal dynamics are captured, residuals should be uncorrelated

2. **Scale Error**: Measures calibration
   - Mean(τ) should be ≈ 1.0 for well-calibrated intensity
   - Scale error < 20% is acceptable

3. **First-Order Residuals**: Compares observed vs baseline-expected counts
   - Excitation ratio = Total observed / Baseline expected
   - Implied spectral radius ρ ≈ 1 - 1/ratio (should be < 1 for stability)

### 2. Prediction Evaluation

Predictions are evaluated by comparing expected vs actual event counts in time bins.

#### Prediction Method

For each time bin `[t_start, t_end]`:

1. **History**: All events before `t_mid = (t_start + t_end) / 2`
2. **Intensity at midpoint**: 
   ```
   λ_{v,k}(t_mid) = μ_{v,k} + α × Σ_{j: t_j < t_mid} K[v, u_j] × M_K[e_j, k] × g(t_mid - t_j)
   ```
3. **Predicted count**: `pred_{v,k} = λ_{v,k}(t_mid) × (t_end - t_start)`

#### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **R²** | `1 - SS_res/SS_tot` | Fraction of variance explained (1.0 = perfect) |
| **Pearson r** | `cov(y, ŷ) / (σ_y × σ_ŷ)` | Linear correlation (-1 to 1) |
| **MAE** | `mean(\|y - ŷ\|)` | Average absolute error |
| **RMSE** | `√mean((y - ŷ)²)` | Root mean squared error |

#### Aggregation Levels

1. **Network-Wide by Type**: Sum predictions across all nodes, compare per mark
   - Tests: "Does the model predict total congestion events well?"
   - Typically highest R² due to averaging

2. **Network-Wide Total**: Sum across nodes and marks
   - Tests: "Does the model predict total event volume?"

3. **Per-Node**: Individual node predictions
   - Tests: "Does the model know WHERE events will occur?"
   - Typically lowest R² - spatial prediction is hardest

4. **Top N Nodes**: Most active nodes only
   - More data → more reliable metrics
   - Joint kernel excels here

#### EMA Smoothing

Exponential Moving Average reduces noise in time series:
```
EMA[i] = α × y[i] + (1 - α) × EMA[i-1]
```

- α = 0.4 (default): Balances responsiveness and smoothness
- Applied to BOTH actual and predicted for fair comparison
- Significantly improves R² by filtering observation noise

---

## Model Details

### Intensity Function

```
λ_{v,k}(t) = μ_{v,k} + α × Σ_{j: t_j < t} K[v, u_j] × M_K[e_j, k] × g(t - t_j) × R[v, u_j]
```

Where:
- **μ_{v,k}**: Baseline rate for (node v, mark k), shape (N, M)
- **α**: Global excitation parameter (scalar)
- **K[v, u]**: Spatial coupling matrix (N × N), distance-based
- **M_K[ℓ, k]**: Mark kernel matrix (M × M)
- **g(τ)**: Temporal kernel on [0, W]
- **R[v, u]**: Reachability mask (1 if u→v within L hops)

### Temporal Kernel

Gaussian mixture representation:
```
g(τ) = Σ_{b=1}^{B_t} w_b × ψ_b(τ)
```

Where each basis function:
```
ψ_b(τ) = (1/Z_b) × exp(-0.5 × ((τ - c_b) / σ_t)²) × 𝟙{0 ≤ τ ≤ W}
```

- Centers `{c_b}` are fixed (equally spaced on [0, W])
- Only weights `{w_b}` are learned (simplex constraint: w_b ≥ 0, Σw = 1)

### Spatial Kernel

Distance-based Gaussian mixture:
```
K̂[v, u] = R[v, u] × Σ_{r=1}^{B_s} β_r × χ_r(d[v, u])
```

Where:
```
χ_r(d) = exp(-0.5 × ((d - c_r) / σ_s)²)
```

Row-normalized: `K[v, u] = K̂[v, u] / Σ_w K̂[v, w]`

### Nonlinear Link Functions

| Name | φ(x) | Properties |
|------|------|------------|
| Linear | max(x, 0) | Simple, interpretable |
| ReLU | max(x, 0) | Same as linear (no negative intensities) |
| Softplus | log(1 + exp(x)) | Smooth approximation to ReLU |
| Exponential | exp(x) | Multiplicative, can explode |

---

## Interpretation Guide

### GOF Interpretation

| Result | Meaning | Action |
|--------|---------|--------|
| Ljung-Box PASS + Scale < 20% | Model captures dynamics well | Use for predictions |
| Ljung-Box PASS + Scale > 20% | Dynamics OK, calibration off | Check baseline estimation |
| Ljung-Box FAIL | Missing temporal structure | Add complexity or check data |

### Prediction R² Interpretation

| R² Range | Quality | Notes |
|----------|---------|-------|
| > 0.8 | Excellent | Network-wide with smoothing typically achieves this |
| 0.5 - 0.8 | Good | Solid for raw predictions |
| 0.3 - 0.5 | Moderate | Typical for per-node spatial prediction |
| < 0.3 | Weak | May need model improvements |

### Excitation Ratio Interpretation

| Ratio | Implied ρ | Meaning |
|-------|-----------|---------|
| 2× | ~0.50 | Moderate self-excitation |
| 5× | ~0.80 | Strong self-excitation |
| 10× | ~0.90 | Very strong excitation (near critical) |
| > 20× | > 0.95 | Near-critical, check stability |

---

## File Locations

### Inference Results
- `inference_result_np_geneva_64nodes_w0p5_linear.pickle` - Linear, W=0.5h
- `inference_result_np_geneva_64nodes_w0p5_relu.pickle` - ReLU, W=0.5h
- `inference_result_np_geneva_64nodes_w0p5_softplus.pickle` - Softplus, W=0.5h
- `inference_result_np_geneva_64nodes_w1p0_linear.pickle` - Linear, W=1.0h
- `inference_result_np_geneva_64nodes_w1p5_linear.pickle` - Linear, W=1.5h
- `inference_result_np_geneva_64nodes_w2p0_linear.pickle` - Linear, W=2.0h
- `inference_result_np3_geneva_64nodes.pickle` - Joint kernel

### Diagnostic Outputs
- `gof_diagnostics.png` - QQ plots, histograms, ACF
- `gof_intensity.png` - Intensity over time
- `gof_results.pickle` - Numerical GOF results

### Scripts
- `gof.py` - Goodness-of-fit testing
- `evaluate_predictions.py` - Prediction evaluation (separable kernel)
- `evaluate_predictions_np3.py` - Prediction evaluation (joint kernel)
- `nonpm_window_6.py` - Separable kernel model training
- `nonpm_window_3.py` - Joint kernel model training

---

## Running the Analysis

### GOF Test
```bash
python gof.py --result inference_result_np_geneva_64nodes_w0p5_linear.pickle \
              --data geneva_64nodes.pickle
```

### Prediction Evaluation
```bash
# Separable kernel models
python evaluate_predictions.py --result inference_result_np_geneva_64nodes_w0p5_linear.pickle \
                               --data geneva_64nodes.pickle \
                               --bin-size 0.25

# Joint kernel model
python evaluate_predictions_np3.py --result inference_result_np3_geneva_64nodes.pickle \
                                   --data geneva_64nodes.pickle \
                                   --bin-size 0.25
```

---

*Generated: January 2026*

