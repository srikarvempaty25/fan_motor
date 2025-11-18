# Motor Anomaly Detection — Documentation

## Overview

This document describes the methodology, rationale, and implementation details for building an anomaly-detection pipeline for your BLDC motor telemetry. It is written for engineers and data scientists who will run, review, or adapt the Jupyter notebook and production pipeline. The approach assumes *no labelled failures* and uses a combination of real seed data and realism-focused synthetic augmentation to build and validate unsupervised models and a drift-aware baseline.

---

## Contents

1. Goals and constraints
2. Data available (seed dataset)
3. High-level strategy
4. Synthetic data modes (detailed)
5. How environment & temperature affect signals (physical explanation + modelling)
6. Feature engineering
7. Anomaly detectors to use and why
8. Drift detection & re-baselining policies
9. Health score definition and alerting tiers
10. Evaluation strategy (with synthetic degradations)
11. Implementation notes and configuration
12. Next steps and recommended experiments

---

## 1. Goals and constraints

**Goals**

* Build an anomaly detection system that: learns a healthy baseline, detects deviations in firmware-reported telemetry, outputs per-sample and per-window health scores, and supports conservative adaptive re-baselining.
* Operate without additional hardware; synthetic data augments the small seed dataset.

**Constraints**

* Only firmware telemetry is available (columns: `timestamp`, `speed_setting`, `current`, `bemf`, `inductance`, `resistance`), and the values are firmware-scaled/normalized rather than raw physical units.
* No labelled failures available (unsupervised approach required).
* A minimal seed dataset (~60 rows) is provided — synthetic augmentation will be used to expand coverage.

---

## 2. Data available (seed dataset)

* Three operating clusters observed: `speed_setting` ≈ 0, 25, 50 (PWM-like commands).
* `current` values are very small (firmware units), `bemf` is stable per cluster; `inductance` and `resistance` appear constant (firmware placeholders).
* Data is temporally ordered and smooth with few transients in the seed.

Recommend storing original seed CSV and not overwriting it; all augmentation and experiments should reference the seed as canonical "healthy" data.

---

## 3. High-level strategy

1. Use the seed dataset to estimate empirical distributions for each operating cluster.
2. Generate a large synthetic *normal* dataset by sampling around those distributions and injecting realistic temporal structure (time gaps, startup ramps). This produces a robust baseline for unsupervised model training.
3. Separately generate synthetic *degraded* datasets of multiple kinds (gradual drift, step changes, intermittent spikes, sensor bias, correlated environmental effects) to evaluate sensitivity and tune thresholds.
4. Train an ensemble of unsupervised detectors on the expanded normal dataset (IsolationForest, LOF, One-Class SVM) and use simple rules (z-score) for sanity checking.
5. Compute a continuous health score per-sample and derive per-window metrics to drive alerts.
6. Implement drift detection and a conservative re-baselining procedure so the system can adapt to slow, benign aging while still detecting sudden faults.

---

## 4. Synthetic data modes (detailed)

To make synthetic data useful for realistic evaluation and model building, produce multiple modes. Each mode should be documented and parameterized so you can vary severity and duration.

### A. Normal (baseline) — `normal`

* Sample `current` and `bemf` from cluster-specific Gaussian/empirical distributions.
* Add small temporal correlations (e.g., AR(1) behaviour or short ramps) to make successive rows realistic.
* Insert occasional short startup/transition ramps to mimic speed changes.
* Randomize sampling intervals using an exponential distribution so time gaps vary.

### B. Gradual degradation — `gradual`

* Model long-term aging by slowly **increasing current** and **decreasing bemf** over hours-weeks.
* Parameterize by `degradation_amount` (relative change) and `drift_length_seconds` (duration over which it occurs).
* Use linear or nonlinear ramps (e.g., logistic growth) to simulate realistic wear-in or lubrication loss.

### C. Step / sudden degradation — `step`

* Sudden persistent shift in the signal (e.g., current jumps up 30–100% and bemf drops) that persists until intervention.
* Useful to mimic sudden mechanical binding, a damaged bearing, or a partial obstruction.

### D. Intermittent spikes / noisy faults — `noisy_fault`

* Sparse, high-amplitude current spikes (3–8×) or bemf mismatches appearing intermittently.
* Useful to simulate transient events (loose wiring, commutation hiccup) that may precede larger faults.

### E. Sensor bias/drift — `sensor_bias`

* The telemetry source slowly drifts due to ADC offset or amplifier temperature drift.
* Model as additive bias on `current` or `bemf` that increases slowly; optionally add periodic calibration resets.

### F. Environmental-correlated drift — `env_correlated`

* Add variations correlated with external variables like ambient temperature, humidity, or supply voltage.
* Example: during a simulated hot period, slightly increase winding resistance, slightly reduce air density effect, change bemf scale, and add sensor bias.

### G. Missing data / packet loss — `missing`

* Randomly drop rows or set values to NaN to test robustness of feature calculations and the deployment pipeline.

### H. Combined scenarios — `combined`

* Mix of the above, e.g., a gradual degradation with intermittent noisy spikes and occasional sensor bias.

**Important:** label synthetic degraded rows in your *evaluation* set but never include them in training. Keep all synthetic normal data + seed data as the training corpus.

---

## 5. How environment & temperature affect signals (physical explanation + modelling)

Below are the key physics and practical effects you should model and monitor. These are simplified but capture the dominant behavior for small BLDC fans.

### Physics fundamentals

The motor electrical equation (simplified):

```
V_supply - V_commutation_losses - V_bemf = I * R_winding
```

Where:

* `V_supply` = driver average voltage (function of PWM, supply rail)
* `V_bemf` = back EMF (proportional to angular speed, `K_e * ω`) — increases with RPM
* `R_winding` = winding resistance (temperature dependent)
* `I` = winding current (related to torque demand)

Rearranged:

```
I = (V_supply - V_bemf - V_losses) / R_winding
```

So current depends on: supply voltage, bemf (speed), resistance and losses.

### Temperature effects

1. **Winding resistance increases with temperature**

   * Copper resistance increases roughly +0.0039 per °C (≈ 0.39% / °C) near room temperature.
   * Higher `R_winding` tends to reduce current for the same (V_supply - V_bemf). However, the motor may draw *more* current if mechanical losses increase (e.g., lubricant viscosity changes or bearing friction increases with temperature extremes).

2. **Magnet strength & bemf**

   * Permanent magnet strength slightly reduces with temperature; this lowers `K_e` and thus reduces bemf at the same RPM, effectively increasing current for a given applied voltage.

3. **Air density (aerodynamic load) changes with ambient temperature and pressure**

   * Air density ∝ 1 / T (Kelvin). Hotter air is less dense → aerodynamic load on a blade decreases → torque required falls → current drops and RPM increases for a given PWM.
   * Conversely, colder air is denser → higher load → more current for same PWM.

4. **Lubrication and bearing friction**

   * Temperature affects lubricant viscosity: very low temps can stiffen grease and increase friction (current up); very high temps can thin grease or dry it out over time, increasing wear and friction (current up and variance up).

5. **Sensor & ADC drift**

   * Temperature affects analog components (op-amps, ADC references) causing offset and gain drift in `current` and `bemf` measurements. Model this as additive and multiplicative bias that correlates with an ambient temperature variable.

6. **Supply voltage variations**

   * Batteries or power supplies can sag at high ambient temperatures or under load, changing `V_supply` and thus altering `I` and `bemf` behaviour.

### Practical modelling suggestions (how to add temperature/env to synthetic data)

* Create a synthetic `ambient_temp` time series (seasonal + random noise + occasional heat events).
* Apply rules:

  * `R_winding(t) = R0 * (1 + alpha * (T(t) - T0))` with alpha ≈ 0.0039/°C.
  * `bemf_scale(t) = K_e * (1 - beta * (T(t) - T0))` with small beta (e.g., 0.0005–0.002/°C) to capture magnet weakening.
  * Aerodynamic load factor `L(t) = L0 * (rho(t)/rho0)` where `rho(t)` uses standard atmosphere: `rho ∝ P/(R_specific*T)` (approx inverse with T).
  * Sensor bias: `sensor_offset(t) = offset0 + offset_slope * (T - T0)` and `sensor_gain(t) = 1 + gain_slope*(T-T0)`.
* Combine these to compute synthetic `current` and `bemf` from a simplified motor model, or apply them as multiplicative/additive transforms to your empirical samples.

**Net effect examples:**

* A +20°C ambient spike might increase winding resistance by ≈ 8% (20°C * 0.39%/°C). That could slightly reduce current if bemf and V_supply unchanged, but if lubricant thins or magnets weaken you could see a net *increase* or more variable current. Modelling both effects (resistance up, magnet strength/down, air density down) yields realistic mixed outcomes that your anomaly detectors must tolerate.

---

## 6. Feature engineering

Recommended derived features:

* `delta_current`, `delta_bemf` (differences) — capture transients
* `rolling_mean_current`, `rolling_std_current` (3–10 sample windows) — capture steady-state stability
* `current / bemf` ratio or `bemf/current` — captures mismatch behaviour
* `speed_setting` (categorical or numerical) — include as context variable or build cluster-specific models
* `time_since_last_speed_change` — to treat transients differently
* `ambient_temp` (synthetic) and `supply_voltage` (if modelled) — include to explain systematic drifts

Normalization: use StandardScaler on model features; maintain scaler parameters to apply to new data.

---

## 7. Anomaly detectors to use and why

Train an ensemble and use voting / health-score fusion:

* **Isolation Forest** (global multivariate outliers)
* **Local Outlier Factor (novelty=True)** (local density / subtle deviations)
* **One-Class SVM** (tight boundary for very stable signals)
* **Z-score thresholds** on current and bemf (interpretable sanity checks)

Consider later additions when you have more synthetic/real data:

* **Autoencoder (dense or LSTM)** for reconstruction-based detection
* **PCA / Mahalanobis** for fast baseline checks
* **Sequence models (LSTM / Transformer)** for temporal anomalies

---

## 8. Drift detection & re-baselining policies

Use a three-layer policy:

1. **Short-term anomaly detection:** flag sudden events using the detectors per-sample or per-window.
2. **Drift monitor:** compute a continuous `health` metric and run a change detector (Page-Hinkley, ADWIN) on it to detect slow baseline shifts.
3. **Re-baseline gate:** when drift is slow and the recent anomaly rate is low (e.g., <1%), allow incremental baseline updates by adding recent non-anomalous data to the training buffer. Retrain periodically or use incremental-fit models.

Guidelines:

* Never retrain on data containing >5% anomalies without human review.
* Keep an immutable archive of old baselines to allow rollbacks.
* Use conservative thresholds early (reduce false-positives), then tighten once field data is available.

---

## 9. Health score definition and alert tiers

A continuous health score (0..1) combines model confidence and anomaly votes. Example:

```
health = 0.7 * normalized_iso_score + 0.3 * (1 - anomaly_votes/num_models)
```

Alerting tiers (example):

* `health >= 0.7`: OK
* `0.5 <= health < 0.7`: INFO / investigate
* `0.3 <= health < 0.5`: WARNING / schedule inspection
* `health < 0.3`: CRITICAL / immediate action

Tune thresholds with synthetic degraded datasets until working points match expected sensitivity.

---

## 10. Evaluation strategy

* **Train** only on seed + synthetic normal.
* **Hold out** degraded simulations (gradual, step, intermittent, sensor bias). Evaluate recall/precision on these sets.
* Use injected anomalies only for offline model comparison; do not use them for production retraining.
* Compute metrics per event (event detection within a time window) and per-sample metrics.

---

## 11. Implementation notes and configuration

* Keep configuration parameters (contamination, nu, thresholds) in a single YAML/JSON file so you can iterate.
* Keep separate scripts for `generate_synthetic.py`, `train_models.py`, `score.py`, and `drift_monitor.py`.
* Persist models, scalers, and baseline training buffers with versions.
* Log all model decisions and flagged timestamps for human review.

Suggested hyperparameters (starting point):

* IsolationForest n_estimators=200, contamination=0.01
* LOF n_neighbors=20, novelty=True, contamination=0.01
* OneClassSVM nu=0.02, kernel=rbf
* EWMA alpha=0.005, Page-Hinkley delta=0.0005, threshold tuned from synthetic runs

---

## 12. Next steps and recommended experiments

1. Run the updated notebook to produce synthetic normals & degradations and evaluate detectors.
2. Tune contamination/nu per-model, and choose health thresholds that balance false alarms vs missed events.
3. Add realistic environment modeling (ambient temp, supply voltage) to augment robustness.
4. If possible later: collect a short duration of blade-attached data (even a single motor run for 60–120s) to anchor the upper-load behaviour.
5. Create a small human-in-the-loop workflow: every flagged event is reviewed and labelled; over time you will accumulate labeled failures and can move to semi-supervised / supervised approaches.

---

## Appendix: quick formulas & snippets

**Motor simplified relation:**

```
I = (V_supply - V_bemf - V_losses) / R_winding
V_bemf = K_e * RPM
R_winding(T) = R0 * (1 + alpha*(T - T0))
```

**Ambient air density (approx):**

```
rho ≈ p / (R_specific * T)
```

(so density decreases as temperature increases; aerodynamic load decreases in hot air)

---

If you want, I can export this document as a standalone `.md` file or add it as an introduction section at the top of the Jupyter notebook. I can also insert parameterized code snippets for each synthetic mode and environment model into the notebook so you can run controlled experiments.
