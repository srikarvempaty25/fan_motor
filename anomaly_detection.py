# Motor anomaly detection notebook
# Save this file and run it as a Jupyter notebook or as a script.
# It will:
# 1) Load your motor_data.db (motor_status / motor_data table)
# 2) Inspect and visualize the real 60 rows you provided
# 3) Generate more *normal* synthetic rows (matched to the three clusters seen: 0,25,50)
# 4) Optionally inject a small number of labeled anomalies for evaluation
# 5) Create derived features and visualizations
# 6) Train several unsupervised anomaly detectors: Z-score, IsolationForest, LocalOutlierFactor, OneClassSVM
# 7) Compare their outputs and save flagged rows + plots

# -----------------------
# CONFIG: change DB_PATH to your .db file
DB_PATH = "motor-data.db"   # change to the path of your SQLite DB
TABLE = "motor_data"       # table name in your DB
OUT_PREFIX = "motor_anom_"
SYNTHETIC_MULTIPLIER = 10   # how many times to expand the dataset (e.g., 10 -> ~600 rows total)
INJECT_ANOMALIES = True
ANOMALY_FRACTION = 0.03    # fraction of dataset to inject as anomalies for testing
RANDOM_SEED = 42
# -----------------------

# Imports
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os

np.random.seed(RANDOM_SEED)

# Utility functions

def load_table(db_path, table):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
    conn.close()
    return df


def ensure_timestamp(df):
    # find and rename timestamp-like column
    if 'timestamp' not in df.columns:
        for c in df.columns:
            if df[c].astype(str).str.contains('T').any():
                df = df.rename(columns={c: 'timestamp'})
                break
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


# 1) Load data
print("Loading DB ->", DB_PATH)
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"DB file not found: {DB_PATH}. Put the .db next to this notebook and update DB_PATH.")

df = load_table(DB_PATH, TABLE)
print("Columns:", df.columns.tolist())
df = ensure_timestamp(df)

# canonical column names
for col in ['speed_setting','current','bemf','inductance','resistance']:
    if col not in df.columns:
        for c in df.columns:
            if c.lower() == col:
                df = df.rename(columns={c: col})
                break

# cast numeric
for col in ['speed_setting','current','bemf','inductance','resistance']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(df.head())
print(f"Loaded {len(df)} rows")

# 2) Quick EDA
plt.figure(figsize=(10,3))
plt.plot(df['timestamp'], df['current'], marker='.', linestyle='-', label='current')
plt.title('Current (firmware units) over time')
plt.xlabel('time'); plt.ylabel('current'); plt.tight_layout(); plt.savefig(OUT_PREFIX+'current_time.png')
print('Saved', OUT_PREFIX+'current_time.png')

plt.figure(figsize=(8,4))
plt.scatter(df['speed_setting'], df['current'])
plt.xlabel('speed_setting'); plt.ylabel('current'); plt.title('current vs speed')
plt.tight_layout(); plt.savefig(OUT_PREFIX+'current_vs_speed.png')
print('Saved', OUT_PREFIX+'current_vs_speed.png')

# show basic stats
print(df[['speed_setting','current','bemf']].describe())

# 3) Generate synthetic normal-like rows
# We'll model the dataset as 3 clusters (0,25,50) and draw noisy samples from each cluster's empirical distribution.
base = df.copy()
clusters = base.groupby('speed_setting')
cluster_params = {}
for k, g in clusters:
    cluster_params[k] = {
        'n': len(g),
        'current_mean': g['current'].mean(),
        'current_std': g['current'].std(ddof=0) if len(g)>1 else 1e-6,
        'bemf_mean': g['bemf'].mean(),
        'bemf_std': g['bemf'].std(ddof=0) if len(g)>1 else 1e-6,
        'inductance_mean': g['inductance'].mean() if 'inductance' in g else np.nan,
        'resistance_mean': g['resistance'].mean() if 'resistance' in g else np.nan
    }

print('Cluster params:')
for k,v in cluster_params.items():
    print(k, v)

# Build synthetic dataset
total_original = len(base)
target = int(total_original * SYNTHETIC_MULTIPLIER)
needed = max(0, target - total_original)
print(f'Expanding from {total_original} to {target} rows; creating {needed} synthetic rows')

synth_rows = []
cluster_keys = sorted(cluster_params.keys())
# produce samples proportionally to original cluster sizes
cluster_weights = np.array([cluster_params[k]['n'] for k in cluster_keys], dtype=float)
cluster_weights /= cluster_weights.sum()

for _ in range(needed):
    k = np.random.choice(cluster_keys, p=cluster_weights)
    p = cluster_params[k]
    cur = np.random.normal(loc=p['current_mean'], scale=max(1e-6, p['current_std']))
    bem = np.random.normal(loc=p['bemf_mean'], scale=max(1e-6, p['bemf_std']))
    row = {
        'timestamp': base['timestamp'].max() + pd.Timedelta(seconds=np.random.uniform(1,3600)),
        'speed_setting': k,
        'current': float(max(0, cur)),
        'bemf': float(bem),
        'inductance': p['inductance_mean'],
        'resistance': p['resistance_mean']
    }
    synth_rows.append(row)

synth_df = pd.DataFrame(synth_rows)
if len(synth_df):
    df_aug = pd.concat([base, synth_df], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
else:
    df_aug = base.copy()

print('Augmented dataset size:', len(df_aug))

df_aug.to_csv(OUT_PREFIX+'augmented.csv', index=False)
print('Saved', OUT_PREFIX+'augmented.csv')

# 4) Optionally inject anomalies for evaluation
if INJECT_ANOMALIES:
    n_anom = max(1, int(len(df_aug) * ANOMALY_FRACTION))
    print('Injecting', n_anom, 'anomalies')
    anom_idx = np.random.choice(len(df_aug), size=n_anom, replace=False)
    df_aug['is_injected_anom'] = False
    for idx in anom_idx:
        # create different anomaly types randomly
        typ = np.random.choice(['high_current','low_bemf','discordant'])
        if typ == 'high_current':
            df_aug.at[idx, 'current'] = df_aug.at[idx, 'current'] * np.random.uniform(3,8) + 0.01
        elif typ == 'low_bemf':
            df_aug.at[idx, 'bemf'] = df_aug.at[idx, 'bemf'] * np.random.uniform(0.2,0.6)
        else:
            # mismatch between speed and bemf/current
            df_aug.at[idx, 'speed_setting'] = np.random.choice(cluster_keys)
            df_aug.at[idx, 'current'] = df_aug.at[idx, 'current'] * np.random.uniform(0.1,0.3)
            df_aug.at[idx, 'bemf'] = df_aug.at[idx, 'bemf'] * np.random.uniform(1.5,3)
        df_aug.at[idx, 'is_injected_anom'] = True
    df_aug.to_csv(OUT_PREFIX+'augmented_with_injected_anoms.csv', index=False)
    print('Saved', OUT_PREFIX+'augmented_with_injected_anoms.csv')
else:
    df_aug['is_injected_anom'] = False

# 5) Feature engineering
df2 = df_aug.copy()
# sort by timestamp
if 'timestamp' in df2.columns:
    df2 = df2.sort_values('timestamp').reset_index(drop=True)
# differences and rolling stats
df2['delta_current'] = df2['current'].diff().fillna(0)
df2['delta_bemf'] = df2['bemf'].diff().fillna(0)
df2['on'] = (df2['speed_setting'] > 0).astype(int)
# rolling 3-point mean/std
df2['rm_current'] = df2['current'].rolling(3, min_periods=1).mean()
 df2['rs_current'] = df2['current'].rolling(3, min_periods=1).std().fillna(0)

# fill NaNs
for c in ['delta_current','delta_bemf','rm_current','rs_current']:
    df2[c] = df2[c].fillna(0)

# 6) Prepare feature matrix
features = ['current','bemf','speed_setting','delta_current','delta_bemf','rm_current','rs_current']
X = df2[features].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7) Train unsupervised detectors
results = pd.DataFrame(index=df2.index)

# Z-score univariate (current and bemf)
df2['current_z'] = stats.zscore(df2['current'].fillna(0))
df2['bemf_z'] = stats.zscore(df2['bemf'].fillna(0))
df2['anom_z'] = (df2['current_z'].abs() > 3) | (df2['bemf_z'].abs() > 3)
results['anom_z'] = df2['anom_z']

# Isolation Forest
iso = IsolationForest(n_estimators=200, contamination=ANOMALY_FRACTION if INJECT_ANOMALIES else 0.05, random_state=RANDOM_SEED)
iso.fit(X_scaled)
iso_scores = iso.decision_function(X_scaled)
iso_pred = iso.predict(X_scaled) == -1
results['iso_score'] = iso_scores
results['anom_iso'] = iso_pred

# Local Outlier Factor (unsupervised) - note: fit_predict returns -1 for outliers
lof = LocalOutlierFactor(n_neighbors=20, contamination=ANOMALY_FRACTION if INJECT_ANOMALIES else 0.05)
lof_pred = lof.fit_predict(X_scaled)
results['anom_lof'] = lof_pred == -1

# One Class SVM
ocsvm = OneClassSVM(gamma='scale', nu=0.05)
ocsvm.fit(X_scaled)
oc_pred = ocsvm.predict(X_scaled) == -1
results['anom_ocsvm'] = oc_pred

# union of anomalies
results['any_anom'] = results[['anom_z','anom_iso','anom_lof','anom_ocsvm']].any(axis=1)

# attach results to df2
df2 = pd.concat([df2.reset_index(drop=True), results.reset_index(drop=True)], axis=1)

# 8) Save and report
df2.to_csv(OUT_PREFIX+'with_anom_scores.csv', index=False)
print('Saved', OUT_PREFIX+'with_anom_scores.csv')

# summary counts
print('Counts:')
print(results.sum())

# 9) Visualizations
plt.figure(figsize=(12,4))
plt.plot(df2['timestamp'], df2['current'], label='current')
plt.scatter(df2[df2['any_anom']]['timestamp'], df2[df2['any_anom']]['current'], color='red', label='anomaly')
plt.title('Current with anomalies flagged')
plt.legend(); plt.tight_layout(); plt.savefig(OUT_PREFIX+'current_flagged.png')
print('Saved', OUT_PREFIX+'current_flagged.png')

plt.figure(figsize=(8,4))
plt.scatter(df2['bemf'], df2['current'], c=df2['any_anom'].map({True: 'red', False: 'blue'}))
plt.xlabel('bemf'); plt.ylabel('current'); plt.title('bemf vs current (anomalies red)')
plt.tight_layout(); plt.savefig(OUT_PREFIX+'bemf_vs_current_flagged.png')
print('Saved', OUT_PREFIX+'bemf_vs_current_flagged.png')

# 10) Show injected anomaly examples (if any)
if INJECT_ANOMALIES:
    print('Injected anomalies example rows:')
    print(df2[df2['is_injected_anom']].head())

print('Notebook run complete. Outputs saved with prefix', OUT_PREFIX)
