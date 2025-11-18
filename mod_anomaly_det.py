"""
Motor Anomaly Detection — Complete Jupyter Notebook
--------------------------------------------------
Run this notebook (or save as .py and run) to:
 - Load seed data from your SQLite .db
 - Exploratory Data Analysis
 - Generate realistic synthetic normal data (multiple modes)
 - Generate synthetic degraded datasets for evaluation
 - Feature engineering
 - Train an ensemble of unsupervised detectors (IsolationForest, LOF, OneClassSVM)
 - Score datasets and compute health score
 - Drift detection (EWMA + Page-Hinkley) and re-baselining policy skeleton
 - Save outputs and plots

IMPORTANT:
 - Update DB_PATH and TABLE at the top to point to your file
 - This notebook is designed to run without additional hardware
 - It is parameterized for flexibility; adjust hyperparameters as needed

"""

# CONFIG -----------------------------------------------------------------
DB_PATH = "motor-data.db"     # path to your sqlite database
TABLE = "motor_data"          # table name
OUT_DIR = "notebook_outputs"  # output directory for csvs/plots
SYNTH_MULT = 50                # how many times to expand seed data
RANDOM_SEED = 42
INJECT_ANOMALIES = False       # keep False for production-style; True if you want injected anoms for tests
ANOMALY_FRACTION = 0.02        # fraction used for anomaly injection when enabled
WINDOW_SECONDS = 60            # window length for health scoring (conceptual)

# Imports ----------------------------------------------------------------
import os, sqlite3, math
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from scipy import stats

np.random.seed(RANDOM_SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# Utility functions ------------------------------------------------------
def load_table(db_path, table):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
    conn.close()
    # detect timestamp-like column
    if 'timestamp' not in df.columns:
        for c in df.columns:
            s = df[c].astype(str)
            if s.str.contains('T').any() or s.str.match('\\d{4}-\\d{2}-\\d{2}').any():
                df = df.rename(columns={c: 'timestamp'})
                break
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# Quick EDA ---------------------------------------------------------------
print("Loading DB:", DB_PATH)
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"DB file not found: {DB_PATH}")

df_seed = load_table(DB_PATH, TABLE)
print("Columns:", df_seed.columns.tolist())
# canonicalize expected names
for col in ['speed_setting','current','bemf','inductance','resistance']:
    if col not in df_seed.columns:
        for c in df_seed.columns:
            if c.lower() == col:
                df_seed = df_seed.rename(columns={c: col})
                break
# ensure numeric
for col in ['speed_setting','current','bemf','inductance','resistance']:
    if col in df_seed.columns:
        df_seed[col] = pd.to_numeric(df_seed[col], errors='coerce')

print('Seed rows:', len(df_seed))
print(df_seed.describe(include='all'))

# plot basics
plt.figure(figsize=(10,3))
plt.plot(df_seed['timestamp'], df_seed['current'], marker='o')
plt.title('Seed: current over time')
plt.xlabel('timestamp'); plt.ylabel('current')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'seed_current_time.png'))
print('Saved seed_current_time.png')

plt.figure(figsize=(6,4))
plt.scatter(df_seed['speed_setting'], df_seed['current'])
plt.xlabel('speed_setting'); plt.ylabel('current'); plt.title('current vs speed')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'seed_current_vs_speed.png'))
print('Saved seed_current_vs_speed.png')

# Fit cluster empirical stats --------------------------------------------
cluster_stats = {}
for k, g in df_seed.groupby('speed_setting'):
    cluster_stats[k] = {
        'n': len(g),
        'current_mean': float(g['current'].mean()),
        'current_std': float(max(g['current'].std(ddof=0), 1e-6)),
        'bemf_mean': float(g['bemf'].mean()),
        'bemf_std': float(max(g['bemf'].std(ddof=0), 1e-6))
    }

print('Cluster stats:')
for k,v in cluster_stats.items():
    print(k, v)

# Synthetic normal generation --------------------------------------------
def generate_synthetic_normal(seed_df, cluster_stats, multiplier=50, last_time_pad_secs=1):
    keys = sorted(cluster_stats.keys())
    weights = np.array([cluster_stats[k]['n'] for k in keys], dtype=float)
    weights /= weights.sum()
    out = []
    last_time = seed_df['timestamp'].max() + pd.Timedelta(seconds=last_time_pad_secs)
    total = len(seed_df) * multiplier
    for i in range(total):
        k = np.random.choice(keys, p=weights)
        st = cluster_stats[k]
        cur = np.random.normal(loc=st['current_mean'], scale=st['current_std'])
        bem = np.random.normal(loc=st['bemf_mean'], scale=st['bemf_std'])
        # time interval variability
        delta_t = np.random.exponential(scale=10)
        last_time = last_time + pd.Timedelta(seconds=delta_t)
        # occasional short ramp sequences
        if np.random.rand() < 0.02:
            ramp_len = np.random.randint(2,6)
            for j in range(ramp_len):
                cur2 = cur * (1 + np.random.uniform(-0.05, 0.15) + 0.02*j)
                bem2 = bem * (1 + np.random.uniform(-0.02, 0.02))
                out.append({'timestamp': last_time + pd.Timedelta(seconds=j), 'speed_setting': k,
                            'current': max(0, float(cur2)), 'bemf': float(bem2),
                            'inductance': cluster_stats[k].get('inductance_mean', np.nan),
                            'resistance': cluster_stats[k].get('resistance_mean', np.nan)})
            last_time = last_time + pd.Timedelta(seconds=ramp_len)
            continue
        out.append({'timestamp': last_time, 'speed_setting': k, 'current': max(0, float(cur)), 'bemf': float(bem),
                    'inductance': cluster_stats[k].get('inductance_mean', np.nan),
                    'resistance': cluster_stats[k].get('resistance_mean', np.nan)})
    return pd.DataFrame(out)

print('Generating synthetic normal...')
df_synth_normal = generate_synthetic_normal(df_seed, cluster_stats, multiplier=SYNTH_MULT)
df_aug = pd.concat([df_seed, df_synth_normal], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
print('Augmented dataset size:', len(df_aug))
df_aug.to_csv(os.path.join(OUT_DIR,'augmented_normal.csv'), index=False)
print('Saved augmented_normal.csv')

# Environmental modeling & synthetic modes --------------------------------
# We'll build functions to simulate temperature & env effects and create degraded modes

def make_ambient_temp_series(base_ts, length, base_temp=25.0, daily_amp=3.0, noise_std=0.5):
    # generate a simple temperature series with slow cycles + noise
    # length = number of timestamps
    t = np.arange(length)
    # daily cycle at arbitrary period (simulate slow changes)
    temp = base_temp + daily_amp * np.sin(2 * np.pi * t / max(1, int(length/10)))
    temp = temp + np.random.randn(length) * noise_std
    return temp

# Motor/environment interaction model (simplified)
ALPHA_RESIST_TEMP = 0.0039  # per degC change in copper resistivity
BETA_BEMF_TEMP = 0.0008     # per degC effect on bemf scale (small)

def apply_environmental_effects(df_in, ambient_temp):
    df = df_in.copy().reset_index(drop=True)
    # assume ambient_temp aligned with df rows
    # R increases -> small effect; magnet weakening reduces bemf slightly
    df['ambient_temp'] = ambient_temp
    df['R_factor'] = 1 + ALPHA_RESIST_TEMP * (df['ambient_temp'] - 25.0)
    df['bemf_factor'] = 1 - BETA_BEMF_TEMP * (df['ambient_temp'] - 25.0)
    # apply multipliers (conservative)
    df['current_env'] = df['current'] * df['R_factor'] * (1 + np.random.normal(0, 0.01, size=len(df)))
    df['bemf_env'] = df['bemf'] * df['bemf_factor'] * (1 + np.random.normal(0, 0.005, size=len(df)))
    return df

# Generate degraded scenarios --------------------------------------------
def simulate_degraded(df_normal, pattern='gradual', degradation_amount=0.4, duration_seconds=3600):
    df = df_normal.copy().sort_values('timestamp').reset_index(drop=True)
    start_ts = df['timestamp'].min()
    df['sec_from_start'] = (df['timestamp'] - start_ts).dt.total_seconds().clip(lower=0)
    total_seconds = duration_seconds
    frac = (df['sec_from_start'] / total_seconds).clip(0,1)

    if pattern == 'gradual':
        df['current'] = df['current'] * (1 + frac * degradation_amount)
        df['bemf'] = df['bemf'] * (1 - frac * degradation_amount * 0.8)
    elif pattern == 'step':
        df.loc[frac > 0.5, 'current'] = df.loc[frac > 0.5, 'current'] * (1 + degradation_amount)
        df.loc[frac > 0.5, 'bemf'] = df.loc[frac > 0.5, 'bemf'] * (1 - degradation_amount * 0.8)
    elif pattern == 'spikes':
        df['current'] = df['current'] * (1 + 0.01*np.random.randn(len(df)))
        spikes = np.random.choice(len(df), size=max(1,int(0.01*len(df))), replace=False)
        df.loc[spikes, 'current'] = df.loc[spikes, 'current'] * np.random.uniform(3,8,len(spikes))
    elif pattern == 'sensor_bias':
        df['current'] = df['current'] + 0.005 * (1 + 0.5*np.sin(2*np.pi*df['sec_from_start']/3600))
    return df.drop(columns=['sec_from_start'])

# Generate ambient temp aligned with augmented normal
print('Applying environmental model to a portion of data (for example)')
# sample a subset for env modelling to keep compute small
sample_df = df_aug.sample(n=min(1000, len(df_aug)), random_state=RANDOM_SEED).sort_values('timestamp').reset_index(drop=True)
ambient = make_ambient_temp_series(sample_df['timestamp'], len(sample_df), base_temp=25.0, daily_amp=5.0)
sample_env = apply_environmental_effects(sample_df, ambient)
sample_env.to_csv(os.path.join(OUT_DIR,'sample_env_effects.csv'), index=False)
print('Saved sample_env_effects.csv')

# Create degraded eval sets ------------------------------------------------
df_deg_gradual = simulate_degraded(df_aug.copy(), pattern='gradual', degradation_amount=0.4, duration_seconds=3600)
df_deg_step = simulate_degraded(df_aug.copy(), pattern='step', degradation_amount=0.6, duration_seconds=1800)
df_deg_spikes = simulate_degraded(df_aug.copy(), pattern='spikes')
print('Created degraded sets')

# Optional injection of small labeled anomalies for validation --------------
if INJECT_ANOMALIES:
    n_inject = max(1,int(len(df_aug)*ANOMALY_FRACTION))
    idxs = np.random.choice(len(df_aug), size=n_inject, replace=False)
    df_aug['is_injected'] = False
    for i in idxs:
        if np.random.rand() < 0.5:
            df_aug.at[i,'current'] = df_aug.at[i,'current'] * np.random.uniform(3,6)
        else:
            df_aug.at[i,'bemf'] = df_aug.at[i,'bemf'] * np.random.uniform(0.2,0.6)
        df_aug.at[i,'is_injected'] = True
    df_aug.to_csv(os.path.join(OUT_DIR,'augmented_with_injected.csv'), index=False)

# Feature engineering ------------------------------------------------------
def enrich_features(df):
    d = df.sort_values('timestamp').reset_index(drop=True).copy()
    d['delta_current'] = d['current'].diff().fillna(0)
    d['delta_bemf'] = d['bemf'].diff().fillna(0)
    d['rm_current'] = d['current'].rolling(3, min_periods=1).mean()
    d['rs_current'] = d['current'].rolling(3, min_periods=1).std().fillna(0)
    # time since last speed change
    d['prev_speed'] = d['speed_setting'].shift(1).fillna(d['speed_setting'])
    d['time_since_speed_change'] = d['timestamp'].diff().dt.total_seconds().fillna(0)
    d['speed_change'] = (d['speed_setting'] != d['prev_speed']).astype(int)
    features = ['current','bemf','speed_setting','delta_current','delta_bemf','rm_current','rs_current']
    return d, features

print('Engineering features on training data...')
df_train_enriched, features = enrich_features(df_aug)
X = df_train_enriched[features].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('Feature matrix shape:', X_scaled.shape)

# Train models --------------------------------------------------------------
print('Training models...')
iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=RANDOM_SEED)
iso.fit(X_scaled)

# LOF with novelty=True requires sklearn >= 0.22 for .decision_function; use novelty=True and fit on normal data
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
lof.fit(X_scaled)

ocsvm = OneClassSVM(nu=0.02, kernel='rbf', gamma='scale')
ocsvm.fit(X_scaled)

models = {'iso': iso, 'lof': lof, 'ocsvm': ocsvm}
print('Models trained')

# Scoring function ---------------------------------------------------------
def score_df(df_to_score, scaler, models, features):
    d, _ = enrich_features(df_to_score)
    X = d[features].fillna(0).values
    Xs = scaler.transform(X)
    out = d[['timestamp','speed_setting','current','bemf']].copy()
    out['iso_score'] = models['iso'].decision_function(Xs)
    out['iso_anom'] = models['iso'].predict(Xs) == -1
    out['lof_score'] = -models['lof'].decision_function(Xs)  # higher => more anomalous
    out['lof_anom'] = models['lof'].predict(Xs) == -1
    out['ocsvm_anom'] = models['ocsvm'].predict(Xs) == -1
    out['anom_votes'] = out[['iso_anom','lof_anom','ocsvm_anom']].sum(axis=1)
    # normalize iso_score to 0..1
    iso_min, iso_max = out['iso_score'].min(), out['iso_score'].max()
    out['iso_norm'] = (out['iso_score'] - iso_min) / (iso_max - iso_min + 1e-9)
    out['health'] = 0.7 * out['iso_norm'] + 0.3 * (1 - (out['anom_votes'] / 3.0))
    return out

# Score training(normal) and degraded sets
print('Scoring normal and degraded examples...')
scored_train = score_df(df_aug, scaler, models, features)
scored_gradual = score_df(df_deg_gradual, scaler, models, features)
scored_step = score_df(df_deg_step, scaler, models, features)
scored_spikes = score_df(df_deg_spikes, scaler, models, features)

scored_train.to_csv(os.path.join(OUT_DIR,'scored_train.csv'), index=False)
scored_gradual.to_csv(os.path.join(OUT_DIR,'scored_gradual.csv'), index=False)
scored_step.to_csv(os.path.join(OUT_DIR,'scored_step.csv'), index=False)
scored_spikes.to_csv(os.path.join(OUT_DIR,'scored_spikes.csv'), index=False)
print('Saved scored CSVs')

# Simple evaluation metrics (for synthetic degraded sets) ------------------
def eval_scored(scored, label):
    print('\n---', label, '---')
    frac_flagged = (scored[['iso_anom','lof_anom','ocsvm_anom']].any(axis=1).sum()) / len(scored)
    print('Fraction rows flagged (any):', frac_flagged)
    print('Health mean / min:', scored['health'].mean(), scored['health'].min())
    # quick plot
    plt.figure(figsize=(10,3))
    plt.plot(scored['timestamp'], scored['health'], label='health')
    plt.ylim(-0.1,1.1)
    plt.title(f'Health over time: {label}'); plt.xlabel('time'); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,f'health_{label}.png'))
    print('Saved', os.path.join(OUT_DIR,f'health_{label}.png'))

for df_s, lab in [(scored_train,'train_normal'), (scored_gradual,'gradual'), (scored_step,'step'), (scored_spikes,'spikes')]:
    eval_scored(df_s, lab)

# Drift detection ----------------------------------------------------------
def ewma(series, alpha=0.005):
    s = series.reset_index(drop=True).astype(float)
    out = np.zeros(len(s))
    out[0] = s.iloc[0]
    for i in range(1,len(s)):
        out[i] = alpha * s.iloc[i] + (1-alpha) * out[i-1]
    return out

def page_hinkley(series, delta=0.0005, threshold=0.03):
    mT = 0.0
    detected = np.zeros(len(series), dtype=bool)
    mean = 0.0
    for i, x in enumerate(series):
        mean = mean + (x - mean) / (i+1)
        mT = mT + x - mean - delta
        if mT > threshold:
            detected[i] = True
            mT = 0
    return detected

# apply PH to health of gradual case
h = scored_gradual['health'].reset_index(drop=True)
detected = page_hinkley(h.values)
plt.figure(figsize=(10,3))
plt.plot(scored_gradual['timestamp'], h, label='health')
plt.scatter(scored_gradual['timestamp'][detected], h[detected], color='red', label='PH detect')
plt.title('Drift detection on gradual degradation'); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'ph_gradual.png'))
print('Saved ph_gradual.png')

# Re-baselining policy (sketch & helper)
# - Maintain a rolling baseline buffer of non-anomalous points
# - If PH detects slow drift AND anomaly_ratio in buffer is low, append recent good data to baseline and retrain

class BaselineManager:
    def __init__(self, models, scaler, features, buffer_limit=10000):
        self.models = models
        self.scaler = scaler
        self.features = features
        self.buffer = pd.DataFrame()
        self.buffer_limit = buffer_limit
    def add_good_data(self, df_good):
        self.buffer = pd.concat([self.buffer, df_good], ignore_index=True)
        if len(self.buffer) > self.buffer_limit:
            self.buffer = self.buffer.iloc[-self.buffer_limit:]
    def retrain(self):
        if len(self.buffer) < 50:
            print('Not enough buffer to retrain')
            return
        print('Retraining models on buffer size', len(self.buffer))
        d, _ = enrich_features(self.buffer)
        X = d[self.features].fillna(0).values
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.models['iso'] = IsolationForest(n_estimators=200, contamination=0.01, random_state=RANDOM_SEED)
        self.models['iso'].fit(Xs)
        self.models['lof'] = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
        self.models['lof'].fit(Xs)
        self.models['ocsvm'] = OneClassSVM(nu=0.02, kernel='rbf', gamma='scale')
        self.models['ocsvm'].fit(Xs)

# Example: initialize manager with initial normal data
bm = BaselineManager(models, scaler, features)
# add initial high-confidence normal points (e.g., low anom_votes)
good_initial = scored_train[scored_train['anom_votes'] == 0][['timestamp','speed_setting','current','bemf']]
bm.add_good_data(good_initial.sample(n=min(len(good_initial), 1000), random_state=RANDOM_SEED))
print('Baseline buffer size after init:', len(bm.buffer))

# Save models and scalers (simple pickle)
import pickle
with open(os.path.join(OUT_DIR,'models_scaler.pkl'), 'wb') as f:
    pickle.dump({'models':models, 'scaler':scaler, 'features':features}, f)
print('Saved models_scaler.pkl')

# Wrap-up -----------------------------------------------------------------
print('\nNotebook complete. Outputs saved to', OUT_DIR)
print('Key files: augmented_normal.csv, scored_*.csv, health_*.png, models_scaler.pkl')

# End of notebook
