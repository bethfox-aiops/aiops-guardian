import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("[INFO] Loading collected data...")

df = pd.read_csv("aiops_data/metrics.csv")

print(f"[INFO] Loaded {len(df)} samples.")

# Drop timestamp
df = df.drop(columns=["timestamp"])

# Select features
features = ['disk','cpu','mem','net_kbps','disk_w_kbps','gpu_util','gpu_mem_mib','gpu_temp_c']

X = df[features]

print(f"[INFO] Training on columns: {features}")

print("[INFO] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("[INFO] Training Isolation Forest anomaly detector...")

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)

print("[INFO] Training complete.")

joblib.dump(model, "iforest_model.pkl")
joblib.dump(scaler, "iforest_scaler.pkl")

print("[INFO] Saved model to: iforest_model.pkl")
print("[INFO] Saved scaler to: iforest_scaler.pkl")
print("[INFO] Done.")
