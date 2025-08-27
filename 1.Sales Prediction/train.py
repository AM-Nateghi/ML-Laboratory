import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from kneed import KneeLocator

cores = round(joblib.cpu_count() * 0.9)

# ===== 1. Load & Clean Data =====
df = pd.read_csv("./RoS_train.csv").drop("Open", axis=1)
print("--> File read")

# Promo2 handling
df.loc[df.Promo2 == 0, ["Promo2SinceYear", "Promo2SinceWeek"]] = 0
df.loc[df.Promo2 == 0, "PromoInterval"] = "None"

# Competition flags
cdn_condition = df.CompetitionDistance.isna()
df.loc[cdn_condition, "HasCompetition"] = 0
df.loc[~cdn_condition, "HasCompetition"] = 1
df["HasCompetition"] = df["HasCompetition"].astype("category")
df.loc[df["HasCompetition"] == 0, "CompetitionDistance"] = -1

# Category conversion
categorical_cols = [
    "Promo","StateHoliday","SchoolHoliday","StoreType","Assortment",
    "Promo2","PromoInterval","month","CompetitionOpenSinceMonth",
    "Promo2SinceYear","year"
]
df[categorical_cols] = df[categorical_cols].astype("category")
for col in ["Sales","CompetitionOpenSinceYear",
            "CompetitionDistance","Promo2SinceWeek","Customers"]:
    df[col] = df[col].astype("Float32")
print("--> data was optimized")

# Label Encoding
label_encodes = {}
for col in ["PromoInterval","StoreType","Assortment","StateHoliday"]:
    le = LabelEncoder().fit(df[col])
    df[col] = le.transform(df[col])
    label_encodes[col] = le

# ===== 2. Features/Target split =====
X = df.drop("Sales", axis=1)
y = df["Sales"]

# Fill NaNs in categoricals
def MakeNaNFill(data):
    _cols = data.select_dtypes(include="category").columns
    for c in _cols:
        data[c] = data[c].astype("Float32")
    return data.fillna(-1)

X_filled = MakeNaNFill(X.copy())
print("--> NaNs filled and encoded")

# ===== 3. Scaling & PCA =====
scaler = RobustScaler().fit(X_filled)
X_scaled = scaler.transform(X_filled)
print("--> data was scaled")

pca = PCA(n_components=8).fit(X_scaled)
X_reduced = pca.transform(X_scaled)
print("--> data was reduced")

# ===== 4. DBSCAN Clustering =====
count_samples = X.shape[1] + 1
distances, _ = NearestNeighbors(n_neighbors=count_samples).fit(X_reduced).kneighbors(X_reduced)
distances = np.sort(distances[:, count_samples - 1])

eps_value_index = KneeLocator(
    range(len(distances)),
    distances,
    curve="convex",
    direction="increasing",
).knee
print(f"[INFO] DBSCAN eps value: {distances[eps_value_index]:.4f}")

dbscan = DBSCAN(eps=distances[eps_value_index], min_samples=3)
labels = dbscan.fit_predict(X_reduced)
# Clear attributes that are not needed at inference
if hasattr(dbscan, 'components_'):
    del dbscan.components_
if hasattr(dbscan, 'core_sample_indices_'):
    del dbscan.core_sample_indices_
print(f"[INFO] DBSCAN labels: {np.unique(labels)}")


# ===== 5. Train models per cluster =====
labeled_X = np.column_stack((X_scaled, labels))
labeled_y = np.column_stack((y, labels))
mask = labels != -1
labeled_X, labeled_y = labeled_X[mask], labeled_y[mask]

cluster_labels = np.unique(labels[mask])

models = {}
for l in cluster_labels:
    idx = labeled_X[:, -1] == l
    train_x, test_x, train_y, test_y = train_test_split(
        labeled_X[idx][:, :-1],
        labeled_y[idx][:, 0],
        test_size=0.2,
        random_state=42,
    )
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=cores)
    rf_model.fit(train_x, train_y)
    print(f"[Cluster {l}] R2 score:", r2_score(test_y, rf_model.predict(test_x)))

    if hasattr(rf_model, 'oob_prediction_'):
        del rf_model.oob_prediction_
    models[l] = rf_model

# ===== 6. Save artifacts =====
artifact = {
    "scaler": scaler,
    "pca": pca,
    "cluster_model": dbscan,
    "models": models,
    "label_encoders": label_encodes
}
joblib.dump(artifact, "sales_cluster_models.pkl", compress=5)
print("✅ model and its components saved → sales_cluster_models.pkl")
