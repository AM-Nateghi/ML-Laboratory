import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, silhouette_score
from kneed import KneeLocator
from hdbscan import HDBSCAN

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
df.loc[df["HasCompetition"] == 0, "CompetitionDistance"] = 0


# ===== بهتر: پر کردن داده‌های گمشده =====
def better_nan_fill(data):
    for col in data.select_dtypes(include=["category", "object"]).columns:
        mode = data[col].mode()[0]
        data[col] = data[col].fillna(mode)
    for col in data.select_dtypes(include=["float", "int", "Float32"]).columns:
        median = data[col].median()
        data[col] = data[col].fillna(median)
    return data


df = better_nan_fill(df)

# ===== type optimization =====
categorical_cols = [
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "StoreType",
    "Assortment",
    "Promo2",
    "PromoInterval",
    "month",
    "CompetitionOpenSinceMonth",
    "Promo2SinceYear",
    "year",
]
df[categorical_cols] = df[categorical_cols].astype("category")
for col in [
    "Sales",
    "CompetitionOpenSinceYear",
    "CompetitionDistance",
    "Promo2SinceWeek",
    "Customers",
]:
    df[col] = df[col].astype("Float32")
print("--> data was optimized")

# Label Encoding
label_encodes = {}
for col in ["PromoInterval", "StoreType", "Assortment", "StateHoliday"]:
    le = LabelEncoder().fit(df[col])
    df[col] = le.transform(df[col])
    label_encodes[col] = le

# ===== 2. Features/Target split =====
X = df.drop("Sales", axis=1)
y = df["Sales"]

# ===== 3. Scaling & PCA =====
scaler = RobustScaler().fit(X.values)
X_scaled = scaler.transform(X.values)
print("--> data was scaled")

pca = PCA(n_components=8).fit(X_scaled)
X_reduced = pca.transform(X_scaled)
print("--> data was reduced")

# ===== 4b. KMeans Clustering =====
# تعیین تعداد خوشه‌ها با elbow method
print("--> KMeans loading...")
inertia = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_reduced)
    inertia.append(kmeans.inertia_)
# انتخاب تعداد خوشه بهینه (elbow)
knee_kmeans = (
    KneeLocator(K_range, inertia, curve="convex", direction="decreasing").knee or 3
)
print(f"[INFO] KMeans optimal clusters: {knee_kmeans}")
kmeans = KMeans(n_clusters=knee_kmeans, random_state=42)
kmeans_labels = kmeans.fit_predict(X_reduced)
print(f"[INFO] KMeans labels: {np.unique(kmeans_labels)}")

# ===== 4c. Spectral Clustering =====
print("--> HDBSCAN parameter search (using sample)...")
sample_size = min(25000, X_reduced.shape[0])
sample_idx = np.random.default_rng(42).choice(
    X_reduced.shape[0], sample_size, replace=False
)
X_reduced_sample = X_reduced[sample_idx]

print("--> SpectralClustering loading...")
spectral_n_clusters = knee_kmeans  # Use same optimal clusters as KMeans for consistency
spectral = SpectralClustering(
    n_clusters=spectral_n_clusters,
    assign_labels="kmeans",
    random_state=42,
    n_jobs=cores,
    affinity="nearest_neighbors",
)
labels = spectral.fit_predict(X_reduced_sample)
print(f"[INFO] SpectralClustering labels: {np.unique(labels)}")

# emission labels into all of out data
rf_x_train, rf_x_test, rf_y_train, rf_y_test = train_test_split(X_reduced_sample, labels, test_size=0.2, random_state=42)
rf_emission = RandomForestClassifier(max_depth=10, n_jobs=cores, random_state=42)
rf_emission.fit(rf_x_train, rf_y_train)
print(f"[INFO] RandomForestClassifier accuracy: {rf_emission.score(rf_x_test, rf_y_test)}", end="\n\n")
labels = rf_emission.predict(X_reduced)

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

    if hasattr(rf_model, "oob_prediction_"):
        del rf_model.oob_prediction_
    models[l] = rf_model

# Calculate cluster centers (ignore noise label -1)
cluster_centers = np.array(
    [X_reduced[labels == i].mean(axis=0) for i in np.unique(labels) if i != -1]
)

# ===== 5b. Train models per KMeans cluster =====
kmeans_models = {}
for l in np.unique(kmeans_labels):
    idx = kmeans_labels == l
    train_x, test_x, train_y, test_y = train_test_split(
        X_scaled[idx],
        y[idx],
        test_size=0.2,
        random_state=42,
    )
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=cores)
    rf_model.fit(train_x, train_y)
    print(f"[KMeans Cluster {l}] R2 score:", r2_score(test_y, rf_model.predict(test_x)))
    if hasattr(rf_model, "oob_prediction_"):
        del rf_model.oob_prediction_
    kmeans_models[l] = rf_model
cluster_centers_kmeans = kmeans.cluster_centers_


# ===== 6. Save artifacts =====

# ذخیره مدل KMeans
kmeans_artifact = {
    "scaler": scaler,
    "pca": pca,
    "cluster_model": kmeans,
    "models": kmeans_models,
    "label_encoders": label_encodes,
    "cluster_centers": cluster_centers_kmeans,
}
joblib.dump(kmeans_artifact, "sales_kmeans_models.pkl", compress=5)
print("✅ KMeans model and its components saved → sales_kmeans_models.pkl")

# ذخیره مدل SpectralClustering
spectral_artifact = {
    "scaler": scaler,
    "pca": pca,
    "cluster_model": spectral,
    "models": models,
    "label_encoders": label_encodes,
    "cluster_centers": cluster_centers,
}
joblib.dump(spectral_artifact, "sales_spectral_models.pkl", compress=5)
print(
    "✅ SpectralClustering model and its components saved → sales_spectral_models.pkl"
)
