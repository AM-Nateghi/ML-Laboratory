import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# بارگذاری مدل‌ها
spectral_artifact = joblib.load("saved_models/sales_spectral_models.pkl")
kmeans_artifact = joblib.load("saved_models/sales_kmeans_models.pkl")

# بارگذاری داده تست
# فرض بر این است که فایل RoS_test.csv مشابه ساختار train است
TEST_PATH = "RoS_train.csv"
df_test = pd.read_csv(TEST_PATH).drop("Open", axis=1).sample(n=10000, random_state=42, axis=0)

# پرکردن داده‌های گمشده و type optimization
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


def better_nan_fill(data):
    for col in data.select_dtypes(include=["category", "object"]).columns:
        mode = data[col].mode()[0]
        data[col] = data[col].fillna(mode)
    for col in data.select_dtypes(include=["float", "int", "Float32"]).columns:
        median = data[col].median()
        data[col] = data[col].fillna(median)
    return data


cdn_condition = df_test.CompetitionDistance.isna()
df_test.loc[cdn_condition, "HasCompetition"] = 0
df_test.loc[~cdn_condition, "HasCompetition"] = 1
df_test["HasCompetition"] = df_test["HasCompetition"].astype("category")
df_test.loc[df_test["HasCompetition"] == 0, "CompetitionDistance"] = 0

df_test = better_nan_fill(df_test)
df_test[categorical_cols] = df_test[categorical_cols].astype("category")
for col in [
    "Sales",
    "CompetitionOpenSinceYear",
    "CompetitionDistance",
    "Promo2SinceWeek",
    "Customers",
]:
    df_test[col] = df_test[col].astype("Float32")

X_test = df_test.drop("Sales", axis=1)
y_true = df_test["Sales"]

# Label Encoding
for col, le in spectral_artifact["label_encoders"].items():
    X_test[col] = le.transform(X_test[col])

# Scaling & PCA
scaler = spectral_artifact["scaler"]
pca = spectral_artifact["pca"]
X_scaled = scaler.transform(X_test.values)
X_reduced = pca.transform(X_scaled)

# تست مدل SpectralForest
cluster_model = spectral_artifact["cluster_model"]
labels = cluster_model.fit_predict(X_reduced)
preds = np.full_like(y_true, np.nan, dtype=np.float32)
for l, model in spectral_artifact["models"].items():
    idx = labels == l
    if np.any(idx):
        preds[idx] = model.predict(X_scaled[idx])
mask = ~np.isnan(preds)
print(f"spectral R2 score: {r2_score(y_true[mask], preds[mask]):.3f}")

# تست مدل KMeans
kmeans_model = kmeans_artifact["cluster_model"]
kmeans_labels = kmeans_model.predict(X_reduced)
preds_kmeans = np.full_like(y_true, np.nan, dtype=np.float32)
for l, model in kmeans_artifact["models"].items():
    idx = kmeans_labels == l
    if np.any(idx):
        preds_kmeans[idx] = model.predict(X_scaled[idx])
mask_kmeans = ~np.isnan(preds_kmeans)
print(
    f"KMeans R2 score: {r2_score(y_true[mask_kmeans], preds_kmeans[mask_kmeans]):.3f}"
)
