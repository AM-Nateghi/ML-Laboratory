from time import time
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

print("Loading...")

# بارگذاری artifact ذخیره شده
artifact = joblib.load("sales_cluster_models.pkl")
scaler = artifact["scaler"]
pca = artifact["pca"]
dbscan = artifact["cluster_model"]
models = artifact["models"]
label_encoders = artifact["label_encoders"]

print("Loading complete.")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class InputData(BaseModel):
    Customers: float = Field(..., ge=20, le=6000)
    Promo: bool
    StateHoliday: bool
    SchoolHoliday: bool
    StoreType: int = Field(..., ge=0, le=3)
    Assortment: int = Field(..., ge=0, le=2)

    HasCompetition: bool
    CompetitionDistance: Optional[float] = -1
    CompetitionOpenSinceMonth: Optional[float] = -1
    CompetitionOpenSinceYear: Optional[float] = -1

    Promo2: bool
    Promo2SinceWeek: Optional[float] = -1
    Promo2SinceYear: Optional[float] = -1
    PromoInterval: Optional[float] = -1

    month: float
    year: float


@app.get("/")
def read_index():
    return FileResponse("static/sales_prediction_ui.html")


@app.post("/predict")
def predict(data: InputData):
    # آماده‌سازی ورودی برای مدل
    x = np.array(
        [
            data.Customers,
            int(data.Promo),
            int(data.StateHoliday),
            int(data.SchoolHoliday),
            data.StoreType,
            data.Assortment,
            data.CompetitionDistance,
            data.CompetitionOpenSinceMonth,
            data.CompetitionOpenSinceYear,
            int(data.Promo2),
            data.Promo2SinceWeek,
            data.Promo2SinceYear,
            data.PromoInterval,
            data.month,
            data.year,
            int(data.HasCompetition),
        ]
    ).reshape(1, -1)

    # پیش‌پردازش مشابه آموزش
    x_scaled = scaler.transform(x)
    x_reduced = pca.transform(x_scaled)

    # تشخیص خوشه
    cluster_label = dbscan.fit_predict(x_reduced)[0]

    if cluster_label == -1 or cluster_label not in models:
        return {"error": "مدل برای این خوشه موجود نیست یا داده نویز است."}

    prediction = models[cluster_label].predict(x_scaled)[0]
    return {"cluster": int(cluster_label), "prediction": float(prediction)}
