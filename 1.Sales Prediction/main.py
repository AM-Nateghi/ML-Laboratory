import warnings

warnings.filterwarnings("ignore")

from time import time
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import Optional, Union
from fastapi.responses import FileResponse
import uvicorn
import os

filename = "sales_kmeans_models.pkl"


def get_cluster_label_manual(x_reduced, cluster_centers, eps):
    # x_reduced: داده جدید بعد از pca، shape=(1, n_features)
    # cluster_centers: لیست مراکز خوشه‌ها، shape=(n_clusters, n_features)
    # eps: آستانه فاصله برای نویز
    distances = np.linalg.norm(cluster_centers - x_reduced, axis=1)
    min_dist = np.min(distances)
    label = np.argmin(distances)
    if min_dist > eps:
        return -1  # نویز
    return label


print("Loading models...")

try:
    # بارگذاری artifact ذخیره شده
    artifact = joblib.load(
        os.path.join(os.path.dirname(__file__), "saved_models", filename)
    )
    scaler = artifact["scaler"]
    pca = artifact["pca"]
    cluster_model = artifact["cluster_model"]
    models = artifact["models"]
    label_encoders = artifact["label_encoders"]
    print("✅ Models loaded successfully")
except FileNotFoundError:
    print(f"❌ Error: {filename} not found!")
    exit(1)
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

app = FastAPI(title="Sales Prediction API", description="سیستم پیش‌بینی فروش")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class InputData(BaseModel):
    Customers: Union[int, float, str]
    Promo: Union[bool, str]
    StateHoliday: Union[bool, str]
    SchoolHoliday: Union[bool, str]
    StoreType: Union[int, str]
    Assortment: Union[int, str]

    HasCompetition: Union[bool, str]
    CompetitionDistance: Optional[Union[float, str]] = 0
    CompetitionOpenSinceMonth: Optional[Union[float, str]] = 0
    CompetitionOpenSinceYear: Optional[Union[float, str]] = 0

    Promo2: Union[bool, str]
    Promo2SinceWeek: Optional[Union[float, str]] = 0
    Promo2SinceYear: Optional[Union[float, str]] = 0
    PromoInterval: Optional[Union[float, str]] = 0

    month: Union[int, float, str]
    year: Union[int, float, str]

    @validator("Customers", pre=True)
    def validate_customers(cls, v):
        try:
            customers = float(v) if v != "" else None
            if customers is None or customers < 20 or customers > 6000:
                raise ValueError("تعداد مشتریان باید بین 20 تا 6000 باشد")
            return customers
        except (ValueError, TypeError):
            raise ValueError("تعداد مشتریان باید عدد معتبری باشد")

    @validator("StoreType", pre=True)
    def validate_store_type(cls, v):
        try:
            store_type = int(v) if v != "" else None
            if store_type is None or store_type < 0 or store_type > 3:
                raise ValueError("نوع فروشگاه باید بین 0 تا 3 باشد")
            return store_type
        except (ValueError, TypeError):
            raise ValueError("نوع فروشگاه باید عدد معتبری باشد")

    @validator("Assortment", pre=True)
    def validate_assortment(cls, v):
        try:
            assortment = int(v) if v != "" else None
            if assortment is None or assortment < 0 or assortment > 2:
                raise ValueError("تنوع کالا باید بین 0 تا 2 باشد")
            return assortment
        except (ValueError, TypeError):
            raise ValueError("تنوع کالا باید عدد معتبری باشد")

    @validator("month", pre=True)
    def validate_month(cls, v):
        try:
            month = int(v) if v != "" else None
            if month is None or month < 1 or month > 12:
                raise ValueError("ماه باید بین 1 تا 12 باشد")
            return month
        except (ValueError, TypeError):
            raise ValueError("ماه باید عدد معتبری باشد")

    @validator("year", pre=True)
    def validate_year(cls, v):
        try:
            year = int(v) if v != "" else None
            if year is None or year < 10 or year > 30:
                raise ValueError("سال باید بین 10 تا 30 باشد")
            return year
        except (ValueError, TypeError):
            raise ValueError("سال باید عدد معتبری باشد")

    @validator(
        "Promo", "StateHoliday", "SchoolHoliday", "HasCompetition", "Promo2", pre=True
    )
    def validate_boolean_fields(cls, v):
        if isinstance(v, str):
            if v.lower() == "true":
                return True
            elif v.lower() == "false":
                return False
            elif v == "on":
                return True
        return bool(v)

    @validator(
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
        pre=True,
    )
    def validate_optional_numeric(cls, v):
        if v == "" or v is None:
            return 0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0


@app.get("/")
def read_index():
    return FileResponse(os.path.join(static_dir, "sales_prediction_ui.html"))


@app.post("/predict")
def predict(data: InputData):
    try:
        # تبدیل داده‌ها به فرمت مناسب
        input_array = np.array(
            [
                float(data.Customers),
                int(data.Promo),
                int(data.StateHoliday),
                int(data.SchoolHoliday),
                int(data.StoreType),
                int(data.Assortment),
                float(data.CompetitionDistance),
                float(data.CompetitionOpenSinceMonth),
                float(data.CompetitionOpenSinceYear),
                int(data.Promo2),
                float(data.Promo2SinceWeek),
                float(data.Promo2SinceYear),
                float(data.PromoInterval),
                float(data.month),
                float(data.year),
                int(data.HasCompetition),
            ]
        ).reshape(1, -1)

        # بررسی شرایط اجباری
        if data.HasCompetition and (
            data.CompetitionDistance == 0
            or data.CompetitionOpenSinceMonth == 0
            or data.CompetitionOpenSinceYear == 0
        ):
            return {"error": "در صورت وجود رقیب، تمام فیلدهای مربوطه را پر کنید"}

        if data.Promo2 and (
            data.Promo2SinceWeek == 0
            or data.Promo2SinceYear == 0
            or data.PromoInterval == 0
        ):
            return {
                "error": "در کدهای پروژه شما ساختار خوبی دارند و مراحل استاندارد یادگیری ماشین (پیش‌پرصورت فعال بودن پروموشن پیشرفته، تمام فیلدهای مربوطه را پر کنید"
            }

        # پیش‌پردازش مشابه آموزش
        x_scaled = scaler.transform(input_array)
        x_reduced = pca.transform(x_scaled)

        # دسته‌بندی دستی
        # cluster_label = get_cluster_label_manual(x_reduced, artifact["cluster_centers"], artifact["cluster_model"].eps)
        cluster_label = cluster_model.predict(x_reduced)[0]  # kmeans

        if cluster_label == -1:
            return {
                "error": "داده شما در دسته نویز قرار گرفته و قابل پیش‌بینی نیست",
                "cluster": -1,
            }

        if cluster_label not in models:
            return {
                "error": f"مدل برای خوشه {cluster_label} موجود نیست",
                "cluster": int(cluster_label),
            }

        # پیش‌بینی
        prediction = models[cluster_label].predict(x_scaled)[0]

        return {
            "success": True,
            "cluster": int(cluster_label),
            "prediction": float(prediction),
            "formatted_prediction": f"{prediction:,.0f}",
            "message": "پیش‌بینی با موفقیت انجام شد",
        }

    except ValueError as ve:
        return {"error": str(ve)}
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": f"خطای غیرمنتظره: {str(e)}"}


# اجرای سرور
if __name__ == "__main__":
    print("🚀 setup server...")
    print("📍 Address: http://localhost:2007")
    print("📁 Static files: /static")
    print("🔄 To stop: Ctrl+C")

    uvicorn.run("main:app", host="127.0.0.1", port=2007, reload=True, log_level="info")
