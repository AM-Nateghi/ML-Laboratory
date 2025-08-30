# 📧 Spam Email Detection Project

این پروژه یک سیستم تشخیص ایمیل اسپم با استفاده از TF-IDF vectorization و مدل‌های مختلف یادگیری ماشین پیاده‌سازی می‌کند.

## ویژگی‌ها

- **پیش پردازش متن**: تمیزکاری، توکن‌سازی و stemming
- **TF-IDF Vectorization**: استفاده از sklearn برای تبدیل متن به بردار
- **مدل‌های متعدد**: مقایسه RandomForest، GradientBoosting و NaiveBayes
- **ارزیابی جامع**: محاسبه Jaccard Score، Accuracy و سایر متریک‌ها

## مدل‌های استفاده شده

1. **RandomForest**: مدل ensemble با درختان تصمیم‌گیری متعدد
2. **GradientBoosting**: مدل boosting برای بهبود تدریجی عملکرد
3. **NaiveBayes**: مدل احتمالاتی مناسب برای طبقه‌بندی متن

## معیارهای ارزیابی

- **Jaccard Score**: معیار اصلی برای انتخاب بهترین مدل
- **Accuracy Score**: دقت کلی مدل
- **F1 Score**: ترکیب precision و recall
- **Confusion Matrix**: تجزیه و تحلیل دقیق خطاها

## نحوه استفاده

1. اطمینان حاصل کنید که تمام کتابخانه‌های لازم نصب شده‌اند:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn nltk jupyter
   ```

2. فایل `train.ipynb` را در Jupyter Notebook اجرا کنید

3. نتایج شامل مقایسه مدل‌ها و انتخاب بهترین مدل نمایش داده می‌شود

## ساختار دیتاست

دیتاست شامل 10,000 ایمیل با فرمت زیر است:
- `id`: شناسه یکتا
- `email`: متن کامل ایمیل
- `label`: برچسب (ham/spam)

## نتایج

مدل‌ها با دقت بالایی عمل می‌کنند و بهترین مدل بر اساس Jaccard Score انتخاب می‌شود.

---

## English Summary

This project implements a spam email detection system using TF-IDF vectorization and multiple machine learning models including RandomForest, GradientBoosting, and NaiveBayes. The system achieves high accuracy in classifying emails as spam or ham, with comprehensive evaluation using Jaccard Score as the primary metric.