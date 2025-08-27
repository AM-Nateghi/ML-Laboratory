/**
 * Sales Prediction Form Handler
 * تسکریپت فرم پیش‌بینی فروش
 */

class SalesPredictionForm {
    constructor() {
        this.form = $("#predictForm");
        this.loadingElement = $("#loading");
        this.resultElement = $("#result");
        this.competitionFields = $("#competitionFields");
        this.promo2Fields = $("#promo2Fields");

        this.initializeEventHandlers();
    }

    /**
     * راه‌اندازی event handlerها
     */
    initializeEventHandlers() {
        // Toggle conditional fields
        $("#HasCompetition").on("change", (e) => {
            this.toggleConditionalFields(this.competitionFields, e.target.checked);
        });

        $("#Promo2").on("change", (e) => {
            this.toggleConditionalFields(this.promo2Fields, e.target.checked);
        });

        // Form submission
        this.form.on("submit", (e) => {
            e.preventDefault();
            this.handleFormSubmission();
        });
    }

    /**
     * نمایش/مخفی کردن فیلدهای شرطی
     */
    toggleConditionalFields(fieldContainer, isVisible) {
        if (isVisible) {
            fieldContainer.addClass("show");
        } else {
            fieldContainer.removeClass("show");
            // پاک کردن مقادیر فیلدها هنگام مخفی شدن
            fieldContainer.find("input").val("");
        }
    }

    /**
     * اعتبارسنجی فرم
     */
    validateForm() {
        const validationResults = {
            isValid: true,
            errors: {}
        };

        // پاک کردن خطاهای قبلی
        this.clearAllErrors();

        // اعتبارسنجی تعداد مشتریان
        const customers = this.getInputValue("Customers");
        if (!this.validateCustomers(customers)) {
            validationResults.isValid = false;
            validationResults.errors.Customers = "تعداد مشتریان باید بین 20 تا 6000 باشد";
        }

        // اعتبارسنجی نوع فروشگاه
        const storeType = this.getSelectValue("StoreType");
        if (!this.validateStoreType(storeType)) {
            validationResults.isValid = false;
            validationResults.errors.StoreType = "لطفاً نوع فروشگاه را انتخاب کنید";
        }

        // اعتبارسنجی تنوع کالا
        const assortment = this.getSelectValue("Assortment");
        if (!this.validateAssortment(assortment)) {
            validationResults.isValid = false;
            validationResults.errors.Assortment = "لطفاً تنوع کالا را انتخاب کنید";
        }

        // اعتبارسنجی فیلدهای رقبا
        if (this.isCheckboxChecked("HasCompetition")) {
            const competitionValidation = this.validateCompetitionFields();
            if (!competitionValidation.isValid) {
                validationResults.isValid = false;
                validationResults.errors.Competition = competitionValidation.error;
            }
        }

        // اعتبارسنجی فیلدهای پروموشن پیشرفته
        if (this.isCheckboxChecked("Promo2")) {
            const promo2Validation = this.validatePromo2Fields();
            if (!promo2Validation.isValid) {
                validationResults.isValid = false;
                validationResults.errors.Promo2 = promo2Validation.error;
            }
        }

        // اعتبارسنجی تاریخ
        const dateValidation = this.validateDateFields();
        if (!dateValidation.isValid) {
            validationResults.isValid = false;
            validationResults.errors.Date = dateValidation.error;
        }

        return validationResults;
    }

    /**
     * اعتبارسنجی تعداد مشتریان
     */
    validateCustomers(customers) {
        return customers && customers >= 20 && customers <= 6000;
    }

    /**
     * اعتبارسنجی نوع فروشگاه
     */
    validateStoreType(storeType) {
        return storeType !== "" && storeType !== null;
    }

    /**
     * اعتبارسنجی تنوع کالا
     */
    validateAssortment(assortment) {
        return assortment !== "" && assortment !== null;
    }

    /**
     * اعتبارسنجی فیلدهای رقبا
     */
    validateCompetitionFields() {
        const requiredFields = [
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear"
        ];

        for (const field of requiredFields) {
            if (!this.getInputValue(field)) {
                return {
                    isValid: false,
                    error: "لطفاً تمام فیلدهای رقبا را پر کنید"
                };
            }
        }

        return { isValid: true };
    }

    /**
     * اعتبارسنجی فیلدهای پروموشن پیشرفته
     */
    validatePromo2Fields() {
        const requiredFields = [
            "Promo2SinceWeek",
            "Promo2SinceYear",
            "PromoInterval"
        ];

        for (const field of requiredFields) {
            if (!this.getInputValue(field)) {
                return {
                    isValid: false,
                    error: "لطفاً تمام فیلدهای پروموشن پیشرفته را پر کنید"
                };
            }
        }

        return { isValid: true };
    }

    /**
     * اعتبارسنجی فیلدهای تاریخ
     */
    validateDateFields() {
        const month = this.getInputValue("month");
        const year = this.getInputValue("year");

        if (!month || !year || month < 1 || month > 12) {
            return {
                isValid: false,
                error: "ماه باید بین 1 تا 12 باشد و سال را وارد کنید"
            };
        }

        return { isValid: true };
    }

    /**
     * آماده‌سازی داده‌ها برای ارسال
     */
    prepareFormData() {
        const formArray = this.form.serializeArray();
        const formData = {};

        // تبدیل داده‌های فرم
        formArray.forEach(item => {
            formData[item.name] = this.convertValue(item.value);
        });

        // اضافه کردن checkbox های unchecked
        this.addUncheckedCheckboxes(formData);

        return formData;
    }

    /**
     * تبدیل مقادیر به نوع صحیح
     */
    convertValue(value) {
        // اگر checkbox است
        if (value === "on") {
            return true;
        }

        // اگر boolean string است
        if (value === "true") return true;
        if (value === "false") return false;

        // اگر عدد است
        if (!isNaN(value) && value !== "") {
            return Number(value);
        }

        // در غیر این صورت string برگردان
        return value;
    }

    /**
     * اضافه کردن checkbox های unchecked
     */
    addUncheckedCheckboxes(formData) {
        const checkboxes = ["HasCompetition", "Promo2"];

        checkboxes.forEach(checkboxName => {
            if (!(checkboxName in formData)) {
                formData[checkboxName] = false;
            }
        });
    }

    /**
     * ارسال درخواست به سرور
     */
    async sendPredictionRequest(data) {
        return new Promise((resolve, reject) => {
            $.ajax({
                url: "/predict",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(data),
                timeout: 10000, // 10 seconds timeout
                success: resolve,
                error: reject
            });
        });
    }

    /**
     * مدیریت ارسال فرم
     */
    async handleFormSubmission() {
        // اعتبارسنجی
        const validation = this.validateForm();

        if (!validation.isValid) {
            this.displayValidationErrors(validation.errors);
            return;
        }

        // نمایش loading
        this.showLoading();

        try {
            // آماده‌سازی داده‌ها
            const formData = this.prepareFormData();

            // ارسال درخواست
            const response = await this.sendPredictionRequest(formData);

            // نمایش نتیجه
            this.displayResult(response);

        } catch (error) {
            console.error("Prediction error:", error);
            this.displayError("خطا در ارسال درخواست. لطفاً دوباره تلاش کنید.");
        } finally {
            this.hideLoading();
        }
    }

    /**
     * نمایش خطاهای اعتبارسنجی
     */
    displayValidationErrors(errors) {
        Object.keys(errors).forEach(errorKey => {
            this.showErrorMessage(`err_${errorKey}`, errors[errorKey]);
        });
    }

    /**
     * نمایش نتیجه موفقیت‌آمیز
     */
    displayResult(response) {
        if (response.error) {
            this.displayError(response.error);
            return;
        }

        const resultHtml = this.buildSuccessResultHTML(response);
        this.resultElement
            .removeClass("error")
            .addClass("show")
            .html(resultHtml);
    }

    /**
     * ساخت HTML نتیجه موفق
     */
    buildSuccessResultHTML(response) {
        return `
            <div style="margin-bottom: 1rem;">
                <strong>✅ نتیجه پیش‌بینی:</strong>
            </div>
            <div style="font-size: 1.3rem; color: #22543d;">
                📈 میزان فروش پیش‌بینی شده: 
                <strong>${this.formatNumber(response.prediction)}</strong>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #4a5568;">
                🎯 خوشه تشخیص داده شده: ${response.cluster}
            </div>
        `;
    }

    /**
     * نمایش پیام خطا
     */
    displayError(errorMessage) {
        this.resultElement
            .addClass("error show")
            .html(`❌ ${errorMessage}`);
    }

    /**
     * Helper Methods
     */
    getInputValue(name) {
        return $(`input[name='${name}']`).val();
    }

    getSelectValue(name) {
        return $(`select[name='${name}']`).val();
    }

    isCheckboxChecked(name) {
        return $(`#${name}`).is(":checked");
    }

    showErrorMessage(elementId, message) {
        $(`#${elementId}`).addClass("show").text(message);
    }

    clearAllErrors() {
        $(".error-message").removeClass("show").text("");
        this.resultElement.removeClass("show error");
    }

    showLoading() {
        this.loadingElement.show();
    }

    hideLoading() {
        this.loadingElement.hide();
    }

    formatNumber(num) {
        return num.toLocaleString('fa-IR');
    }
}

// راه‌اندازی هنگام آماده شدن صفحه
$(document).ready(() => {
    new SalesPredictionForm();
    console.log("🚀 Sales Prediction Form initialized");
});