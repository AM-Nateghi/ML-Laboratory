$(document).ready(function () {
    // Toggle conditional fields
    $("#HasCompetition").change(function () {
        const fields = $("#competitionFields");
        if (this.checked) {
            fields.addClass("show");
        } else {
            fields.removeClass("show");
        }
    });

    $("#Promo2").change(function () {
        const fields = $("#promo2Fields");
        if (this.checked) {
            fields.addClass("show");
        } else {
            fields.removeClass("show");
        }
    });

    // Form validation and submission
    $("#predictForm").submit(function (e) {
        e.preventDefault();
        let valid = true;

        // Reset errors
        $(".error-message").removeClass("show").text("");
        $("#result").removeClass("show error");

        // Validation
        const customers = $("input[name='Customers']").val();
        if (!customers || customers < 20 || customers > 6000) {
            $("#err_Customers")
                .addClass("show")
                .text("تعداد مشتریان باید بین 20 تا 6000 باشد");
            valid = false;
        }

        const storeType = $("select[name='StoreType']").val();
        if (!storeType) {
            $("#err_StoreType")
                .addClass("show")
                .text("لطفاً نوع فروشگاه را انتخاب کنید");
            valid = false;
        }

        const assortment = $("select[name='Assortment']").val();
        if (!assortment) {
            $("#err_Assortment")
                .addClass("show")
                .text("لطفاً تنوع کالا را انتخاب کنید");
            valid = false;
        }

        if ($("#HasCompetition").is(":checked")) {
            const compFields = [
                "CompetitionDistance",
                "CompetitionOpenSinceMonth",
                "CompetitionOpenSinceYear",
            ];
            for (let field of compFields) {
                if (!$(`input[name='${field}']`).val()) {
                    $("#err_Competition")
                        .addClass("show")
                        .text("لطفا تمام فیلدهای رقبا را پر کنید");
                    valid = false;
                    break;
                }
            }
        }

        if ($("#Promo2").is(":checked")) {
            const promo2Fields = [
                "Promo2SinceWeek",
                "Promo2SinceYear",
                "PromoInterval",
            ];
            for (let field of promo2Fields) {
                if (!$(`input[name='${field}']`).val()) {
                    $("#err_Promo2")
                        .addClass("show")
                        .text("لطفا تمام فیلدهای پروموشن پیشرفته را پر کنید");
                    valid = false;
                    break;
                }
            }
        }

        const month = $("input[name='month']").val();
        const year = $("input[name='year']").val();
        if (!month || !year || month < 1 || month > 12) {
            $("#err_Date")
                .addClass("show")
                .text("ماه باید بین 1 تا 12 باشد و سال را وارد کنید");
            valid = false;
        }

        if (!valid) return;

        // Show loading
        $("#loading").show();

        // Prepare data
        const formData = $(this).serializeArray();
        const data = {};
        formData.forEach((item) => {
            data[item.name] =
                item.value === "on"
                    ? true
                    : !isNaN(item.value) && item.value !== ""
                        ? Number(item.value)
                        : item.value === "true"
                            ? true
                            : item.value === "false"
                                ? false
                                : item.value;
        });

        // Send request
        $.ajax({
            url: "/predict",
            method: "POST",
            contentType: "application/json",
            data: JSON.stringify(data),
            success: function (response) {
                $("#loading").hide();
                const result = $("#result");

                if (response.error) {
                    result.addClass("error show");
                    result.html(`❌ ${response.error}`);
                } else {
                    result.removeClass("error").addClass("show");
                    result.html(`
                  <div style="margin-bottom: 1rem;">
                    <strong>✅ نتیجه پیش‌بینی:</strong>
                  </div>
                  <div style="font-size: 1.3rem; color: #22543d;">
                    📈 میزان فروش پیش‌بینی شده: <strong>${response.prediction.toLocaleString()}</strong>
                  </div>
                  <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #4a5568;">
                    🎯 خوشه تشخیص داده شده: ${response.cluster}
                  </div>
                `);
                }
            },
            error: function () {
                $("#loading").hide();
                $("#result")
                    .addClass("error show")
                    .html("❌ خطا در ارسال درخواست");
            },
        });
    });
});