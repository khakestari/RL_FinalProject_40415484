# RL_FinalProject_40415484

## پروژه پایانی یادگیری تقویتی: طراحی و تحلیل عامل هوشمند در هزارتوی پویا

**شماره دانشجویی:** 40415484  
**Seed:** 8 (رقم یکی مانده به آخر)  
**اندازه نقشه:** 15×15 (محاسبه: 15 + (8 mod 4) = 15 + 0 = 15)

---

## 📋 خلاصه پروژه

این پروژه شامل طراحی یک محیط هزارتوی پویا و پیاده‌سازی سه الگوریتم یادگیری تقویتی است:
- **Value Iteration** (model-based)
- **Q-Learning** (model-free, off-policy)
- **SARSA(λ)** (model-free, on-policy با Eligibility Trace)

همچنین شامل بخش انتقال یادگیری (Transfer Learning) و رابط گرافیکی تعاملی می‌باشد.

---

## 🗂️ ساختار پروژه

```
RL_FinalProject_40415484/
├── environments/          # محیط هزارتو
│   ├── maze.py           # کلاس اصلی محیط
│   ├── generator.py      # تولید نقشه
│   └── maps/             # نقشه‌های ذخیره شده
├── agents/               # الگوریتم‌های RL
│   ├── value_iteration.py
│   ├── q_learning.py
│   └── sarsa_lambda.py
├── transfer/             # انتقال یادگیری
│   └── transfer_learning.py
├── gui/                  # رابط گرافیکی
│   ├── app.py
│   └── renderer.py
├── experiments/          # آزمایش‌ها و تحلیل
│   ├── run_experiments.py
│   ├── analysis.py
│   └── configs/
├── results/              # نتایج
│   ├── raw_data/        # داده‌های خام CSV
│   ├── models/          # مدل‌های ذخیره شده
│   ├── figures/         # نمودارها
│   └── videos/          # ویدیوها
├── tests/                # تست‌های واحد
├── report.pdf            # گزارش نهایی
├── requirements.txt      # وابستگی‌ها
├── README.md             # این فایل
└── main.py               # نقطه ورود اصلی
```

---

## 🚀 نصب و راه‌اندازی

### پیش‌نیازها
- Python 3.8 یا بالاتر
- pip

### نصب وابستگی‌ها
```bash
pip install -r requirements.txt
```

---

## 💻 نحوه اجرا

### اجرای رابط گرافیکی (GUI)
```bash
python main.py
# یا
python gui/app.py
```

**قابلیت‌های GUI:**
- نمایش بصری هزارتو و عامل
- انتخاب الگوریتم (VI, Q-Learning, SARSA)
- بارگذاری مدل‌های آموزش‌دیده
- کنترل‌های Play/Pause/Reset/Step
- تنظیم سرعت اجرا
- نمایش value heatmap و policy arrows
- آمار لحظه‌ای (reward, steps, success rate)

### اجرای Transfer Learning
```bash
# Quick demo
python transfer/quick_demo_transfer.py

# Full experiment
python transfer/transfer_learning.py
```

### اجرای آزمایش‌ها
```bash
python experiments/run_experiments.py
```

### اجرای تست‌ها
```bash
pytest tests/
```

---

## 🎯 ویژگی‌های محیط

- **اندازه:** 15×15
- **Seed:** 8
- **عناصر:**
  - دیوارها (حداقل 15% از خانه‌ها)
  - خانه‌های جریمه (حداقل 5 خانه)
  - نقطه شروع
  - کلید
  - در (باز/بسته)
  - هدف
- **قابلیت اضافی:** [در حال تکمیل]
- **احتمال انتقال:**
  - 0.8: حرکت در جهت انتخابی
  - 0.1: انحراف به راست
  - 0.1: انحراف به چپ

---

## 📊 نتایج

### فاز 1-6 تکمیل شد ✅
- **محیط هزارتو:** 18 تست ✓
- **Value Iteration:** 12 تست ✓
- **Q-Learning:** 14 تست ✓
- **SARSA(λ):** 15 تست ✓
- **مقایسه الگوریتم‌ها:** 8 تست ✓
- **Transfer Learning:** 16 تست ✓

**مجموع:** 90 تست پاس شده 🎉

### Transfer Learning
- **سناریوها:** Scratch, Full Transfer, Scaled Transfer (β=0.25,0.5,0.75), Selective Transfer
- **هدف‌ها:** Similar Target (15-20% تغییر), Different Target (35%+ تغییر)
- **معیارها:** Initial performance, Learning speed, Final performance, Negative transfer analysis

---

## 📖 مستندات

برای جزئیات کامل پیاده‌سازی و تحلیل‌ها، فایل `report.pdf` را مطالعه کنید.

---

## 📝 یادداشت‌ها

- تمام نتایج قابل بازتولید هستند
- داده‌های خام در `results/raw_data/` ذخیره می‌شوند
- مدل‌های آموزش دیده در `results/models/` قابل دسترسی هستند

---

## 🔗 منابع

[منابع استفاده شده در پروژه]

---

**تاریخ آخرین به‌روزرسانی:** [تاریخ]
