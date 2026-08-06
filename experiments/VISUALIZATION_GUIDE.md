# Visualization Guide - راهنمای نمودارها

راهنمای کامل برای تولید و تفسیر نمودارهای پروژه

## 📊 انواع نمودارها

### 1. Learning Curves (منحنی‌های یادگیری)

**هدف:** نمایش پیشرفت agent در طول آموزش

**Metrics:**

- **Reward:** پاداش تجمعی در هر episode
- **Episode Length:** تعداد گام‌ها تا پایان
- **Success Rate:** درصد موفقیت در رسیدن به هدف

**تفسیر:**

- شیب تند → یادگیری سریع
- همگرایی → رسیدن به سیاست بهینه
- نوسانات زیاد → exploration بالا یا محیط stochastic
- Plateau → رسیدن به حد بهینه یا گیر کردن در local optimum

**فایل خروجی:**

- `learning_curves_reward.png`
- `learning_curves_length.png`
- `learning_curves_success.png`

---

### 2. Value Heatmap (نقشه حرارتی ارزش)

**هدف:** نمایش تابع ارزش V(s) یا max Q(s,a) برای هر state

**رنگ‌بندی:**

- 🔴 قرمز → ارزش بالا (نزدیک به هدف)
- 🟡 زرد → ارزش متوسط
- 🟢 سبز → ارزش متوسط
- 🔵 آبی → ارزش پایین (دور از هدف)
- ⬛ سیاه → دیوار (غیرقابل عبور)

**تفسیر:**

- Gradient روان → یادگیری خوب
- مقادیر ناگهانی تغییر → ممکن است مشکل باشد
- ارزش بالا در goal → درست ✓
- ارزش بالا در penalty → اشتباه ✗

**فایل خروجی:**

- `heatmap_value_iteration.png`
- `heatmap_q_learning.png`
- `heatmap_sarsa_lambda.png`

---

### 3. Policy Map (نقشه سیاست)

**هدف:** نمایش بهترین action در هر state

**فلش‌ها:**

- ⬆️ → Up
- ⬇️ → Down
- ⬅️ → Left
- ➡️ → Right

**Markers:**

- 🟢 دایره سبز → Start
- 🟡 دایره زرد → Key
- 🟣 دایره بنفش → Door
- ⭐ ستاره قرمز → Goal

**تفسیر:**

- فلش‌ها به سمت goal → سیاست منطقی
- فلش‌های متناقض → سیاست suboptimal
- حلقه‌های فلش → ممکن است loop باشد

**فایل خروجی:**

- `policy_value_iteration.png`
- `policy_q_learning.png`
- `policy_sarsa_lambda.png`

---

### 4. Comparison Chart (نمودار مقایسه)

**هدف:** مقایسه عملکرد نهایی الگوریتم‌ها

**Metrics:**

- Success Rate (%)
- Mean Reward
- Mean Episode Length

**تفسیر:**

- Success Rate بالاتر → بهتر
- Reward بالاتر → بهتر
- Length کمتر → کارآمدتر

**مقایسه الگوریتم‌ها:**

- **Value Iteration:** سریع‌تر، نیاز به model
- **Q-Learning:** بدون نیاز به model، off-policy
- **SARSA(λ):** on-policy، با eligibility traces

**فایل خروجی:**

- `comparison_algorithms.png`

---

### 5. Convergence Analysis (تحلیل همگرایی)

**هدف:** بررسی سرعت و پایداری همگرایی

**دو پلات:**

1. **Smoothed Rewards:** منحنی‌های هموار شده
2. **Final Performance:** میانگین 100 episode آخر با error bars

**تفسیر:**

- Convergence سریع‌تر → الگوریتم کارآمدتر
- Error bars کوچک → پایدارتر
- Final performance بالا → سیاست بهتر

**فایل خروجی:**

- `convergence_analysis.png`

---

### 6. Transfer Learning Plots (نمودارهای انتقال یادگیری)

**هدف:** تحلیل موفقیت انتقال دانش

**4 پلات:**

1. **Initial Performance (Similar):** عملکرد اولیه در target مشابه
2. **Final Performance (Similar):** عملکرد نهایی در target مشابه
3. **Learning Speed (Similar):** سرعت یادگیری
4. **Similar vs Different:** مقایسه دو نوع target

**استراتژی‌ها:**

- **Scratch:** آموزش از صفر (baseline)
- **Full:** انتقال کامل Q-table
- **Scaled:** Q × β (مثلاً β=0.5)
- **Selective:** انتقال انتخابی بر اساس neighborhood

**تفسیر:**

- Initial performance بالاتر → transfer موفق
- Learning speed کمتر → transfer مفید
- Similar بهتر از Different → طبیعی ✓
- Full بهتر از Scaled → انتقال مثبت
- Scratch بهتر از Full → انتقال منفی ✗

**فایل خروجی:**

- `transfer_learning.png`

---

### 7. Visit Frequency Map (نقشه فرکانس بازدید)

**هدف:** نمایش تعداد بازدید از هر state

**رنگ‌بندی:**

- 🟨 زرد روشن → بازدید کم
- 🟧 نارنجی → بازدید متوسط
- 🟥 قرمز تیره → بازدید زیاد

**تفسیر:**

- بازدید یکنواخت → exploration خوب
- تمرکز در ناحیه خاص → exploitation
- states نزدیک goal بازدید بیشتر → طبیعی
- states دور از goal بازدید کم → ممکن است underfitting باشد

**فایل خروجی:**

- `visit_frequency.png`

---

## 🔧 نحوه استفاده

### تولید تمام نمودارها:

```bash
# با داده‌های واقعی (اگر موجود باشد)
python experiments/generate_plots.py

# با داده‌های نمونه (demo)
python experiments/demo_plots.py
```

### تولید نمودار خاص:

```python
from experiments.visualize import Visualizer

viz = Visualizer(save_dir="results/figures")

# Learning curves
viz.plot_learning_curves(agents_data, metric='reward',
                         save_name='my_learning_curves.png')

# Value heatmap
viz.plot_value_heatmap(value_map, maze,
                      title='My Heatmap',
                      save_name='my_heatmap.png')

# Policy map
viz.plot_policy_map(policy_map, maze, metadata,
                   title='My Policy',
                   save_name='my_policy.png')

# Comparison
viz.plot_comparison_bar(comparison_data,
                       metrics=['success_rate', 'mean_reward'],
                       save_name='my_comparison.png')

# Convergence
viz.plot_convergence_analysis(convergence_data,
                             save_name='my_convergence.png')

# Transfer learning
viz.plot_transfer_learning(transfer_data,
                          save_name='my_transfer.png')
```

---

## 📁 ساختار داده‌ها

### Learning Curves Data:

```python
agents_data = {
    'Agent Name': {
        'episodes': [0, 1, 2, ...],
        'reward': [-100, -95, -80, ...],
        'length': [500, 450, 320, ...],
        'success_rate': [0.0, 0.1, 0.5, ...]
    }
}
```

### Value Map:

```python
value_map = {
    (x, y): value,
    (0, 0): -50.5,
    (1, 2): 10.3,
    ...
}
```

### Policy Map:

```python
policy_map = {
    (x, y): action,  # 0=Up, 1=Down, 2=Left, 3=Right
    (0, 0): 3,  # Right
    (1, 2): 0,  # Up
    ...
}
```

### Comparison Data:

```python
comparison_data = {
    'Agent Name': {
        'success_rate': 0.85,
        'mean_reward': -25.3,
        'mean_length': 120.5
    }
}
```

### Transfer Learning Data:

```python
transfer_data = {
    'similar': {
        'strategy_name': {
            'initial_success_rate': 0.3,
            'final_success_rate': 0.85,
            'learning_speed': 200.0
        }
    },
    'different': {
        ...
    }
}
```

---

## 🎨 سفارشی‌سازی

### تغییر اندازه figure:

```python
viz = Visualizer()
viz.fig_size = (12, 8)  # عرض، ارتفاع
viz.dpi = 150  # کیفیت بالاتر
```

### تغییر style:

```python
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # یا 'seaborn', 'bmh', etc.
```

### تغییر colormap:

```python
# در کد visualize.py
im = ax.imshow(value_grid, cmap='viridis')  # یا 'plasma', 'coolwarm'
```

---

## 📊 نمودارهای پیشنهادی برای گزارش

### 1. Section: محیط و تعریف مسئله

- نمایش maze با markers
- Policy map (یکی از الگوریتم‌ها)

### 2. Section: پیاده‌سازی الگوریتم‌ها

- Value heatmap (هر 3 الگوریتم)
- Policy map (هر 3 الگوریتم)

### 3. Section: نتایج و مقایسه

- Learning curves (reward)
- Comparison bar chart
- Convergence analysis

### 4. Section: Transfer Learning

- Transfer learning plots (4 subplots)
- Initial vs Final performance

### 5. Section: تحلیل و بحث

- Visit frequency map
- Learning curves (success rate)

---

## 💡 نکات مهم

1. **Smoothing:** منحنی‌های خام نوسان دارند، از smoothing استفاده کنید
2. **Scale:** محورها را مناسب scale کنید
3. **Labels:** همیشه label، title و legend اضافه کنید
4. **Resolution:** برای گزارش DPI=150 یا بیشتر
5. **Format:** PNG برای گزارش، PDF برای چاپ
6. **Color Blind:** از رنگ‌های مناسب برای color blind استفاده کنید

---

## 🔍 تفسیر نتایج

### الگوی خوب:

- ✅ Convergence به reward بالا
- ✅ Success rate بالا (>80%)
- ✅ Episode length کاهش یابد
- ✅ Value gradient روان
- ✅ Policy منطقی به سمت goal

### الگوی مشکل‌دار:

- ❌ Reward convergence نکند
- ❌ Success rate پایین (<50%)
- ❌ نوسانات شدید
- ❌ Value map غیرمنطقی
- ❌ Policy loops یا contradictions

---

## 📚 منابع

- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/
- RL Visualization Best Practices
- Color Theory for Data Visualization

---