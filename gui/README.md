# GUI Module - رابط گرافیکی

رابط کاربری تعاملی برای نمایش و کنترل عامل‌های یادگیری تقویتی در محیط هزارتو.

---

## 📂 فایل‌ها

### `renderer.py`
موتور رندرینگ برای نمایش هزارتو و عامل

**کلاس اصلی:** `MazeRenderer`

**قابلیت‌ها:**
- رسم هزارتو با رنگ‌بندی
- نمایش عامل با انیمیشن
- نمایش value heatmap
- نمایش policy arrows
- نمایش visited cells
- نمایش energy bar

**رنگ‌ها:**
- دیوار: `#2C3E50` (خاکستری تیره)
- مسیر: `#ECF0F1` (روشن)
- عامل: `#E74C3C` (قرمز)
- کلید: `#F1C40F` (زرد)
- در: `#8B4513` (قهوه‌ای)
- هدف: `#2ECC71` (سبز)
- جریمه: `#E67E22` (نارنجی)

### `app.py`
برنامه اصلی GUI با Tkinter

**کلاس اصلی:** `MazeGUI`

**بخش‌های رابط:**
1. **Canvas** - نمایش هزارتو (سمت چپ)
2. **Control Panel** - کنترل‌ها (سمت راست)

**کنترل‌ها:**
- انتخاب الگوریتم (VI, Q-Learning, SARSA)
- بارگذاری مدل آموزش‌دیده
- Play/Pause/Reset
- Single Step
- Speed slider (0.1x - 10x)
- Visualization toggles

### `demo_gui.py`
تست اجزای GUI بدون باز کردن پنجره

---

## 🚀 نحوه استفاده

### راه‌اندازی GUI:

```bash
# روش 1: از طریق main.py
python main.py

# روش 2: مستقیم
python gui/app.py
```

### تست اجزا:
```bash
python gui/demo_gui.py
```

---

## 🎮 راهنمای استفاده

### 1. انتخاب الگوریتم
- Value Iteration
- Q-Learning  
- SARSA(λ)

### 2. بارگذاری مدل
- کلیک روی "Load Model"
- انتخاب فایل `.pkl` از `results/models/`
- مدل‌های پیش‌فرض:
  - `value_iteration.pkl`
  - `q_learning.pkl`
  - `sarsa_lambda.pkl`

### 3. کنترل اجرا

**Play:** 
- شروع اجرای مداوم episodes
- عامل به صورت خودکار حرکت می‌کند

**Pause:**
- توقف اجرا
- امکان تغییر تنظیمات

**Reset:**
- ریست محیط
- شروع episode جدید

**Single Step:**
- اجرای یک گام
- مناسب برای دیباگ و مشاهده دقیق

**Speed:**
- کنترل سرعت اجرا (0.1x تا 10x)
- 1x = یک گام در ثانیه

### 4. گزینه‌های نمایش

**Show visited cells:**
- نمایش خانه‌های بازدید شده (خاکستری)

**Show value heatmap:**
- نمایش مقادیر value function
- آبی (پایین) → سبز (متوسط) → قرمز (بالا)
- مقادیر عددی در هر خانه

**Show policy arrows:**
- نمایش فلش‌های سیاست
- جهت بهینه حرکت در هر خانه

### 5. اطلاعات نمایش داده شده

**Episode Info:**
- شماره episode
- تعداد گام‌ها
- پاداش تجمعی
- وضعیت (Ready/Running/Success/Failed)

**Statistics:**
- تعداد کل episodes
- نرخ موفقیت (درصد)

---

## 🎨 طراحی رابط

```
┌─────────────────────────────────────────────────────┐
│  RL Maze Environment - Student ID: 40415484         │
├──────────────────────┬──────────────────────────────┤
│                      │  Algorithm                   │
│                      │  ○ Value Iteration           │
│                      │  ● Q-Learning                │
│     MAZE CANVAS      │  ○ SARSA(λ)                  │
│                      │  [Load Model] model.pkl      │
│    (600x600 px)      │                              │
│                      │  Controls                    │
│   رسم هزارتو          │  [Play] [Pause] [Reset]     │
│   با رنگ‌بندی         │  [Single Step]               │
│                      │  Speed: [======•===] 1.0x    │
│                      │                              │
│                      │  Visualization               │
│                      │  ☑ Show visited cells        │
│                      │  ☐ Show value heatmap        │
│                      │  ☐ Show policy arrows        │
│                      │                              │
│                      │  Episode Info                │
│                      │  Episode: 42                 │
│                      │  Steps: 156                  │
│                      │  Reward: -23.5               │
│                      │  Status: Running             │
│                      │                              │
│                      │  Statistics                  │
│                      │  Total Episodes: 100         │
│                      │  Success Rate: 68.0%         │
└──────────────────────┴──────────────────────────────┘
```

---

## 🔧 مثال استفاده برنامه‌نویسی

### ایجاد Renderer:

```python
from gui.renderer import MazeRenderer
from environments.generator import MazeGenerator

# Load maze
maze, metadata = MazeGenerator.load_maze("environments/maps/maze_seed8_size15.npz")

# Create renderer
renderer = MazeRenderer(maze, metadata, cell_size=30)

# Get canvas size
width, height = renderer.get_canvas_size()

# Draw maze
renderer.draw_maze(canvas, show_visited=True)

# Update agent position
renderer.update_agent(canvas, row=5, col=5, energy=80)
```

### راه‌اندازی GUI:

```python
import tkinter as tk
from gui.app import MazeGUI

# Create window
root = tk.Tk()

# Create GUI
app = MazeGUI(root)

# Run
root.mainloop()
```

---

## 📊 ویژگی‌های بصری

### Value Heatmap
- نمایش V(s) یا max Q(s,a)
- رنگ‌بندی gradient
- نمایش اعداد

### Policy Arrows  
- فلش‌های جهت‌دار
- نمایش بهترین action
- رنگ highlighting

### Visited Cells
- Track کردن مسیر عامل
- رنگ متفاوت برای بازدید شده

### Agent Animation
- نمایش موقعیت فعلی
- Energy bar (اگر موجود باشد)
- انیمیشن روان

---

## ⚙️ تنظیمات

### Cell Size
```python
renderer = MazeRenderer(maze, metadata, cell_size=40)  # بزرگتر
```

### رنگ‌ها
رنگ‌ها در `MazeRenderer.COLORS` قابل تغییر هستند.

### سرعت پیش‌فرض
در `MazeGUI.__init__`:
```python
self.speed = 1.0  # تغییر سرعت اولیه
```

---

## 🐛 عیب‌یابی

### مشکل: GUI باز نمی‌شود
```bash
# چک کنید tkinter نصب باشد
python -m tkinter
```

### مشکل: مدل لود نمی‌شود
- مسیر فایل را چک کنید
- فرمت pickle صحیح باشد
- الگوریتم درست انتخاب شده باشد

### مشکل: کندی اجرا
- سرعت را کاهش دهید
- heatmap/policy را خاموش کنید

---

## 📝 یادداشت‌ها

- GUI با Tkinter (built-in Python) ساخته شده
- سازگار با Windows/Linux/Mac
- نیاز به نصب اضافی ندارد
- مناسب برای demo و debugging

---

**Student ID:** 40415484
