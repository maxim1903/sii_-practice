# 🏍️ YOLOv8 Motorcycle Detector Web App

Веб-приложение для детекции мотоциклов на изображениях с использованием обученной модели YOLOv8.


## 🚀 Как запустить

1. Установи зависимости:

```bash
pip install -r requirements.txt
```

2. Запусти сервер:

```bash
cd webapp
python app.py
```

3. Открой в браузере:

```
http://127.0.0.1:5000
```

4. Загрузите изображение — и получите результат детекции мотоцикла!

---

## 🧠 Обучение модели

Модель YOLOv8 обучена с помощью команды:

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Лучшая модель (`best.pt`) сохраняется в:

```
runs/detect/train10/weights/best.pt
```

---

## ⚙️ Загрузка модели в Flask

Убедитесь, что путь к модели в `app.py` указан корректно:

```python
model = YOLO('../runs/detect/train10/weights/best.pt')
```

---


## 📬 Автор

**Максим Дашкевич БФИ2102**  
🔗 [GitHub: maxim1903](https://github.com/maxim1903)

---
