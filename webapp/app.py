from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
from datetime import datetime
from PIL import Image
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Загружаем модель
model = YOLO('../runs/detect/train10/weights/last.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filename = datetime.now().strftime("%Y%m%d%H%M%S_") + image.filename
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(image_path)

            # Запуск детекции
            results = model(image_path)
            results[0].save(filename=os.path.join(RESULT_FOLDER, filename))

            # Логирование
            log = {
                "timestamp": datetime.now().isoformat(),
                "filename": filename,
                "detections": len(results[0].boxes),
                "classes": list(set([int(cls) for cls in results[0].boxes.cls.cpu().numpy()]))
            }

            with open("history.json", "a") as f:
                f.write(json.dumps(log) + "\n")

            return render_template("index.html", uploaded_image=filename, result_image=filename)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)