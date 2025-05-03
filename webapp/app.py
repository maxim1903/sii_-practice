from flask import Flask, render_template, request, redirect, url_for, send_file
from ultralytics import YOLO
import os
from datetime import datetime
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

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

            try:
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
            except Exception as e:
                return str(e)  # Покажет ошибку, если что-то пойдет не так

    return render_template("index.html")

@app.route('/download-report')
def download_report():
    rows = []
    if os.path.exists("history.json"):
        with open("history.json", "r") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Пропускаем строку, если она некорректна

    # Генерация PDF отчета с использованием Table
    pdf_path = "static/report.pdf"
    document = SimpleDocTemplate(pdf_path, pagesize=letter)

    data = [["Дата", "Файл", "Объекты", "Классы"]]  # Заголовки таблицы
    for r in rows:
        data.append([r['timestamp'], r['filename'], str(r['detections']), ', '.join(map(str, r['classes']))])

    table = Table(data)
    
    # Настроим стили таблицы
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('LINEBEFORE', (0, 0), (0, -1), 1, colors.black),
        ('LINEAFTER', (-1, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ])
    
    table.setStyle(style)

    # Создаем PDF с таблицей
    elements = [table]
    document.build(elements)

    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)