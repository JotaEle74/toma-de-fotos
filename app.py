from flask import Flask, render_template, request, url_for
import os
import random
import string
from datetime import datetime
import cv2
import numpy as np
import io
from PIL import Image
from rembg import remove, new_session

app = Flask(__name__)

UPLOAD_DIR = os.path.join(app.root_path, "static", "uploads")
ORIGINAL_DIR = os.path.join(UPLOAD_DIR, "original")
PROCESSED_DIR = os.path.join(UPLOAD_DIR, "processed")
DATA_DIR = os.path.join(app.root_path, "data")
RECORDS_FILE = os.path.join(DATA_DIR, "records.txt")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

print("Cargando modelo rembg (primera vez descarga ~170MB)...")
REMBG_SESSION = new_session("u2net")
print("Modelo listo.")


def remove_background(image_bytes: bytes) -> np.ndarray:
    result_bytes = remove(image_bytes, session=REMBG_SESSION)
    pil_img = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
    white_bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
    white_bg.paste(pil_img, mask=pil_img.split()[3])
    return cv2.cvtColor(np.array(white_bg.convert("RGB")), cv2.COLOR_RGB2BGR)


def crop_to_passport(image: np.ndarray, output_width=400, output_height=500) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_center_x = x + w // 2
        face_center_y = y + h // 2
    else:
        face_center_x = image.shape[1] // 2
        face_center_y = image.shape[0] // 2
        w = min(image.shape[1], image.shape[0] // 2)
        h = w

    target_ratio = output_width / output_height
    crop_height = max(h * 2, int(image.shape[0] * 0.5))
    crop_width = int(crop_height * target_ratio)

    if crop_width > image.shape[1]:
        crop_width = image.shape[1]
        crop_height = int(crop_width / target_ratio)
    if crop_height > image.shape[0]:
        crop_height = image.shape[0]
        crop_width = int(crop_height * target_ratio)

    x1 = int(face_center_x - crop_width / 2)
    y1 = int(face_center_y - crop_height * 0.45)
    x2 = x1 + crop_width
    y2 = y1 + crop_height

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    crop = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
    return crop


def make_unique_filename(prefix: str, ext: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{timestamp}_{suffix}.{ext}"


def make_full_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return request.host_url.rstrip("/") + path


def save_record(dni: str, cell: str, original_url: str, processed_url: str) -> None:
    with open(RECORDS_FILE, "a", encoding="utf-8") as record_file:
        timestamp = datetime.now().isoformat(sep=' ', timespec='seconds')
        record_file.write(
            f"{timestamp} | DNI: {dni} | Celular: {cell} | Original: {original_url} | Procesada: {processed_url}\n"
        )


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    success_message = None
    original_image = None
    processed_image = None
    original_name = None
    output_name = "carnet.jpg"
    file_size_kb = None
    width = None
    height = None
    success = False

    action = request.form.get("action", "process")
    dni = request.form.get("dni", "").strip()
    cell = request.form.get("cell", "").strip()

    if request.method == "POST":
        if action == "save":
            original_image = request.form.get("original_url")
            processed_image = request.form.get("processed_url")
            output_name = request.form.get("output_name", output_name)
            original_name = request.form.get("original_name", "imagen subida")

            if not dni or not cell or not original_image or not processed_image:
                error = "Faltan datos para guardar. Asegúrate de procesar la imagen primero."
            else:
                original_link = make_full_url(original_image)
                processed_link = make_full_url(processed_image)
                save_record(dni, cell, original_link, processed_link)
                success_message = "Datos guardados correctamente en archivo TXT."
                success = True
                original_image = None
                processed_image = None
                original_name = None
                output_name = "carnet.jpg"
                file_size_kb = None
                width = None
                height = None
                dni = ""
                cell = ""
        else:
            file = request.files.get("image")
            if not file or file.filename == "":
                error = "Seleccione una imagen para procesar."
            elif not dni or not cell:
                error = "Ingrese DNI y número de celular antes de procesar."
            else:
                image_bytes = file.read()
                if len(image_bytes) == 0:
                    error = "El archivo está vacío."
                elif len(image_bytes) > 10 * 1024 * 1024:
                    error = "La imagen supera 10 MB. Usa una imagen más ligera."
                else:
                    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        error = "No se pudo leer la imagen."
                    else:
                        original_name = file.filename
                        original_ext = os.path.splitext(file.filename)[1].lstrip('.').lower() or 'jpg'
                        if original_ext not in ['jpg', 'jpeg', 'png', 'webp', 'bmp']:
                            original_ext = 'jpg'

                        original_filename = make_unique_filename('original', original_ext)
                        original_path = os.path.join(ORIGINAL_DIR, original_filename)
                        with open(original_path, 'wb') as original_file:
                            original_file.write(image_bytes)

                        processed = crop_to_passport(remove_background(image_bytes))
                        is_success, buffer = cv2.imencode('.jpg', processed)
                        if not is_success:
                            error = "Error al procesar la imagen."
                        else:
                            processed_filename = make_unique_filename('processed', 'jpg')
                            processed_path = os.path.join(PROCESSED_DIR, processed_filename)
                            with open(processed_path, 'wb') as processed_file:
                                processed_file.write(buffer.tobytes())

                            original_image = url_for('static', filename=f'uploads/original/{original_filename}')
                            processed_image = url_for('static', filename=f'uploads/processed/{processed_filename}')
                            file_size_kb = f"{len(image_bytes) / 1024:.1f}"
                            width = image.shape[1]
                            height = image.shape[0]
                            success_message = "Imagen procesada correctamente. Usa el botón guardar para registrar los datos."

    return render_template(
        "index.html",
        error=error,
        success_message=success_message,
        original_image=original_image,
        processed_image=processed_image,
        original_name=original_name,
        output_name=output_name,
        file_size_kb=file_size_kb,
        width=width,
        height=height,
        dni=dni,
        cell=cell,
        success=success,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)