import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pydicom
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from pydicom.errors import InvalidDicomError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from train import IMAGE_SIZE, SmallUNet, normalize_ct


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_model.pt"
WINDOW_MIN = -1024.0
WINDOW_MAX = 600.0
THRESHOLD = 0.5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallUNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()


def png_data_url(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def read_ct_slice(upload):
    dataset = pydicom.dcmread(BytesIO(upload.read()))

    if str(getattr(dataset, "Modality", "")).upper() != "CT":
        raise ValueError("The uploaded DICOM is not marked as a CT image.")
    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("Only monochrome CT images are supported.")
    if "PixelData" not in dataset:
        raise ValueError("The DICOM does not contain pixel data.")
    if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
        raise ValueError("Please upload one single-frame axial CT slice.")
    if int(getattr(dataset, "SamplesPerPixel", 1)) != 1:
        raise ValueError("Only monochrome CT images are supported.")

    rows = int(getattr(dataset, "Rows", 0))
    columns = int(getattr(dataset, "Columns", 0))
    if rows <= 0 or columns <= 0 or rows * columns > 4096 * 4096:
        raise ValueError("The DICOM image dimensions are missing or too large.")

    image_type = {str(value).upper() for value in getattr(dataset, "ImageType", [])}
    if "LOCALIZER" in image_type or "SCOUT" in image_type:
        raise ValueError("Scout/localizer images are not supported.")
    orientation = getattr(dataset, "ImageOrientationPatient", None)
    if orientation is not None and len(orientation) == 6:
        directions = np.asarray(orientation, dtype=np.float32).reshape(2, 3)
        normal = np.cross(directions[0], directions[1])
        if abs(float(normal[2])) < 0.9:
            raise ValueError("Please upload an axial CT slice.")

    if "RescaleSlope" not in dataset or "RescaleIntercept" not in dataset:
        raise ValueError("The DICOM is missing HU rescale metadata.")
    slope = float(dataset.RescaleSlope)
    intercept = float(dataset.RescaleIntercept)
    if not np.isfinite([slope, intercept]).all() or slope == 0:
        raise ValueError("The DICOM contains invalid HU rescale metadata.")

    pixels = dataset.pixel_array
    if pixels.ndim != 2:
        raise ValueError("Please upload one 2D axial CT slice.")

    hu = pixels.astype(np.float32) * slope + intercept
    if not np.isfinite(hu).all():
        raise ValueError("The DICOM contains invalid pixel values.")

    normalized = np.clip(hu, WINDOW_MIN, WINDOW_MAX)
    normalized = (normalized - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    return normalized, photometric == "MONOCHROME1", False


def read_png_slice(upload):
    try:
        with Image.open(BytesIO(upload.read())) as image:
            if image.format != "PNG":
                raise ValueError("The uploaded file is not a valid PNG image.")
            if image.width <= 0 or image.height <= 0 or image.width * image.height > 4096 * 4096:
                raise ValueError("The PNG dimensions are missing or too large.")
            if image.mode not in {"L", "RGB", "RGBA", "P"}:
                raise ValueError("Please use an 8-bit PNG image.")
            grayscale = image.convert("L")
            normalized = np.asarray(grayscale, dtype=np.float32).copy() / 255.0
    except UnidentifiedImageError as error:
        raise ValueError("The uploaded file is not a valid PNG image.") from error
    # The downloaded processed PNGs are vertically flipped relative to the
    # standard DICOM orientation used by the corrected training pairs.
    return np.flipud(normalized).copy(), False, True


def predict(normalized_ct, invert_preview, restore_vertical):
    ct_uint8 = np.rint(normalized_ct * 255.0).astype(np.uint8)
    model_image = Image.fromarray(ct_uint8)
    resized = model_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    input_array = normalize_ct(
        np.asarray(resized, dtype=np.float32).copy() / 255.0
    )
    input_tensor = torch.from_numpy(input_array[None, None]).to(device)

    with torch.inference_mode():
        probability = torch.sigmoid(model(input_tensor))[0, 0].cpu().numpy()
    mask_256 = probability >= THRESHOLD

    preview_uint8 = 255 - ct_uint8 if invert_preview else ct_uint8
    mask_image = Image.fromarray(mask_256.astype(np.uint8) * 255)
    mask_image = mask_image.resize(model_image.size, Image.Resampling.NEAREST)
    mask = np.asarray(mask_image, dtype=np.uint8) > 0
    if restore_vertical:
        preview_uint8 = np.flipud(preview_uint8).copy()
        mask = np.flipud(mask).copy()

    original_image = Image.fromarray(preview_uint8)
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)

    overlay = np.repeat(preview_uint8[..., None], 3, axis=2)
    if mask.any():
        overlay_pixels = overlay[mask].astype(np.float32)
        red = np.array([255.0, 45.0, 70.0], dtype=np.float32)
        overlay[mask] = (0.45 * overlay_pixels + 0.55 * red).astype(np.uint8)

    return {
        "original": png_data_url(original_image),
        "mask": png_data_url(mask_image),
        "overlay": png_data_url(Image.fromarray(overlay)),
        "has_segmentation": bool(mask.any()),
        "predicted_pixels": int(mask.sum()),
        "maximum_probability": round(float(probability.max()), 4),
    }


@app.get("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.post("/predict")
def predict_dicom():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify(error="Choose a PNG or DICOM file first."), 400

    suffix = Path(upload.filename).suffix.lower()
    if suffix not in {".png", ".dcm"}:
        return jsonify(error="Only .png and .dcm files are accepted."), 415

    try:
        if suffix == ".png":
            normalized_ct, invert_preview, restore_vertical = read_png_slice(upload)
        else:
            normalized_ct, invert_preview, restore_vertical = read_ct_slice(upload)
    except InvalidDicomError:
        return jsonify(error="This is not a valid DICOM file."), 422
    except Exception as error:
        return jsonify(error=f"Could not process this image: {error}"), 422

    try:
        result = predict(normalized_ct, invert_preview, restore_vertical)
    except Exception:
        return jsonify(error="Model inference failed for this DICOM."), 500

    result["filename"] = secure_filename(upload.filename) or "uploaded.dcm"
    result["device"] = str(device)
    return jsonify(result)


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(_error):
    return jsonify(error="The image file is larger than the 32 MB limit."), 413


@app.errorhandler(500)
def internal_server_error(_error):
    if request.path == "/predict":
        return jsonify(error="The prediction server failed. Check its terminal output."), 500
    return "Internal server error", 500


if __name__ == "__main__":
    print(f"Model loaded on {device}. Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
