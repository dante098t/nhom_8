import os, io, base64, json, traceback
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image
import requests

import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# KHỞI TẠO APP
app = FastAPI(title="Pneumonia EfficientNetB0 API", version="1.0.0")

# CẤU HÌNH CƠ BẢN

MODEL_PATH = os.path.join("models", "effb0_best.h5")

# Điều chỉnh theo bài của bạn (3 lớp). Nếu model thực tế binary, đổi CLASS_NAMES cho phù hợp.
CLASS_NAMES = ["NORMAL", "BACTERIAL", "VIRAL"]
N_CLASSES   = len(CLASS_NAMES)
INPUT_SIZE  = (224, 224)  # EfficientNetB0 mặc định 224
LAST_CONV_FALLBACK = "top_conv"  # EfficientNetB0


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# HÀM TIỆN ÍCH

def _load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def _pil_to_base64(img: Image.Image, fmt="JPEG", q=92) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=q)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _preprocess_batch(pil_img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Trả về (rgb_uint8, batch_float32_after_preprocess)"""
    rgb = np.array(pil_img, dtype=np.uint8)
    resized = pil_img.resize(INPUT_SIZE)
    x = np.array(resized, dtype=np.float32)[None, ...]
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return rgb, x

def _build_effb0(input_shape=(224,224,3), n_classes=3, classifier_activation="softmax") -> tf.keras.Model:
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=input_shape
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    out = tf.keras.layers.Dense(n_classes, activation=classifier_activation)(x)
    model = tf.keras.Model(base.input, out, name="EfficientNetB0_head")
    return model

def _try_load_fullmodel(path: str) -> tf.keras.Model:
    try:
        m = tf.keras.models.load_model(path, compile=False)
        return m
    except Exception as e:
        raise RuntimeError(f"fullmodel failed: {e}")

def _try_load_weights(path: str) -> tf.keras.Model:
    # Thử 3 lớp softmax, sau đó fallback 1 lớp sigmoid (binary)
    for n_cls, act in [(N_CLASSES, "softmax"), (1, "sigmoid")]:
        try:
            model_tmp = _build_effb0(input_shape=INPUT_SIZE + (3,), n_classes=n_cls, classifier_activation=act)
            model_tmp.load_weights(path)
            return model_tmp
        except Exception:
            continue
    # Cho nổ để thấy lỗi thật
    model_tmp = _build_effb0(input_shape=INPUT_SIZE + (3,), n_classes=N_CLASSES, classifier_activation="softmax")
    model_tmp.load_weights(path)
    return model_tmp

def load_model_robust(path: str) -> Tuple[tf.keras.Model, str]:
    errors = []
    for how in ["fullmodel", "weights_only"]:
        try:
            if how == "fullmodel":
                m = _try_load_fullmodel(path)
            else:
                m = _try_load_weights(path)
            return m, how
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError("❌ Không load được model. Details:\n" + "\n-----\n".join(errors))

def _infer_last_conv(model: tf.keras.Model, fallback: str = LAST_CONV_FALLBACK) -> str:
    try:
        model.get_layer(fallback)
        return fallback
    except Exception:
        pass
    for layer in reversed(model.layers):
        shp = getattr(layer, "output_shape", None)
        if isinstance(shp, tuple) and len(shp) == 4:
            return layer.name
    raise ValueError("Không tìm thấy lớp conv 4D cuối để làm Grad-CAM.")

def make_gradcam(rgb_uint8: np.ndarray, pred_batch: np.ndarray, model: tf.keras.Model,
                 last_conv_name: Optional[str] = None, alpha: float = 0.5) -> Image.Image:
    if last_conv_name is None:
        last_conv_name = _infer_last_conv(model, LAST_CONV_FALLBACK)

    last_conv_layer = model.get_layer(last_conv_name)
    grad_model = tf.keras.models.Model(model.inputs, [last_conv_layer.output, model.output])

    pil_img = Image.fromarray(rgb_uint8)
    _, x = _preprocess_batch(pil_img)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if preds.shape[-1] == 1:  # binary sigmoid
            target = preds[:, 0]
        else:                      # multiclass softmax
            class_idx = int(np.argmax(preds[0]))
            target = preds[:, class_idx]

    grads = tape.gradient(target, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]  # HxWxC
    heatmap = tf.reduce_sum(tf.multiply(conv_out, pooled_grads), axis=-1).numpy()

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    heatmap = (heatmap * 255).astype(np.uint8)

    hm = cv2.resize(heatmap, (rgb_uint8.shape[1], rgb_uint8.shape[0]), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
    overlay = cv2.addWeighted(rgb_uint8, 1.0, hm_color, alpha, 0)
    return Image.fromarray(overlay)

def postprocess_probs(raw: np.ndarray) -> List[float]:
    raw = raw.flatten()
    if raw.shape[0] == 1:   # binary sigmoid -> quy về [p_normal, p_pneumonia]
        p1 = float(raw[0])
        return [1.0 - p1, p1]
    return [float(x) for x in raw]


# NẠP MODEL KHI KHỞI ĐỘNG

try:
    model, LOADED_AS = load_model_robust(MODEL_PATH)
    try:
        model_input_size = tuple(model.input_shape[1:3])
        if model_input_size != INPUT_SIZE:
            print(f"[WARN] Model input {model_input_size} ≠ {INPUT_SIZE}. Dùng {model_input_size}.")
            INPUT_SIZE = model_input_size   # <-- chỉ gán, KHÔNG dùng 'global' ở đây
    except Exception:
        pass
    LAST_CONV = _infer_last_conv(model, LAST_CONV_FALLBACK)
except Exception as e:
    trace = traceback.format_exc()
    raise RuntimeError(f"Khởi động thất bại: {e}\n{trace}")

# =========================
# SCHEMAS
# =========================
class URLBody(BaseModel):
    url: str
    threshold: float = 0.5
    last_conv: Optional[str] = None
    return_overlay: bool = True
    return_image: bool = True  

# ---- Khuyến nghị theo lớp dự đoán
def _advice_for(label: str) -> dict:
    up = (label or "").upper()
    if up == "BACTERIAL":
        return {
            "details": "Gợi ý viêm phổi do vi khuẩn: tổn thương có thể khu trú/đông đặc rõ.",
            "recommendations": [
                "Khám bác sĩ sớm để đánh giá và cân nhắc kháng sinh.",
                "Có thể làm công thức máu/CRP theo chỉ định.",
                "Uống đủ nước, theo dõi sốt và khó thở."
            ],
        }
    if up == "VIRUS":
        return {
            "details": "Gợi ý viêm phổi do virus: có thể lan tỏa, kính mờ hai phổi (tùy ảnh).",
            "recommendations": [
                "Tha    m vấn bác sĩ; thường ưu tiên điều trị triệu chứng và theo dõi sát.",
                "Cân nhắc test virus hô hấp (Influenza/RSV/SARS-CoV-2) theo chỉ định.",
                "Nghỉ ngơi, bù nước; theo dõi SpO₂ và dấu hiệu nặng."
            ],
        }
    # NORMAL
    return {
        "details": "Mô hình không thấy dấu hiệu viêm phổi rõ rệt trên ảnh.",
        "recommendations": [
            "Tiếp tục theo dõi triệu chứng (ho, sốt, khó thở).",
            "Khám sức khỏe định kỳ theo khuyến cáo."
        ],
    }

# =========================
# ENDPOINTS
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "loaded_as": LOADED_AS,
        "input_size": INPUT_SIZE,
        "last_conv": LAST_CONV,
        "class_names": CLASS_NAMES,
    }

@app.post("/predict/url")
def predict_from_url(body: URLBody):
    # tải ảnh
    try:
        pil = _load_image_from_url(body.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không tải được ảnh: {e}")

    # tiền xử lý & dự đoán
    rgb, x = _preprocess_batch(pil)
    preds = model.predict(x, verbose=0)
    probs = postprocess_probs(preds)  # list float

    # xác định nhãn top-1
    if len(probs) == 2 and len(CLASS_NAMES) == 3:
        label_idx = int(np.argmax([probs[0], probs[1], 0.0]))
    else:
        label_idx = int(np.argmax(probs))
    label = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else str(label_idx)

    # map xác suất theo lớp + top_conf
    conf_map = dict(zip(CLASS_NAMES[:len(probs)], [float(p) for p in probs]))
    top_conf = float(max(probs) if probs else 0.0)

    # kết quả
    result = {
        "label": label,                                # NORMAL | BACTERIAL | VIRAL
        "confidence_per_class": conf_map,              # {NORMAL:..., BACTERIAL:..., VIRAL:...}
        "top_confidence": top_conf,                    # số lớn nhất
        "probs": conf_map,                             # giữ tương thích cũ
    }
    result.update(_advice_for(label))

    # trả ảnh gốc & grad-cam (nếu cần)
    if body.return_image:
        result["image_base64"] = _pil_to_base64(Image.fromarray(rgb))
    if body.return_overlay:
        overlay = make_gradcam(rgb, preds, model, last_conv_name=body.last_conv or LAST_CONV, alpha=0.5)
        result["gradcam_base64"] = _pil_to_base64(overlay)

    return result

@app.post("/predict/file")
def predict_from_file(
    file: UploadFile = File(...),
    return_overlay: bool = True,
    last_conv: Optional[str] = None,
    return_image: bool = True
):
    # đọc ảnh upload
    try:
        pil = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được ảnh tải lên: {e}")

    # tiền xử lý & dự đoán
    rgb, x = _preprocess_batch(pil)
    preds = model.predict(x, verbose=0)
    probs = postprocess_probs(preds)

    # xác định nhãn top-1
    if len(probs) == 2 and len(CLASS_NAMES) == 3:
        label_idx = int(np.argmax([probs[0], probs[1], 0.0]))
    else:
        label_idx = int(np.argmax(probs))
    label = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else str(label_idx)

    # map xác suất theo lớp + top_conf
    conf_map = dict(zip(CLASS_NAMES[:len(probs)], [float(p) for p in probs]))
    top_conf = float(max(probs) if probs else 0.0)

    # kết quả
    result = {
        "label": label,
        "confidence_per_class": conf_map,
        "top_confidence": top_conf,
        "probs": conf_map,
    }
    result.update(_advice_for(label))

    if return_image:
        result["image_base64"] = _pil_to_base64(Image.fromarray(rgb))
    if return_overlay:
        overlay = make_gradcam(rgb, preds, model, last_conv_name=last_conv or LAST_CONV, alpha=0.5)
        result["gradcam_base64"] = _pil_to_base64(overlay)

    return result