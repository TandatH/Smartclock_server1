from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import cv2
import os
import requests

app = Flask(__name__)

# ================================
# 🔥 Firebase Realtime Database
# ================================
FIREBASE_URL = "https://smartclock-2025-default-rtdb.firebaseio.com/users/dat/embedding.json"

# ================================
# 🔥 Load model
# ================================
MODEL_URL = "https://huggingface.co/pherodat1104/face_model/resolve/main/face_model.onnx"
MODEL_PATH = "/tmp/models/face_model.onnx"
os.makedirs("/tmp/models", exist_ok=True)

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

download_model()

ort_session = ort.InferenceSession(
    MODEL_PATH, providers=["CPUExecutionProvider"]
)

# ================================
# 🔥 HÀM COSINE SIMILARITY
# ================================
def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2 + 1e-6)

# ================================
# 🔥 ROUTE UPLOAD ẢNH
# ================================
@app.route("/upload", methods=["POST"])
def upload_image():

    # ====== Kiểm tra file ======
    if "image" not in request.files:
        return jsonify({"error": "Không có file image!"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Ảnh lỗi không decode được!"}), 400

    # ====== Chuẩn hoá ảnh ======
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = img.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, 0)

    # ====== Generate embedding ======
    input_name = ort_session.get_inputs()[0].name
    emb = ort_session.run(None, {input_name: tensor})[0][0]  # shape (512,)

    # ====== Lấy embedding lưu trong Firebase ======
    fb = requests.get(FIREBASE_URL)
    saved_emb = fb.json()

    if not saved_emb:
        return jsonify({"error": "Firebase chưa có embedding!"}), 500

    # ====== Xử lý dữ liệu từ Firebase ======
    saved_emb_list = None

    if isinstance(saved_emb, list):
        saved_emb_list = saved_emb
    elif isinstance(saved_emb, dict):
        if "embedding" in saved_emb:
            saved_emb_list = saved_emb["embedding"]
        else:
            # fallback: sắp xếp theo key số
            try:
                saved_emb_list = [saved_emb[k] for k in sorted(saved_emb.keys(), key=int)]
            except Exception as e:
                return jsonify({"error": "Dữ liệu embedding Firebase không hợp lệ!", "detail": str(e)}), 500
    else:
        return jsonify({"error": "Dữ liệu embedding Firebase không hợp lệ!"}), 500

    if len(saved_emb_list) != 512:
        return jsonify({"error": "Embedding Firebase không đủ 512 chiều!", "length": len(saved_emb_list)}), 500

    # ====== Tính Similarity ======
    similarity = cosine_similarity(emb, saved_emb_list)
    print("🔥 Similarity:", similarity)

    # ====== Ngưỡng nhận diện ======
    THRESHOLD = 0.55
    match = similarity > THRESHOLD

    return jsonify({
        "status": "success",
        "similarity": float(similarity),
        "match": match,
        "threshold": THRESHOLD
    })


# ================================
# 🔥 CHẠY SERVER
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
