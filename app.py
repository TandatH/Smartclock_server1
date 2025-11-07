from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import cv2
import os
import requests

app = Flask(__name__)

# ===========================
# 🔹 1. TẢI MODEL TỪ HUGGINGFACE
# ===========================
MODEL_URL = "https://huggingface.co/pherodat1104/face_model/resolve/main/face_model.onnx"
MODEL_PATH = "/tmp/models/face_model.onnx"
os.makedirs("/tmp/models", exist_ok=True)

def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model đã có sẵn, bỏ qua tải lại.")
        return
    print("📥 Đang tải model từ HuggingFace...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(MODEL_URL, headers=headers, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Model tải xong: {MODEL_PATH}")
        print(f"📏 Kích thước: {os.path.getsize(MODEL_PATH)} bytes")
    else:
        print(f"❌ Lỗi tải model: {response.status_code}")
        raise Exception(f"Lỗi tải model từ HuggingFace ({response.status_code})")

# Tải model khi khởi động
download_model()

# ===========================
# 🔹 2. LOAD MODEL
# ===========================
print("🔄 Đang load model ONNX...")
ort_session = ort.InferenceSession(MODEL_PATH)
print("✅ Model ONNX đã load thành công!")

# ===========================
# 🔹 3. ROUTE GỐC — KIỂM TRA SERVER
# ===========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "✅ SmartClock Server đang hoạt động!",
        "status": "online",
        "model_loaded": os.path.exists(MODEL_PATH)
    })

# ===========================
# 🔹 4. ROUTE UPLOAD ẢNH
# ===========================
@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Không có file 'image' trong request!"}), 400

        file = request.files["image"]
        image_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Không đọc được ảnh!"}), 400

        # Resize ảnh cho khớp model (ví dụ: 112x112)
        img_resized = cv2.resize(img, (112, 112))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = img_resized.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))  # CHW
        img_tensor = np.expand_dims(img_tensor, axis=0)  # NCHW

        # Dự đoán
        ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
        emb = ort_session.run(None, ort_inputs)[0]

        emb_mean = np.mean(emb)
        print(f"✅ Nhận ảnh OK - mean embedding: {emb_mean:.6f}")

        return jsonify({
            "status": "success",
            "embedding_mean": float(emb_mean)
        })

    except Exception as e:
        print(f"❌ Lỗi xử lý upload: {e}")
        return jsonify({"error": str(e)}), 500

# ===========================
# 🔹 5. KHỞI CHẠY
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server đang chạy tại cổng {port}")
    app.run(host="0.0.0.0", port=port)
