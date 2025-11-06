from flask import Flask, request, jsonify
import base64, cv2, numpy as np
import onnxruntime as ort
import os, time, requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # ảnh max 10MB

# ====================== CONFIG ======================
# 🔗 Link tải model từ Google Drive (chỉ cần đổi ID)
#   Link Drive: https://drive.google.com/file/d/1ABCDefGhIJKlmnop/view?usp=sharing
#   ID = 1ABCDefGhIJKlmnop
MODEL_URL = "https://drive.google.com/uc?export=download&id=1yvDBiywqOYTOBQ0mspwMqseQ4ccN_L2y"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_model.onnx")

os.makedirs(MODEL_DIR, exist_ok=True)

# ====================== DOWNLOAD MODEL ======================
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Đang tải model từ Google Drive...")
        r = requests.get(MODEL_URL, allow_redirects=True)
        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
            print(f"✅ Model tải thành công → {MODEL_PATH}")
        else:
            raise RuntimeError(f"❌ Lỗi tải model ({r.status_code})")

download_model()

# ====================== LOAD MODEL ======================
print("🔄 Đang load model...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print("✅ Model đã load:", MODEL_PATH)

# ====================== PREPROCESS ======================
def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 127.5 - 1.0
    return img

# ====================== UPLOAD ROUTE ======================
@app.route("/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json()
        img_base64 = data.get("image")
        rfid = data.get("rfid", "unknown")

        if not img_base64:
            return jsonify({"error": "Thiếu ảnh base64"}), 400

        img_data = base64.b64decode(img_base64)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Không giải mã được ảnh"}), 400

        os.makedirs("uploads", exist_ok=True)
        filename = f"uploads/{rfid}_{int(time.time())}.jpg"
        cv2.imwrite(filename, img)

        # Tạo embedding từ ảnh
        input_tensor = preprocess(img)
        embedding = session.run(None, {input_name: input_tensor})[0][0]

        # ✅ Lưu embedding ra file .npy (để ESP32-S3 tải về)
        emb_path = filename.replace(".jpg", ".npy")
        np.save(emb_path, embedding)

        print(f"✅ Nhận ảnh từ {rfid} | {len(embedding)} chiều | Lưu tại {filename}")

        return jsonify({
            "status": "ok",
            "embedding_dim": len(embedding),
            "embedding_url": request.host_url + emb_path,  # Đường dẫn tải embedding
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ====================== MAIN ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
