from flask import Flask, request, jsonify, send_from_directory
import base64, cv2, numpy as np, onnxruntime as ort
import os, time, requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Giới hạn 10MB ảnh upload

# ======================
# ĐƯỜNG DẪN VÀ MODEL
# ======================
FILE_ID = "1yvDBiywqOYTOBQ0mspwMqseQ4ccN_L2y"
MODEL_DIR = "/tmp/models"
UPLOAD_DIR = "/tmp/uploads"
MODEL_PATH = os.path.join(MODEL_DIR, "face_model.onnx")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ======================
# HÀM TẢI MODEL GOOGLE DRIVE
# ======================
def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model đã tồn tại:", MODEL_PATH)
        print("📏 Kích thước:", os.path.getsize(MODEL_PATH), "bytes")
        return

    print("📥 Đang tải model từ Google Drive...")
    gdrive_api = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

    session = requests.Session()
    response = session.get(gdrive_api, stream=True)

    # Nếu có token xác nhận (file >100MB hoặc cần confirm)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v
            gdrive_api = f"{gdrive_api}&confirm={confirm_token}"
            response = session.get(gdrive_api, stream=True)
            break

    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(1024):
                if chunk:
                    f.write(chunk)
        print("✅ Model tải thành công:", MODEL_PATH)
        print("📏 Kích thước:", os.path.getsize(MODEL_PATH), "bytes")

        if os.path.getsize(MODEL_PATH) < 1000000:
            raise RuntimeError("❌ File tải quá nhỏ (<1MB) — có thể Google trả về HTML, hãy kiểm tra link chia sẻ!")

    else:
        raise RuntimeError(f"❌ Lỗi tải model ({response.status_code})")

# ======================
# LOAD MODEL
# ======================
download_model()
print("🔄 Đang load model ONNX...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print("✅ Model đã load thành công!")

# ======================
# XỬ LÝ ẢNH
# ======================
def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 127.5 - 1.0
    return img

# ======================
# API UPLOAD ẢNH
# ======================
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

        filename = f"{rfid}_{int(time.time())}.jpg"
        img_path = os.path.join(UPLOAD_DIR, filename)
        cv2.imwrite(img_path, img)

        input_tensor = preprocess(img)
        embedding = session.run(None, {input_name: input_tensor})[0][0]
        embedding = embedding / np.linalg.norm(embedding)

        emb_path = img_path.replace(".jpg", ".npy")
        np.save(emb_path, embedding)

        print(f"✅ Nhận ảnh {rfid} | vector {embedding.shape[0]} chiều")

        return jsonify({
            "status": "ok",
            "embedding_dim": int(embedding.shape[0]),
            "embedding_url": request.host_url + "uploads/" + os.path.basename(emb_path),
        })

    except Exception as e:
        print("❌ Lỗi xử lý ảnh:", e)
        return jsonify({"error": str(e)}), 500

# ======================
# ROUTE TRẢ FILE
# ======================
@app.route("/uploads/<path:filename>")
def serve_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# ======================
# CHẠY LOCAL
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
