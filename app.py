from flask import Flask, request, jsonify, send_from_directory
import base64, cv2, numpy as np, onnxruntime as ort
import os, time, requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # ảnh tối đa 10MB

MODEL_URL = "https://drive.google.com/uc?export=download&id=1yvDBiywqOYTOBQ0mspwMqseQ4ccN_L2y"
MODEL_DIR = "/tmp/models"
UPLOAD_DIR = "/tmp/uploads"
MODEL_PATH = os.path.join(MODEL_DIR, "face_model.onnx")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Đang tải model...")
        r = requests.get(MODEL_URL, allow_redirects=True)
        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
            print("✅ Model tải xong:", MODEL_PATH)
        else:
            raise RuntimeError(f"❌ Lỗi tải model ({r.status_code})")

download_model()

print("🔄 Load model...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print("✅ Model đã load.")

def preprocess(img):
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 127.5 - 1.0
    return img

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

        print(f"✅ Nhận ảnh {rfid} | {embedding.shape[0]} chiều")

        return jsonify({
            "status": "ok",
            "embedding_dim": embedding.shape[0],
            "embedding_url": request.host_url + "uploads/" + os.path.basename(emb_path),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/uploads/<path:filename>")
def serve_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
