from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
import os
import io
from PIL import Image
import base64

# Import pipeline đã tối ưu (Load model 1 lần dùng mãi mãi)
from Project_1.pipeline.prediction import PredictPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        # Khởi tạo Pipeline một lần duy nhất khi Server start
        # Model sẽ được load vào RAM và nằm chờ ở đó
        self.predict_pipeline = PredictPipeline()
        # Trigger load model ngay lập tức (Warm-up) để request đầu tiên không bị chậm
        self.predict_pipeline.load_model()

# Khởi tạo Global Object bên ngoài __main__ để tương thích với Gunicorn/uWSGI
clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    # CẢNH BÁO: Việc gọi os.system("python main.py") sẽ treo request này
    # cho đến khi train xong (có thể mất hàng giờ).
    # Nginx/AWS Gateway thường timeout sau 60s -> Lỗi 504 Gateway Timeout.
    # Giải pháp tốt nhất là dùng Celery worker, nhưng demo thì tạm chấp nhận.

    # os.system("python main.py") # Uncomment nếu muốn chạy thật
    # os.system("dvc repro")
    return "Training process started (Check server logs)..."

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        # 1. Nhận chuỗi Base64 từ JSON request
        image_data = request.json['image']

        # 2. Decode Base64 trực tiếp trong RAM (không lưu file inputImage.jpg nữa)
        # Loại bỏ phần header nếu có (vd: "data:image/jpeg;base64,...")
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # 3. Gọi hàm predict (Truyền thẳng object PIL Image vào)
        # Lưu ý: Hàm predict trong class PredictPipeline (bản mới) phải nhận tham số đầu vào
        result = clApp.predict_pipeline.predict(image)

        # 4. Trả kết quả chuẩn JSON
        response_data = [
            {"class": result['class']},
            {"confidence": result['confidence']},
            {"image": result['image']} # Ảnh Segmentation (Base64) trả về
        ]
        return jsonify(response_data)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Chạy host 0.0.0.0 để Docker/AWS access được
    app.run(host='0.0.0.0', port=8080)