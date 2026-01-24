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
        self.predict_pipeline = PredictPipeline()
        self.predict_pipeline.load_model()

clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    return "Training process started (Check server logs)..."

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image_data = request.json['image']

        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        result = clApp.predict_pipeline.predict(image)

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