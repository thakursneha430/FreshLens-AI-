import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# 🔥 our PyTorch prediction function
from src.predict_image import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    # Save uploaded image
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # 🔮 AI Prediction
    result = predict_image(filepath)

    image_path = "static/uploads/" + filename
    return render_template("result.html", prediction=result, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)