from flask import Flask, request, render_template, redirect
import os
import uuid
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load trained model
model = load_model("BloodCell.h5")

# Class labels
class_labels = [
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil"
]

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict_image_class(image_path, model):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Unable to read the uploaded image.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img_rgb, (224, 224))

    img_preprocessed = preprocess_input(
        img_resized.astype(np.float32)
    )

    img_preprocessed = np.expand_dims(
        img_preprocessed,
        axis=0
    )

    predictions = model.predict(img_preprocessed)

    predicted_class_idx = np.argmax(predictions, axis=1)[0]

    predicted_class_label = class_labels[predicted_class_idx]

    return predicted_class_label, img_rgb


@app.route("/", methods=["GET", "POST"])
def upload_file():

    if request.method == "POST":

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        # Secure filename
        original_filename = secure_filename(file.filename)

        # Create unique filename
        filename = f"{uuid.uuid4()}_{original_filename}"

        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(file_path)

        try:

            predicted_class_label, img_rgb = predict_image_class(
                file_path,
                model
            )

            return render_template(
                "result.html",
                class_label=predicted_class_label,
                image_path="/" + file_path.replace("\\", "/")
            )

        except Exception as e:

            return f"Error processing image: {str(e)}"

    return render_template("home.html")


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
