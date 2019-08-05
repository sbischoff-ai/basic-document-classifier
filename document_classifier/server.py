from os import environ

from flask import Flask
from flask import request
from flask_cors import CORS
from flask_yoloapi import endpoint, parameter

from document_classifier.image_classification.cnn import CNN

app = Flask("document classification service")
CORS(app)

@app.route("/predict", methods=["POST"])
@endpoint.api(parameter("image", type=bytes, required=True))
def parse_text(image):
    """Predict category of document image."""
    return {
        "class": "test",
        "confidence": 1.0
    }

PORT = environ["APP_PORT"] if "APP_PORT" in environ else 8080
HOST = environ["APP_HOST"] if "APP_HOST" in environ else "127.0.0.1"

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
