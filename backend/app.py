from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

# TensorFlow Lite interpreter
interpreter = None
reference_embeddings = []  # Temporary storage for reference embeddings


def load_model():
    """
    Load the TensorFlow Lite model.
    """
    global interpreter
    if interpreter is None:
        model_path = "cnn_model.tflite"  # Update with your model path
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()


def preprocess_base64_image(base64_image):
    """
    Decode and preprocess a base64 image string for inference.
    """
    decoded_image = base64.b64decode(base64_image.split(",")[1])
    img = Image.open(io.BytesIO(decoded_image)).convert("L")  # Grayscale
    img = img.resize((92, 112))  # Resize to model's input size
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    return img_array


def predict_embedding(image_tensor):
    """
    Run inference and return the embedding vector.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], image_tensor)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]["index"])
    return embedding.flatten()


def calculate_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.
    """
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)


@app.route("/register", methods=["POST"])
def register():
    """
    Register a user by storing reference embeddings.
    """
    global reference_embeddings
    data = request.get_json()

    if "frames" not in data or len(data["frames"]) != 10:
        return jsonify({"error": "Invalid input. Provide 10 frames."}), 400

    load_model()
    reference_embeddings = [
        predict_embedding(preprocess_base64_image(frame))
        for frame in data["frames"]
    ]

    return jsonify({"message": "Registration successful"})


@app.route("/login", methods=["POST"])
def login():
    """
    Log in a user by comparing embeddings with reference data.
    """
    global reference_embeddings
    data = request.get_json()

    if "frames" not in data or len(data["frames"]) != 10:
        return jsonify({"error": "Invalid input. Provide 10 frames."}), 400

    if not reference_embeddings:
        return jsonify({"error": "No reference embeddings found. Register first."}), 400

    load_model()
    matches = 0
    for frame in data["frames"]:
        login_embedding = predict_embedding(preprocess_base64_image(frame))
        for ref_embedding in reference_embeddings:
            similarity = calculate_similarity(login_embedding, ref_embedding)
            if similarity > 0.8:  # Threshold for similarity
                matches += 1
                break

    is_match = matches >= 7  # At least 7/10 frames must match
    return jsonify({"match": is_match, "message": "Login successful" if is_match else "No match found."})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
