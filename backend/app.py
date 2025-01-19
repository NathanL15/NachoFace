from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import os
import io

app = Flask(__name__)

# TensorFlow Lite interpreter
interpreter = None

# Directories for storing images
REFERENCE_DIR = "images/reference/"
CURRENT_DIR = "images/current/"

# Ensure directories exist
os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(CURRENT_DIR, exist_ok=True)


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
    Save reference images and store their embeddings.
    """
    data = request.get_json()

    if "frames" not in data or len(data["frames"]) != 10:
        return jsonify({"error": "Invalid input. Provide 10 frames."}), 400

    load_model()
    reference_embeddings = []

    for i, frame in enumerate(data["frames"]):
        # Save image to reference directory
        decoded_image = base64.b64decode(frame.split(",")[1])
        with open(os.path.join(REFERENCE_DIR, f"reference_{i}.jpg"), "wb") as f:
            f.write(decoded_image)

        # Process and store embedding
        image_tensor = preprocess_base64_image(frame)
        embedding = predict_embedding(image_tensor)
        reference_embeddings.append(embedding)

    # Save embeddings as a NumPy array
    np.save(os.path.join(REFERENCE_DIR, "reference_embeddings.npy"), reference_embeddings)

    return jsonify({"message": "Reference images registered successfully"})


@app.route("/login", methods=["POST"])
def login():
    """
    Save current images and compare their embeddings with reference embeddings.
    """
    data = request.get_json()

    if "frames" not in data or len(data["frames"]) != 10:
        return jsonify({"error": "Invalid input. Provide 10 frames."}), 400

    # Check if reference embeddings exist
    embeddings_path = os.path.join(REFERENCE_DIR, "reference_embeddings.npy")
    if not os.path.exists(embeddings_path):
        return jsonify({"error": "No reference data found. Please register first."}), 400

    # Load reference embeddings
    reference_embeddings = np.load(embeddings_path, allow_pickle=True)

    load_model()
    matches = 0

    for i, frame in enumerate(data["frames"]):
        # Save image to current directory
        decoded_image = base64.b64decode(frame.split(",")[1])
        with open(os.path.join(CURRENT_DIR, f"current_{i}.jpg"), "wb") as f:
            f.write(decoded_image)

        # Process and compare embeddings
        image_tensor = preprocess_base64_image(frame)
        current_embedding = predict_embedding(image_tensor)

        for ref_embedding in reference_embeddings:
            similarity = calculate_similarity(current_embedding, ref_embedding)
            if similarity > 0.8:  # Example threshold
                matches += 1
                break

    is_match = matches >= 7  # Require at least 7/10 frames to match
    return jsonify({
        "match": is_match,
        "message": "Login successful" if is_match else "No match found."
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)