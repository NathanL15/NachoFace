import tensorflow as tf
import numpy as np
import os
from PIL import Image

def load_image(image_path):
    """
    Load and preprocess an image for model inference.
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((92, 112))  # Resize to match model input size (width x height)
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img




def load_model(model_path):
    """
    Load a TensorFlow Lite model for inference from the file directly.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def predict(interpreter, image):
    """
    Run inference on an image using the given TFLite interpreter.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data)

def find_match(model_path, reference_images_path, current_image_path):
    """
    Compare a current image with a set of reference images using a TFLite model.
    """
    interpreter = load_model(model_path)
    current_image = load_image(current_image_path)
    current_prediction = predict(interpreter, current_image)

    match_found = False
    for ref_image_name in os.listdir(reference_images_path):
        ref_image_path = os.path.join(reference_images_path, ref_image_name)
        if os.path.isfile(ref_image_path):  # Ensure it is a file
            ref_image = load_image(ref_image_path)
            ref_prediction = predict(interpreter, ref_image)
            if ref_prediction == current_prediction:
                match_found = True
                print(f"Match found with reference image: {ref_image_name}")
                break

    if not match_found:
        print("No match found.")

    return match_found

if __name__ == "__main__":
    print("Current Directory:", os.getcwd())
    model_path = os.path.join('Models', 'CNN', 'model', 'cnn_model.tflite')
    reference_images_path = os.path.join('Models', 'Images', 'Reference')
    current_image_path = os.path.join('Models', 'Images', 'Current', 'current1.jpg')

    # Debugging prints
    print("Model path:", model_path)
    print("Reference images path:", reference_images_path)
    print("Current image path:", current_image_path)

    # Ensure the current image exists
    if not os.path.exists(current_image_path):
        print(f"File not found: {current_image_path}")
        exit(1)

    match = find_match(model_path, reference_images_path, current_image_path)
    print("Match found:", match)
