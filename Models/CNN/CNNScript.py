import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

def evaluate_images(model_path, reference_dir, current_dir):
    # Load the saved model
    model = load_model(model_path)

    # Define the data generator for evaluation
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Load the reference image
    reference_data = datagen.flow_from_directory(
        reference_dir,
        target_size=(224, 224),  # Adjust based on your model input size
        batch_size=1,
        class_mode='categorical',  # Adjust based on your problem type
        shuffle=False
    )

    # Load the current images to evaluate
    current_data = datagen.flow_from_directory(
        current_dir,
        target_size=(224, 224),  # Adjust based on your model input size
        batch_size=1,
        class_mode='categorical',  # Adjust based on your problem type
        shuffle=False
    )

    # Get the reference class
    reference_predictions = model.predict(reference_data)
    reference_class = np.argmax(reference_predictions, axis=1)[0]

    # Evaluate the current images
    for i in range(len(current_data)):
        current_image = current_data[i][0]
        current_prediction = model.predict(current_image)
        current_class = np.argmax(current_prediction, axis=1)[0]
        
        if current_class == reference_class:
            print('Same person detected.')
            print(f'Predicted class: {current_class}')
            print(f'Reference class: {reference_class}')
            print('Returning True')
            return True

    print('No matching person detected.')
    print('Returning False')
    return False

if __name__ == "__main__":
    model_path = 'model/cnn_model.keras'
    reference_dir = '../Images/Reference'
    current_dir = '../Images/Current'
    result = evaluate_images(model_path, reference_dir, current_dir)
    exit(result)