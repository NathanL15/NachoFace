import os
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN

# Define the face detector (assumed to be initialized outside)
device = torch.device('cpu')  # cpu
mtcnn = MTCNN(keep_all=False, device=device)

def preprocess_images(input_dir, output_dir):
    """
    Preprocess images in the dataset by detecting faces, cropping, resizing.
    Args:
        input_dir (str): Path to the input dataset directory.
        output_dir (str): Path to save the preprocessed dataset.
    """
    # Walk through the directory
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.endswith(('.jpg', '.png')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, os.path.relpath(root, input_dir), file)

                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    # Read the image
                    img = Image.open(input_path).convert('RGB')
                    
                    # Detect faces
                    boxes, probs = mtcnn.detect(img)

                    # Check if faces were detected
                    if boxes is not None and len(boxes) > 0:
                        # Get coordinates of the first face
                        box = boxes[0]
                        x1, y1, x2, y2 = [int(b) for b in box]
                        # Crop the face from the original RGB image
                        face = img.crop((x1, y1, x2, y2))
                        face = face.resize((224, 224))  # Resize only
                        face.save(output_path)
                        print(f"Saved face to {output_path}")
                    else:
                        print(f"No faces detected in {input_path}")

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    # Paths for input and output directories
    input_dir = "./vggface2/train"  # Replace with your train dataset path
    output_dir = "./vggface2_preprocessed/train"  # Output path for cropped faces
    preprocess_images(input_dir, output_dir)
