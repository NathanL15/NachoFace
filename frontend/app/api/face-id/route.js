export const config = {
  runtime: "nodejs", // Ensure this API route uses the Node.js runtime
};

import * as tf from "@tensorflow/tfjs-node";


import { NextResponse } from "next/server";

// TensorFlow Lite interpreter instance
let interpreter;

// Temporary storage for reference embeddings (in a real app, use a database)
let referenceEmbeddings = [];

// Load TensorFlow Lite model
async function loadModel() {
  if (!interpreter) {
    const modelPath = `${process.cwd()}/public/cnn_model.tflite`;
    const modelBuffer = await fs.readFile(modelPath);
    interpreter = new tf.Interpreter(modelBuffer);
  }
  return interpreter;
}

// Preprocess image from base64 to Tensor
async function preprocessBase64Image(base64) {
  const buffer = Buffer.from(base64.split(",")[1], "base64");
  const image = await tf.node.decodeImage(buffer, 1); // Grayscale
  const resizedImage = tf.image.resizeBilinear(image, [112, 92]);
  const normalizedImage = resizedImage.div(tf.scalar(255.0)); // Normalize to [0, 1]
  return normalizedImage.expandDims(0); // Add batch dimension
}

// Predict embedding from image tensor
async function predictEmbedding(interpreter, imageTensor) {
  const input = interpreter.inputTensor(0);
  const output = interpreter.outputTensor(0);

  input.copyFrom(imageTensor.flatten().dataSync());
  interpreter.invoke();
  const prediction = output.dataSync(); // Embedding or logits
  return prediction; // Return the embedding as a Float32Array
}

// Calculate similarity between two embeddings
function calculateSimilarity(embedding1, embedding2) {
  // Cosine similarity or Euclidean distance
  const dotProduct = embedding1.reduce(
    (sum, val, i) => sum + val * embedding2[i],
    0
  );
  const norm1 = Math.sqrt(embedding1.reduce((sum, val) => sum + val ** 2, 0));
  const norm2 = Math.sqrt(embedding2.reduce((sum, val) => sum + val ** 2, 0));
  return dotProduct / (norm1 * norm2); // Cosine similarity
}

// Handle API requests
export async function POST(req) {
  const { mode, frames } = await req.json();

  if (!frames || frames.length !== 10) {
    return NextResponse.json({ error: "Invalid input" }, { status: 400 });
  }

  try {
    // Load TensorFlow Lite model
    const interpreter = await loadModel();

    // Process each frame and predict embeddings
    const embeddings = [];
    for (const base64 of frames) {
      const imageTensor = await preprocessBase64Image(base64);
      const embedding = await predictEmbedding(interpreter, imageTensor);
      embeddings.push(embedding);
    }

    if (mode === "register") {
      // Store embeddings as the reference set
      referenceEmbeddings = embeddings;
      return NextResponse.json({ message: "Registration successful" });
    } else if (mode === "login") {
      if (referenceEmbeddings.length === 0) {
        return NextResponse.json(
          { error: "No reference embeddings found. Register first." },
          { status: 400 }
        );
      }

      // Compare login embeddings with reference embeddings
      let matches = 0;
      for (const loginEmbedding of embeddings) {
        for (const refEmbedding of referenceEmbeddings) {
          const similarity = calculateSimilarity(loginEmbedding, refEmbedding);
          if (similarity > 0.8) {
            matches++;
            break;
          }
        }
      }

      const isMatch = matches >= 7; // Example threshold: 7/10 must match
      return NextResponse.json({
        match: isMatch,
        message: isMatch ? "Login successful" : "No match found.",
      });
    }

    return NextResponse.json({ error: "Invalid mode" }, { status: 400 });
  } catch (error) {
    console.error("Error:", error);
    return NextResponse.json(
      { error: "An error occurred while processing the images" },
      { status: 500 }
    );
  }
}