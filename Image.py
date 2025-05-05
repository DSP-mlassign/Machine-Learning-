import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import argparse
from PIL import Image, ImageOps

def load_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.resize(img, (28, 28))
        
        # Invert image if it has white digits on black background
        if np.mean(img) > 127:
            img = 255 - img
        img = img.astype('float32') / 255.0
        _, img = cv2.threshold(img, 0.3, 1, cv2.THRESH_BINARY)
        img = img.reshape(1, 28, 28, 1)
        
        return img
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(model, image_path):
    processed_img = preprocess_image(image_path)
    
    if processed_img is None:
        return None, 0, None
    prediction = model.predict(processed_img, verbose=0)[0]
    predicted_digit = np.argmax(prediction)
    confidence = prediction[predicted_digit]
    
    return predicted_digit, confidence, processed_img

def visualize_prediction(image_path, predicted_digit, confidence, processed_img):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    orig_img = plt.imread(image_path)
    plt.imshow(orig_img, cmap='gray' if len(orig_img.shape) == 2 else None)
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(processed_img.reshape(28, 28), cmap='gray')
    plt.title("Preprocessed Image (28x28)")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, f"Prediction: {predicted_digit}\nConfidence: {confidence:.4f}",
             horizontalalignment='center', verticalalignment='center',
             fontsize=20)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict handwritten digits from custom images')
    parser.add_argument('--model', type=str, default='mnist_cnn_model.h5',
                        help='mnist_cnn_model.h5')
    parser.add_argument('--image', type=str, required=True,
                        help='5.jpeg')
    args = parser.parse_args()
    model = load_model(args.model)
    if model is None:
        return
    digit, confidence, processed_img = predict_image(model, args.image)
    
    if digit is None:
        print("Prediction failed.")
        return
    print(f"Predicted digit: {digit}")
    print(f"Confidence: {confidence:.4f}")
    visualize_prediction(args.image, digit, confidence, processed_img)

if __name__ == "__main__":
    main()