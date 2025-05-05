import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import argparse

class DigitRecognizer:
    def __init__(self, model_path='mnist_cnn_model.h5'):
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        self.cap = None
        self.drawing = False
        self.canvas = None
        self.last_point = None
        
    def preprocess_for_prediction(self, img):
        roi = img
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype('float32') / 255.0
        roi = roi.reshape(1, 28, 28, 1)
        
        return roi
    
    def predict_digit(self, img):
        if self.model is None:
            return -1, 0.0
        prediction = self.model.predict(img, verbose=0)[0]
        predicted_digit = np.argmax(prediction)
        confidence = prediction[predicted_digit]
        
        return predicted_digit, confidence
    
    def draw_canvas(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(self.canvas, self.last_point, (x, y), 255, 15)
                self.last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
    
    def run(self):
        if self.model is None:
            print("No model loaded. Exiting.")
            return
        self.canvas = np.zeros((400, 400), dtype=np.uint8)
        cv2.namedWindow('Handwritten Digit Recognition')
        cv2.setMouseCallback('Handwritten Digit Recognition', self.draw_canvas)
        print("\nInstructions:")
        print("- Draw a digit (0-9) using your mouse")
        print("- Press 'c' to clear the canvas")
        print("- Press 'q' to quit")
        
        while True:
            display = self.canvas.copy()
            preprocessed = self.preprocess_for_prediction(self.canvas)
            digit, confidence = self.predict_digit(preprocessed)
            cv2.putText(display, f"Predicted: {digit}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(display, f"Confidence: {confidence:.4f}", (10, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            cv2.imshow('Handwritten Digit Recognition', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros((400, 400), dtype=np.uint8)
        
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Real-time handwritten digit recognition')
    parser.add_argument('--model', type=str, default='mnist_cnn_model.h5',
                        help='Path to the trained model file')
    args = parser.parse_args()

    recognizer = DigitRecognizer(args.model)
    recognizer.run()

if __name__ == "__main__":
    main()