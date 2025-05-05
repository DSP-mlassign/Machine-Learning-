import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
np.random.seed(42)
# Function to load and preprocess the MNIST dataset
def load_and_preprocess_data():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Testing labels shape: {y_test.shape}")
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return X_train, y_train, X_test, y_test

# Function to visualize sample images
def visualize_samples(X_train, y_train):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        # Get the first image of each digit
        digit_indices = np.where(np.argmax(y_train, axis=1) == i)[0]
        img = X_train[digit_indices[0]].reshape(28, 28)
        
        plt.subplot(2, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Digit: {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_digits.png')
    plt.show()

# Function to build the CNN model
def build_cnn_model():

    model = keras.Sequential([
        # First Convolutional Layer
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Layer
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        # Fully connected layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  
        layers.Dense(10, activation='softmax')
    ])
    
   
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model

# Function to train the model
def train_model(model, X_train, y_train, X_test, y_test):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
   
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    
    return history
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return test_loss, test_acc
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

# Function to make predictions on new data
def predict_digit(model, image):
    processed_image = image.reshape(1, 28, 28, 1).astype('float32') / 255.0
    prediction = model.predict(processed_image)[0]
    predicted_digit = np.argmax(prediction)
    confidence = prediction[predicted_digit]
    
    return predicted_digit, confidence

def visualize_predictions(model, X_test, y_test, num_samples=10):
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(28, 28)
        true_label = np.argmax(y_test[idx])
        pred_digit, confidence = predict_digit(model, X_test[idx])
        plt.subplot(2, 5, i+1)
        plt.imshow(img, cmap='gray')
        color = 'green' if pred_digit == true_label else 'red'
        plt.title(f"True: {true_label}, Pred: {pred_digit}\nConf: {confidence:.2f}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.show()
def save_model(model, filename='mnist_cnn_model.h5'):
    model.save(filename)
    print(f"Model saved to {filename}")
def main():
    print("MNIST Handwritten Digit Recognition Project")
    print("-------------------------------------------")
    print("\nStep 1: Loading and preprocessing the data...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    print("\nStep 2: Visualizing sample images...")
    visualize_samples(X_train, y_train)
    print("\nStep 3: Building the CNN model...")
    model = build_cnn_model()
    print("\nStep 4: Training the model...")
    history = train_model(model, X_train, y_train, X_test, y_test)
    print("\nStep 5: Evaluating the model...")
    evaluate_model(model, X_test, y_test)
    print("\nStep 6: Visualizing training history...")
    plot_training_history(history)
    print("\nStep 7: Generating confusion matrix...")
    plot_confusion_matrix(model, X_test, y_test)
    print("\nStep 8: Visualizing predictions...")
    visualize_predictions(model, X_test, y_test)
    print("\nStep 9: Saving the model...")
    save_model(model)
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()