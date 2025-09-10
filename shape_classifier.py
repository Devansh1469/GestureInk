import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

# Define the shape classes
classes = ["circle", "kite", "parallelogram", "square", "rectangle", "rhombus", "trapezoid", "triangle"]

def convert_to_uniform_gray_outline(image, gray_value=128, outline_thickness=3):
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Threshold to separate background (white) and outline (black)
    _, binary_image = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Step 2: Create a new blank white image
    white_background = np.full_like(image, 255)
    
    # Step 3: Find contours of the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 4: Draw contours with the specified gray outline on the white background
    cv2.drawContours(white_background, contours, -1, gray_value, thickness=outline_thickness)
    
    return white_background

def degrade_image(image):
    # Apply Gaussian Blur with a smaller kernel
    image = cv2.medianBlur(image, 3)
    
    # Add mild Gaussian Noise
    noise = np.random.normal(0, 30, image.shape).astype(np.uint8)  # Reduced noise intensity
    noisy_image = cv2.add(image, noise)
    
    # Simulate JPEG Compression Artifacts with higher quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]  # Reduced compression
    _, compressed_image = cv2.imencode('.jpg', noisy_image, encode_param)
    degraded_image = cv2.imdecode(compressed_image, cv2.IMREAD_GRAYSCALE)
    
    return degraded_image


# Function to load and preprocess images from a specific folder
def load_images_from_folder(folder_path):
    images = []
    labels = []
    
    for class_name in classes:
        class_folder = os.path.join(folder_path, class_name)
        image_files = [f for f in os.listdir(class_folder) if f.endswith('.png') or f.endswith('.jpg')]
        
        for filename in image_files:
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            img = cv2.resize(img, (224, 224))  # Resize to 224x224
            img = convert_to_uniform_gray_outline(img)  # Convert to uniform gray outline
            img = degrade_image(img)  # Degrade the image
            images.append(img)
            labels.append(class_name)
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize images to the range [0, 1]
    images = images / 255.0
    
    return images, labels

# Load dataset folders
dataset_path = "C:/Users/DEVANSH PRATAP/Downloads/archive (2)/dataset"
train_images, train_labels = load_images_from_folder(os.path.join(dataset_path, "train"))
val_images, val_labels = load_images_from_folder(os.path.join(dataset_path, "val"))
test_images, test_labels = load_images_from_folder(os.path.join(dataset_path, "test"))

# Encode labels as integers
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)
test_labels = label_encoder.transform(test_labels)

# Expand dimensions to fit CNN input requirements (224x224x1 for grayscale)
X_train = np.expand_dims(train_images, axis=-1)
X_val = np.expand_dims(val_images, axis=-1)
X_test = np.expand_dims(test_images, axis=-1)

# Define the CNN model
def create_model(input_shape=(224, 224, 1), num_classes=8):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.05),  # Dropout added before the output layer
        layers.Dense(num_classes, activation='softmax')  # Output layer with softmax for classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = create_model()

# Train the model
model.fit(X_train, train_labels, epochs=5, batch_size=100  , validation_data=(X_val, val_labels))

# Save the trained model
model.save('shape_classifier_model.h5')

# Test the model
test_loss, test_acc = model.evaluate(X_test, test_labels)
print(f"Test accuracy: {test_acc}")
