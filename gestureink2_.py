import cv2
import mediapipe as mp
import numpy as np  
from tensorflow.keras.models import load_model

# Load the pre-trained shape classifier model
model = load_model('shape_classifier_model.h5')
classes = ["circle", "kite", "parallelogram", "square", "rectangle", "rhombus", "trapezoid", "triangle"]

def classify_shape(image):
    # Resize the image to 224x224 (model's expected input size)
    img = cv2.resize(image, (224, 224),interpolation=cv2.INTER_CUBIC)  # Changed from 200x200 to 224x224
    img = img / 255.0  # Normalize to the range [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    print(class_index)
    shape_type = classes[class_index]
    return shape_type

def straighten_lines(canvas, distance_threshold=10):
    # Convert to grayscale and apply threshold to get binary image
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_canvas, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Approximate each contour to reduce unnecessary points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Iterate over points in the approximated contour
        for i in range(len(approx)):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % len(approx)][0]
            # If points are close enough, draw a straight line between them
            if np.linalg.norm(pt1 - pt2) < distance_threshold:
                cv2.line(canvas, tuple(pt1), tuple(pt2), (255, 255, 255), 5)
    
    return canvas

def enhance_contrast(image):
    # Convert to grayscale and apply histogram equalization for contrast enhancement
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)  # Histogram equalization to enhance contrast
    enhanced_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return enhanced_image

def predict_and_replace_shape(canvas):
   # Straighten lines before predicting the shape
    straightened_canvas = straighten_lines(canvas.copy())
    
    # Normalize contrast to ensure shape-background separation
    enhanced_canvas = enhance_contrast(straightened_canvas)
    
    # Resize the enhanced canvas to 224x224 for model input
    resized_canvas = cv2.resize(enhanced_canvas, (224, 224),interpolation=cv2.INTER_CUBIC)  # Changed to 224x224
    gray_canvas = cv2.cvtColor(resized_canvas, cv2.COLOR_BGR2GRAY)
    gray_canvas = cv2.normalize(gray_canvas, None, 0, 128, cv2.NORM_MINMAX)

    # Invert back to have a white background and gray outline (if needed)
    gray_canvas = cv2.bitwise_not(gray_canvas)
    cv2.imshow("Enhanced Canvas", gray_canvas)
    cv2.imwrite("saved.jpg", gray_canvas)
    cv2.waitKey(0)  # Wait for a key press to close the window
    # Classify the shape based on the straightened canvas
    shape_type = classify_shape(gray_canvas)
    
    # Get the bounding box of the largest contour to estimate shape size
    _, thresh = cv2.threshold(gray_canvas, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        draw_perfect_shape(shape_type, center, w, h, canvas)
        # Overlay the straightened canvas onto the original to retain the user-drawn shape
        canvas = cv2.addWeighted(straightened_canvas, 0.8, canvas, 0.2, 0)
        
        # Draw the perfect shape in a distinct color (e.g., red) and slightly transparent
        
        
        return shape_type
    else:
        return "No shape detected"

def draw_perfect_shape(shape_type, center, width, height, overlay_canvas):
    # Define the drawing functions for each shape type
    if shape_type == "circle":
        radius = min(width, height) // 2
        cv2.circle(overlay_canvas, center, radius, (0, 0, 255), -1)
        
    elif shape_type == "kite":
        pts = np.array([ 
            (center[0], center[1] - height // 2), 
            (center[0] - width // 3, center[1]),
            (center[0], center[1] + height // 2),
            (center[0] + width // 3, center[1])
        ])
        cv2.fillPoly(overlay_canvas, [pts], (0, 0, 255))
        
    elif shape_type == "parallelogram":
        pts = np.array([
            (center[0] - width // 4, center[1] - height // 2),
            (center[0] + width // 4, center[1] - height // 2),
            (center[0] + width // 2, center[1] + height // 2),
            (center[0] - width // 2, center[1] + height // 2)
        ])
        cv2.fillPoly(overlay_canvas, [pts], (0, 0, 255))
        
    elif shape_type == "square":
        size = min(width, height) // 2
        top_left = (center[0] - size, center[1] - size)
        bottom_right = (center[0] + size, center[1] + size)
        cv2.rectangle(overlay_canvas, top_left, bottom_right, (0, 0, 255), -1)
        
    elif shape_type == "rectangle":
        top_left = (center[0] - width // 2, center[1] - height // 2)
        bottom_right = (center[0] + width // 2, center[1] + height // 2)
        cv2.rectangle(overlay_canvas, top_left, bottom_right, (0, 0, 255), -1)
        
    elif shape_type == "rhombus":
        pts = np.array([ 
            (center[0], center[1] - height // 2),
            (center[0] - width // 2, center[1]),
            (center[0], center[1] + height // 2),
            (center[0] + width // 2, center[1])
        ])
        cv2.fillPoly(overlay_canvas, [pts], (0, 0, 255))
        
    elif shape_type == "trapezoid":
        pts = np.array([
            (center[0] - width // 3, center[1] - height // 2),
            (center[0] + width // 3, center[1] - height // 2),
            (center[0] + width // 2, center[1] + height // 2),
            (center[0] - width // 2, center[1] + height // 2)
        ])
        cv2.fillPoly(overlay_canvas, [pts], (0, 0, 255))
        
    elif shape_type == "triangle":
        pts = np.array([
            (center[0], center[1] - height // 2),
            (center[0] - width // 2, center[1] + height // 2),
            (center[0] + width // 2, center[1] + height // 2)
        ])
        cv2.fillPoly(overlay_canvas, [pts], (0, 0, 255))

# Other parts of the code (drawing, erasing, and webcam processing) remain the same
x1 = x2 = y1 = y2 = x3 = y3 = x4 = y4 = 0
webcam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
draw = False
erase = False
predict_mode = False
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
grid_size = 20

def draw_grid(img):
    for i in range(0, img.shape[1], grid_size):
        cv2.line(canvas, (i, 0), (i, img.shape[0]), (200, 200, 200), 1)
    for i in range(0, img.shape[0], grid_size):
        cv2.line(canvas, (0, i), (img.shape[1], i), (200, 200, 200), 1)

# def snap_to_grid(x, y):
#     snapped_x = round(x / grid_size) * grid_size
#     snapped_y = round(y / grid_size) * grid_size
#     return snapped_x, snapped_y

def toggle_eraser(dist):
    global erase
    if dist < 30:
        erase = not erase
        print(f"Eraser {'on' if erase else 'off'}")

while True:
    _, img = webcam.read()
    img = cv2.flip(img, 1)
    # draw_grid(img)
    frame_height, frame_width, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_img)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:
                    x1, y1 = x, y
                if id == 12:
                    x2, y2 = x, y
                if id == 4:
                    x4, y4 = x, y

            dist = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            dist2 = int(((x4 - x1) ** 2 + (y4 - y1) ** 2) ** 0.5)
            
            toggle_eraser(dist2)
            
            if erase:
                canvas.fill(0)
                x3 = y3 = 0
            else:
                draw = dist > 40
                if draw:
                    if x3 == 0 and y3 == 0:
                        x3, y3 = x1, y1
                    cv2.line(canvas, (x3, y3), (x1, y1), (0, 255, 0), 3)
                    x3, y3 = x1, y1
                else:
                    x3 = y3 = 0

    img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
    cv2.imshow("Writing Pad", img)

    if cv2.waitKey(10) & 0xFF == ord('p'):
        predict_mode = not predict_mode
        print(f"Prediction Mode {'ON' if predict_mode else 'OFF'}")

    if predict_mode:
        shape_type = predict_and_replace_shape(canvas)
        print(f"Replaced with: {shape_type}")
        # cv2.putText(canvas, f"Shape: {shape_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        predict_mode = False

    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
