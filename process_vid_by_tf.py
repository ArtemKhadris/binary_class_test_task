import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse

# Function for cropping the image by polygon
def crop_image_by_polygon(image_path, polygon_coords):
    # Read the image
    try:
        img = cv2.imread(image_path)
    except:
        img = np.uint8(image_path)
    
    # Find the bounding box of the polygon
    x, y, w, h = cv2.boundingRect(np.array(polygon_coords))

    # Create a mask based on the polygon, within the bounding box
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    adjusted_polygon = np.array(polygon_coords) - (x, y)
    cv2.fillPoly(mask, [adjusted_polygon], (255, 255, 255))

    # Crop the image using the bounding box
    cropped_image = img[y:y+h, x:x+w]

    # Resize the mask to match the size of the cropped image
    mask = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))

    # Apply the mask directly to the cropped image
    result = cv2.bitwise_and(cropped_image, mask)

    return result

# Preprocessing the image function; Normalize the pixel values to be between 0 and 1
def preprocess_image(img):
    img_array = np.expand_dims(img, axis=0)
    return img_array / 255.0  

# Making prediction function
def classify_frame(frame, model):
    resized_frame = cv2.resize(frame, (224, 224))
    preprocessed_frame = preprocess_image(resized_frame)
    prediction = model.predict(preprocessed_frame)
    # Threshold at 0.5 for binary classification
    return int(prediction[0][0] > 0.5)

# Unifying function, processes video, calls frame processing and model prediction functions
# Returning list of 0s and 1s (0 if "foreign" object detected in area)
def process_video(video_path, polygons, fixed_model_path):
    model = load_model(fixed_model_path)
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Crop frame by polygon
        cropped_frame = crop_image_by_polygon(frame, polygons)

        # Classify the cropped frame
        prediction = classify_frame(cropped_frame, model)
        predictions.append(prediction)

        # Display the original and cropped frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Cropped Frame', cropped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return predictions

# Function for processing a list of 0s and 1s, making final output
def count_zero_ranges(lst):
    zero_ranges = []
    start = None
    for i, value in enumerate(lst):
        if value == 0:
            if start is None:
                start = i
        elif start is not None:
            zero_ranges.append([start, i - 1])
            start = None
    if start is not None:
        zero_ranges.append([start, len(lst) - 1])
    return zero_ranges

# Function to call a script from the console
def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process video based on a polygon and a trained model.')

    # Define command-line arguments
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('polygon_path', type=str, help='Path to the JSON file containing polygon coordinates')
    parser.add_argument('output_path', type=str, help='Path to save the output')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Fixed model path
    fixed_model_path = r"tf_models\best_model.keras"

    # Load polygon coordinates from the JSON file
    with open(args.polygon_path, 'r') as f:
        polygon_path = json.load(f)

    # Process the video
    predictions = process_video(args.video_path, polygon_path, fixed_model_path)

    # Count and print zero ranges
    output = count_zero_ranges(predictions)
    print(output)
    with open(args.output_path, 'w') as output_file:
        json.dump(output, output_file)

if __name__ == '__main__':
    main()