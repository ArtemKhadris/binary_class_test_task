import cv2
import numpy as np
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import argparse

# Defining PyTorch model class
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ImprovedCNN()

# Define image preprocessing for PyTorch
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),])

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

# Preprocessing the image function
def preprocess_image_for_torch(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img)
    img = img.unsqueeze(0)
    return img

# Classifying image using model
def classify_frame_with_torch(frame, model):
    preprocessed_frame = preprocess_image_for_torch(frame)
    with torch.no_grad():
        model_output = model(preprocessed_frame)
        _, predicted_class = torch.max(model_output, 1)
        return int(predicted_class.item())

# Unifying function, processes video, calls image processing and model prediction functions
# Returning list of 0s and 1s (0 if "foreign" object detected in area)
def process_video_with_torch(video_path, polygon, model):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Crop frame by polygon
        cropped_frame = crop_image_by_polygon(frame, polygon)

        # Classify the cropped frame using PyTorch model
        prediction = classify_frame_with_torch(cropped_frame, model)
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
    fixed_model_path = r"pt_models\model.pth"
    model.load_state_dict(torch.load(fixed_model_path))
    model.eval()

    # Load polygon coordinates from the JSON file
    with open(args.polygon_path, 'r') as f:
        polygon_path = json.load(f)

    # Process the video
    predictions = process_video_with_torch(args.video_path, polygon_path, model)

    # Count and print zero ranges
    output = count_zero_ranges(predictions)
    print(output)
    with open(args.output_path, 'w') as output_file:
        json.dump(output, output_file)

if __name__ == '__main__':
    main()