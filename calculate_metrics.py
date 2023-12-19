import cv2
import json
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, precision_score, recall_score, accuracy_score

# Get numbers of frames in video
def count_frames(video_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.release()
    return total_frames

# Read json
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Create 1 nd 0 list from json
def create_binary_list(intervals, length):
    binary_list = [0] * length
    for start, end in intervals:
        binary_list[start:end + 1] = [1] * (end - start + 1)
    return binary_list

# Padding json using number of frames in video
def create_padded_binary_list(intervals, min_length):
    max_end = max(end for _, end in intervals) if intervals else -1
    length = max(max_end + 1, min_length)
    binary_list = [0] * length
    for start, end in intervals:
        binary_list[start:end + 1] = [1] * (end - start + 1)

    return binary_list

# Calculate metrics
def calculate_metrics(true_labels, predicted_labels):
    # MAE
    mae = mean_absolute_error(true_labels, predicted_labels)
    # MSE
    mse = mean_squared_error(true_labels, predicted_labels)
    # ROC-AUC
    roc_auc = roc_auc_score(true_labels, predicted_labels)
    # Precision, Recall, and Accuracy
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)

    return mae, mse, roc_auc, precision, recall, accuracy

def main(vid_path, pred_lab_path, real_lab_path):
    num_frames = count_frames(vid_path) 
    pred_lab_file = read_json_file(pred_lab_path)
    real_lab_file = read_json_file(real_lab_path)
    pred_lab = create_padded_binary_list(pred_lab_file, num_frames)
    real_lab = create_padded_binary_list(real_lab_file, num_frames)
    
    metrics = calculate_metrics(pred_lab, real_lab)
    print("Metrics:")
    print(f"MAE: {metrics[0]:.4f}")
    print(f"MSE: {metrics[1]:.4f}")
    print(f"ROC-AUC: {metrics[2]:.4f}")
    print(f"Precision: {metrics[3]:.4f}")
    print(f"Recall: {metrics[4]:.4f}")
    print(f"Accuracy: {metrics[5]:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <vid_path> <pred_lab_path> <real_lab_path>")
        sys.exit(1)

    vid_path = sys.argv[1]
    pred_lab_path = sys.argv[2]
    real_lab_path = sys.argv[3]

    main(vid_path, pred_lab_path, real_lab_path)