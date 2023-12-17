# Import required packages
import os
import json
import cv2
import numpy as np
import csv
import random
from PIL import ImageEnhance, Image, ImageFilter
import pandas as pd
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_recall_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import F1Score, AUC, Precision, Recall
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import subprocess

# Necessary variables
usual_epochs = 20
usual_batch = 32
dataset_split_size = 0.2

# Files paths
polygons_path = r'###' # PATH TO AN INPUT POLYGONS JSON FILE
time_intervals_path = r'###' # PATH TO AN INPUT TIME INTERVALS JSON FILE
videos_path = r'###' # PATH TO AN INPUT VIDEOS FILE (FOLDER WITH VIDEOS)
images_path = r'###' # PATH TO FRAMES FROM VIDEOS; FOLDER (WILL BE CREATED BY FUNCTION)
dataset_folder = r'###' # PATH TO THE DATASET FOR TRAINING OUR MODELS; FOLDER (WILL BE CREATED BY FUNCTION)
csv_file_path = r'###' # PATH TO THE CSV FILE DESCRIBING OUR DATASET; SHOULD BE NAMED LIKE "data.csv" (WILL BE CREATED BY FUNCTION)
dataset_for_yolo = r'###' # PATH TO YOLO DATASET; FOLDER (WILL BE CREATED BY FUNCTION)
dataset_for_yolo_train = os.path.join(dataset_for_yolo, 'train') # PATH TO YOLO DATASET; FOLDER (TRAIN, WILL BE CREATED BY FUNCTION)
dataset_for_yolo_val = os.path.join(dataset_for_yolo, 'val') # PATH TO YOLO DATASET; FOLDER (VAL, WILL BE CREATED BY FUNCTION)
tf_checkpoint_path = r'###' # PATH TO TF MODEL; SHOULD BE NAMED LIKE "best_model.keras" (WILL BE CREATED BY FUNCTION)
tf_fig1_path = r'###' # PATH TO 1ST TF GRAPH; SHOULD BE NAMED LIKE "figureN.png" (WILL BE CREATED BY FUNCTION)
tf_fig2_path = r'###' # PATH TO 2ND TF GRAPH; SHOULD BE NAMED LIKE "figureN.png" (WILL BE CREATED BY FUNCTION)
tf_fig3_path = r'###' # PATH TO 3RD TF GRAPH; SHOULD BE NAMED LIKE "figureN.png" (WILL BE CREATED BY FUNCTION)
tf_fig4_path = r'###' # PATH TO 4TH TF GRAPH; SHOULD BE NAMED LIKE "figureN.png" (WILL BE CREATED BY FUNCTION)
tf_fig5_path = r'###' # PATH TO 5TH TF GRAPH; SHOULD BE NAMED LIKE "figureN.png" (WILL BE CREATED BY FUNCTION)
tf_text_output_path = r'###' # PATH TO TXT TF MODEL; SHOULD BE NAMED LIKE "output.txt" (WILL BE CREATED BY FUNCTION)
pt_checkpoint_path = r'###' # PATH TO PYTORCH MODEL; SHOULD BE NAMED LIKE "model.pth" (WILL BE CREATED BY FUNCTION)
pt_fig1_path = r'###' # PATH TO 1ST PT GRAPH; SHOULD BE NAMED LIKE "figureN.png" (WILL BE CREATED BY FUNCTION)
pt_text_output_path = r'###' # PATH TO TXT PT MODEL; SHOULD BE NAMED LIKE "output.txt" (WILL BE CREATED BY FUNCTION)

# Yolo command; yolo training model will be started with this command
yolo_command = "yolo task=classify mode=train model=yolov8m-cls.pt data={} imgsz=244 epochs={} batch={}".format(dataset_for_yolo, usual_epochs, usual_batch)

# Changing picture with random params
def change_pic(img_path):
    # Opening image
    img = Image.open(img_path)

    # Generate random values for contrast, brightness, sharpness, color, saturation, blur, and gamma correction
    contrast_factor = random.uniform(0.5, 1.5)
    brightness_factor = random.uniform(-0.2, 0.2)
    color_factor = random.uniform(0.5, 1.5)
    saturation_factor = random.uniform(0.5, 1.5)
    gamma_factor = random.uniform(0.5, 1.5)

    # Adjust params
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1 + brightness_factor)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(color_factor)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    img = ImageEnhance.Brightness(img).enhance(gamma_factor)

    # Convert to np format
    img = np.array(img, dtype=np.uint8)

    return img

# Cropping image by polygon
def crop_image_by_polygon(image_path, polygon_coords):
    # Read the image
    try:
        img = cv2.imread(image_path)
    except:
        img = np.uint8(image_path)

    # Find the bbox of the polygon
    x, y, w, h = cv2.boundingRect(np.array(polygon_coords))

    # Create a mask based on the polygon, within the bbox
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    adjusted_polygon = np.array(polygon_coords) - (x, y)
    cv2.fillPoly(mask, [adjusted_polygon], (255, 255, 255))

    # Crop the image using the bbox
    cropped_image = img[y:y+h, x:x+w]

    # Resize the mask to match the size of the cropped image
    mask = cv2.resize(mask, (cropped_image.shape[1], cropped_image.shape[0]))

    # Apply the mask directly to the cropped image
    result = cv2.bitwise_and(cropped_image, mask)

    return result

# Savin frames from videos
def makin_frames(videos_path, images_path, create_flag = True):
    # Flag
    if create_flag:
        for video_filename in os.listdir(videos_path):
            # Create the "images" folder, if it doesn't exist
            if not os.path.exists(images_path):
                os.makedirs(images_path)
            
            # Path for each video
            video_path = os.path.join(videos_path, video_filename)

            # Create a subfolder for each video inside the "images" folder
            video_name = os.path.splitext(video_filename)[0]
            video_images_folder = os.path.join(images_path, video_name)
            if not os.path.exists(video_images_folder):
                os.makedirs(video_images_folder)

            # Open the video file
            cap = cv2.VideoCapture(video_path)

            # Read and save each frame as an image
            frame_number = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save the frame as an image
                frame_filename = f"{frame_number}.jpg"
                frame_path = os.path.join(video_images_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_number += 1

            # Release the video capture object
            cap.release()
    else:
        print('Change "create_flag" in "makin_frames" function if you want to save frames from video.')

# Making dataset from frames
def makin_dataset(images_path, polygons_path, time_intervals_path, dataset_folder,
                  dataset_for_yolo_train, dataset_for_yolo_val, csv_file_path, dataset_split_size, create_flag=True):
    # Flag
    if create_flag:
        # Making a counter variable for yolo dataset (for dataset split)
        yolo_percent = dataset_split_size
        yolo_counter = 1 / yolo_percent
        pic_counter = 0

        # Making a dataset folder
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        # Opening jsons
        with open(polygons_path, 'r') as file:
            polygons = json.load(file)
        with open(time_intervals_path, 'r') as file:
            time_intervals = json.load(file)

        # Starting making data for csv file:
        # 2 columns: full path to a picture; result for a "foreign" object
        data = [['file_path', 'res']]

        # Making paths for images folders
        images_folders = [os.path.join(images_path, f) 
                          for f in os.listdir(images_path) 
                          if os.path.isdir(os.path.join(images_path, f))]

        for images_folder in images_folders:
            # List of lists:
            # 1st elem - full path to picture
            # 2nd elem - picture name or number of frame
            # 3rd elem - name of the folder, where picture contains, or the name of the video the frame is taken from
            file_paths = [[os.path.join(images_folder, f), f, os.path.basename(images_folder)] 
                          for f in os.listdir(images_folder) 
                          if os.path.isfile(os.path.join(images_folder, f))]

            # For each list of list (for each frame of all videos)
            for file_path in file_paths:
                # Described above (1st, 2nd and 3rd elems of list)
                pic_path = file_path[0]
                pic_name = file_path[1]
                folder_name = file_path[2]

                # Taking a numbers of frames where there is a "foreign" object from a dictionary (time_intervals json file)
                frame_numbers = time_intervals.get(folder_name + '.mp4', [])
                # Taking polygons for frames from a dictionary (polygons json file)
                polygon = polygons.get(folder_name + '.mp4', [])

                # Check if the frame number is within the specified range
                # 0 - presence of a "foreign" object
                # 1 - its absence
                answer = 0 if any(frame_number[0] <= int(pic_name[:-4]) <= frame_number[1] for frame_number in frame_numbers) else 1

                # Determine the number of operations (makeng repetition 3 times, if THERE IS "foreign" object
                # 2 times if THERE IS NOT)
                num_opers = 3 if answer == 0 else 2

                # Loop for number of operations
                for i in range(num_opers):
                    # Path for the result of pictures for datasets on which OUR models will be trained
                    output_folder_path = os.path.join(dataset_folder, folder_name)
                    if not os.path.exists(output_folder_path):
                        os.makedirs(output_folder_path)

                    # Apply operations based on the number of operations
                    if i == 0:
                        output_pic = crop_image_by_polygon(pic_path, polygon)
                    else:
                        output_pic = change_pic(pic_path)
                        output_pic = crop_image_by_polygon(output_pic, polygon)

                    # Savin picture
                    out_pic_path = os.path.join(output_folder_path, pic_name[:-4] + '_' + str(i) + '.jpg')
                    cv2.imwrite(out_pic_path, output_pic)

                    # Add a row in csv dataset information
                    data.append([out_pic_path, answer])

                    # Save images for YOLO
                    # Using pic_counter, we save every N-th image into a validation dataset
                    if answer == 0:
                        if pic_counter % yolo_counter == 0:
                            yolo_folder = os.path.join(dataset_for_yolo_val, '0')
                        else:
                            yolo_folder = os.path.join(dataset_for_yolo_train, '0')
                    else:
                        if pic_counter % yolo_counter == 0:
                            yolo_folder = os.path.join(dataset_for_yolo_val, '1')
                        else:
                            yolo_folder = os.path.join(dataset_for_yolo_train, '1')

                    if not os.path.exists(yolo_folder):
                        os.makedirs(yolo_folder)

                    out_pic_path_yolo = os.path.join(yolo_folder, folder_name + pic_name[:-4] + '_' + str(i) + '.jpg')
                    cv2.imwrite(out_pic_path_yolo, output_pic)
                    pic_counter += 1

        # Saving information about dataset
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    else:
        print('Change "create_flag" in "makin_dataset" function if you want to create a dataset from pictures.')

# Making TensorFlow model
def makin_model_tf(csv_file_path, dataset_split_size, tf_checkpoint_path, tf_fig1_path, tf_fig2_path, tf_fig3_path, tf_fig4_path, tf_fig5_path,
                   tf_text_output_path, main_epochs=20, batchsz=32, create_flag=True):
    # Flag
    if create_flag:
        # Console output will be in txt-file
        original_stdout = sys.stdout
        with open(tf_text_output_path, "w") as f:
            # Redirect standard output to the file
            sys.stdout = f

            # Opening our csv
            df = pd.read_csv(csv_file_path)

            # Convert 'res' column to strings, cuz the format is int, but it is not processed
            df['res'] = df['res'].astype(str)

            # Split the dataset into train and test sets
            # random_state is fixed for subsequent comparison of model quality when changing its parameters
            train_data, test_data = train_test_split(df, test_size=dataset_split_size, random_state=42)

            # Image data generator without data augmentation
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

            # Define the model
            model = models.Sequential()
            model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(32, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(1, activation='sigmoid'))

            # Compile the model with selected metrics
            model.compile(optimizer='adam', 
                          loss='binary_crossentropy', 
                          metrics=['accuracy', 
                                   Precision(), 
                                   Recall(), 
                                   F1Score(name='f1_score'), 
                                   'mae', 
                                   'mse', 
                                   AUC(name='roc_auc')])

            # Implement learning rate reduction
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                             factor=0.1, 
                                                             patience=3, 
                                                             min_lr=1e-6)

            checkpoint = callbacks.ModelCheckpoint(tf_checkpoint_path, save_best_only=True)

            # Train the model
            train_generator = datagen.flow_from_dataframe(train_data, 
                                                          directory=None, 
                                                          x_col='file_path', 
                                                          y_col='res', 
                                                          target_size=(224, 224), 
                                                          batch_size=batchsz, 
                                                          class_mode='binary')

            test_generator = datagen.flow_from_dataframe(test_data, 
                                                         directory=None, 
                                                         x_col='file_path', 
                                                         y_col='res', 
                                                         target_size=(224, 224), 
                                                         batch_size=batchsz, 
                                                         class_mode='binary')

            history = model.fit(train_generator, 
                                epochs=main_epochs, 
                                validation_data=test_generator, 
                                callbacks=[reduce_lr, checkpoint])

            # Savin graphs, showing metrics
            plt.figure(figsize=(8, 6))
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy over Epochs')
            plt.legend()
            plt.savefig(tf_fig3_path)
            plt.show()

            # Evaluate the model with additional metrics
            test_pred_probs = model.predict(test_generator)
            test_preds = (test_pred_probs > 0.5).astype(int)
            test_labels = test_generator.classes

            accuracy = accuracy_score(test_labels, test_preds)
            f1 = f1_score(test_labels, test_preds)
            roc_auc = roc_auc_score(test_labels, test_pred_probs)
            precision, recall, _ = precision_recall_curve(test_labels, test_pred_probs)
            mae = mean_absolute_error(test_labels, test_preds)
            mse = mean_squared_error(test_labels, test_preds)

            print(f'Test Accuracy: {accuracy}')
            print(f'Test Precision: {precision}')
            print(f'Test Recall: {recall}')
            print(f'Test F1 Score: {f1}')
            print(f'Test ROC AUC: {roc_auc}')
            print(f'Test MAE: {mae}')
            print(f'Test MSE: {mse}')

            # Plotting Precision-Recall curve
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['recall'], 
                     history.history['precision'], 
                     color='darkorange', 
                     lw=2, 
                     label='Precision-Recall curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.savefig(tf_fig1_path)
            plt.show()

            # Plotting ROC-AUC curve
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['roc_auc'], label='Training ROC AUC')
            plt.plot(history.history['val_roc_auc'], label='Validation ROC AUC')
            plt.xlabel('Epoch')
            plt.ylabel('ROC AUC')
            plt.title('Training and Validation ROC AUC over Epochs')
            plt.legend()
            plt.savefig(tf_fig2_path)
            plt.show()

            # Plotting MAE and MSE over epochs
            plt.figure(figsize=(12, 6))
            # MAE
            plt.subplot(1, 2, 1)
            plt.plot(history.history['mae'], label='MAE')
            plt.plot(history.history['val_mae'], label='Val MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.title('Mean Absolute Error (MAE) over Epochs')
            plt.legend(loc='upper right')
            # MSE
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mse'], label='MSE')
            plt.plot(history.history['val_mse'], label='Val MSE')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.title('Mean Squared Error (MSE) over Epochs')
            plt.legend(loc='upper right')
            ###
            plt.savefig(tf_fig4_path)

            # Plot F1 Score over epochs
            plt.figure(figsize=(8, 6))
            plt.plot(history.history['f1_score'], label='Training F1 Score')
            plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.title('Training and Validation F1 Score over Epochs')
            plt.legend()
            plt.savefig(tf_fig5_path)
            plt.show()

            # Saving console output in txt file
            sys.stdout = original_stdout
    else:
        print('Change "create_flag" in "makin_model_tf" function if you want to create a TensorFlow model.')

# Making PyTorch model
def makin_model_pt(csv_file_path, dataset_split_size, pt_checkpoint_path, pt_fig1_path, pt_text_output_path, main_epochs=20, batchsz=32, create_flag=True):
    # Flag
    if create_flag:
        # Console output will be in txt-file
        original_stdout = sys.stdout
        with open(pt_text_output_path, "w") as f:
            # Redirect standard output to the file
            sys.stdout = f

            # Opening our csv
            df = pd.read_csv(csv_file_path)

            # Split the data into training and testing sets
            # random_state is fixed for subsequent comparison of model quality when changing its parameters
            train_df, test_df = train_test_split(df, test_size=dataset_split_size, random_state=42)

            # Creating a class for our dataset
            class MyDataset(Dataset):
                def __init__(self, dataframe, transform=None):
                    self.dataframe = dataframe
                    self.transform = transform

                def __len__(self):
                    return len(self.dataframe)

                def __getitem__(self, idx):
                    img_path = self.dataframe.iloc[idx, 0]
                    image = Image.open(img_path).convert("RGB")
                    label = torch.tensor(self.dataframe.iloc[idx, 1], dtype=torch.long)

                    if self.transform:
                        image = self.transform(image)

                    return image, label

            # Resizing pictures to same size as in TF model
            transform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor()])

            # Tranformig our dataset for our class format
            train_dataset = MyDataset(train_df, transform=transform)
            test_dataset = MyDataset(test_df, transform=transform)

            # Making dataset loaders
            train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batchsz, shuffle=False)

            # Creatin model
            class MyCNN(nn.Module):
                def __init__(self):
                    super(MyCNN, self).__init__()
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

            model = MyCNN()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            epochs = main_epochs

            # Lists for metrics, needed for graphs
            train_losses, test_losses = [], []
            train_accuracies, test_accuracies = [], []
            train_maes, test_maes = [], []
            train_mses, test_mses = [], []
            train_precisions, test_precisions = [], []
            train_recalls, test_recalls = [], []
            train_f1s, test_f1s = [], []
            train_roc_aucs, test_roc_aucs = [], []

            # Train the model, grab the necessary values to fill out the lists for metrics
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                all_preds, all_labels = [], []

                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                # Filling out the train metrics
                train_losses.append(running_loss / len(train_loader))
                train_accuracy = accuracy_score(all_labels, all_preds)
                train_accuracies.append(train_accuracy)
                train_mae = mean_absolute_error(all_labels, all_preds)
                train_maes.append(train_mae)
                train_mse = mean_squared_error(all_labels, all_preds)
                train_mses.append(train_mse)
                train_precision, train_recall, _ = precision_recall_curve(all_labels, all_preds)
                train_precisions.append(train_precision)
                train_recalls.append(train_recall)
                train_f1 = f1_score(all_labels, all_preds)
                train_f1s.append(train_f1)
                train_roc_auc = roc_auc_score(all_labels, all_preds)
                train_roc_aucs.append(train_roc_auc)

                # Evaluation on the test set
                model.eval()
                test_loss = 0.0
                all_preds, all_labels = [], []

                # Same, but for test metrics
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()

                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                test_losses.append(test_loss / len(test_loader))
                test_accuracy = accuracy_score(all_labels, all_preds)
                test_accuracies.append(test_accuracy)
                test_mae = mean_absolute_error(all_labels, all_preds)
                test_maes.append(test_mae)
                test_mse = mean_squared_error(all_labels, all_preds)
                test_mses.append(test_mse)
                test_precision, test_recall, _ = precision_recall_curve(all_labels, all_preds)
                test_precisions.append(test_precision)
                test_recalls.append(test_recall)
                test_f1 = f1_score(all_labels, all_preds)
                test_f1s.append(test_f1)
                test_roc_auc = roc_auc_score(all_labels, all_preds)
                test_roc_aucs.append(test_roc_auc)

            # Save the model
            torch.save(model.state_dict(), pt_checkpoint_path)

            # Plot and save graphs
            plt.figure(figsize=(15, 10))

            # Accuracy
            plt.subplot(3, 2, 1)
            plt.plot(train_accuracies, label='Train Accuracy')
            plt.plot(test_accuracies, label='Test Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Train and Test Accuracy')

            # MAE
            plt.subplot(3, 2, 2)
            plt.plot(train_maes, label='Train MAE')
            plt.plot(test_maes, label='Test MAE')
            plt.xlabel('Epochs')
            plt.ylabel('MAE')
            plt.legend()
            plt.title('Train and Test MAE')

            # MSE
            plt.subplot(3, 2, 3)
            plt.plot(train_mses, label='Train MSE')
            plt.plot(test_mses, label='Test MSE')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.title('Train and Test MSE')

            # Precision-Recall
            plt.subplot(3, 2, 4)
            plt.plot(train_recalls, train_precisions, label='Train Precision-Recall')
            plt.plot(test_recalls, test_precisions, label='Test Precision-Recall')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall')
            plt.legend()

            # F1 Score
            plt.subplot(3, 2, 5)
            plt.plot(train_f1s, label='Train F1 Score')
            plt.plot(test_f1s, label='Test F1 Score')
            plt.xlabel('Epochs')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.title('Train and Test F1 Score')

            # ROC AUC
            plt.subplot(3, 2, 6)
            plt.plot(train_roc_aucs, label='Train ROC AUC')
            plt.plot(test_roc_aucs, label='Test ROC AUC')
            plt.xlabel('Epochs')
            plt.ylabel('ROC AUC')
            plt.legend()
            plt.title('Train and Test ROC AUC')
            ###
            plt.tight_layout()
            plt.savefig(pt_fig1_path)

            # Saving console output in txt file
            sys.stdout = original_stdout
    else:
        print('Change "create_flag" in "makin_model_pt" function if you want to create a PyTorch model.')

# Making PyTorch model; This is done this way due to a packages conflict
def makin_yolo_model(yolo_command, create_flag=True):
    # Flag
    if create_flag:
        def run_command(command):
            try:
                # Run the command and capture the output
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

                # Print the output
                print("Command output:", result.stdout)

                # Check if the command was successful (return code 0)
                if result.returncode == 0:
                    print("Command executed successfully.")
                else:
                    print("Command failed. Error:", result.stderr)

            except Exception as e:
                print("An error occurred:", str(e))

        # Run the command
        run_command(yolo_command)
    else:
        print('Change "create_flag" in "makin_yolo_model" function if you want to create a YoloV8 model.')


# Calling functions
makin_frames(videos_path, images_path, False)

makin_dataset(images_path, polygons_path, time_intervals_path, dataset_folder, 
              dataset_for_yolo_train, dataset_for_yolo_val, csv_file_path, dataset_split_size, False)

makin_model_tf(csv_file_path, dataset_split_size, tf_checkpoint_path, tf_fig1_path, tf_fig2_path, 
               tf_fig3_path, tf_fig4_path, tf_fig5_path, tf_text_output_path, 20, 32, False)

makin_model_pt(csv_file_path, dataset_split_size, pt_checkpoint_path, pt_fig1_path, pt_text_output_path, 20, 32, False)

makin_yolo_model(yolo_command, False)

