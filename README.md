# This is a test task for binary image classification.
## Task
We have a set of videos, and two sets of json files for each video: in the first, the coordinates describe the area on the frame, in the second, the frame numbers when there is a foreign object in this area.

The task is to “classify” the frame - whether there is a “foreign” object in the area indicated by the coordinates of the frame.

The input to the script is the path to the video and the path to the json file with coordinates. The output should be a json file containing information about the frames on which a “foreign” object was detected, in the following form: the number of the first frame on which a foreign object was noticed in the area, and the frame number when all “foreign” objects left the area . There can be several such sequences of frames. For example, for the result set ```[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]```, where 0 means there is a foreign object in the area and 1 means it is there no, the result should be ```[[3, 5], [9, 11]]```.

## Choosing a Solution Approach
In my opinion (but it is still worth testing experimentally), the best approach to solving this problem would be to find “foreign” objects in the frame, and if this object is in the area, then give a negative result (accordingly, if it is not there, then positive). But for this approach there is no dataset of “foreign” objects. Neither such a dataset nor a trained model exists in the public domain due to the narrow focus of the task. The dataset itself can be created using various 3D editors (creating models there, screenshots of various satisfying projections, etc.) or simply photographing them from different angles.
And this approach should be tested because there is no understanding of how quickly and accurately it can work.

Obviously, in the conditions of a test data task, that approach is inappropriate, so the approach of training a classifying model was chosen. Creating a dataset from areas where there are only two classes: area 0 (there is a foreign object in it) and area 1 (it is not there). 

This approach also has disadvantages. There may be an object in the area that is not “foreign”. The area may be blocked by other objects (a fly will land on the camera or a bird will land in front of the camera). There are also no shots in different weather conditions, different times of day, etc. Because of this, false classifications are possible. Also, training the model and testing (in real conditions) will take a lot of time and require powerful and compatible equipment. Therefore, small models are used in the training task.

## Model preparation
### Preparing the dataset
After processing the video, we have a dataset that consists of a frame (cropped area) and a description of this frame, whether there is a foreign object on it or not (0/1). Information about them is recorded in a csv file.

Since the images were not in equal quantities, the dataset was artificially expanded (using filters (contrast, brightness, color, saturation, gamma) whose value was assigned randomly.). The number of images without foreign objects was doubled, and the number of images with a foreign object was tripled. After this, the dataset became in almost equal proportions. 1890 images where there is a foreign object, and 1668 where there is none.

>All of the above is done in functions:
>
>```change_pic``` - Changing the contrast, brightness, color, saturation and gamma of the image input to the function. (path of the image is a parameter, result is an image in np.array format)
>
>```crop_image_by_polygon``` - Cropping a frame by coordinates. (path of the image or an image is np.array fromat and a path to a json file with a coordinates of polygon are a parameters, result is an image in np.array format)
>
>```makin_frames``` - Cutting all frames from a video and saving them along a specific path. (path to an inputed video, path to outputed folder with puctures and a flag is a parameters, result is a folder with frames of video)
>
>```makin_dataset``` - The function creates a folder with a ready-made dataset (path to a folder with images, path to a polygons json file, path to a time intervals of video (which frame is 0 and which is 1), path to an Yolo dataset and a path to a csv file are a parameters, result is two datasets: one dataset for your own models with csv file with paths to pictures and result (0/1), another for Yolo)

### About the selected metrics
- Accuracy - It calculates the ratio of correctly predicted instances to the total instances. Since we have previously leveled the dataset, it is advisable to use this metric. If we didn't align the dataset, it wouldn't be worth using.
- MAE and MSE - measure the average absolute and squared differences, respectively, between predicted and true values. More often used in regression models, but here it is used to evaluate the performance of the model. In this case it is not necessary.

Since the dataset is not fully balanced, it makes sense to use the following metrics:

- Precision and Recall - provide insights into the model's performance with respect to positive instances.
- F1 - harmonic mean of precision and recall, providing a balanced measure between the two.
- ROC AUC - metric that assesses the ability of the model to discriminate between positive and negative instances across different threshold values.

All described metrics are used in the models described below (except YOLO, it uses its own metrics).

Unused metrics:

- Specificity - The focus in binary classification is often on detecting positive instances (recall), making specificity less emphasized.
- F2 and F0.5 - The F1 score already balances precision and recall; additional variations might be unnecessary complexity unless a specific weighting is required.
- AUC PR - While ROC AUC is included, AUC-PR is not explicitly used. The trade-off between precision and recall is already assessed.
- Balanced Accuracy - The code uses overall accuracy, and balancing sensitivity and specificity might be considered indirectly through other metrics.
- MCC - Similar information is captured by precision, recall, and F1 score, making MCC redundant in this context.
- Log Loss - Binary cross-entropy is used, providing a simpler and closely related measure for this classification task.

### TensorFlow model
The first model is a convolutional neural network (CNN) for binary image classification using TensorFlow and Keras. The model is created in a ```makin_model_tf``` function in the ```makin_model.py```. The model architecture consists of three convolutional layers with max-pooling, followed by a flattened layer, a dense layer with ReLU activation and dropout, and a final dense layer with a sigmoid activation for binary classification. The model is compiled with the Adam optimizer and binary cross-entropy loss. Additional metrics such as accuracy, precision, recall, mean absolute error (MAE), and mean squared error (MSE) are tracked during training. The function includes data preparation steps, data generators, early stopping, learning rate reduction, and model checkpointing. After training, the model is evaluated on a test set, and various metrics are computed. Visualization includes plots of accuracy over epochs, precision-recall curve, ROC curve, and MAE/MSE over epochs. The function provides the option to redirect standard output to a file for logging, and its execution is controlled by a conditional flag.
- Graphs:
![figure1](https://github.com/ArtemKhadris/binary_class_test_task/assets/106828028/9ddd0a04-60f2-4e89-95ef-a614974d0661)
![figure2](https://github.com/ArtemKhadris/binary_class_test_task/assets/106828028/947daab7-19cc-4da5-aee5-07ab7acc6406)
![figure3](https://github.com/ArtemKhadris/binary_class_test_task/assets/106828028/e7a7d2e5-46ed-4a1c-a053-b325e992fe4a)
![figure4](https://github.com/ArtemKhadris/binary_class_test_task/assets/106828028/658308b8-3273-4e8a-b360-9e3377d73a29)
![figure5](https://github.com/ArtemKhadris/binary_class_test_task/assets/106828028/e43f21ab-a6a7-4238-a912-1beda217d37f)

- Text output is in /tf_models/output.txt
- Model is in /tf_models/best_model.keras

### PyTorch model
The second model is a binary image classification model in PyTorch, employing a CNN. The model is created in a ```makin_model_pt``` function in the same file as before. The architecture involves two convolutional layers, max-pooling, and two fully connected layers with ReLU activation. Training utilizes CrossEntropyLoss and the Adam optimizer, tracking metrics like accuracy. The script includes data handling, custom data loaders, and a CustomDataset class for loading and transforming images. The training loop iterates over epochs, capturing metrics such as training and testing losses, accuracies, MAE and MSE. Model checkpointing saves the state dictionary for future use. Visualizations encompass plots for training and testing losses, accuracies, MAE, and MSE over epochs, along with ROC-AUC and Precision-Recall curves for performance assessment. The code provides flexibility with optional logging to a file, controlled by the ```create_flag```.
- Graphs:
![figure1](https://github.com/ArtemKhadris/binary_class_test_task/assets/106828028/3af4a1d5-0c29-47af-aa23-1fbbe762af37)

- Text output in /pt_models/output.txt
- Model is in /pt_models/model.pth

### Yolo model
For third model will be used an YoloV8 model for classification images. The premade model is yolov8m-cls.pt (its medium size model)
- Graphs:
![results](https://github.com/ArtemKhadris/binary_class_test_task/assets/106828028/218fadee-1e07-48b2-81df-c020f4ceb409)
![confusion_matrix](https://github.com/ArtemKhadris/binary_class_test_task/assets/106828028/d72ffe99-7d63-4dfd-95d0-601011a5d075)
![confusion_matrix_normalized](https://github.com/ArtemKhadris/binary_class_test_task/assets/106828028/e68f0bef-d0a9-4829-8009-651a6321d9a3)

- All other is in /yolo_model

## Using models
### TensorFlow

### PyTorch

### Yolo
