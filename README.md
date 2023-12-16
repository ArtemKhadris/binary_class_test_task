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

## Preparing the dataset
After processing the video, we have a dataset that consists of a frame (cropped area) and a description of this frame, whether there is a foreign object on it or not (0/1). Information about them is recorded in a csv file.

Since the images were not in equal quantities, the dataset was artificially expanded (using filters (contrast, brightness, color, saturation, gamma) whose value was assigned randomly.). The number of images without foreign objects was doubled, and the number of images with a foreign object was tripled. After this, the dataset became in almost equal proportions. 1890 images where there is a foreign object, and 1668 where there is none.
