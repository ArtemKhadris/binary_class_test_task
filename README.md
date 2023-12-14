# This is a test task for binary image classification.
We have a set of videos, and two sets of json files for each video: in the first, the coordinates describe the area on the frame, in the second, the frame numbers when there is a foreign object in this area.

After processing the video, we have a dataset that consists of a frame (cropped area) and a description of this frame, whether there is a foreign object on it or not (0/1). Information about them is recorded in a csv file.

Since the images were not in equal quantities, the dataset was artificially expanded. The number of images without foreign objects was doubled (using filters), and the number of images with a foreign object was tripled. After this, the dataset became in almost equal proportions. 1890 images where there is a foreign object, and 1668 where there is none.
