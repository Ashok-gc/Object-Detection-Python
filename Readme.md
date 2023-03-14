# Real-time Object Detection using OpenCV and Deep Learning

This project demonstrates how to perform real-time object detection using OpenCV's deep neural network (DNN) module and a pre-trained model trained on the COCO dataset. The code can be used to detect and track a wide range of objects in real-time from a video stream or webcam.

# More Details 

This code uses OpenCV's deep neural network (DNN) module to perform object detection on a video stream from the user's webcam. It loads a pre-trained model called "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt" that is trained on the COCO dataset, which contains 80 different classes of objects.

The code reads in a list of class names from a file called "coco.names", then sets various parameters for the video capture device, including the resolution and brightness. It then enters an infinite loop that reads frames from the video stream, runs object detection on each frame, and draws bounding boxes and labels around the detected objects. The loop exits when the user presses the 's' key.

The object detection is performed using the detect() method of the cv2.dnn_DetectionModel class, which takes as input the image frame, a confidence threshold for detecting objects, and the size of the input images that the model expects. The method returns a list of class IDs, confidence scores, and bounding box coordinates for each detected object in the image.

The code then uses OpenCV functions to draw bounding boxes and labels around each detected object. The class name and confidence score are displayed on the image next to each bounding box.

Overall, this code demonstrates how to perform real-time object detection using a pre-trained deep learning model and a webcam or video stream.


# Getting Started

# Prerequisites
1. Python 3.6 or higher
2. OpenCV 4.5 or higher
3. A pre-trained object detection model (included in this repository)

# Installing
1. Clone the repository to your local machine.
2. Install OpenCV using pip: `pip install opencv-python`.
3. Download the pre-trained object detection model from the TensorFlow Object Detection API and extract the files to the project directory.

# Running the Code
To run the code, execute the following command in your terminal:

`python main.py`

This will launch the application and start detecting objects in real-time from your webcam.


# Customizing the Code
The code can be easily customized to detect different objects or improve the detection performance. Here are some suggestions:

1. Modify the `thres` variable to change the confidence threshold for detecting objects. Lower values will result in more detections but may also produce more false positives.
2. Modify the `classFile` variable to use a different list of class names. You can find other pre-trained models and their corresponding class names.
3. Modify the `configPath` and `weightsPath` variables to use a different pre-trained object detection model. You can find other pre-trained models.
4. Modify the `net.setInputSize()` method to change the input size of the model. Larger inputs will result in better detection accuracy but may also be slower to process.

# License
This project is licensed under the MIT License - see the LICENSE file for details.



