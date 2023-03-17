import cv2
import numpy as np

# Load the object detection model
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')

# Load the classes
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set the input and output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Open the camera
cap = cv2.VideoCapture(0)

# Process each frame
while True:
    # Capture the frame
    ret, frame = cap.read()

    # Resize the frame
    frame = cv2.resize(frame, (640, 480))

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (640, 480), (127.5, 127.5, 127.5), True, False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Show the detected objects
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * 640)
                center_y = int(detection[1] * 480)
                w = int(detection[2] * 640)
                h = int(detection[3] * 480)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
