import cv2
import numpy as np

# Load pre-trained object detection model and the configuration file
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the COCO class labels the YOLO model was trained on
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layer names from the YOLO model
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Generate random colors for each class label for visualization
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load an image or video from the webcam (0) or provide a file path for a video
cap = cv2.VideoCapture(0)  # 0 for webcam, or replace with file path for video

while True:
    # Capture frame-by-frame from the video source
    ret, img = cap.read()
    
    # Get the dimensions of the image
    height, width, channels = img.shape

    # Preprocess the image for the YOLO model (resizing, scaling, mean subtraction)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to hold detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each detection from the YOLO model
    for out in outs:
        for detection in out:
            # Extract the class scores and identify the class with the highest score
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections by ensuring the confidence is above a threshold (0.5)
            if confidence > 0.5:
                # Calculate the coordinates of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append the bounding box coordinates, confidence, and class ID to respective lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to remove overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes and labels on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            # Draw the bounding box on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # Put text label above the bounding box
            cv2.putText(img, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image with detected objects
    cv2.imshow("Image", img)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
