"""
Smart Glass for Blind People Using Machine Learning
----------------------------------------------------
Detects objects in a single input image using a YOLOv3 model trained on the
MS-COCO dataset, estimates each detected object's rough position in the
frame (e.g. "top left cup"), and converts the resulting description into
speech using Google Text-to-Speech (gTTS).

Usage:
    python src/detect_and_speak.py --image path/to/image.jpg --yolo models/ \
        --confidence 0.5 --threshold 0.3

Expected files inside the --yolo directory:
    coco.names      - class label names (one per line)
    yolov3.cfg       - YOLOv3 network configuration
    yolov3.weights   - pretrained YOLOv3 weights (see models/README.md)
"""

import argparse
import os
import time

import cv2
import numpy as np
from gtts import gTTS


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--image", required=True,
                     help="path to input image")
    ap.add_argument("-y", "--yolo", required=True,
                     help="base path to YOLO directory (weights/cfg/names)")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                     help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                     help="threshold used when applying non-maxima suppression")
    ap.add_argument("-o", "--output-audio", default="output.mp3",
                     help="path to save the generated speech audio file")
    return vars(ap.parse_args())


def load_yolo(yolo_dir):
    labels_path = os.path.sep.join([yolo_dir, "coco.names"])
    labels = open(labels_path).read().strip().split("\n")

    weights_path = os.path.sep.join([yolo_dir, "yolov3.weights"])
    config_path = os.path.sep.join([yolo_dir, "yolov3.cfg"])

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    return net, labels


def detect_objects(net, image, confidence_thresh, nms_thresh):
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                  swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    layer_outputs = net.forward(ln)
    end = time.time()
    print(f"[INFO] YOLO took {end - start:.6f} seconds")

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_thresh:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, nms_thresh)
    return boxes, confidences, class_ids, idxs, (W, H)


def describe_positions(boxes, class_ids, idxs, labels, frame_size):
    (W, H) = frame_size
    descriptions = []

    if len(idxs) == 0:
        return descriptions

    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        center_x = round((2 * x + w) / 2)
        center_y = round((2 * y + h) / 2)

        if center_x <= W / 3:
            w_pos = "left"
        elif center_x <= (W / 3 * 2):
            w_pos = "center"
        else:
            w_pos = "right"

        if center_y <= H / 3:
            h_pos = "top"
        elif center_y <= (H / 3 * 2):
            h_pos = "mid"
        else:
            h_pos = "bottom"

        descriptions.append(f"{h_pos} {w_pos} {labels[class_ids[i]]}")

    return descriptions


def speak(text, output_path):
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(output_path)
    print(f"[INFO] speech saved to {output_path}")


def main():
    args = parse_args()

    net, labels = load_yolo(args["yolo"])
    image = cv2.imread(args["image"])
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args['image']}")

    boxes, confidences, class_ids, idxs, frame_size = detect_objects(
        net, image, args["confidence"], args["threshold"]
    )
    descriptions = describe_positions(boxes, class_ids, idxs, labels, frame_size)

    if not descriptions:
        print("[INFO] no objects detected above the confidence threshold")
        return

    description_text = ", ".join(descriptions)
    print(f"[RESULT] {description_text}")
    speak(description_text, args["output_audio"])


if __name__ == "__main__":
    main()
