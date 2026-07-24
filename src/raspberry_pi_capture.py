"""
Raspberry Pi hardware integration for the Smart Glass project.
----------------------------------------------------------------
This is the piece that ties the software pipeline (detect_and_speak.py) to
the physical hardware described in the project report:

    Push Button (GPIO) --press--> Webcam captures a frame
                                        |
                                        v
                              YOLOv3 detection + gTTS
                                        |
                                        v
                         Audio played through wired headphones
                         (via the Pi's built-in 3.5mm audio jack)

Wiring (matches the report's hardware section):
    - Webcam: any USB webcam plugged into one of the Pi's USB ports.
    - Push button: one leg to a GPIO pin (default: GPIO 17), the other leg
      to GND, using the Pi's internal pull-up resistor (no external
      resistor/breadboard strictly required, though the report's prototype
      used a breadboard + jumper wires for a cleaner build).
    - Headphones: plugged into the Pi's 3.5mm audio jack (this is exactly
      why the report chose wired headphones over USB/Bluetooth — it's a
      free port that doesn't compete with the webcam for USB).

Run on the Raspberry Pi (not on a laptop — this needs RPi.GPIO + picamera2
or a USB webcam accessible to OpenCV):

    python3 src/raspberry_pi_capture.py --yolo models/ --confidence 0.5

Press the button to capture + describe a frame. Ctrl+C to exit.
"""

import argparse
import os
import time

import cv2
import numpy as np
from gtts import gTTS

try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    # Lets you dry-run the capture/detection logic on a laptop without a Pi.
    ON_PI = False
    print("[WARN] RPi.GPIO not found — running in keyboard-trigger fallback mode "
          "(press ENTER instead of the physical button).")

BUTTON_PIN = 17  # BCM numbering — change if you wired the button elsewhere


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5)
    ap.add_argument("-t", "--threshold", type=float, default=0.3)
    ap.add_argument("--camera-index", type=int, default=0,
                     help="OpenCV camera index for the webcam (default 0)")
    ap.add_argument("--button-pin", type=int, default=BUTTON_PIN,
                     help="BCM GPIO pin the push button is wired to")
    return vars(ap.parse_args())


def setup_button(pin):
    if not ON_PI:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def wait_for_trigger(pin):
    """Blocks until the button is pressed (or ENTER, in fallback mode)."""
    if not ON_PI:
        input("Press ENTER to capture a frame (fallback mode)...")
        return
    print("Waiting for button press...")
    GPIO.wait_for_edge(pin, GPIO.FALLING)


def capture_frame(camera_index):
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}")
    # Give the sensor a moment to adjust exposure before grabbing the frame
    for _ in range(5):
        cam.read()
    ok, frame = cam.read()
    cam.release()
    if not ok:
        raise RuntimeError("Failed to capture frame from camera")
    return frame


def load_yolo(yolo_dir):
    labels_path = os.path.join(yolo_dir, "coco.names")
    labels = open(labels_path).read().strip().split("\n")
    weights_path = os.path.join(yolo_dir, "yolov3.weights")
    config_path = os.path.join(yolo_dir, "yolov3.cfg")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    return net, labels


def detect_objects(net, image, confidence_thresh, nms_thresh):
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

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
        w_pos = "left" if center_x <= W / 3 else "center" if center_x <= (W / 3 * 2) else "right"
        h_pos = "top" if center_y <= H / 3 else "mid" if center_y <= (H / 3 * 2) else "bottom"
        descriptions.append(f"{h_pos} {w_pos} {labels[class_ids[i]]}")
    return descriptions


def speak_through_headphones(text, path="output.mp3"):
    """
    Saves the TTS output and plays it back through whatever audio output
    the Pi is currently configured to use. Set the Pi to use the 3.5mm jack
    (not HDMI) with:  sudo raspi-config -> System Options -> Audio -> Headphones
    """
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(path)
    # mpg321 / omxplayer / mpg123 are common lightweight players on Raspberry Pi OS.
    # Install with: sudo apt-get install mpg321
    os.system(f"mpg321 -q {path}")


def main():
    args = parse_args()
    setup_button(args["button_pin"])
    net, labels = load_yolo(args["yolo"])

    print("Smart Glass ready.")
    try:
        while True:
            wait_for_trigger(args["button_pin"])
            print("[INFO] capturing frame...")
            frame = capture_frame(args["camera_index"])

            boxes, confidences, class_ids, idxs, frame_size = detect_objects(
                net, frame, args["confidence"], args["threshold"]
            )
            descriptions = describe_positions(boxes, class_ids, idxs, labels, frame_size)

            if descriptions:
                text = ", ".join(descriptions)
                print(f"[RESULT] {text}")
                speak_through_headphones(text)
            else:
                print("[INFO] no objects detected")
                speak_through_headphones("No objects detected")

            time.sleep(0.5)  # small debounce before listening for the next press
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        if ON_PI:
            GPIO.cleanup()


if __name__ == "__main__":
    main()
