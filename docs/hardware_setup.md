# Hardware Integration Guide

This explains how the software pipeline (`src/detect_and_speak.py` /
`src/raspberry_pi_capture.py`) connects to the physical components described
in the project report.

## Components and their role in the software

| Component | Connects via | Role in code |
|---|---|---|
| Webcam | USB port | `cv2.VideoCapture(camera_index)` grabs a live frame on demand |
| Push button | GPIO pin (BCM 17 by default) + GND | `RPi.GPIO` waits for a falling edge (button press) to trigger a capture |
| Raspberry Pi 3 (B/B+) | — | Runs Python, OpenCV/YOLOv3 inference, and gTTS |
| Headphones (wired) | Pi's 3.5mm audio jack | Plays back the `mpg321`-rendered `.mp3` from gTTS |
| MicroSD card | — | Holds Raspberry Pi OS + this project + model weights |
| Breadboard / jumper wires | — | Physically wires the button to GPIO 17 and GND |

## Wiring the push button

```
Push Button                Raspberry Pi (BCM numbering)
   Leg 1  ─────────────────  GPIO 17 (physical pin 11)
   Leg 2  ─────────────────  GND      (physical pin 9, or any GND pin)
```

The script enables the Pi's **internal pull-up resistor**
(`GPIO.PUD_UP`), so no external resistor is strictly required — a direct
button-to-GPIO-and-GND wire (via breadboard + jumpers, as in the report) is
enough. The pin reads HIGH normally and drops LOW when pressed, which is
what `GPIO.wait_for_edge(pin, GPIO.FALLING)` listens for.

If you wire the button to a different GPIO pin, pass it in:

```bash
python3 src/raspberry_pi_capture.py --yolo models/ --button-pin 27
```

## Setting up the webcam

Plug the USB webcam into any of the Pi's USB ports. Confirm it's detected:

```bash
lsusb                 # should list the webcam
ls /dev/video*        # usually /dev/video0
```

`cv2.VideoCapture(0)` (the default `--camera-index`) maps to `/dev/video0`.
If you have multiple video devices, adjust `--camera-index` accordingly.

## Routing audio to the headphones (not HDMI)

By default, a Raspberry Pi connected to a monitor may route audio through
HDMI instead of the 3.5mm jack. Force it to the headphone jack:

```bash
sudo raspi-config
# System Options -> Audio -> Headphones
```

or directly:

```bash
amixer cset numid=3 1   # 1 = headphone jack, 0 = auto, 2 = HDMI
```

Install a lightweight command-line audio player (used by
`speak_through_headphones()` in the script) so the generated `.mp3` can be
played without a desktop environment:

```bash
sudo apt-get update
sudo apt-get install mpg321
```

## Putting it together

1. Wire the button to GPIO 17 and GND as above.
2. Plug in the webcam and headphones.
3. Copy this repo onto the Pi (or clone it directly there) and install
   dependencies (`pip install -r requirements.txt`, plus `RPi.GPIO` — this
   is preinstalled on Raspberry Pi OS).
4. Download `yolov3.weights` into `models/` (see `models/README.md`).
5. Run:

   ```bash
   python3 src/raspberry_pi_capture.py --yolo models/ --confidence 0.5
   ```

6. Press the button — the webcam captures a frame, YOLOv3 detects objects,
   and the spoken description plays through the headphones.

## Running the logic without a Pi (development/testing)

`raspberry_pi_capture.py` detects whether `RPi.GPIO` is importable. If it
isn't (e.g. you're testing on a laptop), it falls back to pressing **ENTER**
in the terminal instead of the physical button, so you can validate the
capture → detect → speak flow before deploying to the actual hardware. For
a single static image instead of a live webcam trigger, use
`src/detect_and_speak.py` directly.
