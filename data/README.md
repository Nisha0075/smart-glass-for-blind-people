# Data

## `sample.jpg`

A real test frame used in the project (objects on a desk — cup, remote,
dining table), included so `notebooks/smart_glass_pipeline.ipynb` and
`src/detect_and_speak.py` have something to run detection on out of the box:

```bash
python src/detect_and_speak.py --image data/sample.jpg --yolo models/
```

## Training dataset (not included)

The YOLOv3 weights in this project were trained on **MS-COCO** (Common
Objects in Context) — Microsoft's large-scale object detection dataset:

- ~330K images, 200K+ labeled
- 1.5M object instances across 80 categories
- Annotations in JSON (`info`, `licenses`, `images`, `categories`,
  `annotations` sections)

The dataset itself (tens of GB) is not part of this repo. If you want to
retrain or fine-tune the model rather than use the pretrained weights linked
in `models/README.md`, download it from the official source:

- https://cocodataset.org/#download

Class names matching this project's model are already included at
`models/coco.names`.
