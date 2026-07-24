# Model Files

This directory holds the YOLOv3 files used for object detection.

| File | Description | Committed to repo? |
|---|---|---|
| `yolov3.cfg` | YOLOv3 network **architecture** (layers, filters, anchors) | ✅ Yes — small text file |
| `coco.names` | 80 MS-COCO class labels | ✅ Yes — small text file |
| `yolov3.weights` | Pretrained **weights** (COCO-trained) | ❌ No — ~236 MB, excluded via `.gitignore` |

`yolov3.cfg` and `coco.names` are already in this folder and pushed with the
repo, since they're just plain text (architecture definition + label list),
not trained parameters. Only `yolov3.weights` needs a separate download —
GitHub blocks files over 100MB without Git LFS.

Download it and place it in this folder before running detection:

```bash
cd models
wget https://pjreddie.com/media/files/yolov3.weights
```

Full source links for reference:

| File | Source |
|---|---|
| `yolov3.cfg` | https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg |
| `yolov3.weights` | https://pjreddie.com/media/files/yolov3.weights |
| `coco.names` | https://github.com/pjreddie/darknet/blob/master/data/coco.names |

Quick download of all three (if you ever need to re-fetch cfg/names too):

```bash
cd models
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

If you'd rather commit the weights to GitHub, use
[Git LFS](https://git-lfs.com/):

```bash
git lfs install
git lfs track "*.weights"
git add .gitattributes
```
